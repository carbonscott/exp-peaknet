import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import zarr

from more_itertools import chunked

from functools  import partial
from contextlib import nullcontext

import math

from einops import rearrange, repeat

from peaknet.tensor_transforms import (
    InstanceNorm,
)
from peaknet.utils.checkpoint import Checkpoint

import gc

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -- Model
# --- Linear
class LinearAE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=False)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        self._init_weights()

    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)

        x_enc = self.encoder(x_flat)
        x_dec = self.decoder(x_enc)

        x_out = x_dec.view(x.shape)
        return x_out

    @torch.no_grad()
    def encode(self, x):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        return self.encoder(x_flat)

    def _init_weights(self):
        nn.init.orthogonal_(self.encoder.weight)  # Orthogonal initialization for better initial projections

        # Initialize decoder weights as transpose of encoder
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.t())

# --- Conv
class ConvAE(nn.Module):
    def __init__(self, input_size, latent_dim):
        super().__init__()

        # Calculate the number of downsample steps needed
        # We want to reduce 1920x1920 to roughly 8x8 before flattening
        self.input_size = input_size
        target_size = 8
        self.n_steps = int(math.log2(input_size[0] // target_size))

        # Calculate initial number of channels (increase gradually)
        init_channels = 16  # Start small and increase

        # Encoder
        encoder_layers = []
        in_channels = 1  # Assuming grayscale input
        curr_channels = init_channels

        # Use large initial kernel to capture more global structure
        encoder_layers.extend([
            nn.Conv2d(in_channels, curr_channels, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(8, curr_channels),  # GroupNorm as efficient LayerNorm alternative
            nn.ReLU(inplace=True)
        ])

        # Progressive downsampling with increasing channels
        for i in range(self.n_steps - 1):
            next_channels = curr_channels * 2
            encoder_layers.extend([
                nn.Conv2d(curr_channels, next_channels, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, next_channels),  # GroupNorm as efficient LayerNorm alternative
                nn.ReLU(inplace=True)
            ])
            curr_channels = next_channels

        self.encoder_conv = nn.Sequential(*encoder_layers)

        # Calculate the size after convolutions
        with torch.no_grad():
            test_input = torch.zeros(1, 1, *input_size)
            test_output = self.encoder_conv(test_input)
            self.conv_flat_dim = test_output.numel() // test_output.size(0)
            self.conv_spatial_shape = test_output.shape[1:]

        # Final linear layer to get to latent_dim
        self.encoder_linear = nn.Linear(self.conv_flat_dim, latent_dim, bias=False)

        # Decoder starts with linear layer
        self.decoder_linear = nn.Linear(latent_dim, self.conv_flat_dim, bias=False)

        # Decoder convolutions
        decoder_layers = []
        curr_channels = self.conv_spatial_shape[0]

        # Progressive upsampling
        for i in range(self.n_steps - 1):
            next_channels = curr_channels // 2
            decoder_layers.extend([
                nn.ConvTranspose2d(curr_channels, next_channels, kernel_size=3, stride=2,
                                 padding=1, output_padding=1),
                nn.GroupNorm(8, next_channels),
                nn.ReLU(inplace=True)
            ])
            curr_channels = next_channels

        # Final upsampling to original size
        decoder_layers.extend([
            nn.ConvTranspose2d(curr_channels, 1, kernel_size=7, stride=2,
                             padding=3, output_padding=1)
        ])

        self.decoder_conv = nn.Sequential(*decoder_layers)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with orthogonal initialization"""
        def init_ortho(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init_ortho)

        # Initialize decoder linear weights as transpose of encoder
        with torch.no_grad():
            self.decoder_linear.weight.copy_(self.encoder_linear.weight.t())

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded

    @torch.no_grad()
    def encode(self, x):
        # Convolutional encoding
        x = self.encoder_conv(x)
        # Flatten and project to latent space
        x = x.view(x.size(0), -1)
        return self.encoder_linear(x)

    def decode(self, z):
        # Project from latent space and reshape
        x = self.decoder_linear(z)
        x = x.view(x.size(0), *self.conv_spatial_shape)
        # Convolutional decoding
        return self.decoder_conv(x)

# --- VQ
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Initialize embeddings
        self.embeddings = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        nn.init.xavier_uniform_(self.embeddings)

        # Add EMA tracking for codebook updates
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = 0.99
        self._epsilon = 1e-5
        self.register_buffer('_ema_dw', torch.zeros((num_embeddings, embedding_dim)))

    def forward(self, x):
        # Flatten input
        flat_x = x.reshape(-1, self.embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_x**2, dim=1, keepdim=True)
                    + torch.sum(self.embeddings**2, dim=1)
                    - 2 * torch.matmul(flat_x, self.embeddings.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()

        # Quantize
        quantized = torch.matmul(encodings, self.embeddings)

        # Reshape quantized values to match input shape
        quantized = quantized.reshape(x.shape)

        # EMA codebook update
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._ema_w + \
                                   (1 - self._ema_w) * torch.sum(encodings, dim=0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self.num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_x)
            self._ema_dw = self._ema_w * self._ema_dw + (1 - self._ema_w) * dw

            self.embeddings.data = self._ema_dw / self._ema_cluster_size.unsqueeze(1)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight through estimator
        quantized = x + (quantized - x).detach()

        return quantized, loss, encoding_indices

# --- Transformers
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, use_flash=False):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        self.use_flash = use_flash

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        b, n, d = x.shape
        h = self.heads

        # Project to q, k, v
        qkv = rearrange(self.to_qkv(x), 'b n (three h d) -> three b h n d', three=3, h=h)
        q, k, v = qkv

        if self.use_flash:
            # Flash attention implementation
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
                attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        else:
            # Regular attention
            dots = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dim_head)
            attn = dots.softmax(dim=-1)
            attn_output = torch.matmul(attn, v)

        # Merge heads and project
        out = rearrange(attn_output, 'b h n d -> b n (h d)')
        return self.to_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, use_flash=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, heads, dim_head, use_flash=use_flash)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4, bias=False),
            nn.GELU(),
            nn.Linear(dim * 4, dim, bias=False)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads=8, dim_head=64, use_flash=False):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(dim, heads, dim_head, use_flash=use_flash) for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads=8, dim_head=64, use_flash=False):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(dim, heads, dim_head, use_flash=False) for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ViTAutoencoder(nn.Module):
    def __init__(
        self,
        image_size=(1920, 1920),
        patch_size=128,
        latent_dim=256,
        dim=1024,
        depth=2,
        heads=8,
        dim_head=64,
        use_flash=True,
        norm_pix=True  # Flag for patch normalization
    ):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.norm_pix = norm_pix

        # Calculate patches
        self.num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        patch_dim = 1 * patch_size * patch_size

        # Encoder components
        self.patch_embed = nn.Sequential(
            nn.Linear(patch_dim, dim, bias=True),
            nn.LayerNorm(dim)
        )

        ## # Learnable position embeddings
        ## self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
        # Fixed positional embedding
        pos_embed = self._get_sinusoidal_pos_embed(self.num_patches, dim)
        self.register_buffer('pos_embedding', pos_embed.unsqueeze(0))

        # Transformer encoder
        self.encoder = nn.ModuleList([
            nn.Sequential(
                TransformerBlock(dim, heads, dim_head, use_flash),
                nn.Dropout(0.1)
            ) for _ in range(depth)
        ])

        # Projection to latent space
        self.to_latent = nn.Sequential(
            nn.LayerNorm(dim * self.num_patches),
            nn.Linear(dim * self.num_patches, latent_dim, bias=True)
        )

        # Decoder components
        self.from_latent = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, dim * self.num_patches, bias=True)
        )

        # Transformer decoder
        self.decoder = nn.ModuleList([
            nn.Sequential(
                TransformerBlock(dim, heads, dim_head, use_flash),
                nn.Dropout(0.1)
            ) for _ in range(depth)
        ])

        # Patch reconstruction
        self.to_pixels = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, patch_dim, bias=True)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with orthogonal initialization"""
        def init_ortho(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init_ortho)

        # Get the last Linear layer from each Sequential
        to_latent_linear = [m for m in self.to_latent if isinstance(m, nn.Linear)][-1]
        from_latent_linear = [m for m in self.from_latent if isinstance(m, nn.Linear)][-1]

        # Initialize decoder projection as transpose of encoder
        with torch.no_grad():
            from_latent_linear.weight.copy_(to_latent_linear.weight.t())

    def _get_sinusoidal_pos_embed(self, num_pos, dim, max_period=10000):
        """
        Generate fixed sinusoidal position embeddings.

        Args:
            num_pos   : Number of positions (patches)
            dim       : Embedding dimension
            max_period: Maximum period for the sinusoidal functions. Controls the
                        range of wavelengths from 2π to max_period⋅2π. Higher values
                        create longer-range position sensitivity.

        Returns:
            torch.Tensor: Position embeddings of shape (num_pos, dim)
        """
        assert dim % 2 == 0, "Embedding dimension must be even"

        # Use half dimension for sin and half for cos
        omega = torch.arange(dim // 2, dtype=torch.float32) / (dim // 2 - 1)
        omega = 1. / (max_period**omega)  # geometric progression of wavelengths

        pos = torch.arange(num_pos, dtype=torch.float32)
        pos = pos.view(-1, 1)  # Shape: (num_pos, 1)
        omega = omega.view(1, -1)  # Shape: (1, dim//2)

        # Now when we multiply, broadcasting will work correctly
        angles = pos * omega  # Shape: (num_pos, dim//2)

        # Compute sin and cos embeddings
        pos_emb_sin = torch.sin(angles)  # Shape: (num_pos, dim//2)
        pos_emb_cos = torch.cos(angles)  # Shape: (num_pos, dim//2)

        # Concatenate to get final embeddings
        pos_emb = torch.cat([pos_emb_sin, pos_emb_cos], dim=1)  # Shape: (num_pos, dim)
        return pos_emb

    def normalize_patches(self, patches):
        """
        Normalize each patch independently
        patches: (B, N, P*P) where N is number of patches, P is patch size
        """
        if not self.norm_pix:
            return patches

        # Calculate mean and var over patch pixels
        mean = patches.mean(dim=-1, keepdim=True)
        var = patches.var(dim=-1, keepdim=True)
        patches = (patches - mean) / (var + 1e-6).sqrt()

        return patches

    def denormalize_patches(self, patches, orig_mean, orig_var):
        """
        Denormalize patches using stored statistics
        """
        if not self.norm_pix:
            return patches

        patches = patches * (orig_var + 1e-6).sqrt() + orig_mean
        return patches

    def patchify(self, x):
        """Convert image to patches"""
        return rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                        p1=self.patch_size, p2=self.patch_size)

    def unpatchify(self, patches):
        """Convert patches back to image"""
        h_patches = w_patches = int(math.sqrt(self.num_patches))
        return rearrange(patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                        h=h_patches, w=w_patches, p1=self.patch_size, p2=self.patch_size)

    def encode(self, x):
        # Convert image to patches
        patches = self.patchify(x)

        # Store original statistics for denormalization if needed
        if self.norm_pix:
            self.orig_mean = patches.mean(dim=-1, keepdim=True)
            self.orig_var = patches.var(dim=-1, keepdim=True)
            patches = self.normalize_patches(patches)

        # Patch embedding
        tokens = self.patch_embed(patches)

        # Add positional embedding
        x = tokens + self.pos_embedding

        # Transformer encoding
        for encoder_block in self.encoder:
            x = x + encoder_block(x)

        # Project to latent space
        latent = self.to_latent(rearrange(x, 'b n d -> b (n d)'))
        return latent

    def decode(self, z):
        # Project from latent space
        x = self.from_latent(z)
        x = rearrange(x, 'b (n d) -> b n d', n=self.num_patches)

        # Transformer decoding
        for decoder_block in self.decoder:
            x = x + decoder_block(x)

        # Reconstruct patches
        patches = self.to_pixels(x)

        # Denormalize if needed
        if self.norm_pix:
            patches = self.denormalize_patches(patches, self.orig_mean, self.orig_var)

        # Convert patches back to image
        return self.unpatchify(patches)

    def forward(self, x):
        latent = self.encode(x)
        return self.decode(latent)

## input_dim = 1920*1920
## latent_dim = 256
## model = LinearAE(input_dim, latent_dim)

num_codebook_vectors = 16
latent_dim = 256
commitment_cost = 0.25
vq = VectorQuantizer(
    num_embeddings=num_codebook_vectors,
    embedding_dim=latent_dim,
    commitment_cost=commitment_cost
)

model = ViTAutoencoder(
    image_size=(1920, 1920),
    patch_size=128,
    latent_dim=256,
    dim=1024,
    depth=2,
    use_flash=True,
    norm_pix=True,
)
logger.info(f"{sum(p.numel() for p in model.parameters())/1e6} M pamameters.")

# -- Loss
## criterion = nn.MSELoss()
## criterion = nn.L1Loss()

class LatentDiversityLoss(nn.Module):
    def __init__(self, min_distance=0.1):
        super().__init__()
        self.min_distance = min_distance

    def forward(self, z):
        batch_size = z.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0, device=z.device)

        # Normalize latent vectors
        z_normalized = F.normalize(z, p=2, dim=1)

        # Compute cosine similarity
        similarity = torch.mm(z_normalized, z_normalized.t())
        similarity = torch.clamp(similarity, -1.0, 1.0)

        # Mask out diagonal
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=z.device)
        similarity = similarity[mask].view(batch_size, -1)

        # Convert to distance and compute loss
        distance = 1.0 - similarity
        loss = F.relu(self.min_distance - distance).mean()

        return loss

class AdaptiveWeightedLoss(nn.Module):
    def __init__(self, kernel_size=15, weight_factor=2.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight_factor = weight_factor

    def compute_local_contrast(self, x):
        # Compute local mean using average pooling
        padding = self.kernel_size // 2
        local_mean = F.avg_pool2d(
            F.pad(x, (padding, padding, padding, padding), mode='reflect'),
            self.kernel_size,
            stride=1
        )

        # Compute local standard deviation
        local_var = F.avg_pool2d(
            F.pad((x - local_mean)**2, (padding, padding, padding, padding), mode='reflect'),
            self.kernel_size,
            stride=1
        )
        local_std = torch.sqrt(local_var + 1e-6)

        # Normalize to create weight map
        weight_map = 1.0 + self.weight_factor * (local_std / local_std.mean())
        return weight_map

    def forward(self, pred, target):
        # Compute base L1 loss
        base_loss = torch.abs(pred - target)

        # Compute weight map based on local contrast of target
        weight_map = self.compute_local_contrast(target)

        # Apply weights to loss
        weighted_loss = base_loss * weight_map

        return weighted_loss.mean()

class TotalLoss(nn.Module):
    def __init__(self, kernel_size, weight_factor, min_distance, div_weight):
        super().__init__()
        self.adaptive_criterion  = AdaptiveWeightedLoss(kernel_size, weight_factor)
        self.diversity_criterion = LatentDiversityLoss(min_distance)
        self.div_weight = div_weight

    def forward(self, batch, latent, batch_logits):
        rec_loss = self.adaptive_criterion(batch_logits, batch)
        div_loss = self.diversity_criterion(latent)
        total_loss = rec_loss + self.div_weight * div_loss
        return total_loss

kernel_size = 5
weight_factor = 10
min_distance = 100
div_weight = 0.1
criterion = TotalLoss(kernel_size, weight_factor, min_distance, div_weight)

# -- Optim
def cosine_decay(initial_lr: float, current_step: int, total_steps: int, final_lr: float = 0.0) -> float:
    # Ensure we don't go past total steps
    current_step = min(current_step, total_steps)

    # Calculate cosine decay
    cosine_decay = 0.5 * (1 + math.cos(math.pi * current_step / total_steps))

    # Calculate decayed learning rate
    decayed_lr = final_lr + (initial_lr - final_lr) * cosine_decay

    return decayed_lr

init_lr = 1e-3
weight_decay = 0
adam_beta1 = 0.9
adam_beta2 = 0.999
param_iter = model.parameters()
optim_arg_dict = dict(
    lr           = init_lr,
    weight_decay = weight_decay,
    betas        = (adam_beta1, adam_beta2),
)
optimizer = optim.AdamW(param_iter, **optim_arg_dict)

# -- Dataset
zarr_path = 'peaknet10k/mfxl1025422_r0313_peaknet.0031.zarr'
z_store = zarr.open(zarr_path, mode='r')
batch_size = 4

# -- Device
device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
if device != 'cpu': torch.cuda.set_device(device)
vq.to(device)
model.to(device)

# -- Misc
# --- Mixed precision
dist_dtype = 'bfloat16'
mixed_precision_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dist_dtype]

# --- Autocast
device_type = 'cuda' if 'cuda' in device else 'cpu'
autocast_context = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type = device_type, dtype = mixed_precision_dtype)

scaler_func = torch.cuda.amp.GradScaler
scaler = scaler_func(enabled=(dist_dtype == 'float16'))

# --- Grad clip
grad_clip = 1.0

# --- Normlization
normalizer = InstanceNorm(scales_variance=True)

# --- Checkpoint
checkpointer = Checkpoint()
path_chkpt = 'chkpt_ae_lite'

# --- Memory
def log_memory():
    if torch.cuda.is_available():
        logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        logger.info(f"CUDA memory cached: {torch.cuda.memory_cached() / 1e9:.2f} GB")

# -- Trainig loop
iteration_counter = 0
total_iterations  = 500
loss_min = float('inf')
while True:
    torch.cuda.synchronize()

    # Adjust learning rate
    lr = cosine_decay(init_lr, iteration_counter, total_iterations*0.5, init_lr*1e-1)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    batches = chunked(z_store['images'], batch_size)
    for enum_idx, batch in enumerate(batches):
        if enum_idx % 10 == 0:  # Log every epoch
            log_memory()

        # Turn list of arrays into a single array with the batch dim
        batch = torch.from_numpy(np.stack(batch)).unsqueeze(1).to(device, non_blocking=True)
        ## batch = normalizer(batch)

        batch[...,:10,:]=0
        batch[...,-10:,:]=0
        batch[...,:,:10]=0
        batch[...,:,-10:]=0

        # Fwd/Bwd
        with autocast_context:
            latent = model.encode(batch)
            z_quantized, vq_loss, _ = vq(latent)
            batch_logits = model.decode(z_quantized)
            recon_loss = criterion(batch, latent, batch_logits)
            loss = vq_loss + recon_loss
        scaler.scale(loss).backward()

        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # Update parameters
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad(set_to_none=True)

        # Log
        log_data = {
            "logevent"        : "LOSS:TRAIN",
            "iteration"       : iteration_counter,
            "lr"              : f"{lr:06f}",
            "grad_norm"       : f"{grad_norm:.6f}",
            "mean_train_loss" : f"{loss:.6f}",
        }
        log_msg = " | ".join([f"{k}={v}" for k, v in log_data.items()])
        logger.info(log_msg)

        iteration_counter += 1

    if iteration_counter > 0.2*total_iterations and loss_min > loss.item():
        loss_min = loss.item()
        checkpointer.save(0, model, optimizer, None, None, path_chkpt)
    if iteration_counter > total_iterations:
        break


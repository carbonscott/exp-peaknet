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

# VQVAE

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Initialize embeddings
        self.embeddings = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        nn.init.xavier_uniform_(self.embeddings)

        # EMA tracking for codebook updates
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = 0.99
        self._epsilon = 1e-5
        self.register_buffer('_ema_dw', torch.zeros((num_embeddings, embedding_dim)))

    def forward(self, x):
        # Reshape input to (batch * spatial_dim, embedding_dim)
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

        # EMA codebook update during training
        if self.training:
            # Update cluster size
            self._ema_cluster_size = self._ema_cluster_size * self._ema_w + \
                                   (1 - self._ema_w) * torch.sum(encodings, dim=0)

            # Laplace smoothing
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self.num_embeddings * self._epsilon) * n)

            # Update embeddings
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

        qkv = rearrange(self.to_qkv(x), 'b n (three h d) -> three b h n d', three=3, h=h)
        q, k, v = qkv

        if self.use_flash:
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
                out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        else:
            dots = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dim_head)
            attn = dots.softmax(dim=-1)
            out = torch.matmul(attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
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
        # Single residual connection per block
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class ViTAutoencoder(nn.Module):
    def __init__(
        self,
        image_size=(1920, 1920),
        patch_size=128,
        embedding_dim=256,
        dim=1024,
        depth=2,
        heads=8,
        dim_head=64,
        use_flash=True,
        norm_pix=True,
        eps=1e-6
    ):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.norm_pix = norm_pix
        self.eps = eps

        # Calculate patches
        self.h_patches = image_size[0] // patch_size
        self.w_patches = image_size[1] // patch_size
        self.num_patches = self.h_patches * self.w_patches
        patch_dim = patch_size * patch_size

        # Patch-level normalization parameters (learned)
        if norm_pix:
            self.patch_norm_scale = nn.Parameter(torch.ones(1, 1, patch_dim))
            self.patch_norm_bias = nn.Parameter(torch.zeros(1, 1, patch_dim))

        # Encoder components with layer norm
        self.patch_embed = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        # Position embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim) * 0.02)

        # Transformer blocks
        self.transformers = nn.ModuleList([
            TransformerBlock(dim, heads, dim_head, use_flash)
            for _ in range(depth)
        ])

        # Project to embedding space with normalization
        self.to_embedding = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

        # Decoder components
        self.from_embedding = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, dim),
            nn.LayerNorm(dim)
        )

        # Decoder transformer blocks
        self.decoder_transformers = nn.ModuleList([
            TransformerBlock(dim, heads, dim_head, use_flash)
            for _ in range(depth)
        ])

        # Final projection with denormalization
        self.to_pixels = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, patch_dim)
        )

        self._init_weights()

    def _init_weights(self):
        def init_ortho(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        self.apply(init_ortho)

        # Initialize position embeddings
        nn.init.normal_(self.pos_embedding, std=0.02)

    def patchify(self, x):
        return rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                        p1=self.patch_size, p2=self.patch_size)

    def unpatchify(self, patches):
        return rearrange(patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                        h=self.h_patches, w=self.w_patches,
                        p1=self.patch_size, p2=self.patch_size)

    def normalize_patches(self, patches):
        if not self.norm_pix:
            return patches

        # Calculate patch statistics
        patch_mean = patches.mean(dim=-1, keepdim=True)
        patch_var = patches.var(dim=-1, keepdim=True, unbiased=False)
        patches = (patches - patch_mean) / torch.sqrt(patch_var + self.eps)

        # Apply learned normalization parameters
        patches = patches * self.patch_norm_scale + self.patch_norm_bias
        return patches

    def denormalize_patches(self, patches, orig_mean, orig_var):
        if not self.norm_pix:
            return patches

        # Remove learned normalization
        patches = (patches - self.patch_norm_bias) / self.patch_norm_scale

        # Restore original scale
        patches = patches * torch.sqrt(orig_var + self.eps) + orig_mean
        return patches

    def encode(self, x):
        # Save original statistics for denormalization
        patches = self.patchify(x)
        if self.norm_pix:
            self.orig_mean = patches.mean(dim=-1, keepdim=True)
            self.orig_var = patches.var(dim=-1, keepdim=True, unbiased=False)
            patches = self.normalize_patches(patches)

        # Embed patches
        x = self.patch_embed(patches)

        # Add positional embeddings
        x = x + self.pos_embedding

        # Transform
        for transformer in self.transformers:
            x = transformer(x)

        # Project to embedding space
        spatial_latent = self.to_embedding(x)
        return spatial_latent

    def decode(self, z):
        # Project back to transformer dim
        x = self.from_embedding(z)

        # Transform
        for transformer in self.decoder_transformers:
            x = transformer(x)

        # Project to pixels
        patches = self.to_pixels(x)

        # Denormalize if needed
        if self.norm_pix:
            patches = self.denormalize_patches(patches, self.orig_mean, self.orig_var)

        # Reshape to image
        return self.unpatchify(patches)

    def forward(self, x):
        # Optional global normalization
        if hasattr(self, 'normalizer'):
            x = self.normalizer(x)

        spatial_latent = self.encode(x)
        return self.decode(spatial_latent), spatial_latent

class VQ_AE(nn.Module):
    def __init__(self, vq, ae):
        super().__init__()
        self.ae = ae
        self.vq = vq

    def forward(self, x):
        spatial_latent = self.ae.encode(x)
        z_quantized, vq_loss, indices = self.vq(spatial_latent)
        decoded = self.ae.decode(z_quantized)
        return decoded, z_quantized, vq_loss

# Model initialization
num_codebook_vectors = 512  # Increased codebook size
embedding_dim = 256
commitment_cost = 0.25
vq_model = VectorQuantizer(
    num_embeddings=num_codebook_vectors,
    embedding_dim=embedding_dim,
    commitment_cost=commitment_cost
)
logger.info(f"VQ: {sum(p.numel() for p in vq_model.parameters())/1e6} M pamameters.")

ae_model = ViTAutoencoder(
    image_size=(1920, 1920),
    patch_size=128,
    embedding_dim=embedding_dim,
    dim=1024,
    depth=4,
    use_flash=True,
    norm_pix=True,
)
logger.info(f"AE: {sum(p.numel() for p in ae_model.parameters())/1e6} M pamameters.")

model = VQ_AE(vq_model, ae_model)

# -- Loss
## criterion = nn.MSELoss()
## criterion = nn.L1Loss()

class SpatialLatentDiversityLoss(nn.Module):
    def __init__(self, min_distance=0.1):
        super().__init__()
        self.min_distance = min_distance

    def forward(self, z):
        # z shape: [batch_size, spatial_h * spatial_w, embedding_dim]
        batch_size = z.size(0)

        if batch_size <= 1:
            return torch.tensor(0.0, device=z.device)

        # Pool spatial dimensions to get single vector per image
        z_pooled = z.mean(dim=1)  # [batch_size, embedding_dim]

        # Normalize pooled vectors
        z_normalized = F.normalize(z_pooled, p=2, dim=1)

        # Compute similarity matrix
        similarity = torch.mm(z_normalized, z_normalized.t())
        similarity = torch.clamp(similarity, -1.0, 1.0)

        # Mask out self-similarity
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=z.device)
        similarity = similarity[mask].view(batch_size, -1)

        # Compute distance-based loss
        distance = 1.0 - similarity
        loss = F.relu(self.min_distance - distance).mean()

        return loss

class AdaptiveWeightedLoss(nn.Module):
    def __init__(self, kernel_size=15, weight_factor=1.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight_factor = weight_factor

    def compute_local_contrast(self, x):
        padding = self.kernel_size // 2
        local_mean = F.avg_pool2d(
            F.pad(x, (padding, padding, padding, padding), mode='reflect'),
            self.kernel_size,
            stride=1
        )
        local_var = F.avg_pool2d(
            F.pad((x - local_mean)**2, (padding, padding, padding, padding), mode='reflect'),
            self.kernel_size,
            stride=1
        )
        local_std = torch.sqrt(local_var + 1e-6)

        # Normalize and scale weights
        weight_map = 1.0 + self.weight_factor * (local_std / (local_std.mean() + 1e-6))
        return weight_map

    def forward(self, pred, target):
        # Basic reconstruction loss
        base_loss = torch.abs(pred - target)

        # Weight map based on local contrast
        weight_map = self.compute_local_contrast(target)

        # Combine
        weighted_loss = base_loss * weight_map
        return weighted_loss.mean()

class TotalLoss(nn.Module):
    def __init__(self, kernel_size=5, weight_factor=1.0, min_distance=0.1, div_weight=0.01):
        super().__init__()
        self.adaptive_criterion = AdaptiveWeightedLoss(kernel_size, weight_factor)
        self.diversity_criterion = SpatialLatentDiversityLoss(min_distance)
        self.div_weight = div_weight

    def forward(self, input_batch, latent, output_batch):
        rec_loss = self.adaptive_criterion(output_batch, input_batch)
        div_loss = self.diversity_criterion(latent)
        total_loss = rec_loss + self.div_weight * div_loss
        return total_loss

kernel_size = 5
weight_factor = 1
min_distance = 0.1
div_weight = 0.1
criterion = TotalLoss(kernel_size, weight_factor, min_distance, div_weight)

vq_weight = 0.5

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
optim_arg_dict = dict(
    lr           = init_lr,
    weight_decay = weight_decay,
    betas        = (adam_beta1, adam_beta2),
)
param_groups = [
    {
        'params': model.ae.parameters(),
        'lr': init_lr,
        'weight_decay': weight_decay,
        'betas': (adam_beta1, adam_beta2),
        'name': 'ae'
    },
    {
        'params': model.vq.parameters(),
        ## 'lr': init_lr * 0.1,
        'lr': init_lr,
        'weight_decay': weight_decay,
        'betas': (adam_beta1, adam_beta2),
        'name': 'vq'
    }
]
optimizer = optim.AdamW(param_groups)

# -- Dataset
zarr_path = 'peaknet10k/mfxl1025422_r0313_peaknet.0031.zarr'
z_store = zarr.open(zarr_path, mode='r')
batch_size = 4

# -- Device
device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
if device != 'cpu': torch.cuda.set_device(device)
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
path_chkpt_vq = 'chkpt_ae_lite_vq'

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
        if param_group['name'] == 'ae':
            param_group['lr'] = lr
        elif param_group['name'] == 'vq':
            ## param_group['lr'] = lr * 0.1
            param_group['lr'] = lr

    batches = chunked(z_store['images'], batch_size)
    for enum_idx, batch in enumerate(batches):
        if enum_idx % 10 == 0:  # Log every epoch
            log_memory()

        # Turn list of arrays into a single array with the batch dim
        batch = torch.from_numpy(np.stack(batch)).unsqueeze(1).to(device, non_blocking=True)
        batch = normalizer(batch)

        batch[...,:10,:]=0
        batch[...,-10:,:]=0
        batch[...,:,:10]=0
        batch[...,:,-10:]=0

        # Fwd/Bwd
        with autocast_context:
            batch_logits, z_quantized, vq_loss = model(batch)
            recon_loss = criterion(batch, z_quantized, batch_logits)
            loss = vq_weight * vq_loss + recon_loss
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


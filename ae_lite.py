import numpy as np

import torch
import torch.nn as nn
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
        patch_size=32,
        latent_dim=256,
        dim=512,
        depth=6,
        heads=8,
        dim_head=64,
        use_flash=False
    ):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size

        # Calculate patches
        self.num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        patch_dim = 1 * patch_size * patch_size

        # Encoder components
        self.patch_embed = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim, bias=False),
            nn.LayerNorm(dim)
        )

        # Learnable position embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))

        # Transformer encoder
        self.encoder = TransformerEncoder(dim, depth, heads, dim_head, use_flash)

        # Projection to latent space
        self.to_latent = nn.Linear(dim * self.num_patches, latent_dim, bias=False)

        # Decoder components
        self.from_latent = nn.Linear(latent_dim, dim * self.num_patches, bias=False)

        # Transformer decoder
        self.decoder = TransformerDecoder(dim, depth, heads, dim_head, use_flash)

        # Patch reconstruction
        self.to_pixels = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, patch_dim, bias=False)
        )

        self._init_weights()

    def _init_weights(self):
        def init_ortho(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)

        self.apply(init_ortho)

        # Initialize decoder projection as transpose of encoder
        with torch.no_grad():
            self.from_latent.weight.copy_(self.to_latent.weight.t())

    def encode(self, x):
        # Convert image to patches
        patches = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)

        # Patch embedding
        tokens = self.patch_embed(patches)

        # Add positional embedding
        tokens = tokens + self.pos_embedding

        # Transformer encoding
        encoded = self.encoder(tokens)

        # Project to latent space
        latent = self.to_latent(rearrange(encoded, 'b n d -> b (n d)'))
        return latent

    def decode(self, z):
        # Project from latent space
        x = self.from_latent(z)
        x = rearrange(x, 'b (n d) -> b n d', n=self.num_patches)

        # Transformer decoding
        x = self.decoder(x)

        # Reconstruct patches
        patches = self.to_pixels(x)

        # Convert patches back to image
        h_patches = w_patches = int(math.sqrt(self.num_patches))
        imgs = rearrange(patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                         h=h_patches, w=w_patches, p1=self.patch_size, p2=self.patch_size)
        return imgs

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

## input_dim = 1920*1920
## latent_dim = 256
## model = LinearAE(input_dim, latent_dim)

model = ViTAutoencoder(
    image_size=(1920, 1920),
    patch_size=64,
    latent_dim=128,
    dim=256,
    depth=4,
    use_flash=True,
)
logger.info(f"{sum(p.numel() for p in model.parameters())/1e6} M pamameters.")

# -- Loss
criterion = nn.MSELoss()

# -- Optim
lr = 1e-3
weight_decay = 0
adam_beta1 = 0.9
adam_beta2 = 0.999
param_iter = model.parameters()
optim_arg_dict = dict(
    lr           = lr,
    weight_decay = weight_decay,
    betas        = (adam_beta1, adam_beta2),
)
optimizer = optim.AdamW(param_iter, **optim_arg_dict)

# -- Dataset
zarr_path = 'peaknet10k/mfxl1025422_r0313_peaknet.0031.zarr'
z_store = zarr.open(zarr_path, mode='r')
batch_size = 64

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
normalizer = InstanceNorm()

# -- Trainig loop
iteration_counter = 0
while True:
    batches = chunked(z_store['images'], batch_size)
    for enum_idx, batch in enumerate(batches):
        # Turn list of arrays into a single array with the batch dim
        batch = torch.from_numpy(np.stack(batch)).unsqueeze(1).to(device)
        batch = normalizer(batch)

        # Fwd/Bwd
        with autocast_context:
            batch_logits = model(batch)
            loss = criterion(batch_logits, batch)
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
            "logevent"           : "LOSS:TRAIN",
            "iteration"          : iteration_counter,
            "grad_norm"          : f"{grad_norm:.6f}",
            "mean_train_loss"    : f"{loss:.6f}",
        }
        log_msg = " | ".join([f"{k}={v}" for k, v in log_data.items()])
        logger.info(log_msg)

        iteration_counter += 1

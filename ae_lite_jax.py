import jax
import jax.numpy as jnp
import jax.random as jr

import equinox as eqx

from functools import partial

import einops

from jaxtyping import Array, Float, Int, PyTree, PRNGKeyArray
from typing import Tuple, Optional

import math

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MultiHeadAttention(eqx.Module):
    dim: int
    heads: int
    dim_head: int
    to_qkv: eqx.nn.Linear
    to_out: eqx.nn.Linear
    use_flash: bool = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        use_flash: bool,
        *,
        key: PRNGKeyArray
    ):
        keys = jr.split(key, num=2)

        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.use_flash = use_flash

        inner_dim = heads * dim_head

        self.to_qkv = eqx.nn.Linear(dim, 3 * inner_dim, use_bias=False, key=keys[0])
        self.to_out = eqx.nn.Linear(inner_dim, dim, use_bias=False, key=keys[1])

    @eqx.filter_jit
    def __call__(self, x: Float[Array, "t e"]):
        """ Pretend input doesn't have the batch dimension
        """
        # Emit q, k, v from x
        q, k, v = einops.rearrange(
            jax.vmap(self.to_qkv)(x),
            't (three h d) -> three h t d',
            three=3,
            h=self.heads
        )

        # Scaled dot product
        if self.use_flash:
            v_attn = jax.nn.scaled_dot_product_attention(
                q, k, v,
                scale=1.0/jnp.sqrt(self.dim_head)
            )
        else:
            w = einops.einsum(
                q, k,
                'h t_q d, h t_k d -> h t_q t_k'
            )
            w /= self.dim_head**0.5
            w = jax.nn.softmax(w, axis=-1) # Each q over all k should be interpreted as prob distribution
            v_attn = einops.einsum(
                w, v,
                'h t_q t_k, h t_k d -> h t_q d'
            )

            # Merge heads
            v_merged = einops.rearrange(
                v_attn,
                'h t d -> t (h d)'
            )

        return jax.vmap(self.to_out)(v_merged)

class TransformerBlock(eqx.Module):
    ln1: eqx.nn.LayerNorm
    ln2: eqx.nn.LayerNorm
    attn: MultiHeadAttention
    ff: eqx.nn.Sequential

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        use_flash: bool,
        *,
        key: PRNGKeyArray
    ):
        keys = jr.split(key, num=3)

        self.ln1 = eqx.nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, heads, dim_head, use_flash, key=keys[0])
        self.ln2 = eqx.nn.LayerNorm(dim)
        self.ff = eqx.nn.Sequential([
            eqx.nn.Linear(dim, 4 * dim, use_bias=False, key=keys[1]),
            eqx.nn.Lambda(jax.nn.gelu),
            eqx.nn.Linear(4 * dim, dim, use_bias=False, key=keys[2]),
        ])

    @eqx.filter_jit
    def __call__(self, x: Float[Array, "t e"]):
        x = x + self.attn(jax.vmap(self.ln1)(x))
        x = x + jax.vmap(self.ff)(jax.vmap(self.ln2)(x))
        return x

class TransformerEncoder(eqx.Module):
    layers: list[TransformerBlock]

    def __init__(
        self,
        depth: int,
        dim: int,
        heads: int,
        dim_head: int,
        use_flash: bool,
        *,
        key: PRNGKeyArray
    ):
        keys = jr.split(key, num=1)
        self.layers = [
            TransformerBlock(dim, heads, dim_head, use_flash, key=keys[0]) for _ in range(depth)
        ]

    @eqx.filter_jit
    def __call__(self, x: Float[Array, "t e"]):
        for layer in self.layers:
            x = layer(x)
        return x

class ViTAutoencoder(eqx.Module):
    patch_size: int
    image_size: int
    norm_pix: bool
    num_patches: int
    patch_embed: eqx.nn.Sequential
    pos_embed: jnp.ndarray
    encoder: list[TransformerBlock]
    to_latent: eqx.nn.Sequential
    from_latent: eqx.nn.Sequential
    decoder: list[TransformerBlock]
    to_pixel: eqx.nn.Sequential

    def __init__(
        self,
        image_size: Tuple[int],
        patch_size: int,
        latent_dim: int,
        depth: int,
        dim: int,
        heads: int,
        dim_head: int,
        use_flash: bool,
        norm_pix: bool,
        *,
        key: PRNGKeyArray
    ):
        keys = jr.split(key, num=6)

        self.patch_size = patch_size
        self.image_size = image_size
        self.norm_pix   = norm_pix

        # Calculate patches
        self.num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        patch_dim = 1 * patch_size * patch_size

        # -- Encoder
        # Patch embedding
        self.patch_embed = eqx.nn.Sequential([
            eqx.nn.Linear(patch_dim, dim, use_bias=False, key=keys[0]),
            eqx.nn.LayerNorm(dim),
        ])

        # Positional embedding
        # Use jax.lax.stop_gradient to mark it un-trainable
        # See more: https://docs.kidger.site/equinox/faq/#how-to-mark-arrays-as-non-trainable-like-pytorchs-buffers
        self.pos_embed = self.get_sinusoidal_pos_embed(self.num_patches, dim)

        self.encoder = [
            TransformerBlock(dim, heads, dim_head, use_flash, key=keys[1]) for _ in range(depth)
        ]

        # -- Latent code
        self.to_latent = eqx.nn.Sequential([
            eqx.nn.LayerNorm(dim * self.num_patches),
            eqx.nn.Linear(dim * self.num_patches, latent_dim, use_bias=True, key=keys[2])
        ])
        self.from_latent = eqx.nn.Sequential([
            eqx.nn.LayerNorm(latent_dim),
            eqx.nn.Linear(latent_dim, dim * self.num_patches, use_bias=True, key=keys[3])
        ])

        # -- Decoder
        self.decoder = [
            TransformerBlock(dim, heads, dim_head, use_flash, key=keys[4]) for _ in range(depth)
        ]

        # -- Embedding to pixel
        self.to_pixel = eqx.nn.Sequential([
            eqx.nn.LayerNorm(dim),
            eqx.nn.Linear(dim, patch_dim, use_bias=True, key=keys[5])
        ])

    @eqx.filter_jit
    def get_sinusoidal_pos_embed(
        self,
        num_pos: int,
        dim: int,
        max_period: int=10000
    ):
        """
        Generate fixed sinusoidal position embeddings.
        Returns:
            Array: Position embeddings of shape (num_pos, dim)
        """
        # Use half dimension for sin and half for cos
        omega = jnp.arange(dim // 2) / (dim // 2 - 1)
        omega = 1. / (max_period**omega)  # geometric progression of wavelengths
        pos = jnp.arange(num_pos)
        angles = einops.einsum(pos, omega, 'num_pos, half_dim -> num_pos half_dim')

        # Compute sin and cos embeddings
        pos_emb_sin = jnp.sin(angles)  # Shape: (num_pos, dim//2)
        pos_emb_cos = jnp.cos(angles)  # Shape: (num_pos, dim//2)

        # Concatenate to get final embeddings
        pos_emb = einops.pack([pos_emb_sin, pos_emb_cos], 'num_pos *')[0] # [num_pos, dim]
                                                                          # einops.pack result is at position 0
        return pos_emb

    @eqx.filter_jit
    def patchify(self, x: Float[Array, "c h w"]):
        return einops.rearrange(x, 'c (h p1) (w p2) -> (h w) (p1 p2 c)',
                                p1=self.patch_size, p2=self.patch_size)

    @eqx.filter_jit
    def unpatchify(self, x: Float[Array, "h_w p1_p2_c"]):
        h_patches = w_patches = int(math.sqrt(self.num_patches))
        return einops.rearrange(x, '(h w) (p1 p2 c) -> c (h p1) (w p2)',
                                p1=self.patch_size, p2=self.patch_size, h=h_patches)

    @eqx.filter_jit
    def normalize_patches(self, patches: Float[Array, "num_patch patch_size_squared"]):
        def _norm(patches):
            mean = patches.mean(axis=-1, keepdims=True)
            var = patches.var(axis=-1, keepdims=True)
            patches = (patches - mean) / (var + 1e-6)**0.5
            return patches
        def _identity(patches):
            return patches
        return jax.lax.cond(self.norm_pix, _norm, _identity, patches)

    @eqx.filter_jit
    def denormalize_patches(self, patches: Float[Array, "num_patch patch_size_squared"], orig_mean: Float[Array, "num_patch 1"], orig_var: Float[Array, "num_patch 1"]):
        return jax.lax.cond(
            self.norm_pix,
            lambda patches: patches * (orig_var + 1e-6)**0.5 + orig_mean,
            lambda patches: patches,
            patches
        )

    @eqx.filter_jit
    def encode(self, x: Float[Array, "c h w"]):
        patches = self.patchify(x)

        # Need original statistics for denormalization
        orig_mean = patches.mean(axis=-1, keepdims=True)
        orig_var = patches.var(axis=-1, keepdims=True)
        patches = self.normalize_patches(patches)

        tokens = jax.vmap(self.patch_embed)(patches) + jax.lax.stop_gradient(self.pos_embed) # vmap handles patch dim
                                                                                             # outer vmap handles batch dim
        for encoder_block in self.encoder:
            tokens = tokens + encoder_block(tokens)
        tokens = einops.rearrange(tokens, 't e -> (t e)')
        latent = self.to_latent(tokens)

        return latent, orig_mean, orig_var

    @eqx.filter_jit
    def decode(self, z: Float[Array, "latent_dim"], orig_mean: Float[Array, "num_patch 1"], orig_var: Float[Array, "num_patch 1"]):
        tokens = self.from_latent(z)
        tokens = einops.rearrange(tokens, '(t e) -> t e', t=self.num_patches)
        for decoder_block in self.decoder:
            tokens = tokens + decoder_block(tokens)
        patches = jax.vmap(self.to_pixel)(tokens)
        patches = self.denormalize_patches(patches, orig_mean, orig_var)
        recon = self.unpatchify(patches)
        return recon

    @eqx.filter_jit
    def __call__(self, x: Float[Array, "c h w"]):
        latent, orig_mean, orig_var = self.encode(x)
        recon  = self.decode(latent, orig_mean, orig_var)
        return recon

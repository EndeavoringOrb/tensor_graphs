# examples/flux_klein_4b.py
"""
FLUX.2 Klein 4B Image Generation on tensor_graphs

Implements the complete diffusion transformer pipeline:
- Text encoding (Qwen3-4B)
- Latent diffusion (Transformer)
- Image decoding (VAE)

Reference: flux.c, flux_transformer.c, flux_vae.c
"""

import os
import math
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass

import numpy as np
from PIL import Image
from tqdm import tqdm

from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ir.graph import GraphBuilder
from tensor_graphs.session import GraphSession
from tensor_graphs.weights import SafetensorsSource


def encode_image(self, image: np.ndarray) -> np.ndarray:
    """
    Encode image to latent using VAE encoder.

    Args:
        image: [H, W, 3] RGB image in [0, 255]

    Returns:
        [1, 128, H/16, W/16] latent
    """
    print("Encoding image...")

    # Normalize to [-1, 1]
    img_tensor = (image.astype(np.float32) / 255.0) * 2.0 - 1.0  # [H, W, 3]
    img_tensor = img_tensor.transpose(2, 0, 1)  # [3, H, W]
    img_tensor = img_tensor[np.newaxis, :, :, :]  # [1, 3, H, W]

    # Build encoder if needed
    self.build_vae_encoder()

    # Run encoder
    latent = self.vae_encoder_session.run({"image": img_tensor})

    return latent


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class FluxConfig:
    """FLUX.2 Klein 4B configuration."""

    # Transformer
    hidden_size: int = 3072
    num_heads: int = 24
    head_dim: int = 128
    mlp_hidden: int = 9216
    num_double_layers: int = 5
    num_single_layers: int = 20
    rope_theta: float = 2000.0
    axis_dim: int = 32  # head_dim / 4

    # VAE
    vae_channels: int = 128
    vae_base_ch: int = 128
    vae_z_channels: int = 32

    # Text Encoder (Qwen3-4B)
    text_dim: int = 7680
    text_vocab_size: int = 151936
    text_hidden_size: int = 2560
    text_num_layers: int = 36
    text_num_heads: int = 32
    text_num_kv_heads: int = 8
    text_head_dim: int = 128
    text_max_seq: int = 4

    # Latent
    latent_channels: int = 128
    patch_size: int = 2

    # Sampling
    num_steps_distilled: int = 4
    num_steps_base: int = 50


# ============================================================================
# Flux Graph Builder
# ============================================================================


class FluxBuilder(GraphBuilder):
    """Graph builder for FLUX.2 architecture."""

    def __init__(self, cfg: FluxConfig):
        super().__init__()
        self.cfg = cfg
        self.eps = self.const([1e-6])

    # --- Fused Operations ---

    def silu(self, x: TensorNode) -> TensorNode:
        """SiLU activation: x * sigmoid(x)"""
        return TensorNode("SiLU", x.dtype, [x], name=self._next_name("silu"))

    def gelu_tanh(self, x: TensorNode) -> TensorNode:
        """GELU with tanh approximation."""
        # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        x3 = self.mul(x, self.mul(x, x))
        inner = self.add(x, self.mul(self.const([0.044715]), x3))
        inner = self.mul(self.const([math.sqrt(2.0 / math.pi)]), inner)
        tanh_out = self.tanh(inner)
        return self.mul(
            self.const([0.5]), self.mul(x, self.add(self.const([1.0]), tanh_out))
        )

    def rms_norm(
        self, x: TensorNode, scale: TensorNode, axis: int = -1, eps: float = 1e-6
    ) -> TensorNode:
        """RMSNorm: x * rsqrt(mean(x^2) + eps) * (1 + scale)"""
        # Use the registered RMSNorm fused op
        eps_node = self.const([eps])
        return TensorNode(
            "RMSNorm",
            x.dtype,
            [x, scale, eps_node],
            name=self._next_name("rmsnorm"),
            attrs={"axis": axis},
        )

    def group_norm(
        self,
        x: TensorNode,
        weight: TensorNode,
        bias: TensorNode,
        num_groups: int,
        eps: float = 1e-6,
    ) -> TensorNode:
        """GroupNorm for VAE."""
        return TensorNode(
            "GroupNorm",
            x.dtype,
            [x, weight, bias],
            name=self._next_name("groupnorm"),
            attrs={"num_groups": num_groups, "eps": eps},
        )

    def rope_2d(self, x: TensorNode, cos: TensorNode, sin: TensorNode) -> TensorNode:
        """2D Rotary Position Embedding."""
        return TensorNode("RoPE", x.dtype, [x, cos, sin], name=self._next_name("rope"))

    def attention(
        self,
        q: TensorNode,
        k: TensorNode,
        v: TensorNode,
        mask: Optional[TensorNode] = None,
    ) -> TensorNode:
        """Scaled dot-product attention."""
        # Q @ K^T / sqrt(d_k)
        d_k = self.cfg.head_dim
        scale = 1.0 / math.sqrt(d_k)

        # k_transposed = permute(k, [0, 1, 3, 2]) for [B, H, L, D]
        k_t = self.permute(k, [0, 1, 3, 2])
        scores = self.dot(q, k_t)
        scores = self.mul(scores, self.const([scale]))

        if mask is not None:
            scores = self.add(scores, mask)

        # Softmax
        probs = TensorNode(
            "Softmax",
            scores.dtype,
            [scores],
            name=self._next_name("softmax"),
            attrs={"axis": -1},
        )

        # @ V
        return self.dot(probs, v)

    def ada_ln_modulation(
        self, t_emb: TensorNode, weight: TensorNode, hidden_size: int
    ) -> Tuple[TensorNode, TensorNode]:
        """AdaLN modulation: shift, scale = linear(silu(t_emb))"""
        t_silu = self.silu(t_emb)
        mod = self.dot(t_silu, self.permute(weight, [1, 0]))
        shift = self.slice(mod, [0], [hidden_size])
        scale = self.slice(mod, [hidden_size], [hidden_size * 2])
        return shift, scale

    def ada_ln_single_modulation(
        self, t_emb: TensorNode, weight: TensorNode, hidden_size: int
    ) -> Tuple[TensorNode, TensorNode, TensorNode]:
        """Single block modulation: shift, scale, gate"""
        t_silu = self.silu(t_emb)
        mod = self.dot(t_silu, self.permute(weight, [1, 0]))
        shift = self.slice(mod, [0], [hidden_size])
        scale = self.slice(mod, [hidden_size], [hidden_size * 2])
        gate = self.slice(mod, [hidden_size * 2], [hidden_size * 3])
        return shift, scale, gate

    # --- Transformer Components ---

    def timestep_embedding(
        self, t: TensorNode, dim: int, max_period: int = 10000
    ) -> TensorNode:
        """Sinusoidal timestep embedding."""
        # Create frequency bands: exp(-log(max_period) * arange(0, dim, 2) / dim)
        half = dim // 2
        freqs = self.arange(self.const(0), self.const(half), self.const(1))
        freqs = self.mul(freqs, self.const([-math.log(max_period) / dim]))
        freqs = self.exp(freqs)

        # args = t * freqs
        args = self.mul(t, freqs)

        # embedding = cat([cos(args), sin(args)])
        cos_emb = self.cos(args)
        sin_emb = self.sin(args)
        return self.concat([cos_emb, sin_emb], axis=0)

    def time_embedder(
        self,
        t: TensorNode,
        fc1_weight: TensorNode,
        fc2_weight: TensorNode,
        dim: int = 256,
        hidden: int = 3072,
    ) -> TensorNode:
        """Full time embedding: sincos -> fc1 -> silu -> fc2"""
        t_sincos = self.timestep_embedding(t, dim)

        # fc1: [hidden, 512]
        h = self.dot(t_sincos, self.permute(fc1_weight, [1, 0]))
        h = self.silu(h)

        # fc2: [hidden, hidden]
        return self.dot(h, self.permute(fc2_weight, [1, 0]))

    def qk_norm(
        self, x: TensorNode, norm_weight: TensorNode, head_dim: int
    ) -> TensorNode:
        """QK normalization (RMSNorm per head)."""
        # Reshape to [B, H, L, D] then normalize per head
        # For simplicity, we use standard RMSNorm
        return self.rms_norm(x, norm_weight)

    def compute_modulation(
        self, t_emb: TensorNode, weight: TensorNode, hidden: int, chunks: int
    ) -> list[TensorNode]:
        """
        Compute shared modulation parameters.
        C Code: iris_linear_nobias(..., t_emb_silu, adaln_weight, ...)
        """
        # t_emb input here is expected to be already SiLU'd if following C code exactly,
        # but FluxTransformer passes raw t_emb. C code does: t_emb_silu = silu(t_emb).
        # We will handle the SiLU inside FluxTransformer._build_graph before calling this.

        # weight shape: [hidden * chunks, hidden]
        # output: [1, hidden * chunks]
        mod = self.dot(t_emb, self.permute(weight, [1, 0]))

        # Split into chunks of size 'hidden'
        results = []
        for i in range(chunks):
            start = i * hidden
            end = (i + 1) * hidden
            results.append(self.slice(mod, [start], [end]))
        return results

    def double_block(
        self,
        img_hidden: TensorNode,
        txt_hidden: TensorNode,
        img_seq_len: TensorNode,
        txt_seq_len: TensorNode,
        # Pre-computed modulation tensors (passed from FluxTransformer)
        img_mods: tuple[TensorNode, TensorNode, TensorNode],  # shift, scale, gate
        txt_mods: tuple[TensorNode, TensorNode, TensorNode],
        weights: Dict[str, TensorNode],
        cos: TensorNode,
        sin: TensorNode,
    ) -> Tuple[TensorNode, TensorNode]:
        """Double stream block with shared modulation inputs."""
        cfg = self.cfg
        hidden = cfg.hidden_size
        head_dim = cfg.head_dim
        num_heads = cfg.num_heads

        img_shift, img_scale, img_gate = img_mods
        txt_shift, txt_scale, txt_gate = txt_mods

        # --- Image Stream ---
        img_norm = self.rms_norm(img_hidden, self.const([1.0]))
        # AdaLN: (1+scale)*norm + shift
        img_mod = self.add(
            self.mul(img_norm, self.add(self.const([1.0]), img_scale)), img_shift
        )

        # Q, K, V
        img_q = self.dot(img_mod, self.permute(weights["img_q"], [1, 0]))
        img_k = self.dot(img_mod, self.permute(weights["img_k"], [1, 0]))
        img_v = self.dot(img_mod, self.permute(weights["img_v"], [1, 0]))

        # QK Norm
        img_q = self.rms_norm(img_q, weights["img_norm_q"])
        img_k = self.rms_norm(img_k, weights["img_norm_k"])

        # RoPE
        img_q = self._reshape_for_attn(img_q, img_seq_len, num_heads, head_dim)
        img_k = self._reshape_for_attn(img_k, img_seq_len, num_heads, head_dim)
        img_v = self._reshape_for_attn(img_v, img_seq_len, num_heads, head_dim)
        img_q = self.rope_2d(img_q, cos, sin)
        img_k = self.rope_2d(img_k, cos, sin)

        # --- Text Stream ---
        txt_norm = self.rms_norm(txt_hidden, self.const([1.0]))
        txt_mod = self.add(
            self.mul(txt_norm, self.add(self.const([1.0]), txt_scale)), txt_shift
        )

        txt_q = self.dot(txt_mod, self.permute(weights["txt_q"], [1, 0]))
        txt_k = self.dot(txt_mod, self.permute(weights["txt_k"], [1, 0]))
        txt_v = self.dot(txt_mod, self.permute(weights["txt_v"], [1, 0]))

        txt_q = self.rms_norm(txt_q, weights["txt_norm_q"])
        txt_k = self.rms_norm(txt_k, weights["txt_norm_k"])

        # Text RoPE (1D - handled by using same rope op but appropriate cos/sin)
        txt_q = self._reshape_for_attn(txt_q, txt_seq_len, num_heads, head_dim)
        txt_k = self._reshape_for_attn(txt_k, txt_seq_len, num_heads, head_dim)
        txt_v = self._reshape_for_attn(txt_v, txt_seq_len, num_heads, head_dim)
        txt_q = self.rope_2d(txt_q, cos, sin)
        txt_k = self.rope_2d(txt_k, cos, sin)

        # --- Joint Attention ---
        # Concat [Text, Image] for K, V
        joint_k = self.concat([txt_k, img_k], axis=2)
        joint_v = self.concat([txt_v, img_v], axis=2)

        img_attn_out = self.attention(img_q, joint_k, joint_v)
        txt_attn_out = self.attention(txt_q, joint_k, joint_v)

        img_attn_out = self._reshape_from_attn(img_attn_out, img_seq_len, hidden)
        txt_attn_out = self._reshape_from_attn(txt_attn_out, txt_seq_len, hidden)

        img_attn_proj = self.dot(
            img_attn_out, self.permute(weights["img_proj"], [1, 0])
        )
        txt_attn_proj = self.dot(
            txt_attn_out, self.permute(weights["txt_proj"], [1, 0])
        )

        # Residual with Gate
        img_hidden = self.add(img_hidden, self.mul(img_gate, img_attn_proj))
        txt_hidden = self.add(txt_hidden, self.mul(txt_gate, txt_attn_proj))

        return img_hidden, txt_hidden

    def double_block_ffn(
        self, x: TensorNode, mods: tuple, weights: Dict[str, TensorNode], prefix: str
    ) -> TensorNode:
        shift, scale, gate = mods

        norm = self.rms_norm(x, self.const([1.0]))
        mod = self.add(self.mul(norm, self.add(self.const([1.0]), scale)), shift)

        # Fused Gate + Up: weights["mlp_in"] is [mlp*2, hidden]
        fused_ff = self.dot(mod, self.permute(weights[f"{prefix}_mlp_in"], [1, 0]))

        mlp_hidden = self.cfg.mlp_hidden
        ff_gate = self.slice(fused_ff, [0], [mlp_hidden])
        ff_up = self.slice(fused_ff, [mlp_hidden], [mlp_hidden * 2])

        inner = self.mul(self.silu(ff_gate), ff_up)

        out = self.dot(inner, self.permute(weights[f"{prefix}_mlp_out"], [1, 0]))

        return self.add(x, self.mul(gate, out))

    def single_block(
        self,
        hidden: TensorNode,
        seq_len: TensorNode,
        mods: tuple[TensorNode, TensorNode, TensorNode],
        weights: Dict[str, TensorNode],
        cos: TensorNode,
        sin: TensorNode,
    ) -> TensorNode:
        """Single stream block."""
        cfg = self.cfg
        h_size = cfg.hidden_size
        mlp_h = cfg.mlp_hidden

        shift, scale, gate = mods

        # Norm + Mod
        norm = self.rms_norm(hidden, self.const([1.0]))
        mod = self.add(self.mul(norm, self.add(self.const([1.0]), scale)), shift)

        # Fused QKV + MLP: weights["qkv_mlp"] is [hidden*3 + mlp*2, hidden]
        fused = self.dot(mod, self.permute(weights["qkv_mlp"], [1, 0]))

        # Split: Q, K, V, Gate, Up
        q = self.slice(fused, [0], [h_size])
        k = self.slice(fused, [h_size], [h_size * 2])
        v = self.slice(fused, [h_size * 2], [h_size * 3])
        mlp_gate = self.slice(fused, [h_size * 3], [h_size * 3 + mlp_h])
        mlp_up = self.slice(fused, [h_size * 3 + mlp_h], [h_size * 3 + mlp_h * 2])

        # QK Norm
        q = self.rms_norm(q, weights["norm_q"])
        k = self.rms_norm(k, weights["norm_k"])

        # RoPE
        q = self._reshape_for_attn(q, seq_len, cfg.num_heads, cfg.head_dim)
        k = self._reshape_for_attn(k, seq_len, cfg.num_heads, cfg.head_dim)
        v = self._reshape_for_attn(v, seq_len, cfg.num_heads, cfg.head_dim)
        q = self.rope_2d(q, cos, sin)
        k = self.rope_2d(k, cos, sin)

        # Attention
        attn_out = self.attention(q, k, v)
        attn_out = self._reshape_from_attn(attn_out, seq_len, h_size)

        # MLP
        mlp_out = self.mul(self.silu(mlp_gate), mlp_up)

        # Concat & Project
        concat = self.concat([attn_out, mlp_out], axis=-1)
        out = self.dot(concat, self.permute(weights["proj_mlp"], [1, 0]))

        return self.add(hidden, self.mul(gate, out))

    def final_layer(
        self,
        img_hidden: TensorNode,
        t_emb: TensorNode,
        weights: Dict[str, TensorNode],
        latent_channels: int,
    ) -> TensorNode:
        """Final layer: norm -> proj to latent channels."""
        hidden_size = self.cfg.hidden_size

        # Modulation (only shift, scale - no gate)
        t_silu = self.silu(t_emb)
        mod = self.dot(t_silu, self.permute(weights["norm"], [1, 0]))
        shift = self.slice(mod, [0], [hidden_size])
        scale = self.slice(mod, [hidden_size], [hidden_size * 2])

        # Norm + mod
        h_norm = self.rms_norm(img_hidden, self.const([1.0]))
        h_mod = self.add(h_norm, shift)
        h_mod = self.mul(h_mod, self.add(self.const([1.0]), scale))

        # Project to latent channels
        return self.dot(h_mod, self.permute(weights["proj"], [1, 0]))

    # --- Helper methods ---

    def _double_block_modulation(
        self,
        t_emb: TensorNode,
        img_weight: TensorNode,
        txt_weight: TensorNode,
        hidden: int,
    ):
        """Extract 6 modulation params for double block."""
        img_mod = self.dot(self.silu(t_emb), self.permute(img_weight, [1, 0]))
        txt_mod = self.dot(self.silu(t_emb), self.permute(txt_weight, [1, 0]))

        img_shift = self.slice(img_mod, [0], [hidden])
        img_scale = self.slice(img_mod, [hidden], [hidden * 2])
        img_gate = self.slice(img_mod, [hidden * 2], [hidden * 3])

        txt_shift = self.slice(txt_mod, [0], [hidden])
        txt_scale = self.slice(txt_mod, [hidden], [hidden * 2])
        txt_gate = self.slice(txt_mod, [hidden * 2], [hidden * 3])

        return img_shift, img_scale, img_gate, txt_shift, txt_scale, txt_gate

    def _double_block_ffn(
        self,
        hidden: TensorNode,
        weights: Dict[str, TensorNode],
        stream: str,
        mlp_hidden: int,
    ) -> TensorNode:
        """Feed-forward network for double block."""
        # Gate + Up projection
        gate = self.dot(hidden, self.permute(weights[f"{stream}_mlp_gate"], [1, 0]))
        up = self.dot(hidden, self.permute(weights[f"{stream}_mlp_up"], [1, 0]))

        # SiLU gate
        gate = self.silu(gate)

        # Element-wise multiply
        hidden_ff = self.mul(gate, up)

        # Down projection
        hidden_ff = self.dot(
            hidden_ff, self.permute(weights[f"{stream}_mlp_down"], [1, 0])
        )

        return self.add(hidden, hidden_ff)

    def _reshape_for_attn(
        self, x: TensorNode, seq_len: TensorNode, num_heads: int, head_dim: int
    ) -> TensorNode:
        """Reshape [B, L, H*D] -> [B, H, L, D]"""
        # Explicitly construct shape node: [1, num_heads, seq_len, head_dim]
        shape = self.concat(
            [
                self.const([1]),
                self.const([num_heads]),
                seq_len,
                self.const([head_dim]),
            ],
            axis=0,
        )
        return self.reshape(x, shape)

    def _reshape_from_attn(
        self, x: TensorNode, seq_len: TensorNode, hidden: int
    ) -> TensorNode:
        """Reshape [B, H, L, D] -> [B, L, H*D]"""
        # Explicitly construct shape node: [1, seq_len, hidden]
        shape = self.concat([self.const([1]), seq_len, self.const([hidden])], axis=0)
        return self.reshape(x, shape)


# ============================================================================
# Flux Transformer
# ============================================================================


class FluxTransformer:
    def __init__(self, cfg: FluxConfig):
        self.cfg = cfg
        self.builder = FluxBuilder(cfg)
        self._build_graph()

    def _build_graph(self):
        b = self.builder
        cfg = self.cfg

        # Inputs
        self.img_latent = b.input("img_latent", (1, cfg.latent_channels, None, None))
        self.txt_emb = b.input("txt_emb", (1, None, cfg.text_dim))
        self.timestep = b.input("timestep", (1,))
        self.img_h = b.input("img_h", (1,))
        self.img_w = b.input("img_w", (1,))
        self.img_seq = b.mul(self.img_h, self.img_w)
        self.txt_seq = b.const([cfg.text_max_seq])  # Simplified for demo

        # 1. Embeddings
        # Timestep
        t_emb_raw = b.time_embedder(
            self.timestep,
            b.param("time_embed.fc1.weight", (cfg.hidden_size, 256)),
            b.param("time_embed.fc2.weight", (cfg.hidden_size, cfg.hidden_size)),
            dim=256,
        )
        t_emb_silu = b.silu(t_emb_raw)  # Shared for modulation

        # Image Projection (NLC)
        img_flat = b.permute(self.img_latent, [0, 2, 3, 1])
        img_flat = b.reshape(
            img_flat,
            b.concat([b.const([1]), self.img_seq, b.const([cfg.latent_channels])], 0),
        )
        img_hidden = b.dot(
            img_flat,
            b.permute(
                b.param("img_in.weight", (cfg.hidden_size, cfg.latent_channels)), [1, 0]
            ),
        )

        # Text Projection
        txt_hidden = b.dot(
            self.txt_emb,
            b.permute(
                b.param("txt_in.weight", (cfg.hidden_size, cfg.text_dim)), [1, 0]
            ),
        )

        # RoPE
        cos, sin = self._build_rope(self.img_seq)  # Simplified call

        # 2. Shared Modulation (Computed Once)
        # Double Blocks (6 components each: shift1, scale1, gate1, shift2, scale2, gate2)
        mod_img_all = b.compute_modulation(
            t_emb_silu,
            b.param("double_mod_img.weight", (cfg.hidden_size * 6, cfg.hidden_size)),
            cfg.hidden_size,
            6,
        )
        mod_txt_all = b.compute_modulation(
            t_emb_silu,
            b.param("double_mod_txt.weight", (cfg.hidden_size * 6, cfg.hidden_size)),
            cfg.hidden_size,
            6,
        )

        # Single Blocks (3 components: shift, scale, gate)
        mod_single_all = b.compute_modulation(
            t_emb_silu,
            b.param("single_mod.weight", (cfg.hidden_size * 3, cfg.hidden_size)),
            cfg.hidden_size,
            3,
        )

        # 3. Double Blocks
        for i in range(cfg.num_double_layers):
            weights = self._get_double_block_weights(i)

            # Attn Modulation (first 3)
            img_hidden, txt_hidden = b.double_block(
                img_hidden,
                txt_hidden,
                self.img_seq,
                self.txt_seq,
                mod_img_all[:3],
                mod_txt_all[:3],
                weights,
                cos,
                sin,
            )

            # FFN Modulation (last 3)
            img_hidden = b.double_block_ffn(img_hidden, mod_img_all[3:], weights, "img")
            txt_hidden = b.double_block_ffn(txt_hidden, mod_txt_all[3:], weights, "txt")

        # 4. Concatenate [Text, Image]
        combined = b.concat([txt_hidden, img_hidden], axis=1)
        total_seq = b.add(self.txt_seq, self.img_seq)

        # 5. Single Blocks
        for i in range(cfg.num_single_layers):
            weights = self._get_single_block_weights(i)
            combined = b.single_block(
                combined, total_seq, mod_single_all, weights, cos, sin
            )

        # 6. Final Layer (Image Only)
        img_out = b.slice(combined, [self.txt_seq], [total_seq])

        # Final Mod (reuse double_mod_img weights logic per C code, but graph needs separate compute for clarity or reuse node)
        # C code reuses buffer but computes: linear(t_emb_silu, final_norm_weight) -> [shift, scale]
        # Wait, C code says: `iris_linear_nobias(final_mod, tf->t_emb_silu, tf->final_norm_weight, ...)`
        # final_norm_weight in C load is `norm_out.linear.weight` [2*hidden, hidden]
        final_mods = b.compute_modulation(
            t_emb_silu,
            b.param("final_norm.weight", (cfg.hidden_size * 2, cfg.hidden_size)),
            cfg.hidden_size,
            2,
        )

        final_shift, final_scale = final_mods
        norm_out = b.rms_norm(img_out, b.const([1.0]))
        mod_out = b.add(
            b.mul(norm_out, b.add(b.const([1.0]), final_scale)), final_shift
        )

        output = b.dot(
            mod_out,
            b.permute(
                b.param("final_proj.weight", (cfg.latent_channels, cfg.hidden_size)),
                [1, 0],
            ),
        )

        # Output Reshape
        output = b.permute(output, [0, 2, 1])
        output = b.reshape(
            output,
            b.concat(
                [b.const([1]), b.const([cfg.latent_channels]), self.img_h, self.img_w],
                0,
            ),
        )
        self.output = output

    def _build_rope(self, img_seq_len):
        # Placeholder for complex 2D RoPE logic, assuming precomputed inputs for this demo
        # In a real impl, this would generate the frequencies graph based on img_h/img_w
        return self.builder.param(
            "rope_cos", (1, 1, None, self.cfg.head_dim)
        ), self.builder.param("rope_sin", (1, 1, None, self.cfg.head_dim))

    def _get_double_block_weights(self, idx: int) -> Dict[str, TensorNode]:
        b = self.builder
        p = f"double_blocks.{idx}"
        return {
            "img_q": b.param(f"{p}.img_attn.q.weight", (None,)),
            "img_k": b.param(f"{p}.img_attn.k.weight", (None,)),
            "img_v": b.param(f"{p}.img_attn.v.weight", (None,)),
            "img_proj": b.param(f"{p}.img_attn.proj.weight", (None,)),
            "img_norm_q": b.param(f"{p}.img_attn.norm_q.weight", (None,)),
            "img_norm_k": b.param(f"{p}.img_attn.norm_k.weight", (None,)),
            "img_mlp_in": b.param(f"{p}.img_mlp.in.weight", (None,)),  # Fused Gate+Up
            "img_mlp_out": b.param(f"{p}.img_mlp.out.weight", (None,)),
            "txt_q": b.param(f"{p}.txt_attn.q.weight", (None,)),
            "txt_k": b.param(f"{p}.txt_attn.k.weight", (None,)),
            "txt_v": b.param(f"{p}.txt_attn.v.weight", (None,)),
            "txt_proj": b.param(f"{p}.txt_attn.proj.weight", (None,)),
            "txt_norm_q": b.param(f"{p}.txt_attn.norm_q.weight", (None,)),
            "txt_norm_k": b.param(f"{p}.txt_attn.norm_k.weight", (None,)),
            "txt_mlp_in": b.param(f"{p}.txt_mlp.in.weight", (None,)),
            "txt_mlp_out": b.param(f"{p}.txt_mlp.out.weight", (None,)),
        }

    def _get_single_block_weights(self, idx: int) -> Dict[str, TensorNode]:
        b = self.builder
        p = f"single_blocks.{idx}"
        return {
            "qkv_mlp": b.param(f"{p}.qkv_mlp.weight", (None,)),
            "proj_mlp": b.param(f"{p}.proj_mlp.weight", (None,)),
            "norm_q": b.param(f"{p}.norm_q.weight", (None,)),
            "norm_k": b.param(f"{p}.norm_k.weight", (None,)),
        }


# ============================================================================
# Euler Sampler
# ============================================================================


class EulerSampler:
    """Euler sampler for rectified flow."""

    def __init__(
        self,
        num_steps: int = 4,
        use_shifted_schedule: bool = True,
        image_seq_len: int = 256,
    ):
        self.num_steps = num_steps
        self.use_shifted_schedule = use_shifted_schedule
        self.image_seq_len = image_seq_len

    def get_schedule(self) -> np.ndarray:
        """Get timestep schedule."""
        if self.use_shifted_schedule:
            # Shifted sigmoid schedule (official FLUX)
            return self._shifted_sigmoid_schedule()
        else:
            # Linear schedule
            return np.linspace(1.0, 0.0, self.num_steps + 1)

    def _shifted_sigmoid_schedule(self) -> np.ndarray:
        """
        Shifted sigmoid schedule for better coverage at small timesteps.
        This matches the official FLUX implementation.
        """
        # Shift parameter based on image sequence length
        shift = 3.0  # Default for 256x256

        # Generate schedule
        t = np.linspace(0, 1, self.num_steps + 1)

        # Shifted sigmoid: sigmoid(logit(t) + shift)
        # logit(t) = log(t / (1 - t))
        # But we use a simpler approximation

        # From flux_sample.c: shifted sigmoid with base and shift
        base = 2.0
        sigma = 1.0

        # For FLUX distilled, the schedule is pre-defined
        # See flux_official_schedule in flux_sample.c
        if self.num_steps == 4:
            return np.array([1.0, 0.75, 0.5, 0.25, 0.0])
        elif self.num_steps == 50:
            # Base model schedule
            return np.linspace(1.0, 0.0, self.num_steps + 1)
        else:
            return np.linspace(1.0, 0.0, self.num_steps + 1)

    def step(
        self, z: np.ndarray, velocity: np.ndarray, t: float, dt: float
    ) -> np.ndarray:
        """
        Euler step: z_{t-dt} = z_t - velocity * dt
        """
        return z - velocity * dt


# ============================================================================
# VAE Decoder
# ============================================================================


class VAEDecoder(GraphBuilder):
    """
    FLUX VAE Decoder using tensor_graphs.

    Architecture from flux_vae.c:
    - Conv in: 32 -> 512
    - Mid block: ResBlock -> AttnBlock -> ResBlock
    - Up blocks (4 levels, reverse of encoder):
      - Level 0 (512ch): 3 ResBlocks + Upsample
      - Level 1 (512ch): 3 ResBlocks + Upsample
      - Level 2 (256ch): 3 ResBlocks + Upsample
      - Level 3 (128ch): 3 ResBlocks (no upsample)
    - GroupNorm -> Swish -> Conv out: 128 -> 3
    """

    def __init__(self, cfg: FluxConfig):
        super().__init__()
        self.cfg = cfg
        self.eps = 1e-6

    def group_norm(
        self, x: TensorNode, weight: TensorNode, bias: TensorNode, num_groups: int
    ) -> TensorNode:
        """GroupNorm with 32 groups."""
        return TensorNode(
            "GroupNorm",
            x.dtype,
            [x, weight, bias],
            name=self._next_name("groupnorm"),
            attrs={"num_groups": num_groups, "eps": self.eps},
        )

    def swish(self, x: TensorNode) -> TensorNode:
        """SiLU/Swish activation."""
        return TensorNode("SiLU", x.dtype, [x], name=self._next_name("silu"))

    def silu(self, x: TensorNode) -> TensorNode:
        """SiLU activation using existing implementation."""
        return TensorNode("SiLU", x.dtype, [x], name=self._next_name("silu"))

    def conv2d(
        self,
        x: TensorNode,
        weight: TensorNode,
        bias: Optional[TensorNode],
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ) -> TensorNode:
        """
        2D Convolution using im2col + matmul.
        """
        return TensorNode(
            "Conv2D",
            x.dtype,
            [x, weight] + ([bias] if bias else []),
            name=self._next_name("conv2d"),
            attrs={"kernel_size": kernel_size, "stride": stride, "padding": padding},
        )

    def upsample_nearest_2x(self, x: TensorNode) -> TensorNode:
        """Nearest neighbor 2x upsample."""
        return TensorNode(
            "Upsample2x",
            x.dtype,
            [x],
            name=self._next_name("upsample"),
            attrs={"mode": "nearest", "scale": 2},
        )

    def resblock(
        self,
        x: TensorNode,
        norm1_w: TensorNode,
        norm1_b: TensorNode,
        conv1_w: TensorNode,
        conv1_b: TensorNode,
        norm2_w: TensorNode,
        norm2_b: TensorNode,
        conv2_w: TensorNode,
        conv2_b: TensorNode,
        skip_w: Optional[TensorNode] = None,
        skip_b: Optional[TensorNode] = None,
        num_groups: int = 32,
    ) -> TensorNode:
        """
        ResBlock: norm1 -> swish -> conv1 -> norm2 -> swish -> conv2 + skip
        """
        # Main path
        h = self.group_norm(x, norm1_w, norm1_b, num_groups)
        h = self.swish(h)
        h = self.conv2d(h, conv1_w, conv1_b, kernel_size=3, stride=1, padding=1)

        h = self.group_norm(h, norm2_w, norm2_b, num_groups)
        h = self.swish(h)
        h = self.conv2d(h, conv2_w, conv2_b, kernel_size=3, stride=1, padding=1)

        # Skip connection
        if skip_w is not None:
            skip = self.conv2d(x, skip_w, skip_b, kernel_size=1, stride=1, padding=0)
            return self.add(h, skip)
        else:
            return self.add(h, x)

    def attnblock(
        self,
        x: TensorNode,
        norm_w: TensorNode,
        norm_b: TensorNode,
        q_w: TensorNode,
        q_b: TensorNode,
        k_w: TensorNode,
        k_b: TensorNode,
        v_w: TensorNode,
        v_b: TensorNode,
        out_w: TensorNode,
        out_b: TensorNode,
        h: TensorNode,  # Added argument
        w: TensorNode,  # Added argument
        num_groups: int = 32,
    ) -> TensorNode:
        """
        Self-attention block for VAE.
        Integrated decomposition: norm -> Q, K, V -> flattened attention -> out_proj + residual
        """
        # Determine channels from weight shape (output channels of Q projection)
        # q_w shape is [out_channels, in_channels, kH, kW] -> [C, C, 1, 1]
        ch = q_w.shape[0] if q_w.shape else self.cfg.vae_channels

        # 1. Normalize
        h_norm = self.group_norm(x, norm_w, norm_b, num_groups)

        # 2. Q, K, V projections (1x1 conv)
        q = self.conv2d(h_norm, q_w, q_b, kernel_size=1, stride=1, padding=0)
        k = self.conv2d(h_norm, k_w, k_b, kernel_size=1, stride=1, padding=0)
        v = self.conv2d(h_norm, v_w, v_b, kernel_size=1, stride=1, padding=0)

        # 3. Flatten spatial dimensions for attention logic
        # [B, C, H, W] -> [B, H, W, C] -> [B, L, C]
        def flatten_spatial(node):
            # Transpose to Channel-last
            perm = self.permute(node, [0, 2, 3, 1])

            # Calculate sequence length explicitly
            seq_len = self.mul(h, w)
            c_dim = node.shape[1] if node.shape else ch

            # Explicit shape construction: [1, seq_len, C]
            shape = self.concat([self.const([1]), seq_len, self.const([c_dim])], axis=0)
            return self.reshape(perm, shape)

        q_flat = flatten_spatial(q)
        k_flat = flatten_spatial(k)
        v_flat = flatten_spatial(v)

        # 4. Compute Attention Scores: (Q @ K^T) * scale
        k_t = self.permute(k_flat, [0, 2, 1])
        scores = self.dot(q_flat, k_t)

        scale_val = float(ch) ** -0.5
        scores_scaled = self.mul(scores, self.const([scale_val]))

        # 5. Softmax along last dimension
        probs = TensorNode(
            "Softmax",
            q.dtype,
            [scores_scaled],
            name=self._next_name("vae_attn_softmax"),
            attrs={"axis": -1},
        )

        # 6. Context: Probs @ V
        attn_out_flat = self.dot(probs, v_flat)

        # 7. Reshape back to NCHW using explicit h, w
        # [B, L, C] -> [B, H, W, C] -> [B, C, H, W]
        out_hwc = self.reshape(
            attn_out_flat,
            self.concat(
                [
                    self.const([1]),
                    h,
                    w,
                    self.const([ch]),
                ],
                axis=0,
            ),
        )

        attn_out = self.permute(out_hwc, [0, 3, 1, 2])

        # 8. Output projection
        out = self.conv2d(attn_out, out_w, out_b, kernel_size=1, stride=1, padding=0)

        # 9. Residual connection
        return self.add(out, x)

    def unpack(
        self, latent: TensorNode, h_in: TensorNode, w_in: TensorNode
    ) -> Tuple[TensorNode, TensorNode, TensorNode]:
        """
        Unpack packed latents: [B, 128, H, W] -> [B, 32, H*2, W*2]
        """
        p = self.cfg.patch_size  # 2
        c = self.cfg.vae_z_channels  # 32

        # 1. Reshape [B, p, p, C, H, W]
        shape1 = self.concat(
            [
                self.const([1]),
                self.const([p]),
                self.const([p]),
                self.const([c]),
                h_in,
                w_in,
            ],
            axis=0,
        )
        h = self.reshape(latent, shape1)

        # 2. Permute to [B, C, H, p, W, p]
        # (Batch, p1, p2, Chan, H, W) -> (Batch, Chan, H, p1, W, p2)
        h = self.permute(h, [0, 3, 4, 1, 5, 2])

        # 3. Reshape to [B, C, H*p, W*p]
        new_h = self.mul(h_in, self.const([p]))
        new_w = self.mul(w_in, self.const([p]))

        shape2 = self.concat([self.const([1]), self.const([c]), new_h, new_w], axis=0)

        return self.reshape(h, shape2), new_h, new_w

    def decode(
        self,
        latent: TensorNode,
        h_in: TensorNode,
        w_in: TensorNode,
        weights: Dict[str, TensorNode],
    ) -> TensorNode:
        """
        Full VAE decoder forward pass.
        """
        # Unpack FLUX latents
        h, h_unpacked, w_unpacked = self.unpack(latent, h_in, w_in)

        # Post-quantization conv (32 -> 32)
        h = self.conv2d(
            h,
            weights["post_quant_conv.weight"],
            weights["post_quant_conv.bias"],
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # Conv in: 32 -> 512
        h = self.conv2d(
            h,
            weights["decoder.conv_in.weight"],
            weights["decoder.conv_in.bias"],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Mid block (ResBlock -> AttnBlock -> ResBlock)
        # Note: Resolution is preserved in the mid block.
        h = self.resblock(
            h,
            weights["decoder.mid_block.resnets.0.norm1.weight"],
            weights["decoder.mid_block.resnets.0.norm1.bias"],
            weights["decoder.mid_block.resnets.0.conv1.weight"],
            weights["decoder.mid_block.resnets.0.conv1.bias"],
            weights["decoder.mid_block.resnets.0.norm2.weight"],
            weights["decoder.mid_block.resnets.0.norm2.bias"],
            weights["decoder.mid_block.resnets.0.conv2.weight"],
            weights["decoder.mid_block.resnets.0.conv2.bias"],
        )

        # Pass the unpacked resolution to the Attention Block
        h = self.attnblock(
            h,
            weights["decoder.mid_block.attentions.0.group_norm.weight"],
            weights["decoder.mid_block.attentions.0.group_norm.bias"],
            weights["decoder.mid_block.attentions.0.to_q.weight"],
            weights["decoder.mid_block.attentions.0.to_q.bias"],
            weights["decoder.mid_block.attentions.0.to_k.weight"],
            weights["decoder.mid_block.attentions.0.to_k.bias"],
            weights["decoder.mid_block.attentions.0.to_v.weight"],
            weights["decoder.mid_block.attentions.0.to_v.bias"],
            weights["decoder.mid_block.attentions.0.to_out.0.weight"],
            weights["decoder.mid_block.attentions.0.to_out.0.bias"],
            h=h_unpacked,
            w=w_unpacked,
        )

        h = self.resblock(
            h,
            weights["decoder.mid_block.resnets.1.norm1.weight"],
            weights["decoder.mid_block.resnets.1.norm1.bias"],
            weights["decoder.mid_block.resnets.1.conv1.weight"],
            weights["decoder.mid_block.resnets.1.conv1.bias"],
            weights["decoder.mid_block.resnets.1.norm2.weight"],
            weights["decoder.mid_block.resnets.1.norm2.bias"],
            weights["decoder.mid_block.resnets.1.conv2.weight"],
            weights["decoder.mid_block.resnets.1.conv2.bias"],
        )

        for level in range(3, -1, -1):
            for r in range(3):
                res_prefix = f"decoder.up_blocks.{3 - level}.resnets.{r}"

                # Check for shortcut weight (channel transition)
                skip_w_key = f"{res_prefix}.conv_shortcut.weight"
                skip_b_key = f"{res_prefix}.conv_shortcut.bias"

                skip_w = weights.get(skip_w_key)
                skip_b = weights.get(skip_b_key)

                h = self.resblock(
                    h,
                    weights[f"{res_prefix}.norm1.weight"],
                    weights[f"{res_prefix}.norm1.bias"],
                    weights[f"{res_prefix}.conv1.weight"],
                    weights[f"{res_prefix}.conv1.bias"],
                    weights[f"{res_prefix}.norm2.weight"],
                    weights[f"{res_prefix}.norm2.bias"],
                    weights[f"{res_prefix}.conv2.weight"],
                    weights[f"{res_prefix}.conv2.bias"],
                    skip_w=skip_w,
                    skip_b=skip_b,
                )

            # Upsample (except last level)
            if level > 0:
                h = self.upsample_nearest_2x(h)
                h = self.conv2d(
                    h,
                    weights[f"decoder.up_blocks.{3 - level}.upsamplers.0.conv.weight"],
                    weights[f"decoder.up_blocks.{3 - level}.upsamplers.0.conv.bias"],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )

        # Output: norm -> swish -> conv
        h = self.group_norm(
            h,
            weights["decoder.conv_norm_out.weight"],
            weights["decoder.conv_norm_out.bias"],
            32,
        )
        h = self.swish(h)
        h = self.conv2d(
            h,
            weights["decoder.conv_out.weight"],
            weights["decoder.conv_out.bias"],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        return h


# ============================================================================
# Qwen3 Text Encoder
# ============================================================================


class Qwen3Encoder(GraphBuilder):
    """
    Qwen3-4B Text Encoder using tensor_graphs.

    Architecture:
    - Word embedding: [vocab_size, hidden_size]
    - 36 Transformer layers with:
      - RMSNorm
      - Self-attention with RoPE
      - MLP (gate, up, down projections with SiLU)
    - Final RMSNorm

    Hidden size: 2560
    Num heads: 20
    Head dim: 128
    Intermediate size: 9728 (for MLP)
    """

    def __init__(self, cfg: FluxConfig):
        super().__init__()
        self.cfg = cfg
        self.eps = 1e-6

    def compute_rope_freqs(
        self, seq_len_node: TensorNode, head_dim: int, theta: float = 10000.0
    ) -> Tuple[TensorNode, TensorNode]:
        """Compute RoPE frequencies using graph operations."""
        b = self
        half_dim = head_dim // 2

        # freqs = 1 / (theta ** (2i / head_dim)) for i in [0, half_dim)
        indices = b.arange(b.const(0), b.const(half_dim), b.const(1))
        indices_fp = b.cast(indices, DType.FP32)
        head_dim_fp = b.const(float(head_dim), DType.FP32)
        theta_const = b.const(theta, DType.FP32)

        # exponent = 2i / head_dim
        exponent = b.divide(b.mul(indices_fp, b.const(2.0, DType.FP32)), head_dim_fp)
        # freqs = theta ** exponent
        freqs = b.power(theta_const, exponent)
        # inv_freq = 1 / freqs
        inv_freq = b.divide(b.const(1.0, DType.FP32), freqs)

        # positions = [0, 1, ..., seq_len-1]
        positions = b.arange(b.const(0), seq_len_node, b.const(1))
        positions_fp = b.cast(positions, DType.FP32)

        # Outer product: angles = positions x inv_freq
        # pos_col: [seq_len, 1], freq_row: [1, half_dim]
        pos_col = b.reshape(
            positions_fp, b.concat([seq_len_node, b.const([1])], axis=0)
        )
        freq_row = b.reshape(
            inv_freq, b.concat([b.const([1]), b.const([half_dim])], axis=0)
        )

        angles_half = b.mul(pos_col, freq_row)  # [seq_len, half_dim]

        # Concat to get full head_dim: [seq_len, head_dim]
        angles = b.concat([angles_half, angles_half], axis=1)

        # Expand to [1, 1, seq_len, head_dim]
        final_shape = b.concat(
            [b.const([1]), b.const([1]), seq_len_node, b.const([head_dim])], axis=0
        )
        cos_out = b.reshape(b.cos(angles), final_shape)
        sin_out = b.reshape(b.sin(angles), final_shape)

        return cos_out, sin_out

    def rms_norm(self, x: TensorNode, weight: TensorNode) -> TensorNode:
        """RMSNorm."""
        eps_node = self.const([self.eps])
        return TensorNode(
            "RMSNorm",
            x.dtype,
            [x, weight, eps_node],
            name=self._next_name("rmsnorm"),
            attrs={"axis": -1},
        )

    def silu(self, x: TensorNode) -> TensorNode:
        """SiLU activation."""
        return TensorNode("SiLU", x.dtype, [x], name=self._next_name("silu"))

    def attention(
        self,
        x: TensorNode,
        q_w: TensorNode,
        k_w: TensorNode,
        v_w: TensorNode,
        o_w: TensorNode,
        q_norm_w: TensorNode,
        k_norm_w: TensorNode,
        cos: TensorNode,
        sin: TensorNode,
        mask: Optional[TensorNode],
        seq_len_node: TensorNode,
    ) -> TensorNode:
        """
        Multi-head self-attention with QK norm and RoPE.
        """
        # 1. Q, K, V projections
        q = self.dot(x, self.permute(q_w, [1, 0]))
        k = self.dot(x, self.permute(k_w, [1, 0]))
        v = self.dot(x, self.permute(v_w, [1, 0]))

        # 2. Reshape to [B, S, H, D]
        batch = self.const([1])
        head_dim_node = self.const([self.cfg.head_dim])
        n_head = self.const([self.cfg.text_num_heads])
        n_kv_head = self.const([self.cfg.text_num_kv_heads])

        q = self.reshape(
            q, self.concat([batch, seq_len_node, n_head, head_dim_node], axis=-1)
        )
        k = self.reshape(
            k, self.concat([batch, seq_len_node, n_kv_head, head_dim_node], axis=-1)
        )
        v = self.reshape(
            v, self.concat([batch, seq_len_node, n_kv_head, head_dim_node], axis=-1)
        )

        # 3. QK Norm
        q = self.rms_norm(q, q_norm_w)
        k = self.rms_norm(k, k_norm_w)

        # 4. Transpose to [B, H, S, D] for Score Calculation
        q = self.permute(q, [0, 2, 1, 3])  # [B, H_q, S, D]
        k = self.permute(k, [0, 2, 1, 3])  # [B, H_kv, S, D]
        v = self.permute(v, [0, 2, 1, 3])  # [B, H_kv, S, D]

        # 5. Apply RoPE
        q = self.rope(q, cos, sin)
        k = self.rope(k, cos, sin)

        # 6. Handle GQA (Grouped Query Attention)
        # Decompose repeat_interleave(k, heads_per_kv, axis=1)
        heads_per_kv = self.cfg.text_num_heads // self.cfg.text_num_kv_heads

        if heads_per_kv > 1:
            # Step A: Expand dims at axis 2 (insert size 1)
            # Shape [B, H_kv, S, D] -> [B, H_kv, 1, S, D]
            shape_expanded = self.concat(
                [batch, n_kv_head, self.const([1]), seq_len_node, head_dim_node],
                axis=-1,
            )
            k_exp = self.reshape(k, shape_expanded)
            v_exp = self.reshape(v, shape_expanded)

            # Step B: Tile along the new dimension
            # Shape [B, H_kv, 1, S, D] -> [B, H_kv, heads_per_kv, S, D]
            k_tiled = self.repeat(k_exp, heads_per_kv, -3)
            v_tiled = self.repeat(v_exp, heads_per_kv, -3)

            # Step C: Flatten to merge H_kv and heads_per_kv
            # Shape [B, H_kv, heads_per_kv, S, D] -> [B, H_q, S, D]
            # Note: n_head is already n_kv_head * heads_per_kv
            shape_final = self.concat(
                [batch, n_head, seq_len_node, head_dim_node], axis=-1
            )
            k = self.reshape(k_tiled, shape_final)
            v = self.reshape(v_tiled, shape_final)

        # 7. Scaled Dot-Product Attention
        scale = self.divide(
            self.const([1.0]), self.sqrt(self.cast(head_dim_node, DType.FP32))
        )

        k_t = self.permute(k, [0, 1, 3, 2])  # [B, H_q, D, S]
        scores = self.dot(q, k_t)  # [B, H_q, S, S]
        scores = self.mul(scores, scale)

        # 8. Masking
        if mask is not None:
            scores = self.add(scores, mask)

        probs = self.softmax(scores)

        # 9. Attention Output
        attn_out = self.dot(probs, v)

        # 10. Reshape back
        attn_out = self.permute(attn_out, [0, 2, 1, 3])
        attn_out_dim = self.const([4096])
        attn_out = self.reshape(
            attn_out, self.concat([batch, seq_len_node, attn_out_dim], axis=-1)
        )

        # 11. Output projection
        return self.dot(attn_out, self.permute(o_w, [1, 0]))

    def mlp(
        self, x: TensorNode, gate_w: TensorNode, up_w: TensorNode, down_w: TensorNode
    ) -> TensorNode:
        """
        MLP with gated activation: down(silu(gate(x)) * up(x))
        """
        gate = self.dot(x, self.permute(gate_w, [1, 0]))
        gate = self.silu(gate)

        up = self.dot(x, self.permute(up_w, [1, 0]))

        hidden = self.mul(gate, up)

        return self.dot(hidden, self.permute(down_w, [1, 0]))

    def transformer_block(
        self,
        x: TensorNode,
        weights: Dict[str, TensorNode],
        layer_idx: int,
        cos: TensorNode,
        sin: TensorNode,
        mask: Optional[TensorNode],
        seq_len_node: TensorNode,
    ) -> TensorNode:
        """Single transformer block."""
        prefix = f"model.layers.{layer_idx}"

        # Pre-norm attention
        h = self.rms_norm(x, weights[f"{prefix}.input_layernorm.weight"])
        h = self.attention(
            h,
            weights[f"{prefix}.self_attn.q_proj.weight"],
            weights[f"{prefix}.self_attn.k_proj.weight"],
            weights[f"{prefix}.self_attn.v_proj.weight"],
            weights[f"{prefix}.self_attn.o_proj.weight"],
            weights[f"{prefix}.self_attn.q_norm.weight"],
            weights[f"{prefix}.self_attn.k_norm.weight"],
            cos,
            sin,
            mask,
            seq_len_node,
        )
        x = self.add(x, h)

        # Pre-norm MLP
        h = self.rms_norm(x, weights[f"{prefix}.post_attention_layernorm.weight"])
        h = self.mlp(
            h,
            weights[f"{prefix}.mlp.gate_proj.weight"],
            weights[f"{prefix}.mlp.up_proj.weight"],
            weights[f"{prefix}.mlp.down_proj.weight"],
        )
        x = self.add(x, h)

        return x

    def forward(
        self,
        input_ids: TensorNode,
        weights: Dict[str, TensorNode],
    ) -> TensorNode:
        """
        Forward pass through Qwen3 encoder.

        input_ids: [B, seq_len] token indices
        Returns: [B, seq_len, hidden_size] embeddings
        """
        seq_len_node = self.input("seq_len", (1,), DType.INT32)

        # Word embedding
        x = TensorNode(
            "Gather",
            DType.FP32,
            [weights["model.embed_tokens.weight"], input_ids],
            name=self._next_name("embedding"),
        )

        # Compute RoPE frequencies
        cos, sin = self.compute_rope_freqs(seq_len_node, self.cfg.text_head_dim)

        # Causal mask
        mask = self.compute_causal_mask(seq_len_node)

        # Transformer layers
        for i in range(self.cfg.text_num_layers):
            x = self.transformer_block(x, weights, i, cos, sin, mask, seq_len_node)

        # Final norm
        x = self.rms_norm(x, weights["model.norm.weight"])

        return x

    def compute_causal_mask(self, seq_len_node: TensorNode) -> TensorNode:
        """Compute causal attention mask."""
        # Lower triangular mask
        mask_shape = self.concat([seq_len_node, seq_len_node], axis=0)
        ones_matrix = self.fill(self.const(1.0, DType.FP32), mask_shape)
        triu_mask = self.triu(ones_matrix, k=1)
        final_shape = self.concat(
            [self.const([1]), self.const([1]), seq_len_node, seq_len_node],
            axis=0,
        )
        return self.reshape(triu_mask, final_shape)


# ============================================================================
# Updated FluxPipeline
# ============================================================================


class FluxPipeline:
    """Complete FLUX.2 Klein 4B pipeline."""

    def __init__(
        self, model_dir: str, cfg: Optional[FluxConfig] = None, device: str = "cpu"
    ):
        self.model_dir = model_dir
        self.cfg = cfg or FluxConfig()
        self.device = device

        # Initialize builders
        self.transformer_builder = FluxBuilder(self.cfg)
        self.vae_decoder = VAEDecoder(self.cfg)
        self.text_encoder = Qwen3Encoder(self.cfg)

        # Sessions (initialized on demand)
        self.transformer_session: Optional[GraphSession] = None
        self.text_encoder_session: Optional[GraphSession] = None
        self.vae_session: Optional[GraphSession] = None
        self.vae_encoder_session: Optional[GraphSession] = None

        # Weight sources
        self.transformer_weights: Optional[SafetensorsSource] = None
        self.text_encoder_weights: Optional[SafetensorsSource] = None
        self.vae_weights: Optional[SafetensorsSource] = None

        # Tokenizer
        self.tokenizer: Optional[Any] = None

    def load_tokenizer(self):
        """Load the Qwen3 tokenizer."""
        if self.tokenizer:
            return

        from tokenizers import Tokenizer

        tokenizer_path = os.path.join(self.model_dir, "tokenizer", "tokenizer.json")
        if os.path.exists(tokenizer_path):
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
        else:
            # Fallback to simple tokenization
            print(f"Warning: Tokenizer not found at {tokenizer_path}")
            self.tokenizer = None

    def load_text_encoder(self):
        """Load Qwen3 text encoder weights."""
        if self.text_encoder_weights:
            return

        encoder_path = os.path.join(self.model_dir, "text_encoder")
        if not os.path.isdir(encoder_path):
            raise FileNotFoundError(f"Text encoder directory not found: {encoder_path}")

        self.text_encoder_weights = SafetensorsSource(encoder_path)
        print(f"Loaded text encoder weights from {encoder_path}")

    def load_vae(self):
        """Load VAE weights."""
        if self.vae_weights:
            return

        vae_path = os.path.join(
            self.model_dir, "vae", "diffusion_pytorch_model.safetensors"
        )
        vae_dir = os.path.join(self.model_dir, "vae")

        # Try single file first, then directory
        if os.path.exists(vae_path):
            self.vae_weights = SafetensorsSource(vae_path)
            print(f"Loaded VAE weights from {vae_path}")
        elif os.path.isdir(vae_dir):
            self.vae_weights = SafetensorsSource(vae_dir)
            print(f"Loaded VAE weights from {vae_dir}")
        else:
            raise FileNotFoundError(f"VAE weights not found at {vae_path} or {vae_dir}")

    def load_transformer(self):
        """Load transformer weights."""
        if self.transformer_weights:
            return

        transformer_path = os.path.join(self.model_dir, "transformer")
        if not os.path.isdir(transformer_path):
            raise FileNotFoundError(
                f"Transformer directory not found: {transformer_path}"
            )

        self.transformer_weights = SafetensorsSource(transformer_path)
        print(f"Loaded transformer weights from {transformer_path}")

    def build_text_encoder(self):
        """Build and compile text encoder graph."""
        if self.text_encoder_session:
            return

        print("Building text encoder graph...")

        # Create input nodes
        input_ids = self.text_encoder.input(
            "input_ids", (1, self.cfg.text_max_seq), DType.INT32
        )

        # Load weights
        self.load_text_encoder()

        # Build graph
        embeddings = self.text_encoder.forward(
            input_ids,
            {
                k: self.text_encoder.param(
                    k, self.text_encoder_weights.get_tensor_metadata(k)[0]
                )
                for k in self.text_encoder_weights.keys()
            },
        )

        # Create session
        self.text_encoder_session = GraphSession(embeddings)

        # Compile
        sample_inputs = {
            "input_ids": np.zeros((1, self.cfg.text_max_seq), dtype=np.int32),
            "seq_len": np.array([self.cfg.text_max_seq], dtype=np.int32),
        }
        self.text_encoder_session.compile(sample_inputs)

        print("Text encoder ready!")

    def build_vae_decoder(self):
        """Build and compile VAE decoder graph."""
        if self.vae_session:
            return

        print("Building VAE decoder graph...")

        self.load_vae()

        # Create input node
        latent = self.vae_decoder.input(
            "latent", (1, self.cfg.vae_channels, None, None), DType.FP32
        )
        # Add height and width inputs for static shape construction
        h_node = self.vae_decoder.input("h", (1,), DType.INT32)
        w_node = self.vae_decoder.input("w", (1,), DType.INT32)

        # Build weights shape dict
        weights = {}
        for key in self.vae_weights.keys():
            weights[key] = self.vae_decoder.param(
                key, self.vae_weights.get_tensor_metadata(key)[0]
            )

        # Build decoder with shape inputs
        image = self.vae_decoder.decode(latent, h_node, w_node, weights)

        # Create session
        self.vae_session = GraphSession(image)

        # Compile with sample input
        sample_latent = np.zeros((1, self.cfg.vae_channels, 8, 8), dtype=np.float32)
        sample_h = np.array([8], dtype=np.int32)
        sample_w = np.array([8], dtype=np.int32)

        self.vae_session.compile(
            {"latent": sample_latent, "h": sample_h, "w": sample_w}
        )

        print("VAE decoder ready!")

    def build_transformer(self):
        """Build and compile the transformer graph."""
        if self.transformer_session:
            return

        print("Building Transformer graph...")
        self.load_transformer()

        # Instantiate wrapper which builds the graph
        self.flux_transformer = FluxTransformer(self.cfg)

        # Create session attached to the output node
        self.transformer_session = GraphSession(self.flux_transformer.output)

        # Prepare dummy inputs for compilation
        # Assuming H=W=16 (latent 8x8) for quick compile, or target resolution
        h_val, w_val = 16, 16
        sample_inputs = {
            "img_latent": np.zeros(
                (1, self.cfg.latent_channels, h_val, w_val), dtype=np.float32
            ),
            "txt_emb": np.zeros(
                (1, self.cfg.text_max_seq, self.cfg.text_dim), dtype=np.float32
            ),
            "timestep": np.array([1.0], dtype=np.float32),
            "img_h": np.array([h_val], dtype=np.int32),
            "img_w": np.array([w_val], dtype=np.int32),
            # Dummy RoPE cache (would need actual computation logic passed or computed inside)
            "rope_cos": np.zeros(
                (1, 1, h_val * w_val + self.cfg.text_max_seq, self.cfg.head_dim),
                dtype=np.float32,
            ),
            "rope_sin": np.zeros(
                (1, 1, h_val * w_val + self.cfg.text_max_seq, self.cfg.head_dim),
                dtype=np.float32,
            ),
        }

        self.transformer_session.compile(sample_inputs)
        print("Transformer ready!")

    def encode_text(self, prompt: str) -> np.ndarray:
        """
        Encode text prompt using Qwen3.

        Returns: [1, 512, text_dim] embeddings
        """
        print(f"Encoding prompt: {prompt[:50]}...")

        # Load tokenizer
        self.load_tokenizer()

        # Tokenize
        if not self.tokenizer:
            raise ValueError("FluxPipeline has no tokenizer")
        # Add BOS token
        tokens = self.tokenizer.encode(prompt).ids
        # Pad/truncate to max_seq_len
        if len(tokens) > self.cfg.text_max_seq:
            tokens = tokens[: self.cfg.text_max_seq]
        else:
            tokens = tokens + [0] * (self.cfg.text_max_seq - len(tokens))

        input_ids = np.array([tokens], dtype=np.int32)

        # Build encoder if needed
        self.build_text_encoder()

        # Run encoder
        inputs = {
            "input_ids": input_ids,
            "seq_len": np.array([self.cfg.text_max_seq], dtype=np.int32),
        }

        self.text_encoder_session.load_weights(self.text_encoder_weights)

        embeddings = self.text_encoder_session.run(inputs)

        return embeddings

    def decode_latent(self, latent: np.ndarray) -> np.ndarray:
        """
        Decode latent to image using VAE decoder.

        Args:
            latent: [1, C, H, W] latent tensor

        Returns:
            [H*16, W*16, 3] RGB image in [0, 255]
        """
        print("Decoding latent...")

        # Build decoder if needed
        self.build_vae_decoder()

        # Extract dimensions
        h_val = latent.shape[2]
        w_val = latent.shape[3]

        self.vae_session.load_weights(self.vae_weights)

        # Run decoder with shape inputs
        image_tensor = self.vae_session.run(
            {
                "latent": latent,
                "h": np.array([h_val], dtype=np.int32),
                "w": np.array([w_val], dtype=np.int32),
            }
        )

        # Convert to image
        # image_tensor is [B, 3, H, W] in [-1, 1] range
        img = image_tensor[0].transpose(1, 2, 0)  # [H, W, 3]
        img = (img + 1.0) * 0.5  # [-1, 1] -> [0, 1]
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

        return img

    def sample(
        self,
        text_emb: np.ndarray,
        height: int = 256,
        width: int = 256,
        num_steps: Optional[int] = None,
        seed: Optional[int] = None,
        progress: bool = True,
    ) -> np.ndarray:
        """Run diffusion sampling."""
        if seed is not None:
            np.random.seed(seed)

        latent_h = height // 16
        latent_w = width // 16

        z = np.random.randn(1, self.cfg.latent_channels, latent_h, latent_w).astype(
            np.float32
        )

        steps = num_steps or self.cfg.num_steps_distilled
        sampler = EulerSampler(steps)
        schedule = sampler.get_schedule()

        # Ensure transformer is built before loop
        self.build_transformer()
        self.transformer_session.load_weights(self.transformer_weights)

        iterator = tqdm(range(steps), desc="Sampling") if progress else range(steps)

        for i in iterator:
            t = schedule[i]
            t_next = schedule[i + 1]
            dt = t_next - t  # Euler: t to t_next

            # Model prediction (v-prediction)
            velocity = self._run_transformer(z, text_emb, t)

            # Step
            z = sampler.step(z, velocity, t, dt)

        return z

    def _run_transformer(
        self, z: np.ndarray, text_emb: np.ndarray, t: float
    ) -> np.ndarray:
        """Execute one step of the transformer."""
        # Prepare inputs
        _, _, h, w = z.shape
        inputs = {
            "img_latent": z,
            "txt_emb": text_emb,
            "timestep": np.array([t], dtype=np.float32),
            "img_h": np.array([h], dtype=np.int32),
            "img_w": np.array([w], dtype=np.int32),
        }

        return self.transformer_session.run(inputs)

    def generate(
        self,
        prompt: str,
        height: int = 256,
        width: int = 256,
        num_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Full generation pipeline.
        """
        print(f"\nGenerating: {prompt[:50]}...")
        print(f"Size: {width}x{height}")

        # Encode text
        # text_emb = self.encode_text(prompt)
        text_emb = np.zeros(
            (1, self.cfg.text_max_seq, self.cfg.text_hidden_size)
        )  # TODO: remove, this is a placeholder to skip expensive text encoding while testing sampling/latent decoding.

        # Sample
        latent = self.sample(text_emb, height, width, num_steps, seed)

        # Decode
        image = self.decode_latent(latent)

        print("Done!")
        return image


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    """Demo: Generate image with FLUX.2 Klein 4B."""

    # Configuration
    model_dir = "flux-klein-4b"  # Path to model weights

    if not os.path.exists(model_dir):
        print(f"Model directory not found: {model_dir}")
        print("Please download model weights first:")
        print("  ./download_model.sh 4b")
        return

    # Create pipeline
    cfg = FluxConfig()
    pipeline = FluxPipeline(model_dir, cfg)

    # Generate
    prompt = "cat"

    image = pipeline.generate(
        prompt=prompt,
        height=128,
        width=128,
        num_steps=4,  # Distilled model
        seed=42,
    )

    # Save result
    output_path = "flux_output.png"
    img = Image.fromarray(image)
    img.save(output_path, format="PNG")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()

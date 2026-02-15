"""
FLUX.2 Klein 4B Image Generation on tensor_graphs

Implements the complete diffusion transformer pipeline:
- Text encoding (Qwen3-4B) with layer concatenation (8, 17, 26)
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
    text_max_seq: int = 512
    text_rope_theta: float = 1000000.0

    # Latent
    latent_channels: int = 128
    patch_size: int = 2

    # Sampling
    num_steps_distilled: int = 4
    num_steps_base: int = 50


class FluxBuilder(GraphBuilder):
    """Graph builder for FLUX.2 architecture."""

    def __init__(self, cfg: FluxConfig):
        super().__init__()
        self.cfg = cfg
        self.eps = self.const([1e-6])

    # --- Fused Operations ---

    def silu(self, x: TensorNode) -> TensorNode:
        return TensorNode("SiLU", x.dtype, [x], name=self._next_name("silu"))

    def rms_norm(
        self, x: TensorNode, scale: TensorNode, eps: float = 1e-6
    ) -> TensorNode:
        eps_node = self.const([eps])
        return TensorNode(
            "RMSNorm",
            x.dtype,
            [x, scale, eps_node],
            name=self._next_name("rmsnorm"),
            attrs={"axis": -1},
        )

    def rope_2d(self, x: TensorNode, cos: TensorNode, sin: TensorNode) -> TensorNode:
        return TensorNode(
            "RoPE2DConsecutive", x.dtype, [x, cos, sin], name=self._next_name("rope")
        )

    def attention(
        self,
        q: TensorNode,
        k: TensorNode,
        v: TensorNode,
    ) -> TensorNode:
        """Scaled dot-product attention."""
        d_k = self.cfg.head_dim
        scale = 1.0 / math.sqrt(d_k)

        # k_transposed = permute(k, [0, 1, 3, 2]) for [B, H, L, D]
        k_t = self.permute(k, [0, 1, 3, 2])
        scores = self.dot(q, k_t)
        scores = self.mul(scores, self.const([scale]))

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

    def layer_norm(self, x: TensorNode, eps: float = 1e-6) -> TensorNode:
        """Standard LayerNorm without affine parameters (centering + scaling)."""
        dim = self.cfg.hidden_size

        # Mean
        mean = TensorNode("Sum", x.dtype, [x], attrs={"axis": -1, "keepdims": True})
        mean = self.divide(mean, self.const([float(dim)]))

        # Sub Mean
        x_sub = self.add(x, self.negate(mean))

        # Var
        sq = self.mul(x_sub, x_sub)
        var_sum = TensorNode("Sum", x.dtype, [sq], attrs={"axis": -1, "keepdims": True})
        var = self.divide(var_sum, self.const([float(dim)]))

        # Div Std
        std = self.sqrt(self.add(var, self.const([eps])))
        one = self.const([1.0])
        inv_std = self.divide(one, std)

        return self.mul(x_sub, inv_std)

    # --- Transformer Components ---

    def timestep_embedding(
        self, t: TensorNode, dim: int, max_period: int = 10000
    ) -> TensorNode:
        """Sinusoidal timestep embedding.
        Input t: (1,)
        Output: (dim,)
        """
        half = dim // 2
        freqs = self.cast(
            self.arange(self.const(0), self.const(half), self.const(1)), DType.FP32
        )
        freqs = self.mul(freqs, self.const([-math.log(max_period) / half]))
        freqs = self.exp(freqs)

        # args = t * freqs -> (half,)
        args = self.mul(t, freqs)

        # embedding = cat([cos(args), sin(args)]) -> (dim,)
        cos_emb = self.cos(args)
        sin_emb = self.sin(args)
        return self.concat([cos_emb, sin_emb], axis=0)

    def time_embedder(
        self,
        t: TensorNode,
        fc1_weight: TensorNode,
        fc2_weight: TensorNode,
        dim: int = 256,
    ) -> TensorNode:
        """Full time embedding: sincos -> fc1 -> silu -> fc2"""
        # t: (1,)
        t_sincos = self.timestep_embedding(t, dim)  # (256,)

        # fc1: [hidden, 256] -> we need permute for dot
        # (256,) @ (256, hidden) -> (hidden,)
        h = self.dot(t_sincos, self.permute(fc1_weight, [1, 0]))
        h = self.silu(h)

        # fc2: [hidden, hidden]
        # (hidden,) @ (hidden, hidden) -> (hidden,)
        return self.dot(h, self.permute(fc2_weight, [1, 0]))

    def compute_modulation(
        self, t_emb: TensorNode, weight: TensorNode, hidden: int, chunks: int
    ) -> list[TensorNode]:
        """
        Compute shared modulation parameters.
        t_emb: (hidden,)
        weight: (hidden*chunks, hidden)
        """
        # (hidden,) @ (hidden, hidden*chunks) -> (hidden*chunks,)
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
        img_mods: tuple,
        txt_mods: tuple,
        weights: Dict[str, TensorNode],
        cos: TensorNode,
        sin: TensorNode,
    ) -> Tuple[TensorNode, TensorNode]:
        """Double stream block."""
        cfg = self.cfg
        hidden = cfg.hidden_size
        head_dim = cfg.head_dim
        num_heads = cfg.num_heads

        img_shift, img_scale, img_gate = img_mods
        txt_shift, txt_scale, txt_gate = txt_mods

        # --- Image Stream ---
        # AdaLN: (1+scale)*LayerNorm(x) + shift
        # Fixed: Use layer_norm instead of rms_norm
        img_norm = self.layer_norm(img_hidden)
        img_mod = self.add(
            self.mul(img_norm, self.add(self.const([1.0]), img_scale)), img_shift
        )

        # Q, K, V
        img_q = self.dot(img_mod, self.permute(weights["img_q"], [1, 0]))
        img_k = self.dot(img_mod, self.permute(weights["img_k"], [1, 0]))
        img_v = self.dot(img_mod, self.permute(weights["img_v"], [1, 0]))
        img_q = self._reshape_for_attn(img_q, img_seq_len, num_heads, head_dim)
        img_k = self._reshape_for_attn(img_k, img_seq_len, num_heads, head_dim)
        img_v = self._reshape_for_attn(img_v, img_seq_len, num_heads, head_dim)

        # QK Norm (Uses RMSNorm in Flux, correctly)
        img_q = self.rms_norm(img_q, weights["img_norm_q"])
        img_k = self.rms_norm(img_k, weights["img_norm_k"])

        # --- Slice RoPE for Image (Image tokens start after Text tokens) ---
        img_cos = self.slice(
            cos, [0, 0, self.cfg.text_max_seq, 0], [None, None, None, None]
        )
        img_sin = self.slice(
            sin, [0, 0, self.cfg.text_max_seq, 0], [None, None, None, None]
        )
        img_q = self.rope_2d(img_q, img_cos, img_sin)
        img_k = self.rope_2d(img_k, img_cos, img_sin)

        # --- Text Stream ---
        # Fixed: Use layer_norm instead of rms_norm
        txt_norm = self.layer_norm(txt_hidden)
        txt_mod = self.add(
            self.mul(txt_norm, self.add(self.const([1.0]), txt_scale)), txt_shift
        )

        txt_q = self.dot(txt_mod, self.permute(weights["txt_q"], [1, 0]))
        txt_k = self.dot(txt_mod, self.permute(weights["txt_k"], [1, 0]))
        txt_v = self.dot(txt_mod, self.permute(weights["txt_v"], [1, 0]))

        txt_q = self._reshape_for_attn(txt_q, txt_seq_len, num_heads, head_dim)
        txt_k = self._reshape_for_attn(txt_k, txt_seq_len, num_heads, head_dim)
        txt_v = self._reshape_for_attn(txt_v, txt_seq_len, num_heads, head_dim)

        # QK Norm (Uses RMSNorm in Flux, correctly)
        txt_q = self.rms_norm(txt_q, weights["txt_norm_q"])
        txt_k = self.rms_norm(txt_k, weights["txt_norm_k"])

        # --- Slice RoPE for Text ---
        txt_cos = self.slice(
            cos, [0, 0, 0, 0], [None, None, self.cfg.text_max_seq, None]
        )
        txt_sin = self.slice(
            sin, [0, 0, 0, 0], [None, None, self.cfg.text_max_seq, None]
        )
        txt_q = self.rope_2d(txt_q, txt_cos, txt_sin)
        txt_k = self.rope_2d(txt_k, txt_cos, txt_sin)

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

        norm = self.layer_norm(x)
        mod = self.add(self.mul(norm, self.add(self.const([1.0]), scale)), shift)

        # Fused Gate + Up
        fused_ff = self.dot(mod, self.permute(weights[f"{prefix}_mlp_in"], [1, 0]))

        mlp_hidden = self.cfg.mlp_hidden
        ff_gate = self.slice(fused_ff, [0, 0, 0], [None, None, mlp_hidden])
        ff_up = self.slice(fused_ff, [0, 0, mlp_hidden], [None, None, mlp_hidden * 2])

        inner = self.mul(self.silu(ff_gate), ff_up)

        out = self.dot(inner, self.permute(weights[f"{prefix}_mlp_out"], [1, 0]))

        return self.add(x, self.mul(gate, out))

    def single_block(
        self,
        hidden: TensorNode,
        seq_len: TensorNode,
        mods: tuple,
        weights: Dict[str, TensorNode],
        cos: TensorNode,
        sin: TensorNode,
    ) -> TensorNode:
        """Single stream block."""
        cfg = self.cfg
        h_size = cfg.hidden_size
        mlp_h = cfg.mlp_hidden

        shift, scale, gate = mods

        norm = self.layer_norm(hidden)
        mod = self.add(self.mul(norm, self.add(self.const([1.0]), scale)), shift)

        # Fused QKV + MLP
        fused = self.dot(mod, self.permute(weights["qkv_mlp"], [1, 0]))

        # Split: Q, K, V, Gate, Up
        q = self.slice(fused, [0, 0, 0], [None, None, h_size])
        k = self.slice(fused, [0, 0, h_size], [None, None, h_size * 2])
        v = self.slice(fused, [0, 0, h_size * 2], [None, None, h_size * 3])
        mlp_gate = self.slice(
            fused, [0, 0, h_size * 3], [None, None, h_size * 3 + mlp_h]
        )
        mlp_up = self.slice(
            fused, [0, 0, h_size * 3 + mlp_h], [None, None, h_size * 3 + mlp_h * 2]
        )

        q = self._reshape_for_attn(q, seq_len, cfg.num_heads, cfg.head_dim)
        k = self._reshape_for_attn(k, seq_len, cfg.num_heads, cfg.head_dim)
        v = self._reshape_for_attn(v, seq_len, cfg.num_heads, cfg.head_dim)

        # QK Norm (Uses RMSNorm in Flux, correctly)
        q = self.rms_norm(q, weights["norm_q"])
        k = self.rms_norm(k, weights["norm_k"])

        # RoPE
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

    def _reshape_for_attn(
        self, x: TensorNode, seq_len: TensorNode, num_heads: int, head_dim: int
    ) -> TensorNode:
        """Reshape [1, L, H*D] -> [1, H, L, D]"""
        reshaped = self.reshape(
            x,
            self.concat(
                [
                    self.const([1]),
                    seq_len,
                    self.const([num_heads]),
                    self.const([head_dim]),
                ],
                axis=0,
            ),
        )
        # Permute to [1, H, L, D] (dims 0, 2, 1, 3)
        return self.permute(reshaped, [0, 2, 1, 3])

    def _reshape_from_attn(
        self, x: TensorNode, seq_len: TensorNode, hidden: int
    ) -> TensorNode:
        """Reshape [1, H, L, D] -> [1, L, H*D]"""
        # Permute -> [1, L, H, D] (dims 0, 2, 1, 3)
        permuted = self.permute(x, [0, 2, 1, 3])
        # Reshape -> [1, L, Hidden]
        return self.reshape(
            permuted,
            self.concat([self.const([1]), seq_len, self.const([hidden])], axis=0),
        )


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
        self.img_h = b.input("img_h", (1,), DType.INT32)
        self.img_w = b.input("img_w", (1,), DType.INT32)

        # Derived sequences
        self.img_seq = b.mul(self.img_h, self.img_w)
        self.txt_seq = b.input("txt_seq", (1,), dtype=DType.INT32)

        # 1. Embeddings
        # Timestep
        self.timestep = b.mul(self.timestep, b.const([1000.0]))
        t_emb_raw = b.time_embedder(
            self.timestep,
            b.param(
                "time_guidance_embed.timestep_embedder.linear_1.weight",
                (cfg.hidden_size, 256),
            ),
            b.param(
                "time_guidance_embed.timestep_embedder.linear_2.weight",
                (cfg.hidden_size, cfg.hidden_size),
            ),
            dim=256,
        )
        t_emb_silu = b.silu(t_emb_raw)  # Shared for modulation

        # Image Projection (NLC)
        # [1, C, H, W] -> [1, H, W, C] -> [1, L, C]
        img_perm = b.permute(self.img_latent, [0, 2, 3, 1])
        img_flat = b.reshape(
            img_perm,
            b.concat([b.const([1]), self.img_seq, b.const([cfg.latent_channels])], 0),
        )
        img_hidden = b.dot(
            img_flat,
            b.permute(
                b.param("x_embedder.weight", (cfg.hidden_size, cfg.latent_channels)),
                [1, 0],
            ),
        )

        # Text Projection
        txt_hidden = b.dot(
            self.txt_emb,
            b.permute(
                b.param("context_embedder.weight", (cfg.hidden_size, cfg.text_dim)),
                [1, 0],
            ),
        )

        # RoPE
        cos, sin = self._build_rope()

        # 2. Shared Modulation
        # Double Blocks
        mod_img_all = b.compute_modulation(
            t_emb_silu,
            b.param(
                "double_stream_modulation_img.linear.weight",
                (cfg.hidden_size * 6, cfg.hidden_size),
            ),
            cfg.hidden_size,
            6,
        )
        mod_txt_all = b.compute_modulation(
            t_emb_silu,
            b.param(
                "double_stream_modulation_txt.linear.weight",
                (cfg.hidden_size * 6, cfg.hidden_size),
            ),
            cfg.hidden_size,
            6,
        )

        # Single Blocks
        mod_single_all = b.compute_modulation(
            t_emb_silu,
            b.param(
                "single_stream_modulation.linear.weight",
                (cfg.hidden_size * 3, cfg.hidden_size),
            ),
            cfg.hidden_size,
            3,
        )

        # 3. Double Blocks
        for i in range(cfg.num_double_layers):
            weights = self._get_double_block_weights(i)

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

            img_hidden = b.double_block_ffn(img_hidden, mod_img_all[3:], weights, "img")
            txt_hidden = b.double_block_ffn(txt_hidden, mod_txt_all[3:], weights, "txt")

        # 4. Concatenate [Text, Image]
        # Text comes first in Flux single blocks
        combined = b.concat([txt_hidden, img_hidden], axis=1)
        total_seq = b.add(self.txt_seq, self.img_seq)

        # 5. Single Blocks
        for i in range(cfg.num_single_layers):
            weights = self._get_single_block_weights(i)
            combined = b.single_block(
                combined, total_seq, mod_single_all, weights, cos, sin
            )

        # 6. Final Layer (Image Only)
        # Extract image part: slice(txt_seq, end)
        img_out = b.slice(combined, [0, self.cfg.text_max_seq, 0], [None, None, None])

        # Final Mod
        final_mods = b.compute_modulation(
            t_emb_silu,
            b.param("norm_out.linear.weight", (cfg.hidden_size * 2, cfg.hidden_size)),
            cfg.hidden_size,
            2,
        )
        final_scale, final_shift = final_mods

        norm_out = b.layer_norm(img_out)
        mod_out = b.add(
            b.mul(norm_out, b.add(b.const([1.0]), final_scale)), final_shift
        )

        output = b.dot(
            mod_out,
            b.permute(
                b.param("proj_out.weight", (cfg.latent_channels, cfg.hidden_size)),
                [1, 0],
            ),
        )

        # Output Reshape: [1, L, C] -> [1, H, W, C] -> [1, C, H, W]
        output = b.reshape(
            output,
            b.concat(
                [b.const([1]), self.img_h, self.img_w, b.const([cfg.latent_channels])],
                0,
            ),
        )
        output = b.permute(output, [0, 3, 1, 2])
        self.output = output

    def _build_rope(self):
        # We pass cos/sin as inputs, concatenated for [Text + Image]
        cos = self.builder.input("rope_cos", (1, 1, None, self.cfg.head_dim))
        sin = self.builder.input("rope_sin", (1, 1, None, self.cfg.head_dim))
        return cos, sin

    def _get_double_block_weights(self, idx: int) -> Dict[str, TensorNode]:
        b = self.builder
        p = f"transformer_blocks.{idx}"
        hidden = self.cfg.hidden_size
        mlp = self.cfg.mlp_hidden
        head_dim = self.cfg.head_dim

        return {
            "img_q": b.param(f"{p}.attn.to_q.weight", (hidden, hidden)),
            "img_k": b.param(f"{p}.attn.to_k.weight", (hidden, hidden)),
            "img_v": b.param(f"{p}.attn.to_v.weight", (hidden, hidden)),
            "img_proj": b.param(f"{p}.attn.to_out.0.weight", (hidden, hidden)),
            "img_norm_q": b.param(f"{p}.attn.norm_q.weight", (head_dim,)),
            "img_norm_k": b.param(f"{p}.attn.norm_k.weight", (head_dim,)),
            # FFN: linear_in is [mlp*2, hidden] (gate + up), linear_out is [hidden, mlp]
            "img_mlp_in": b.param(f"{p}.ff.linear_in.weight", (mlp * 2, hidden)),
            "img_mlp_out": b.param(f"{p}.ff.linear_out.weight", (hidden, mlp)),
            "txt_q": b.param(f"{p}.attn.add_q_proj.weight", (hidden, hidden)),
            "txt_k": b.param(f"{p}.attn.add_k_proj.weight", (hidden, hidden)),
            "txt_v": b.param(f"{p}.attn.add_v_proj.weight", (hidden, hidden)),
            "txt_proj": b.param(f"{p}.attn.to_add_out.weight", (hidden, hidden)),
            "txt_norm_q": b.param(f"{p}.attn.norm_added_q.weight", (head_dim,)),
            "txt_norm_k": b.param(f"{p}.attn.norm_added_k.weight", (head_dim,)),
            "txt_mlp_in": b.param(
                f"{p}.ff_context.linear_in.weight", (mlp * 2, hidden)
            ),
            "txt_mlp_out": b.param(f"{p}.ff_context.linear_out.weight", (hidden, mlp)),
        }

    def _get_single_block_weights(self, idx: int) -> Dict[str, TensorNode]:
        b = self.builder
        p = f"single_transformer_blocks.{idx}"
        hidden = self.cfg.hidden_size
        mlp = self.cfg.mlp_hidden
        head_dim = self.cfg.head_dim

        # fused qkv_mlp: [hidden*3 + mlp*2, hidden]
        qkv_dim = hidden * 3 + mlp * 2

        # fused proj_mlp: [hidden, hidden + mlp]
        proj_dim = hidden + mlp

        return {
            "qkv_mlp": b.param(f"{p}.attn.to_qkv_mlp_proj.weight", (qkv_dim, hidden)),
            "proj_mlp": b.param(f"{p}.attn.to_out.weight", (hidden, proj_dim)),
            "norm_q": b.param(f"{p}.attn.norm_q.weight", (head_dim,)),
            "norm_k": b.param(f"{p}.attn.norm_k.weight", (head_dim,)),
        }


class Sampler:
    """Proper FLUX.2 resolution-dependent Euler sampler."""

    def __init__(self, num_steps: int):
        self.num_steps = num_steps

    def get_schedule(self, image_seq_len: int) -> np.ndarray:
        """Matches iris_schedule_flux and official FLUX.1/2 schedule logic."""
        # Empirical constants from Flux training distribution
        a1, b1 = 8.73809524e-05, 1.89833333
        a2, b2 = 0.00016927, 0.45666666

        if image_seq_len > 4300:
            mu = a2 * image_seq_len + b2
        else:
            # Interpolate between two linear fits
            m_200 = a2 * image_seq_len + b2
            m_10 = a1 * image_seq_len + b1
            a = (m_200 - m_10) / 190.0
            b = m_200 - 200.0 * a
            mu = a * self.num_steps + b

        # Linear timesteps from 1.0 down to 0.0
        t_linear = 1.0 - np.linspace(0, 1, self.num_steps + 1)

        # Apply generalized SNR shift: exp(mu) / (exp(mu) + (1/t - 1)^sigma)
        schedule = np.zeros_like(t_linear)
        for i, t in enumerate(t_linear):
            if t <= 0:
                schedule[i] = 0.0
            elif t >= 1:
                schedule[i] = 1.0
            else:
                # Flux uses sigma=1.0 for the generalized shift
                schedule[i] = math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0))

        return schedule

    def step(self, z: np.ndarray, velocity: np.ndarray, dt: float) -> np.ndarray:
        """Euler step: z_{t+dt} = z_t + dt * v"""
        return z + velocity * dt


class VAEDecoder(GraphBuilder):
    """
    FLUX VAE Decoder using tensor_graphs.
    """

    def __init__(self, cfg: FluxConfig):
        super().__init__()
        self.cfg = cfg
        self.eps = 1e-4

    def group_norm(
        self, x: TensorNode, weight: TensorNode, bias: TensorNode, num_groups: int
    ) -> TensorNode:
        return TensorNode(
            "GroupNorm",
            x.dtype,
            [x, weight, bias],
            name=self._next_name("groupnorm"),
            attrs={"num_groups": num_groups, "eps": self.eps},
        )

    def swish(self, x: TensorNode) -> TensorNode:
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
        inputs = [x, weight]
        if bias:
            inputs.append(bias)
        return TensorNode(
            "Conv2D",
            x.dtype,
            inputs,
            name=self._next_name("conv2d"),
            attrs={"kernel_size": kernel_size, "stride": stride, "padding": padding},
        )

    def upsample_nearest_2x(self, x: TensorNode) -> TensorNode:
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
        h = self.group_norm(x, norm1_w, norm1_b, num_groups)
        h = self.swish(h)
        h = self.conv2d(h, conv1_w, conv1_b, kernel_size=3, stride=1, padding=1)

        h = self.group_norm(h, norm2_w, norm2_b, num_groups)
        h = self.swish(h)
        h = self.conv2d(h, conv2_w, conv2_b, kernel_size=3, stride=1, padding=1)

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
        h: TensorNode,
        w: TensorNode,
        num_groups: int = 32,
    ) -> TensorNode:
        ch = q_w.shape[0] if q_w.shape else self.cfg.vae_channels

        h_norm = self.group_norm(x, norm_w, norm_b, num_groups)

        q = self.conv2d(h_norm, q_w, q_b, kernel_size=1, stride=1, padding=0)
        k = self.conv2d(h_norm, k_w, k_b, kernel_size=1, stride=1, padding=0)
        v = self.conv2d(h_norm, v_w, v_b, kernel_size=1, stride=1, padding=0)

        def flatten_spatial(node):
            perm = self.permute(node, [0, 2, 3, 1])
            seq_len = self.mul(h, w)
            c_dim = node.shape[1] if node.shape else ch
            shape = self.concat([self.const([1]), seq_len, self.const([c_dim])], axis=0)
            return self.reshape(perm, shape)

        q_flat = flatten_spatial(q)
        k_flat = flatten_spatial(k)
        v_flat = flatten_spatial(v)

        k_t = self.permute(k_flat, [0, 2, 1])
        scores = self.dot(q_flat, k_t)

        scale_val = float(ch) ** -0.5
        scores_scaled = self.mul(scores, self.const([scale_val]))

        probs = TensorNode(
            "Softmax",
            q.dtype,
            [scores_scaled],
            name=self._next_name("vae_attn_softmax"),
            attrs={"axis": -1},
        )

        attn_out_flat = self.dot(probs, v_flat)

        out_hwc = self.reshape(
            attn_out_flat,
            self.concat(
                [self.const([1]), h, w, self.const([ch])],
                axis=0,
            ),
        )

        attn_out = self.permute(out_hwc, [0, 3, 1, 2])
        out = self.conv2d(attn_out, out_w, out_b, kernel_size=1, stride=1, padding=0)

        return self.add(out, x)

    def unpack(
        self, latent: TensorNode, h_in: TensorNode, w_in: TensorNode
    ) -> Tuple[TensorNode, TensorNode, TensorNode]:
        p = self.cfg.patch_size
        c = self.cfg.vae_z_channels

        # [B, C, p, p, H, W]
        # The channel dimension (128) is packed as C_base(32) * p(2) * p(2)
        shape1 = self.concat(
            [
                self.const([1]),
                self.const([c]),
                self.const([p]),
                self.const([p]),
                h_in,
                w_in,
            ],
            axis=0,
        )
        h = self.reshape(latent, shape1)

        # [B, C, H, p, W, p]
        # Original dims: 0:B, 1:C, 2:Py, 3:Px, 4:H, 5:W
        # Target dims:   0:B, 1:C, 4:H, 2:Py, 5:W, 3:Px
        h = self.permute(h, [0, 1, 4, 2, 5, 3])

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
        bn_mean = weights["bn.running_mean"]
        bn_var = weights["bn.running_var"]

        # Reshape [128] -> [1, 128, 1, 1] for NCHW broadcasting
        b_shape = self.const(
            np.array([1, self.cfg.latent_channels, 1, 1], dtype=np.int32)
        )
        mu = self.reshape(bn_mean, b_shape)
        var = self.reshape(bn_var, b_shape)

        # Flux VAE uses eps=1e-4 for the latent batch norm
        std = self.sqrt(self.add(var, self.const([self.eps])))
        latent = self.add(self.mul(latent, std), mu)

        h, h_unpacked, w_unpacked = self.unpack(latent, h_in, w_in)

        h = self.conv2d(
            h,
            weights["post_quant_conv.weight"],
            weights["post_quant_conv.bias"],
            kernel_size=1,
            stride=1,
            padding=0,
        )

        h = self.conv2d(
            h,
            weights["decoder.conv_in.weight"],
            weights["decoder.conv_in.bias"],
            kernel_size=3,
            stride=1,
            padding=1,
        )

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


class Qwen3Encoder(GraphBuilder):
    """
    Qwen3-4B Text Encoder with Flux specific layer extraction (8, 17, 26).
    """

    def __init__(self, cfg: FluxConfig):
        super().__init__()
        self.cfg = cfg
        self.eps = 1e-6

    def compute_rope_freqs(
        self, seq_len_node: TensorNode, head_dim: int, theta: float = 1000000.0
    ) -> Tuple[TensorNode, TensorNode]:
        b = self
        half_dim = head_dim // 2

        indices = b.arange(b.const(0), b.const(half_dim), b.const(1))
        indices_fp = b.cast(indices, DType.FP32)
        head_dim_fp = b.const(float(head_dim), DType.FP32)
        theta_const = b.const(theta, DType.FP32)

        exponent = b.divide(b.mul(indices_fp, b.const(2.0, DType.FP32)), head_dim_fp)
        freqs = b.power(theta_const, exponent)
        inv_freq = b.divide(b.const(1.0, DType.FP32), freqs)

        positions = b.arange(b.const(0), seq_len_node, b.const(1))
        positions_fp = b.cast(positions, DType.FP32)

        pos_col = b.reshape(
            positions_fp, b.concat([seq_len_node, b.const([1])], axis=0)
        )
        freq_row = b.reshape(
            inv_freq, b.concat([b.const([1]), b.const([half_dim])], axis=0)
        )

        angles_half = b.mul(pos_col, freq_row)
        angles = b.concat([angles_half, angles_half], axis=1)

        final_shape = b.concat(
            [b.const([1]), b.const([1]), seq_len_node, b.const([head_dim])], axis=0
        )
        cos_out = b.reshape(b.cos(angles), final_shape)
        sin_out = b.reshape(b.sin(angles), final_shape)

        return cos_out, sin_out

    def rms_norm(self, x: TensorNode, weight: TensorNode) -> TensorNode:
        eps_node = self.const([self.eps])
        return TensorNode(
            "RMSNorm",
            x.dtype,
            [x, weight, eps_node],
            name=self._next_name("rmsnorm"),
            attrs={"axis": -1},
        )

    def silu(self, x: TensorNode) -> TensorNode:
        return TensorNode("SiLU", x.dtype, [x], name=self._next_name("silu"))

    def rope(self, x, cos, sin):
        return TensorNode("RoPE", x.dtype, [x, cos, sin], name=self._next_name("rope"))

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
        q = self.dot(x, self.permute(q_w, [1, 0]))
        k = self.dot(x, self.permute(k_w, [1, 0]))
        v = self.dot(x, self.permute(v_w, [1, 0]))

        batch = self.const([1])
        head_dim_node = self.const([self.cfg.text_head_dim])
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

        q = self.rms_norm(q, q_norm_w)
        k = self.rms_norm(k, k_norm_w)

        q = self.permute(q, [0, 2, 1, 3])
        k = self.permute(k, [0, 2, 1, 3])
        v = self.permute(v, [0, 2, 1, 3])

        q = self.rope(q, cos, sin)
        k = self.rope(k, cos, sin)

        heads_per_kv = self.cfg.text_num_heads // self.cfg.text_num_kv_heads
        if heads_per_kv > 1:
            shape_expanded = self.concat(
                [batch, n_kv_head, self.const([1]), seq_len_node, head_dim_node],
                axis=-1,
            )
            k_exp = self.reshape(k, shape_expanded)
            v_exp = self.reshape(v, shape_expanded)
            k_tiled = self.repeat(k_exp, heads_per_kv, -3)
            v_tiled = self.repeat(v_exp, heads_per_kv, -3)
            shape_final = self.concat(
                [batch, n_head, seq_len_node, head_dim_node], axis=-1
            )
            k = self.reshape(k_tiled, shape_final)
            v = self.reshape(v_tiled, shape_final)

        scale = self.divide(
            self.const([1.0]), self.sqrt(self.cast(head_dim_node, DType.FP32))
        )

        k_t = self.permute(k, [0, 1, 3, 2])
        scores = self.dot(q, k_t)
        scores = self.mul(scores, scale)

        if mask is not None:
            scores = self.add(scores, mask)

        probs = self.softmax(scores)
        attn_out = self.dot(probs, v)

        attn_out = self.permute(attn_out, [0, 2, 1, 3])
        attn_out_dim = self.const([4096])  # Specific to Qwen
        attn_out = self.reshape(
            attn_out, self.concat([batch, seq_len_node, attn_out_dim], axis=-1)
        )

        return self.dot(attn_out, self.permute(o_w, [1, 0]))

    def mlp(
        self, x: TensorNode, gate_w: TensorNode, up_w: TensorNode, down_w: TensorNode
    ) -> TensorNode:
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
        prefix = f"model.layers.{layer_idx}"
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
        seq_len_node = self.input("seq_len", (1,), DType.INT32)
        x = TensorNode(
            "Gather",
            DType.FP32,
            [weights["model.embed_tokens.weight"], input_ids],
            name=self._next_name("embedding"),
        )
        cos, sin = self.compute_rope_freqs(
            seq_len_node, self.cfg.text_head_dim, self.cfg.text_rope_theta
        )
        mask = self.compute_causal_mask(seq_len_node)

        # Flux extraction logic: Layers 8, 17, 26 (0-indexed)
        # We need to loop up to 26 inclusive.
        target_layers = {8, 17, 26}
        max_layer = 26
        extracted_states = []

        for i in range(max_layer + 1):
            x = self.transformer_block(x, weights, i, cos, sin, mask, seq_len_node)
            if i in target_layers:
                extracted_states.append(x)

        # Concatenate extracted hidden states along feature dimension
        # Shape: (1, L, 2560) -> (1, L, 2560*3=7680)
        return self.concat(extracted_states, axis=-1)

    def compute_causal_mask(self, seq_len_node: TensorNode) -> TensorNode:
        mask_shape = self.concat([seq_len_node, seq_len_node], axis=0)
        ones_matrix = self.fill(self.const(1.0, DType.FP32), mask_shape)
        triu_mask = self.triu(ones_matrix, k=1)
        final_shape = self.concat(
            [self.const([1]), self.const([1]), seq_len_node, seq_len_node],
            axis=0,
        )
        mask = self.mul(triu_mask, self.const(-1e9, DType.FP32))
        return self.reshape(mask, final_shape)


class FluxPipeline:
    def __init__(
        self, model_dir: str, cfg: Optional[FluxConfig] = None, device: str = "cpu"
    ):
        self.model_dir = model_dir
        self.cfg = cfg or FluxConfig()
        self.device = device
        self.transformer_session: Optional[GraphSession] = None
        self.transformer_weights: Optional[SafetensorsSource] = None

        self.vae_session: Optional[GraphSession] = None
        self.vae_weights: Optional[SafetensorsSource] = None
        self.vae_decoder = VAEDecoder(self.cfg)

        self.text_encoder_session: Optional[GraphSession] = None
        self.text_encoder_weights: Optional[SafetensorsSource] = None
        self.text_encoder = Qwen3Encoder(self.cfg)
        self.tokenizer: Optional[Any] = None

    def load_transformer(self):
        if self.transformer_weights:
            return

        path = os.path.join(
            self.model_dir, "transformer", "diffusion_pytorch_model.safetensors"
        )
        if not os.path.exists(path):
            path = os.path.join(self.model_dir, "diffusion_pytorch_model.safetensors")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Transformer weights not found in {self.model_dir}"
                )

        self.transformer_weights = SafetensorsSource(path)
        print(f"Loaded transformer weights from {path}")

    def load_tokenizer(self):
        if self.tokenizer:
            return
        from tokenizers import Tokenizer

        tokenizer_path = os.path.join(self.model_dir, "tokenizer", "tokenizer.json")
        if os.path.exists(tokenizer_path):
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
        else:
            print(f"Warning: Tokenizer not found at {tokenizer_path}")

    def load_text_encoder(self):
        if self.text_encoder_weights:
            return
        encoder_path = os.path.join(self.model_dir, "text_encoder")
        if os.path.isdir(encoder_path):
            self.text_encoder_weights = SafetensorsSource(encoder_path)
            print(f"Loaded text encoder weights from {encoder_path}")

    def load_vae(self):
        if self.vae_weights:
            return
        vae_path = os.path.join(
            self.model_dir, "vae", "diffusion_pytorch_model.safetensors"
        )
        if os.path.exists(vae_path):
            self.vae_weights = SafetensorsSource(vae_path)
            print(f"Loaded VAE weights from {vae_path}")

    def compute_rope(
        self, txt_seq: int, img_h: int, img_w: int, head_dim: int, theta: float = 2000.0
    ):
        img_seq = img_h * img_w
        total_seq = txt_seq + img_seq
        axis_dim = head_dim // 4

        # Precompute freqs: 1.0 / (theta ** (2i / axis_dim))
        freqs = 1.0 / (
            theta ** (np.arange(0, axis_dim, 2).astype(np.float32) / axis_dim)
        )

        cos_out = np.ones((total_seq, head_dim), dtype=np.float32)
        sin_out = np.zeros((total_seq, head_dim), dtype=np.float32)

        # 1. Text Part
        t_pos = np.arange(txt_seq, dtype=np.float32).reshape(-1, 1)
        args = t_pos * freqs.reshape(1, -1)
        c = np.cos(args)
        s = np.sin(args)
        ax3_start = axis_dim * 3
        # Axis 3 is the last one, so ::2 works here, but bounded is safer
        cos_out[:txt_seq, ax3_start::2] = c
        cos_out[:txt_seq, ax3_start + 1 :: 2] = c
        sin_out[:txt_seq, ax3_start::2] = s
        sin_out[:txt_seq, ax3_start + 1 :: 2] = s

        # 2. Image Part
        y = np.arange(img_h, dtype=np.float32)
        x = np.arange(img_w, dtype=np.float32)
        yy, xx = np.meshgrid(y, x, indexing="ij")
        yy = yy.flatten().reshape(-1, 1)
        xx = xx.flatten().reshape(-1, 1)

        # Axis 1: Height
        ax1_start = axis_dim * 1
        ax1_end = ax1_start + axis_dim
        args_h = yy * freqs.reshape(1, -1)
        ch = np.cos(args_h)
        sh = np.sin(args_h)
        # Fix: bound the slice to [ax1_start, ax1_end)
        cos_out[txt_seq:, ax1_start:ax1_end:2] = ch
        cos_out[txt_seq:, ax1_start + 1 : ax1_end : 2] = ch
        sin_out[txt_seq:, ax1_start:ax1_end:2] = sh
        sin_out[txt_seq:, ax1_start + 1 : ax1_end : 2] = sh

        # Axis 2: Width
        ax2_start = axis_dim * 2
        ax2_end = ax2_start + axis_dim
        args_w = xx * freqs.reshape(1, -1)
        cw = np.cos(args_w)
        sw = np.sin(args_w)
        # Fix: bound the slice to [ax2_start, ax2_end)
        cos_out[txt_seq:, ax2_start:ax2_end:2] = cw
        cos_out[txt_seq:, ax2_start + 1 : ax2_end : 2] = cw
        sin_out[txt_seq:, ax2_start:ax2_end:2] = sw
        sin_out[txt_seq:, ax2_start + 1 : ax2_end : 2] = sw

        return (
            cos_out.reshape(1, 1, total_seq, head_dim),
            sin_out.reshape(1, 1, total_seq, head_dim),
        )

    def build_transformer(self, latent_h: int = 16, latent_w: int = 16):
        """Build and compile the transformer. Resolution is locked at first compile."""
        if self.transformer_session:
            return

        print(f"Building Transformer graph for {latent_w * 16}x{latent_h * 16}...")
        self.load_transformer()
        self.flux_transformer = FluxTransformer(self.cfg)
        self.transformer_session = GraphSession(self.flux_transformer.output)

        txt_len = self.cfg.text_max_seq

        dummy_rope_cos, dummy_rope_sin = self.compute_rope(
            txt_len, latent_h, latent_w, self.cfg.head_dim, self.cfg.rope_theta
        )

        sample_inputs = {
            "img_latent": np.zeros(
                (1, self.cfg.latent_channels, latent_h, latent_w), dtype=np.float32
            ),
            "txt_emb": np.zeros((1, txt_len, self.cfg.text_dim), dtype=np.float32),
            "timestep": np.array([1.0], dtype=np.float32),
            "img_h": np.array([latent_h], dtype=np.int32),
            "img_w": np.array([latent_w], dtype=np.int32),
            "txt_seq": np.array([txt_len], dtype=np.int32),
            "rope_cos": dummy_rope_cos,
            "rope_sin": dummy_rope_sin,
        }

        self.transformer_session.compile(sample_inputs)
        print("Transformer ready!")

    def build_text_encoder(self):
        if self.text_encoder_session:
            return
        print("Building text encoder graph...")
        input_ids = self.text_encoder.input(
            "input_ids", (1, self.cfg.text_max_seq), DType.INT32
        )
        self.load_text_encoder()
        if not self.text_encoder_weights:
            print("Skipping text encoder build (no weights)")
            return

        # Prepare weights dict
        weights_nodes = {}
        for k in self.text_encoder_weights.keys():
            # Check for layers beyond 26 to save compilation time/memory (optimization)
            if "model.layers." in k:
                layer_idx = int(k.split(".")[2])
                if layer_idx > 26:
                    continue

            shape = self.text_encoder_weights.get_tensor_metadata(k)[0]
            weights_nodes[k] = self.text_encoder.param(k, shape)

        embeddings = self.text_encoder.forward(input_ids, weights_nodes)
        self.text_encoder_session = GraphSession(embeddings)
        sample_inputs = {
            "input_ids": np.zeros((1, self.cfg.text_max_seq), dtype=np.int32),
            "seq_len": np.array([self.cfg.text_max_seq], dtype=np.int32),
        }
        self.text_encoder_session.compile(sample_inputs)
        print("Text encoder ready!")

    def build_vae_decoder(self):
        if self.vae_session:
            return
        print("Building VAE decoder graph...")
        self.load_vae()
        if not self.vae_weights:
            print("Skipping VAE build (no weights)")
            return

        latent = self.vae_decoder.input(
            "latent", (1, self.cfg.vae_channels, None, None), DType.FP32
        )
        h_node = self.vae_decoder.input("h", (1,), DType.INT32)
        w_node = self.vae_decoder.input("w", (1,), DType.INT32)
        weights = {}
        for key in self.vae_weights.keys():
            weights[key] = self.vae_decoder.param(
                key, self.vae_weights.get_tensor_metadata(key)[0]
            )

        image = self.vae_decoder.decode(latent, h_node, w_node, weights)
        self.vae_session = GraphSession(image)
        sample_latent = np.zeros((1, self.cfg.vae_channels, 8, 8), dtype=np.float32)
        self.vae_session.compile(
            {
                "latent": sample_latent,
                "h": np.array([8], dtype=np.int32),
                "w": np.array([8], dtype=np.int32),
            }
        )
        print("VAE decoder ready!")

    def encode_text(self, prompt: str) -> np.ndarray:
        print(f"Encoding prompt: {prompt[:50]}...")
        self.load_tokenizer()
        if not self.tokenizer:
            raise ValueError("no tokenizer")

        tokens = self.tokenizer.encode(prompt).ids
        pad_id = 151643
        if len(tokens) > self.cfg.text_max_seq:
            tokens = tokens[: self.cfg.text_max_seq]
        else:
            tokens = tokens + [pad_id] * (self.cfg.text_max_seq - len(tokens))
        input_ids = np.array([tokens], dtype=np.int32)

        self.build_text_encoder()
        if self.text_encoder_session:
            self.text_encoder_session.load_weights(self.text_encoder_weights)
            inputs = {
                "input_ids": input_ids,
                "seq_len": np.array([self.cfg.text_max_seq], dtype=np.int32),
            }
            return self.text_encoder_session.run(inputs)
        raise ValueError("no text_encoder_session")

    def decode_latent(self, latent: np.ndarray) -> np.ndarray:
        print("Decoding latent...")
        self.build_vae_decoder()
        if self.vae_session:
            h_val = latent.shape[2]
            w_val = latent.shape[3]
            self.vae_session.load_weights(self.vae_weights)
            image_tensor = self.vae_session.run(
                {
                    "latent": latent,
                    "h": np.array([h_val], dtype=np.int32),
                    "w": np.array([w_val], dtype=np.int32),
                }
            )
            img = image_tensor[0].transpose(1, 2, 0)
            img = (img + 1.0) * 0.5
            return np.clip(img * 255, 0, 255).astype(np.uint8)

        # Fallback visualization
        img = latent[0, :3, :, :].transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)
        return (img * 255).astype(np.uint8)

    def sample(
        self,
        text_emb: np.ndarray,
        height: int = 256,
        width: int = 256,
        num_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)

        latent_h = height // 16
        latent_w = width // 16
        img_seq_len = latent_h * latent_w

        # Initialize noise
        z = np.random.randn(1, self.cfg.latent_channels, latent_h, latent_w).astype(
            np.float32
        )

        steps = num_steps or self.cfg.num_steps_distilled
        sampler = Sampler(steps)
        # Proper Flux schedule depends on the number of tokens (latent area)
        schedule = sampler.get_schedule(img_seq_len)

        self.build_transformer(latent_h, latent_w)
        self.transformer_session.load_weights(self.transformer_weights)

        txt_seq = text_emb.shape[1]
        rope_cos, rope_sin = self.compute_rope(
            txt_seq, latent_h, latent_w, self.cfg.head_dim, self.cfg.rope_theta
        )

        for i in tqdm(range(steps), desc="Sampling"):
            t_curr = schedule[i]
            t_next = schedule[i + 1]
            dt = t_next - t_curr  # Negative for denoising

            inputs = {
                "img_latent": z,
                "txt_emb": text_emb,
                "timestep": np.array([t_curr], dtype=np.float32),
                "img_h": np.array([latent_h], dtype=np.int32),
                "img_w": np.array([latent_w], dtype=np.int32),
                "txt_seq": np.array([txt_seq], dtype=np.int32),
                "rope_cos": rope_cos,
                "rope_sin": rope_sin,
            }

            # Predict velocity
            velocity = self.transformer_session.run(inputs)

            # Euler update
            z = sampler.step(z, velocity, dt)

        return z

    def generate(
        self,
        prompt: str,
        height: int = 128,
        width: int = 128,
        num_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        print(f"\nGenerating: {prompt[:50]}...")
        print(f"Size: {width}x{height}")

        prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        text_emb = self.encode_text(prompt)
        # text_emb = np.zeros(
        #     (1, self.cfg.text_max_seq, self.cfg.text_dim)
        # )  # TODO: remove, this is a placeholder to skip expensive text encoding while testing sampling/latent decoding.
        latent = self.sample(text_emb, height, width, num_steps, seed)
        image = self.decode_latent(latent)
        # image = None

        print("Done!")
        return image


def main():
    model_dir = "flux-klein-4b"
    if not os.path.exists(model_dir):
        if os.path.exists("diffusion_pytorch_model.safetensors"):
            model_dir = "."
        else:
            print(f"Model directory '{model_dir}' not found.")
            return

    cfg = FluxConfig()
    pipeline = FluxPipeline(model_dir, cfg)

    while True:
        prompt = input("Enter prompt: ")
        image = pipeline.generate(
            prompt=prompt,
            height=128,
            width=128,
            num_steps=4,
            seed=42,
        )

        output_path = "flux_output.png"
        Image.fromarray(image).save(output_path)
        print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()

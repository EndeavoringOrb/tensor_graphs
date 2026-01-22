import os
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, Any

from safetensors.torch import load_file
from tokenizers import Tokenizer

# --- Tensor Graphs Imports ---
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType, TensorSignature
from tensor_graphs.ops.atomic import OpType
from tensor_graphs.backend.reference import evaluate_graph

# Import Fused Ops Definitions
from tensor_graphs.ops.fused.llm import RoPE, Embedding
from tensor_graphs.ops.fused.norm import RMSNorm
from tensor_graphs.ops.fused.activation import GELU, Softmax

# Ensure Kernels are Registered
import tensor_graphs.backend.kernels


# ==============================================================================
# 1. Graph Construction Helpers
# ==============================================================================


class GraphBuilder:
    def __init__(self):
        self.params = {}
        self.inputs = {}

    def input(self, name, shape, dtype=DType.FP32):
        node = TensorNode(OpType.INPUT, shape, dtype, [], name)
        self.inputs[name] = node
        return node

    def param(self, name, shape, dtype=DType.FP32):
        node = TensorNode(OpType.INPUT, shape, dtype, [], name)
        self.params[name] = node
        return node

    # --- Atomic Wrappers ---
    def add(self, a, b):
        return TensorNode(OpType.ADD, a.shape, DType.FP32, [a, b], f"add_{a.name}")

    def mul(self, a, b):
        return TensorNode(OpType.MUL, a.shape, DType.FP32, [a, b], f"mul_{a.name}")

    def matmul(self, a, b):
        out_shape = list(a.shape[:-1]) + [b.shape[-1]]
        return TensorNode(
            OpType.DOT, tuple(out_shape), DType.FP32, [a, b], f"dot_{a.name}"
        )

    def reshape(self, x, target_shape, shape_node):
        return TensorNode(
            OpType.RESHAPE,
            target_shape,
            DType.FP32,
            [x, shape_node],
            f"reshape_{x.name}",
        )

    def permute(self, x, dims, perm_node):
        new_shape = tuple(x.shape[i] for i in dims)
        return TensorNode(
            OpType.PERMUTE, new_shape, DType.FP32, [x, perm_node], f"permute_{x.name}"
        )

    # --- Fused / Composite Wrappers (Using Library Ops) ---
    def embedding(self, indices, weights):
        out_shape = indices.shape + (weights.shape[-1],)
        return TensorNode(
            Embedding.op_type, out_shape, DType.FP32, [indices, weights], "embed"
        )

    def rms_norm(self, x, scale, eps_node):
        return TensorNode(
            RMSNorm.op_type,
            x.shape,
            DType.FP32,
            [x, scale, eps_node],
            f"rmsnorm_{x.name}",
        )

    def gelu(self, x):
        return TensorNode(GELU.op_type, x.shape, DType.FP32, [x], "gelu")

    def softmax(self, x):
        return TensorNode(Softmax.op_type, x.shape, DType.FP32, [x], "softmax")

    def rope(self, x, cos, sin):
        return TensorNode(RoPE.op_type, x.shape, DType.FP32, [x, cos, sin], "rope")

    def repeat(self, x, times, rep_node):
        new_shape = list(x.shape)
        new_shape[1] *= times  # Specific to GQA logic
        return TensorNode(
            OpType.REPEAT,
            tuple(new_shape),
            DType.FP32,
            [x, rep_node],
            f"repeat_{x.name}",
        )


# ==============================================================================
# 2. Model Definition (Gemma 3)
# ==============================================================================


class Gemma3Model:
    def __init__(self, cfg, weights: Dict[str, np.ndarray]):
        self.cfg = cfg
        self.weights = weights
        self.builder = GraphBuilder()
        self.constant_inputs = {}

    def _get_param(self, name, shape):
        if name not in self.weights:
            # Fallback for tied weights if not present under specific name
            return self.builder.param(name, shape)
        return self.builder.param(name, shape)

    def _const(self, value, name, dtype=DType.INT32):
        val_arr = np.array(
            value, dtype=np.int32 if dtype == DType.INT32 else np.float32
        )
        node = self.builder.input(name, val_arr.shape, dtype)
        self.constant_inputs[name] = val_arr
        return node

    def forward(self, input_ids_node, cos, sin):
        cfg = self.cfg
        B, S = input_ids_node.shape

        # Embedding
        w_emb = self._get_param(
            "model.embed_tokens.weight", (cfg["vocab_size"], cfg["emb_dim"])
        )
        x = self.builder.embedding(input_ids_node, w_emb)

        # Scale
        scale_val = self._const([cfg["emb_dim"] ** 0.5], "emb_scale_val", DType.FP32)
        x = self.builder.mul(x, scale_val)

        # Layers
        for i in range(cfg["n_layers"]):
            x = self._transformer_block(x, i, cos, sin, B, S)

        # Final Norm
        w_norm = self._get_param("model.norm.weight", (cfg["emb_dim"],))
        eps_node = self._const([1e-6], "final_norm_eps", DType.FP32)
        x = self.builder.rms_norm(x, w_norm, eps_node)

        # Head (Weight Tying)
        w_head = w_emb
        if "lm_head.weight" in self.weights:
            w_head = self._get_param(
                "lm_head.weight", (cfg["vocab_size"], cfg["emb_dim"])
            )

        # Permute for MatMul
        perm_t = self._const([1, 0], "perm_head_T")
        w_head_t = self.builder.permute(w_head, [1, 0], perm_t)
        logits = self.builder.matmul(x, w_head_t)

        return logits

    def _transformer_block(self, x, layer_idx, cos, sin, B, S):
        cfg = self.cfg
        prefix = f"model.layers.{layer_idx}"
        residual = x

        # Input Norm
        w_ln = self._get_param(f"{prefix}.input_layernorm.weight", (cfg["emb_dim"],))
        eps_ln = self._const([1e-6], f"ln_eps_{layer_idx}", DType.FP32)
        x_norm = self.builder.rms_norm(x, w_ln, eps_ln)

        # Attention
        x_attn = self._attention(x_norm, layer_idx, cos, sin, B, S)

        # Post Attn Norm
        w_post = self._get_param(
            f"{prefix}.post_attention_layernorm.weight", (cfg["emb_dim"],)
        )
        eps_post = self._const([1e-6], f"post_attn_eps_{layer_idx}", DType.FP32)
        x_attn = self.builder.rms_norm(x_attn, w_post, eps_post)

        x = self.builder.add(residual, x_attn)

        # Feed Forward
        residual = x
        w_pre = self._get_param(
            f"{prefix}.pre_feedforward_layernorm.weight", (cfg["emb_dim"],)
        )
        eps_pre = self._const([1e-6], f"pre_ff_eps_{layer_idx}", DType.FP32)
        x_norm = self.builder.rms_norm(x, w_pre, eps_pre)

        x_ff = self._mlp(x_norm, layer_idx)

        w_post_ff = self._get_param(
            f"{prefix}.post_feedforward_layernorm.weight", (cfg["emb_dim"],)
        )
        eps_post_ff = self._const([1e-6], f"post_ff_eps_{layer_idx}", DType.FP32)
        x_ff = self.builder.rms_norm(x_ff, w_post_ff, eps_post_ff)

        return self.builder.add(residual, x_ff)

    def _attention(self, x, layer_idx, cos, sin, B, S):
        cfg = self.cfg
        prefix = f"model.layers.{layer_idx}.self_attn"
        head_dim = cfg["head_dim"]
        n_heads = cfg["n_heads"]
        n_kv = cfg["n_kv_groups"]
        d_model = cfg["emb_dim"]

        # Projections
        wq = self._get_param(f"{prefix}.q_proj.weight", (n_heads * head_dim, d_model))
        wk = self._get_param(f"{prefix}.k_proj.weight", (n_kv * head_dim, d_model))
        wv = self._get_param(f"{prefix}.v_proj.weight", (n_kv * head_dim, d_model))

        perm_w = self._const([1, 0], f"perm_w_{layer_idx}")
        wq_t = self.builder.permute(wq, [1, 0], perm_w)
        wk_t = self.builder.permute(wk, [1, 0], perm_w)
        wv_t = self.builder.permute(wv, [1, 0], perm_w)

        q = self.builder.matmul(x, wq_t)
        k = self.builder.matmul(x, wk_t)
        v = self.builder.matmul(x, wv_t)

        # Reshape & Permute
        shape_q = self._const([B, S, n_heads, head_dim], f"shape_q_{B}_{S}")
        shape_kv = self._const([B, S, n_kv, head_dim], f"shape_kv_{B}_{S}")
        perm_attn = self._const([0, 2, 1, 3], f"perm_attn_{layer_idx}")

        q = self.builder.permute(
            self.builder.reshape(q, (B, S, n_heads, head_dim), shape_q),
            [0, 2, 1, 3],
            perm_attn,
        )
        k = self.builder.permute(
            self.builder.reshape(k, (B, S, n_kv, head_dim), shape_kv),
            [0, 2, 1, 3],
            perm_attn,
        )
        v = self.builder.permute(
            self.builder.reshape(v, (B, S, n_kv, head_dim), shape_kv),
            [0, 2, 1, 3],
            perm_attn,
        )

        # QK Norm
        w_q_norm = self._get_param(f"{prefix}.q_norm.weight", (head_dim,))
        w_k_norm = self._get_param(f"{prefix}.k_norm.weight", (head_dim,))
        eps_qk = self._const([1e-6], f"eps_qk_{layer_idx}", DType.FP32)
        q = self.builder.rms_norm(q, w_q_norm, eps_qk)
        k = self.builder.rms_norm(k, w_k_norm, eps_qk)

        # RoPE
        q = self.builder.rope(q, cos, sin)
        k = self.builder.rope(k, cos, sin)

        # GQA Repeat
        if n_heads != n_kv:
            rep_node = self._const([n_heads // n_kv], f"rep_{layer_idx}")
            k = self.builder.repeat(k, n_heads // n_kv, rep_node)
            v = self.builder.repeat(v, n_heads // n_kv, rep_node)

        # Scale
        scale_node = self._const(
            [cfg["query_pre_attn_scalar"] ** -0.5], "attn_scale", DType.FP32
        )
        q = self.builder.mul(q, scale_node)

        # Scores
        perm_kt = self._const([0, 1, 3, 2], "perm_kt")
        k_t = self.builder.permute(k, [0, 1, 3, 2], perm_kt)
        scores = self.builder.matmul(q, k_t)

        # Mask
        mask_node = self.builder.input("causal_mask", (1, 1, S, S), DType.FP32)
        scores = self.builder.add(scores, mask_node)
        probs = self.builder.softmax(scores)

        # Context
        context = self.builder.matmul(probs, v)
        perm_back = self._const([0, 2, 1, 3], f"perm_back_{layer_idx}")
        context = self.builder.permute(context, [0, 2, 1, 3], perm_back)

        shape_flat = self._const([B, S, n_heads * head_dim], f"shape_flat_{B}_{S}")
        context = self.builder.reshape(context, (B, S, n_heads * head_dim), shape_flat)

        # Output Proj
        wo = self._get_param(f"{prefix}.o_proj.weight", (d_model, n_heads * head_dim))
        wo_t = self.builder.permute(wo, [1, 0], perm_w)
        return self.builder.matmul(context, wo_t)

    def _mlp(self, x, layer_idx):
        cfg = self.cfg
        prefix = f"model.layers.{layer_idx}.mlp"
        d_model = cfg["emb_dim"]
        d_hidden = cfg["hidden_dim"]

        w_gate = self._get_param(f"{prefix}.gate_proj.weight", (d_hidden, d_model))
        w_up = self._get_param(f"{prefix}.up_proj.weight", (d_hidden, d_model))
        w_down = self._get_param(f"{prefix}.down_proj.weight", (d_model, d_hidden))

        perm_mlp = self._const([1, 0], f"perm_mlp_{layer_idx}")
        w_gate_t = self.builder.permute(w_gate, [1, 0], perm_mlp)
        w_up_t = self.builder.permute(w_up, [1, 0], perm_mlp)
        w_down_t = self.builder.permute(w_down, [1, 0], perm_mlp)

        gate = self.builder.gelu(self.builder.matmul(x, w_gate_t))
        up = self.builder.matmul(x, w_up_t)

        return self.builder.matmul(self.builder.mul(gate, up), w_down_t)


# ==============================================================================
# 4. Utilities
# ==============================================================================


def compute_rope_params_numpy(head_dim, theta_base=10000.0, context_length=4096):
    inv_freq = 1.0 / (
        theta_base ** (np.arange(0, head_dim, 2).astype(np.float32) / head_dim)
    )
    positions = np.arange(context_length, dtype=np.float32)
    angles = np.outer(positions, inv_freq)
    angles = np.concatenate([angles, angles], axis=1)  # [S, D]
    cos = np.cos(angles)
    sin = np.sin(angles)
    # Shape for broadcasting: [1, 1, S, D]
    return cos[None, None, :, :], sin[None, None, :, :]


def create_causal_mask(seq_len):
    # 0 for keep, -1e9 for mask
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    return mask[None, None, :, :] * -1e9


GEMMA3_CONFIG_270M = {
    "vocab_size": 262_144,
    "context_length": 32_768,
    "emb_dim": 640,
    "n_heads": 4,
    "n_layers": 18,
    "hidden_dim": 2048,
    "head_dim": 256,
    "n_kv_groups": 1,
    "query_pre_attn_scalar": 256,
}

# ==============================================================================
# 5. Main Execution
# ==============================================================================


def main():
    print("Initializing Gemma 3 (270M) on tensor_graphs...")

    # 1. Download/Load Weights
    tokenizer_path = "resources/tokenizer.json"
    weights_path = "resources/model.safetensors"

    print(f"Loading weights from {weights_path}...")
    state_dict = load_file(weights_path)

    # Convert weights to FP32 numpy
    weights_np = {}
    print("Converting weights to FP32 numpy...")
    for k, v in tqdm(state_dict.items()):
        weights_np[k] = v.to(torch.float32).numpy()

    # 2. Setup Tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # 3. Setup Logic
    prompt = "<start_of_turn>user\nExplain Quantum Mechanics to a 5 year old.<end_of_turn>\n<start_of_turn>model\n"
    input_ids = tokenizer.encode(prompt).ids

    # Generation Loop
    max_new_tokens = 50
    print(f"\nPrompt: {prompt}")
    print("Generating...", end="", flush=True)

    # Precompute RoPE cache (Full Context)
    full_cos, full_sin = compute_rope_params_numpy(
        GEMMA3_CONFIG_270M["head_dim"],
        theta_base=10000.0,  # Approximate for demo
        context_length=4096,
    )

    for _ in range(max_new_tokens):
        seq_len = len(input_ids)

        # Create Graph for current sequence length
        # In a compiled backend, we would use symbolic shapes.
        # Here we rebuild inputs specific to the current iteration.

        # 3a. Prepare Inputs
        input_ids_np = np.array([input_ids], dtype=np.int32)  # [1, S]

        # Slice RoPE to current length
        cos_cur = full_cos[:, :, :seq_len, :]
        sin_cur = full_sin[:, :, :seq_len, :]

        # Causal Mask
        mask_cur = create_causal_mask(seq_len).astype(np.float32)

        # 3b. Build Graph
        # We instantiate a new builder/model wrapper to create a fresh graph structure
        model = Gemma3Model(GEMMA3_CONFIG_270M, weights_np)

        # Define Input Nodes
        in_node = model.builder.input("input_ids", (1, seq_len), DType.INT32)
        cos_node = model.builder.input(
            "cos", (1, 1, seq_len, GEMMA3_CONFIG_270M["head_dim"]), DType.FP32
        )
        sin_node = model.builder.input(
            "sin", (1, 1, seq_len, GEMMA3_CONFIG_270M["head_dim"]), DType.FP32
        )
        mask_node = model.builder.input(
            "causal_mask", (1, 1, seq_len, seq_len), DType.FP32
        )

        # Build Forward Pass
        # Pass required auxiliary nodes
        logits_node = model.forward(in_node, cos_node, sin_node)

        # 3c. Prepare Feed Dict
        feed_dict = {
            "input_ids": input_ids_np,
            "cos": cos_cur,
            "sin": sin_cur,
            "causal_mask": mask_cur,
            **model.weights,  # Bind weights
            **model.constant_inputs,  # Bind generated constants
        }

        # 3d. Execute
        # evaluate_graph returns numpy array
        logits_out = evaluate_graph(logits_node, feed_dict)

        # 3e. Next Token Strategy (Greedy)
        next_token_logits = logits_out[0, -1, :]
        next_token_id = int(np.argmax(next_token_logits))

        input_ids.append(next_token_id)

        # Decode and Print
        word = tokenizer.decode([next_token_id])
        print(word, end="", flush=True)

        if next_token_id == tokenizer.token_to_id("<end_of_turn>"):
            break

    print("\n\nDone.")


if __name__ == "__main__":
    main()

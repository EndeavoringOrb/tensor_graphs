import os
import torch
import numpy as np
from tqdm import tqdm
from importlib.metadata import version
from typing import Dict, Any, List, Optional, Tuple

from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file
from tokenizers import Tokenizer

# --- Tensor Graphs Imports ---
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType, TensorSignature
from tensor_graphs.ops.atomic import OpType
from tensor_graphs.backend.registry import KernelRegistry
from tensor_graphs.backend.reference import evaluate_graph
import tensor_graphs.backend.kernels  # Import to ensure standard kernels are registered

# ==============================================================================
# 1. Custom Op Definitions & Kernel Registrations
# ==============================================================================


# Extended OpTypes for LLM building blocks
class LLMOpType:
    EMBEDDING = "Embedding"
    RMS_NORM = "RMSNorm"
    GELU = "GELU"
    SOFTMAX = "Softmax"
    ROPE = "RoPE"
    REPEAT = "Repeat"


# --- Embedding Kernel ---
@KernelRegistry.register(
    LLMOpType.EMBEDDING,
    [
        TensorSignature(DType.INT32, (None, None)),  # Indices: [Batch, Seq]
        TensorSignature(DType.FP32, (None, None)),  # Weights: [Vocab, Dim]
    ],
)
def embedding_kernel(inputs):
    indices = inputs[0].astype(int)
    weights = inputs[1]
    return weights[indices]


# --- RMSNorm Kernel (Gemma 3 specific: includes 1+scale and bias) ---
@KernelRegistry.register(
    LLMOpType.RMS_NORM,
    [
        TensorSignature(DType.FP32, shape=None),  # X: Match any rank
        TensorSignature(DType.FP32, (None,)),  # Scale: [D]
        TensorSignature(DType.FP32, (1,)),  # Eps: [1]
    ],
)
def rms_norm_kernel(inputs):
    x = inputs[0]
    scale = inputs[1]
    eps = inputs[2][0]

    # Float32 accumulation for precision
    x_f = x.astype(np.float32)
    var = np.mean(x_f**2, axis=-1, keepdims=True)
    x_norm = x_f * (1.0 / np.sqrt(var + eps))

    # Gemma 3 uses (1 + scale)
    out = x_norm * (1.0 + scale)
    return out


# --- GELU (Tanh Approximation) ---
@KernelRegistry.register(LLMOpType.GELU, [TensorSignature(DType.FP32, shape=None)])
def gelu_kernel(inputs):
    x = inputs[0]
    # 0.5 * x * (1 + Tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    c1 = np.sqrt(2.0 / np.pi)
    c2 = 0.044715
    inner = c1 * (x + c2 * np.power(x, 3))
    return 0.5 * x * (1.0 + np.tanh(inner))


# --- Softmax ---
@KernelRegistry.register(LLMOpType.SOFTMAX, [TensorSignature(DType.FP32, shape=None)])
def softmax_kernel(inputs):
    x = inputs[0]
    # Stability fix: sub max
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    sum_exp = np.sum(exp_x, axis=-1, keepdims=True)
    return exp_x / sum_exp


# --- RoPE Kernel ---
@KernelRegistry.register(
    LLMOpType.ROPE,
    [
        TensorSignature(DType.FP32, shape=None),  # X: [B, H, S, D]
        TensorSignature(DType.FP32, shape=None),  # Cos: [1, 1, S, D]
        TensorSignature(DType.FP32, shape=None),  # Sin: [1, 1, S, D]
    ],
)
def rope_kernel(inputs):
    x = inputs[0]
    cos = inputs[1]
    sin = inputs[2]

    # x is [B, Heads, Seq, HeadDim]
    # Split last dim
    head_dim = x.shape[-1]
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]

    rotated = np.concatenate((-x2, x1), axis=-1)

    # Broadcast cos/sin (assumes they are pre-sliced/shaped correctly)
    return (x * cos) + (rotated * sin)


# --- Repeat Interleave (for GQA) ---
@KernelRegistry.register(
    LLMOpType.REPEAT,
    [
        TensorSignature(DType.FP32, shape=None),  # Data
        TensorSignature(DType.INT32, (1,)),  # Repeats
    ],
)
def repeat_kernel(inputs):
    x = inputs[0]
    repeats = int(inputs[1][0])
    # Assume repeating dim 1 (Heads) for GQA: [B, KV_Heads, S, D] -> [B, KV_Heads * Rep, S, D]
    # Note: Reference repeats dim 1.
    return np.repeat(x, repeats, axis=1)


# ==============================================================================
# 2. Graph Construction Helpers
# ==============================================================================


class GraphBuilder:
    def __init__(self):
        self.params = {}  # Map param_name -> TensorNode
        self.inputs = {}  # Map input_name -> TensorNode

    def input(self, name, shape, dtype=DType.FP32):
        node = TensorNode(OpType.INPUT, shape, dtype, [], name)
        self.inputs[name] = node
        return node

    def param(self, name, shape, dtype=DType.FP32):
        node = TensorNode(OpType.INPUT, shape, dtype, [], name)
        self.params[name] = node
        return node

    def add(self, a, b):
        return TensorNode(
            OpType.ADD, a.shape, DType.FP32, [a, b], f"add_{a.name}_{b.name}"
        )

    def mul(self, a, b):
        return TensorNode(
            OpType.MUL, a.shape, DType.FP32, [a, b], f"mul_{a.name}_{b.name}"
        )

    def matmul(self, a, b):
        # Infer shape: A[..., M, K], B[K, N] -> [..., M, N]
        out_shape = list(a.shape[:-1]) + [b.shape[-1]]
        return TensorNode(
            OpType.DOT, tuple(out_shape), DType.FP32, [a, b], f"dot_{a.name}"
        )

    def embedding(self, indices, weights):
        # [B, S], [V, D] -> [B, S, D]
        out_shape = indices.shape + (weights.shape[-1],)
        return TensorNode(
            LLMOpType.EMBEDDING, out_shape, DType.FP32, [indices, weights], "embed"
        )

    def rms_norm(self, x, scale, eps_node):
        return TensorNode(
            LLMOpType.RMS_NORM,
            x.shape,
            DType.FP32,
            [x, scale, eps_node],
            f"rmsnorm_{x.name}",
        )

    def gelu(self, x):
        return TensorNode(LLMOpType.GELU, x.shape, DType.FP32, [x], "gelu")

    def softmax(self, x):
        return TensorNode(LLMOpType.SOFTMAX, x.shape, DType.FP32, [x], "softmax")

    def rope(self, x, cos, sin):
        return TensorNode(LLMOpType.ROPE, x.shape, DType.FP32, [x, cos, sin], "rope")

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

    def repeat(self, x, times, rep_node):
        new_shape = list(x.shape)
        new_shape[1] *= times
        return TensorNode(
            LLMOpType.REPEAT,
            tuple(new_shape),
            DType.FP32,
            [x, rep_node],
            f"repeat_{x.name}",
        )


# ==============================================================================
# 3. Model Definition
# ==============================================================================


class Gemma3Model:
    def __init__(self, cfg, weights: Dict[str, np.ndarray]):
        self.cfg = cfg
        self.weights = weights
        self.builder = GraphBuilder()
        self.constant_inputs = {}  # Stores constants like eps, shapes, perms

    def _get_param(self, name, shape):
        if name not in self.weights:
            raise ValueError(f"Weight '{name}' not found!")
        # Validate shape if possible (skip for flexibility)
        return self.builder.param(name, shape)

    def _const(self, value, name, dtype=DType.INT32):
        # Helper to create constant inputs needed for Reshape/Permute
        val_arr = np.array(
            value, dtype=np.int32 if dtype == DType.INT32 else np.float32
        )
        node = self.builder.input(name, val_arr.shape, dtype)
        self.constant_inputs[name] = val_arr
        return node

    def forward(self, input_ids_node, cos, sin):
        # input_ids_node: [B, S]
        # cos, sin: [1, 1, S, D] (passed as nodes)

        cfg = self.cfg
        B, S = input_ids_node.shape

        # Embedding
        w_emb = self._get_param(
            "model.embed_tokens.weight", (cfg["vocab_size"], cfg["emb_dim"])
        )
        x = self.builder.embedding(input_ids_node, w_emb)
        # Scale
        scale_val = self._const([cfg["emb_dim"] ** 0.5], "emb_scale_val", DType.FP32)
        # Broadcast scale manually or rely on generic MUL?
        # Generic MUL kernel supports Scalar * Matrix broadcasting?
        # Let's assume yes based on 'mul_scalar' kernel or generic fallback.
        # Actually generic mul is (None,) * (None,).
        # We need a Reshape on scale if it fails, but let's try direct.
        # For safety, let's just multiply in the kernel or assume numpy broadcasting works.
        # tensor_graphs mul.py has mul_scalar for (1,).
        x = self.builder.mul(x, scale_val)

        # Layers
        for i in range(cfg["n_layers"]):
            x = self._transformer_block(x, i, cos, sin, B, S)

        # Final Norm
        w_norm = self._get_param("model.norm.weight", (cfg["emb_dim"],))
        eps_node = self._const([1e-6], "final_norm_eps", DType.FP32)
        x = self.builder.rms_norm(x, w_norm, eps_node)

        # Output Head (Weight Tying)
        # Gemma 3 ties weights? "model.embed_tokens.weight" usually.
        # The reference checks if "lm_head.weight" exists, else uses embed tokens.
        if "lm_head.weight" in self.weights:
            w_head = self._get_param(
                "lm_head.weight", (cfg["vocab_size"], cfg["emb_dim"])
            )
        else:
            w_head = w_emb  # Reuse embedding node

        # MatMul: [B, S, D] @ [V, D].T -> [B, S, V]
        # Our DOT kernel is MatMul(A, B).
        # We need to transpose w_head.
        # Or we can implement a Linear Op.
        # Let's Permute w_head.
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

        x_attn = self._attention(x_norm, layer_idx, cos, sin, B, S)

        # Post Attn Norm
        w_post_attn = self._get_param(
            f"{prefix}.post_attention_layernorm.weight", (cfg["emb_dim"],)
        )
        eps_post_attn = self._const([1e-6], f"post_attn_eps_{layer_idx}", DType.FP32)
        x_attn = self.builder.rms_norm(x_attn, w_post_attn, eps_post_attn)

        x = self.builder.add(residual, x_attn)

        # Feed Forward
        residual = x
        w_pre_ff = self._get_param(
            f"{prefix}.pre_feedforward_layernorm.weight", (cfg["emb_dim"],)
        )
        eps_pre_ff = self._const([1e-6], f"pre_ff_eps_{layer_idx}", DType.FP32)
        x_norm = self.builder.rms_norm(x, w_pre_ff, eps_pre_ff)

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

        # QKV Projections
        wq = self._get_param(f"{prefix}.q_proj.weight", (n_heads * head_dim, d_model))
        wk = self._get_param(f"{prefix}.k_proj.weight", (n_kv * head_dim, d_model))
        wv = self._get_param(f"{prefix}.v_proj.weight", (n_kv * head_dim, d_model))

        # Transpose weights for MatMul: x @ W.T
        perm_w_node = self._const(
            [1, 0], f"perm_w_{layer_idx}"
        )  # Create once for reuse
        wq_t = self.builder.permute(wq, [1, 0], perm_w_node)
        wk_t = self.builder.permute(wk, [1, 0], perm_w_node)
        wv_t = self.builder.permute(wv, [1, 0], perm_w_node)

        # Add constants for perms
        self._const([1, 0], f"perm_w_{layer_idx}")

        q = self.builder.matmul(x, wq_t)  # [B, S, H*D]
        k = self.builder.matmul(x, wk_t)  # [B, S, KV*D]
        v = self.builder.matmul(x, wv_t)  # [B, S, KV*D]

        # Reshape to [B, S, H, D]
        shape_q = self._const([B, S, n_heads, head_dim], f"shape_q_{B}_{S}")
        shape_kv = self._const([B, S, n_kv, head_dim], f"shape_kv_{B}_{S}")

        q = self.builder.reshape(q, (B, S, n_heads, head_dim), shape_q)
        k = self.builder.reshape(k, (B, S, n_kv, head_dim), shape_kv)
        v = self.builder.reshape(v, (B, S, n_kv, head_dim), shape_kv)

        # Transpose to [B, H, S, D]
        perm_attn_node = self._const([0, 2, 1, 3], f"perm_attn_{layer_idx}")
        q = self.builder.permute(q, [0, 2, 1, 3], perm_attn_node)
        k = self.builder.permute(k, [0, 2, 1, 3], perm_attn_node)
        v = self.builder.permute(v, [0, 2, 1, 3], perm_attn_node)

        # QK Norm (Gemma 3)
        w_q_norm = self._get_param(f"{prefix}.q_norm.weight", (head_dim,))
        w_k_norm = self._get_param(f"{prefix}.k_norm.weight", (head_dim,))
        eps_qk_node = self._const([1e-6], f"eps_qk_{layer_idx}", DType.FP32)
        q = self.builder.rms_norm(q, w_q_norm, eps_qk_node)
        k = self.builder.rms_norm(k, w_k_norm, eps_qk_node)

        # RoPE
        q = self.builder.rope(q, cos, sin)
        k = self.builder.rope(k, cos, sin)

        # GQA Repeat
        if n_heads != n_kv:
            rep_node = self._const([n_heads // n_kv], f"rep_{layer_idx}")
            k = self.builder.repeat(k, n_heads // n_kv, rep_node)
            v = self.builder.repeat(v, n_heads // n_kv, rep_node)

        # Scale Queries
        scale_factor = cfg["query_pre_attn_scalar"] ** -0.5
        scale_node = self._const([scale_factor], "attn_scale", DType.FP32)
        q = self.builder.mul(q, scale_node)

        # Attention Score: Q @ K.T -> [B, H, S, D] @ [B, H, D, S] -> [B, H, S, S]
        # Transpose K last two dims
        perm_k_t = self._const([0, 1, 3, 2], "perm_kt")
        k_t = self.builder.permute(k, [0, 1, 3, 2], perm_k_t)  # FIXED: Added perm_k_t

        scores = self.builder.matmul(q, k_t)

        # Masking (Causal)
        # We need to add -inf to upper triangle.
        # We assume 'mask' input node is provided which has -inf in future positions.
        mask_node = self.builder.input("causal_mask", (1, 1, S, S), DType.FP32)
        scores = self.builder.add(scores, mask_node)

        probs = self.builder.softmax(scores)

        # Context: Probs @ V -> [B, H, S, S] @ [B, H, S, D] -> [B, H, S, D]
        context = self.builder.matmul(probs, v)

        # Transpose back: [B, S, H, D]
        perm_back_node = self._const([0, 2, 1, 3], f"perm_back_{layer_idx}")
        context = self.builder.permute(context, [0, 2, 1, 3], perm_back_node)

        # Flatten: [B, S, H*D]
        shape_flat = self._const([B, S, n_heads * head_dim], f"shape_flat_{B}_{S}")
        context = self.builder.reshape(context, (B, S, n_heads * head_dim), shape_flat)

        # Output Proj
        wo = self._get_param(f"{prefix}.o_proj.weight", (d_model, n_heads * head_dim))
        wo_t = self.builder.permute(wo, [1, 0], perm_w_node)
        out = self.builder.matmul(context, wo_t)

        return out

    def _mlp(self, x, layer_idx):
        cfg = self.cfg
        prefix = f"model.layers.{layer_idx}.mlp"
        d_model = cfg["emb_dim"]
        d_hidden = cfg["hidden_dim"]

        w_gate = self._get_param(f"{prefix}.gate_proj.weight", (d_hidden, d_model))
        w_up = self._get_param(f"{prefix}.up_proj.weight", (d_hidden, d_model))
        w_down = self._get_param(f"{prefix}.down_proj.weight", (d_model, d_hidden))

        # Transposes
        perm_mlp = self._const([1, 0], f"perm_mlp_{layer_idx}")
        w_gate_t = self.builder.permute(w_gate, [1, 0], perm_mlp)
        w_up_t = self.builder.permute(w_up, [1, 0], perm_mlp)
        w_down_t = self.builder.permute(w_down, [1, 0], perm_mlp)

        gate = self.builder.matmul(x, w_gate_t)
        up = self.builder.matmul(x, w_up_t)

        gate = self.builder.gelu(gate)

        act = self.builder.mul(gate, up)
        out = self.builder.matmul(act, w_down_t)

        return out


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

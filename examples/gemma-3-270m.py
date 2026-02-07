import os
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict

from safetensors.torch import load_file
from tokenizers import Tokenizer

# --- Tensor Graphs Imports ---
from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ir.buffer import StorageType
from tensor_graphs.ops.atomic_types import OpType
from tensor_graphs.backend.executor import evaluate_graph

# Compiler Imports

# Ensure Kernels are Registered

# ==============================================================================
# 1. Graph Construction Helpers
# ==============================================================================


class GraphBuilder:
    def __init__(self):
        self.params = {}
        self.inputs = {}
        self._count = 0

    def _next_name(self, op_name):
        self._count += 1
        return f"{op_name}_{self._count}"

    def input(self, name, shape, dtype=DType.FP32):
        node = TensorNode(
            OpType.INPUT, dtype, [], shape, name, storage_type=StorageType.TRANSIENT
        )
        self.inputs[name] = node
        return node

    def constant(self, value, shape, dtype, name):
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        node = TensorNode(
            OpType.CONSTANT,
            dtype,
            [],
            shape,
            name,
            attrs={"value": value},
            storage_type=StorageType.PERSISTENT,
        )
        return node

    def param(self, name, shape, dtype=DType.FP32):
        node = TensorNode(
            OpType.INPUT, dtype, [], shape, name, storage_type=StorageType.PERSISTENT
        )
        self.params[name] = node
        return node

    # --- Fixed Atomic Wrappers with Unique Names ---
    def add(self, a, b):
        return TensorNode(OpType.ADD, a.dtype, [a, b], name=self._next_name("add"))

    def mul(self, a, b):
        return TensorNode(
            OpType.MUL, a.dtype, [a, b], a.shape, name=self._next_name("mul")
        )

    def divide(self, a, b):
        return TensorNode(
            OpType.DIVIDE, a.dtype, [a, b], a.shape, name=self._next_name("div")
        )

    def matmul(self, a, b):
        return TensorNode(OpType.DOT, a.dtype, [a, b], name=self._next_name("dot"))

    def reshape(self, x, target_shape, shape_node):
        return TensorNode(
            OpType.RESHAPE,
            x.dtype,
            [x, shape_node],
            target_shape,
            name=self._next_name("reshape"),
        )

    def permute(self, x, dims, perm_node=None):
        # Using a more descriptive name for weight permutations
        return TensorNode(
            OpType.PERMUTE,
            x.dtype,
            [x],
            name=self._next_name("permute"),
            attrs={"dims": dims},
        )

    def concat(self, inputs, axis_node, axis_idx, output_shape):
        return TensorNode(
            OpType.CONCAT,
            inputs[0].dtype,
            inputs,
            output_shape,
            name=self._next_name("concat"),
            attrs={"axis": axis_idx},
        )

    def arange(self, start_node, stop_node, step_node):
        return TensorNode(
            OpType.ARANGE,
            DType.INT32,
            [start_node, stop_node, step_node],
            (None,),
            name=self._next_name("arange"),
        )

    def power(self, base, exp):
        return TensorNode(
            OpType.POWER,
            base.dtype,
            [base, exp],
            base.shape,
            name=self._next_name("pow"),
        )

    def triu(self, x, k_node):
        return TensorNode(
            OpType.TRIU, x.dtype, [x], x.shape, name=self._next_name("triu")
        )

    def cast(self, x, target_dtype):
        return TensorNode(
            OpType.CAST,
            target_dtype,
            [x],
            x.shape,
            name=self._next_name("cast"),
            attrs={"to": target_dtype},
        )

    def cos(self, x):
        return TensorNode(
            OpType.COS, x.dtype, [x], x.shape, name=self._next_name("cos")
        )

    def sin(self, x):
        return TensorNode(
            OpType.SIN, x.dtype, [x], x.shape, name=self._next_name("sin")
        )

    def fill(self, value_node, shape_node, target_shape):
        return TensorNode(
            OpType.FILL,
            value_node.dtype,
            [value_node, shape_node],
            target_shape,
            name=self._next_name("fill"),
        )

    def embedding(self, indices, weights):
        out_shape = (
            indices.shape + (weights.shape[-1],)
            if indices.shape and weights.shape
            else None
        )
        return TensorNode(
            OpType.GATHER,
            weights.dtype,
            [weights, indices],
            out_shape,
            name=self._next_name("embed"),
        )

    def rms_norm(self, x, scale, eps_node):
        return TensorNode(
            "RMSNorm",
            x.dtype,
            [x, scale, eps_node],
            x.shape,
            name=self._next_name("rmsnorm"),
        )

    def gelu(self, x):
        return TensorNode("GELU", x.dtype, [x], x.shape, name=self._next_name("gelu"))

    def softmax(self, x):
        return TensorNode(
            "Softmax", x.dtype, [x], x.shape, name=self._next_name("softmax")
        )

    def rope(self, x, cos, sin):
        return TensorNode(
            "RoPE", x.dtype, [x, cos, sin], x.shape, name=self._next_name("rope")
        )

    def repeat(self, x, repeats, axis=1):
        return TensorNode(
            OpType.REPEAT,
            x.dtype,
            [x],
            name=self._next_name("repeat"),
            attrs={"repeats": repeats, "axis": axis},
        )

    def sum(self, x, axis=1, keepdims=True):
        return TensorNode(
            OpType.SUM,
            x.dtype,
            [x],
            name=self._next_name("sum"),
            attrs={"axis": axis, "keepdims": keepdims},
        )

    # --- RoPE and Mask Computation Inside Graph ---

    def compute_rope(self, seq_len_node, head_dim, theta_base=10000.0):
        """
        Build RoPE (cos, sin) computation into the graph.
        Returns (cos_node, sin_node) with shape (1, 1, seq_len, head_dim)

        Formula:
        - inv_freq = (theta_base ** (arange(0, head_dim, 2) / head_dim)) ** -1
        - positions = arange(seq_len)
        - angles = outer(positions, inv_freq)  # (seq_len, head_dim//2)
        - angles = concat([angles, angles], axis=1)  # (seq_len, head_dim)
        - cos = cos(angles), sin = sin(angles)
        - reshape to (1, 1, seq_len, head_dim)
        """
        b = self

        def _const(val, dtype=DType.INT32):
            arr = np.array(val, dtype=np.int32 if dtype == DType.INT32 else np.float32)
            if arr.ndim == 0:
                arr = arr.reshape(1)
            node = b.constant(arr, (1,), dtype, self._next_name("const"))
            return node

        # 1. arange(0, head_dim, 2) -> indices
        start, stop, step = _const(0), _const(head_dim), _const(2)
        indices_int = b.arange(start, stop, step)
        indices = b.cast(indices_int, DType.FP32)

        # 2. inv_freq = theta_base ** (indices / head_dim) ** -1
        h_dim_fp = _const(float(head_dim), DType.FP32)
        exponent = b.divide(indices, h_dim_fp)

        theta_node = _const(theta_base, DType.FP32)
        base_to_exponent = b.power(theta_node, exponent)

        one_node = _const(1.0, DType.FP32)
        inv_freq = b.divide(one_node, base_to_exponent)

        # 3. positions = arange(seq_len)
        p_start, p_stop, p_step = _const(0), seq_len_node, _const(1)
        pos_int = b.arange(p_start, p_stop, p_step)
        pos = b.cast(pos_int, DType.FP32)

        # 4. Reshape pos to (seq_len, 1) and inv_freq to (1, head_dim//2)
        seq_len_1 = b.concat([seq_len_node, _const(1)], _const([1]), 1, (None, 1))
        pos_col = b.reshape(pos, (None, 1), seq_len_1)

        half_dim = head_dim // 2
        freq_shape = b.concat([_const(1), _const(half_dim)], _const([1]), 1, (1, None))
        freq_row = b.reshape(inv_freq, (1, None), freq_shape)

        # 5. Outer product: angles = pos_col * freq_row -> (seq_len, head_dim//2)
        angles = b.mul(pos_col, freq_row)

        # 6. Concat [angles, angles] -> (seq_len, head_dim)
        angles = b.concat([angles, angles], _const([1]), 1, (None, head_dim))

        # 7. Cos, Sin
        cos_t = b.cos(angles)
        sin_t = b.sin(angles)

        # 8. Final Reshape to (1, 1, seq_len, head_dim) for broadcasting
        final_shape = b.concat(
            [_const(1), _const(1), seq_len_node, _const(head_dim)],
            _const([1]),
            1,
            (1, 1, None, head_dim),
        )
        cos_out = b.reshape(cos_t, (1, 1, None, head_dim), final_shape)
        sin_out = b.reshape(sin_t, (1, 1, None, head_dim), final_shape)

        return cos_out, sin_out

    def compute_causal_mask(self, seq_len_node, max_seq_len, mask_val=-1e9):
        """
        Build causal mask computation into the graph.
        Returns mask with shape (1, 1, max_seq_len, max_seq_len)

        The mask is computed as:
        - fill ones matrix (seq_len, seq_len)
        - triu(k=1) to get upper triangle (strictly causal)
        - scale by mask_val

        Then reshape to (1, 1, seq_len, seq_len) for broadcasting.
        """
        b = self

        def _const(val, dtype=DType.INT32):
            arr = np.array(val, dtype=np.int32 if dtype == DType.INT32 else np.float32)
            if arr.ndim == 0:
                arr = arr.reshape(1)
            node = b.constant(arr, (1,), dtype, self._next_name("const"))
            return node

        # Shape for mask matrix: (seq_len, seq_len)
        mask_shape = b.concat(
            [seq_len_node, seq_len_node], _const([1]), 1, (None, None)
        )

        # Fill with ones
        ones_val = _const(1.0, DType.FP32)
        ones_matrix = b.fill(ones_val, mask_shape, (None, None))

        # Apply triu with k=1
        k_node = _const(1, DType.INT32)
        triu_mask = b.triu(ones_matrix, k_node)

        # Scale by mask_val (convert to const)
        mask_scale = _const(mask_val, DType.FP32)
        scaled_mask = b.mul(triu_mask, mask_scale)

        # Reshape to (1, 1, seq_len, seq_len) for broadcasting
        final_shape = b.concat(
            [_const(1), _const(1), seq_len_node, seq_len_node],
            _const([1]),
            1,
            (1, 1, None, None),
        )
        mask_out = b.reshape(scaled_mask, (1, 1, None, None), final_shape)

        return mask_out


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
        return self.builder.param(name, shape)

    def _const(self, value, name, dtype=DType.INT32):
        val_arr = np.array(
            value, dtype=np.int32 if dtype == DType.INT32 else np.float32
        )
        if val_arr.ndim == 0:
            val_arr = val_arr.reshape(1)

        node = self.builder.constant(val_arr, (1,), dtype, name)
        # Store for initialization
        self.constant_inputs[name] = val_arr
        return node

    def forward(
        self, input_ids_node, seq_len_node, max_seq_len, shapes: Dict[str, TensorNode]
    ):
        """
        Forward pass with integrated RoPE and mask computation.

        Args:
            input_ids_node: Input token IDs
            seq_len_node: Current sequence length (runtime value)
            max_seq_len: Maximum sequence length (compile-time constant)
            shapes: Shape nodes for reshape operations
        """
        cfg = self.cfg

        # Embedding
        w_emb = self._get_param(
            "model.embed_tokens.weight", (cfg["vocab_size"], cfg["emb_dim"])
        )
        x = self.builder.embedding(input_ids_node, w_emb)

        # Scale
        scale_val = self._const([cfg["emb_dim"] ** 0.5], "emb_scale_val", DType.FP32)
        x = self.builder.mul(x, scale_val)

        # Compute RoPE (cos, sin) internally
        cos, sin = self.builder.compute_rope(
            seq_len_node,
            cfg["head_dim"],
            theta_base=10000.0,
        )

        # Compute causal mask internally
        mask = self.builder.compute_causal_mask(
            seq_len_node,
            max_seq_len,
            mask_val=-1e9,
        )

        # Layers
        for i in range(cfg["n_layers"]):
            x = self._transformer_block(x, i, cos, sin, mask, shapes)

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

    def _transformer_block(self, x, layer_idx, cos, sin, mask, shapes):
        cfg = self.cfg
        prefix = f"model.layers.{layer_idx}"
        residual = x

        # Input Norm
        w_ln = self._get_param(f"{prefix}.input_layernorm.weight", (cfg["emb_dim"],))
        eps_ln = self._const([1e-6], f"ln_eps_{layer_idx}", DType.FP32)
        x_norm = self.builder.rms_norm(x, w_ln, eps_ln)

        # Attention
        x_attn = self._attention(x_norm, layer_idx, cos, sin, mask, shapes)

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

    def _attention(self, x, layer_idx, cos, sin, mask, shapes):
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
        perm_attn = self._const([0, 2, 1, 3], f"perm_attn_{layer_idx}")

        q = self.builder.permute(
            self.builder.reshape(q, (None, None, n_heads, head_dim), shapes["q_shape"]),
            [0, 2, 1, 3],
            perm_attn,
        )
        k = self.builder.permute(
            self.builder.reshape(k, (None, None, n_kv, head_dim), shapes["kv_shape"]),
            [0, 2, 1, 3],
            perm_attn,
        )
        v = self.builder.permute(
            self.builder.reshape(v, (None, None, n_kv, head_dim), shapes["kv_shape"]),
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
            k = self.builder.repeat(k, n_heads // n_kv, axis=1)
            v = self.builder.repeat(v, n_heads // n_kv, axis=1)

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
        scores = self.builder.add(scores, mask)
        probs = self.builder.softmax(scores)

        # Context
        context = self.builder.matmul(probs, v)
        perm_back = self._const([0, 2, 1, 3], f"perm_back_{layer_idx}")
        context = self.builder.permute(context, [0, 2, 1, 3], perm_back)

        context = self.builder.reshape(
            context, (None, None, n_heads * head_dim), shapes["flat_shape"]
        )

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
# 4. Main Execution
# ==============================================================================


def main():
    print("Initializing Gemma 3 (270M) on tensor_graphs...")

    # 1. Download/Load Weights
    tokenizer_path = "resources/tokenizer.json"
    weights_path = "resources/model.safetensors"

    if not os.path.exists(weights_path):
        print(f"Weights not found at {weights_path}. Please ensure the file exists.")
        return

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

    # Generation Loop Params
    max_new_tokens = 50
    MAX_SEQ_LEN = 128

    print(f"\nPrompt: {prompt}")
    print("Generating...", end="", flush=True)

    # --- BUILD GRAPH ---
    cfg = GEMMA3_CONFIG_270M
    model = Gemma3Model(cfg, weights_np)

    # Define Input Nodes
    in_node = model.builder.input("input_ids", (1, MAX_SEQ_LEN), DType.INT32)
    seq_len_node = model.builder.input("seq_len", (1,), DType.INT32)

    q_shape_node = model.builder.input("q_shape", (4,), DType.INT32)
    kv_shape_node = model.builder.input("kv_shape", (4,), DType.INT32)
    flat_shape_node = model.builder.input("flat_shape", (3,), DType.INT32)

    shapes = {
        "q_shape": q_shape_node,
        "kv_shape": kv_shape_node,
        "flat_shape": flat_shape_node,
    }

    # Build the computational graph (RoPE and mask computed internally)
    logits_node = model.forward(in_node, seq_len_node, MAX_SEQ_LEN, shapes)

    # Combine weights and constant inputs into a base dictionary
    base_inputs = {**model.weights, **model.constant_inputs}
    q_shape = np.array(
        [1, MAX_SEQ_LEN, cfg["n_heads"], cfg["head_dim"]], dtype=np.int32
    )
    kv_shape = np.array(
        [1, MAX_SEQ_LEN, cfg["n_kv_groups"], cfg["head_dim"]], dtype=np.int32
    )
    flat_shape = np.array(
        [1, MAX_SEQ_LEN, cfg["n_heads"] * cfg["head_dim"]], dtype=np.int32
    )

    # --- RUN LOOP ---
    for _ in range(max_new_tokens):
        seq_len = len(input_ids)
        if seq_len > MAX_SEQ_LEN:
            break

        # Prepare iteration-specific inputs
        input_ids_padded = np.zeros((1, MAX_SEQ_LEN), dtype=np.int32)
        input_ids_padded[0, :seq_len] = input_ids

        # Define concrete values for sequence length
        seq_len_val = np.array([seq_len], dtype=np.int32)

        # Final input dictionary for this step
        step_inputs = {
            **base_inputs,
            "input_ids": input_ids_padded,
            "seq_len": seq_len_val,
            "q_shape": q_shape,
            "kv_shape": kv_shape,
            "flat_shape": flat_shape,
        }

        # --- EXECUTE VIA evaluate_graph ---
        # This helper handles optimization, compilation, and execution in one go.
        logits_out = evaluate_graph(logits_node, step_inputs)

        # Greedy Decoding
        next_token_logits = logits_out[0, seq_len - 1, :]
        next_token_id = int(np.argmax(next_token_logits))

        input_ids.append(next_token_id)
        word = tokenizer.decode([next_token_id])
        print(word, end="", flush=True)

        if next_token_id == tokenizer.token_to_id("<end_of_turn>"):
            break

    print("\n\nDone.")


if __name__ == "__main__":
    main()

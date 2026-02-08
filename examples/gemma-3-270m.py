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
from tensor_graphs.session import GraphSession

from tensor_graphs.backend.kernels import *

# ==============================================================================
# 1. Graph Construction Helpers
# ==============================================================================


class GraphBuilder:
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
        indices_int = arange.arange_ref([start, stop, step])
        indices = cast.cast_ref([indices_int], {"to": DType.FP32})

        # 2. inv_freq = theta_base ** (indices / head_dim) ** -1
        h_dim_fp = _const(float(head_dim), DType.FP32)
        exponent = divide.divide_ref([indices, h_dim_fp])

        theta_node = _const(theta_base, DType.FP32)
        base_to_exponent = power.power_ref([theta_node, exponent])

        one_node = _const(1.0, DType.FP32)
        inv_freq = divide.divide_ref([one_node, base_to_exponent])

        # 3. positions = arange(seq_len)
        p_start, p_stop, p_step = _const(0), seq_len_node, _const(1)
        pos_int = arange.arange_ref([p_start, p_stop, p_step])
        pos = cast.cast_ref([pos_int], {"to": DType.FP32})

        # 4. Reshape pos to (seq_len, 1) and inv_freq to (1, head_dim//2)
        seq_len_1 = concat.concat_ref([seq_len_node, _const(1)], {"axis": 0})
        pos_col = reshape.reshape_ref([pos, seq_len_1])

        half_dim = head_dim // 2
        freq_shape = concat.concat_ref([_const(1), _const(half_dim)], {"axis": 0})
        freq_row = reshape.reshape_ref([inv_freq, freq_shape])

        # 5. Outer product: angles = pos_col * freq_row -> (seq_len, head_dim//2)
        angles = mul.mul_ref([pos_col, freq_row])

        # 6. Concat [angles, angles] -> (seq_len, head_dim)
        angles = concat.concat_ref([angles, angles], {"axis": 1})

        # 7. Cos, Sin
        cos_t = cos.cos_ref([angles])
        sin_t = sin.sin_ref([angles])

        # 8. Final Reshape to (1, 1, seq_len, head_dim) for broadcasting
        final_shape = concat.concat_ref(
            [_const(1), _const(1), seq_len_node, _const(head_dim)], {"axis": 0}
        )
        cos_out = reshape.reshape_ref([cos_t, final_shape])
        sin_out = reshape.reshape_ref([sin_t, final_shape])

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
        mask_shape = concat.concat_ref([seq_len_node, seq_len_node], {"axis": 0})

        # Fill with ones
        ones_val = _const(1.0, DType.FP32)
        ones_matrix = fill.fill_ref([ones_val, mask_shape])

        # Apply triu with k=1
        triu_mask = triu.triu_ref([ones_matrix], {"k": 1})

        # Scale by mask_val (convert to const)
        mask_scale = _const(mask_val, DType.FP32)
        scaled_mask = mul.mul_ref([triu_mask, mask_scale])

        # Reshape to (1, 1, seq_len, seq_len) for broadcasting
        final_shape = concat.concat_ref(
            [_const(1), _const(1), seq_len_node, seq_len_node], {"axis": 0}
        )
        mask_out = reshape.reshape_ref([scaled_mask, final_shape])

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
        x = gather.gather_ref([w_emb, input_ids_node])

        # Scale
        scale_val = self._const([cfg["emb_dim"] ** 0.5], "emb_scale_val", DType.FP32)
        x = mul.mul_ref([x, scale_val])

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
        x = rms_norm.rms_norm(x, w_norm, eps_node)

        # Head (Weight Tying)
        w_head = w_emb
        if "lm_head.weight" in self.weights:
            w_head = self._get_param(
                "lm_head.weight", (cfg["vocab_size"], cfg["emb_dim"])
            )

        # Permute for MatMul
        w_head_t = permute.permute_ref([w_head], {"dims": [1, 0]})
        logits = dot.dot_ref([x, w_head_t])

        return logits

    def _transformer_block(self, x, layer_idx, cos, sin, mask, shapes):
        cfg = self.cfg
        prefix = f"model.layers.{layer_idx}"
        residual = x

        # Input Norm
        w_ln = self._get_param(f"{prefix}.input_layernorm.weight", (cfg["emb_dim"],))
        eps_ln = self._const([1e-6], f"ln_eps_{layer_idx}", DType.FP32)
        x_norm = rms_norm.rms_norm(x, w_ln, eps_ln)

        # Attention
        x_attn = self._attention(x_norm, layer_idx, cos, sin, mask, shapes)

        # Post Attn Norm
        w_post = self._get_param(
            f"{prefix}.post_attention_layernorm.weight", (cfg["emb_dim"],)
        )
        eps_post = self._const([1e-6], f"post_attn_eps_{layer_idx}", DType.FP32)
        x_attn = rms_norm.rms_norm(x_attn, w_post, eps_post)

        x = add.add_ref([residual, x_attn])

        # Feed Forward
        residual = x
        w_pre = self._get_param(
            f"{prefix}.pre_feedforward_layernorm.weight", (cfg["emb_dim"],)
        )
        eps_pre = self._const([1e-6], f"pre_ff_eps_{layer_idx}", DType.FP32)
        x_norm = rms_norm.rms_norm(x, w_pre, eps_pre)

        x_ff = self._mlp(x_norm, layer_idx)

        w_post_ff = self._get_param(
            f"{prefix}.post_feedforward_layernorm.weight", (cfg["emb_dim"],)
        )
        eps_post_ff = self._const([1e-6], f"post_ff_eps_{layer_idx}", DType.FP32)
        x_ff = rms_norm.rms_norm(x_ff, w_post_ff, eps_post_ff)

        return add.add_ref([residual, x_ff])

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
        wq_t = permute.permute_ref([wq], {"dims": [1, 0]})
        wk_t = permute.permute_ref([wk], {"dims": [1, 0]})
        wv_t = permute.permute_ref([wv], {"dims": [1, 0]})

        q = dot.dot_ref([x, wq_t])
        k = dot.dot_ref([x, wk_t])
        v = dot.dot_ref([x, wv_t])

        # Reshape & Permute
        q = permute.permute_ref(
            [reshape.reshape_ref([q, shapes["q_shape"]])],
            {"dims": [0, 2, 1, 3]},
        )
        k = permute.permute_ref(
            [reshape.reshape_ref([k, shapes["kv_shape"]])],
            {"dims": [0, 2, 1, 3]},
        )
        v = permute.permute_ref(
            [reshape.reshape_ref([v, shapes["kv_shape"]])],
            {"dims": [0, 2, 1, 3]},
        )

        # QK Norm
        w_q_norm = self._get_param(f"{prefix}.q_norm.weight", (head_dim,))
        w_k_norm = self._get_param(f"{prefix}.k_norm.weight", (head_dim,))
        eps_qk = self._const([1e-6], f"eps_qk_{layer_idx}", DType.FP32)
        q = rms_norm.rms_norm(q, w_q_norm, eps_qk)
        k = rms_norm.rms_norm(k, w_k_norm, eps_qk)

        # RoPE
        q = rope.rope(q, cos, sin)
        k = rope.rope(k, cos, sin)

        # GQA Repeat
        if n_heads != n_kv:
            k = repeat.repeat_ref([k], {"repeats": n_heads // n_kv, "axis": 1})
            v = repeat.repeat_ref([v], {"repeats": n_heads // n_kv, "axis": 1})

        # Scale
        scale_node = self._const(
            [cfg["query_pre_attn_scalar"] ** -0.5], "attn_scale", DType.FP32
        )
        q = mul.mul_ref([q, scale_node])

        # Scores
        k_t = permute.permute_ref([k], {"dims": [0, 1, 3, 2]})
        scores = dot.dot_ref([q, k_t])

        # Mask
        scores = add.add_ref([scores, mask])
        probs = softmax.softmax(scores)

        # Context
        context = dot.dot_ref([probs, v])
        context = permute.permute_ref([context], {"dims": [0, 2, 1, 3]})

        context = reshape.reshape_ref(
            [context, shapes["flat_shape"]]
        )

        # Output Proj
        wo = self._get_param(f"{prefix}.o_proj.weight", (d_model, n_heads * head_dim))
        wo_t = permute.permute_ref([wo], {"dims": [1, 0]})
        return dot.dot_ref([context, wo_t])

    def _mlp(self, x, layer_idx):
        cfg = self.cfg
        prefix = f"model.layers.{layer_idx}.mlp"
        d_model = cfg["emb_dim"]
        d_hidden = cfg["hidden_dim"]

        w_gate = self._get_param(f"{prefix}.gate_proj.weight", (d_hidden, d_model))
        w_up = self._get_param(f"{prefix}.up_proj.weight", (d_hidden, d_model))
        w_down = self._get_param(f"{prefix}.down_proj.weight", (d_model, d_hidden))

        w_gate_t = permute.permute_ref([w_gate], {"dims": [1, 0]})
        w_up_t = permute.permute_ref([w_up], {"dims": [1, 0]})
        w_down_t = permute.permute_ref([w_down], {"dims": [1, 0]})

        gate = gelu.gelu(dot.dot_ref([x, w_gate_t]))
        up = dot.dot_ref([x, w_up_t])

        return dot.dot_ref([mul.mul_ref([gate, up]), w_down_t])


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

    # Build the computational graph
    logits_node = model.forward(in_node, seq_len_node, MAX_SEQ_LEN, shapes)

    # Initialize Session
    session = GraphSession(logits_node)

    # Base inputs
    base_inputs = {**model.weights, **model.constant_inputs}
    # Shapes constants
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

        input_ids_padded = np.zeros((1, MAX_SEQ_LEN), dtype=np.int32)
        input_ids_padded[0, :seq_len] = input_ids
        seq_len_val = np.array(
            [MAX_SEQ_LEN], dtype=np.int32
        )  # Note: using MAX for consistent shape

        step_inputs = {
            **base_inputs,
            "input_ids": input_ids_padded,
            "seq_len": seq_len_val,
            "q_shape": q_shape,
            "kv_shape": kv_shape,
            "flat_shape": flat_shape,
        }

        # USE SESSION
        logits_out = session.run(step_inputs)

        # Decoding
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

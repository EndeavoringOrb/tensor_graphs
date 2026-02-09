import os
import time
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict

from safetensors.torch import load_file
from tokenizers import Tokenizer

from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ir.graph import GraphBuilder
from tensor_graphs.session import GraphSession
from tensor_graphs.config import DEBUG_EXECUTION


class Gemma3Model:
    def __init__(self, cfg, weights: Dict[str, np.ndarray]):
        self.cfg = cfg
        self.weights = weights
        self.builder = GraphBuilder()
        self.eps = self.builder.const([1e-6], DType.FP32)

    def compute_rope(self, seq_len_node, head_dim, theta_base=10000.0):
        b = self.builder

        # 1. arange(0, head_dim, 2)
        start, stop, step = b.const(0), b.const(head_dim), b.const(2)
        indices_int = b.arange(start, stop, step)
        indices = b.cast(indices_int, DType.FP32)

        # 2. inv_freq = theta_base ** (indices / head_dim) ** -1
        h_dim_fp = b.const(float(head_dim), DType.FP32)
        exponent = b.divide(indices, h_dim_fp)
        base_to_exponent = b.power(b.const(theta_base, DType.FP32), exponent)
        inv_freq = b.divide(b.const(1.0, DType.FP32), base_to_exponent)

        # 3. positions = arange(seq_len)
        pos_int = b.arange(b.const(0), seq_len_node, b.const(1))
        pos = b.cast(pos_int, DType.FP32)

        # 4. Reshaping for outer product
        pos_col = b.reshape(
            pos, b.concat([seq_len_node, b.const([1])], axis=0)
        )  # Changed to [1]
        freq_row = b.reshape(
            inv_freq,
            b.concat(
                [b.const([1]), b.const([head_dim // 2])], axis=0
            ),  # Changed to [1] and [head_dim // 2]
        )

        # 5. Outer product: angles = pos_col * freq_row
        angles_half = b.mul(pos_col, freq_row)

        # 6. Concat [angles, angles] -> (seq_len, head_dim)
        angles = b.concat([angles_half, angles_half], axis=1)

        # 7. Cos, Sin and final broadcast reshape
        final_shape = b.concat(
            [b.const([1]), b.const([1]), seq_len_node, b.const([head_dim])],
            axis=0,  # Changed to [1], [1], and [head_dim]
        )
        cos_out = b.reshape(b.cos(angles), final_shape)
        sin_out = b.reshape(b.sin(angles), final_shape)

        return cos_out, sin_out

    def compute_causal_mask(self, seq_len_node, max_seq_len, mask_val=-1e9):
        b = self.builder
        mask_shape = b.concat([seq_len_node, seq_len_node], axis=0)
        ones_matrix = b.fill(b.const(1.0, DType.FP32), mask_shape)
        triu_mask = b.triu(ones_matrix, k=1)
        scaled_mask = b.mul(triu_mask, b.const(mask_val, DType.FP32))

        final_shape = b.concat(
            [b.const([1]), b.const([1]), seq_len_node, seq_len_node],
            axis=0,  # Changed to [1] and [1]
        )
        return b.reshape(scaled_mask, final_shape)

    def forward(
        self, input_ids_node, seq_len_node, max_seq_len, shapes: Dict[str, TensorNode]
    ):
        cfg = self.cfg
        b = self.builder

        # Embedding & Scale
        w_emb = b.param(
            "model.embed_tokens.weight", (cfg["vocab_size"], cfg["emb_dim"])
        )
        x = b.gather(w_emb, input_ids_node)
        x = b.mul(x, b.const([cfg["emb_dim"] ** 0.5], DType.FP32))

        # Internally generated RoPE and Mask
        cos, sin = self.compute_rope(seq_len_node, cfg["head_dim"])
        mask = self.compute_causal_mask(seq_len_node, max_seq_len)

        for i in range(cfg["n_layers"]):
            x = self._transformer_block(x, i, cos, sin, mask, shapes)

        # Final Head
        x = b.rms_norm(x, b.param("model.norm.weight", (cfg["emb_dim"],)), self.eps)
        w_head = w_emb  # Weight tying
        logits = b.dot(x, b.permute(w_head, [1, 0]))

        return logits

    def _transformer_block(self, x, layer_idx, cos, sin, mask, shapes):
        b = self.builder
        prefix = f"model.layers.{layer_idx}"

        # Self Attention Path
        residual = x
        x_norm = b.rms_norm(
            x,
            b.param(f"{prefix}.input_layernorm.weight", (self.cfg["emb_dim"],)),
            self.eps,
        )
        x_attn = self._attention(x_norm, layer_idx, cos, sin, mask, shapes)
        x_attn = b.rms_norm(
            x_attn,
            b.param(
                f"{prefix}.post_attention_layernorm.weight", (self.cfg["emb_dim"],)
            ),
            self.eps,
        )
        x = b.add(residual, x_attn)

        # MLP Path
        residual = x
        x_norm = b.rms_norm(
            x,
            b.param(
                f"{prefix}.pre_feedforward_layernorm.weight", (self.cfg["emb_dim"],)
            ),
            self.eps,
        )
        x_ff = self._mlp(x_norm, layer_idx)
        x_ff = b.rms_norm(
            x_ff,
            b.param(
                f"{prefix}.post_feedforward_layernorm.weight", (self.cfg["emb_dim"],)
            ),
            self.eps,
        )
        return b.add(residual, x_ff)

    def _attention(self, x, layer_idx, cos, sin, mask, shapes):
        cfg = self.cfg
        b = self.builder
        prefix = f"model.layers.{layer_idx}.self_attn"

        # Projections
        q = b.dot(
            x,
            b.permute(
                b.param(
                    f"{prefix}.q_proj.weight",
                    (cfg["n_heads"] * cfg["head_dim"], cfg["emb_dim"]),
                ),
                [1, 0],
            ),
        )
        k = b.dot(
            x,
            b.permute(
                b.param(
                    f"{prefix}.k_proj.weight",
                    (cfg["n_kv_groups"] * cfg["head_dim"], cfg["emb_dim"]),
                ),
                [1, 0],
            ),
        )
        v = b.dot(
            x,
            b.permute(
                b.param(
                    f"{prefix}.v_proj.weight",
                    (cfg["n_kv_groups"] * cfg["head_dim"], cfg["emb_dim"]),
                ),
                [1, 0],
            ),
        )

        # Reshape & Permute to [B, H, S, D]
        q = b.permute(b.reshape(q, shapes["q_shape"]), [0, 2, 1, 3])
        k = b.permute(b.reshape(k, shapes["kv_shape"]), [0, 2, 1, 3])
        v = b.permute(b.reshape(v, shapes["kv_shape"]), [0, 2, 1, 3])

        # QK Norm & RoPE
        q = b.rope(
            b.rms_norm(
                q, b.param(f"{prefix}.q_norm.weight", (cfg["head_dim"],)), self.eps
            ),
            cos,
            sin,
        )
        k = b.rope(
            b.rms_norm(
                k, b.param(f"{prefix}.k_norm.weight", (cfg["head_dim"],)), self.eps
            ),
            cos,
            sin,
        )

        if cfg["n_heads"] != cfg["n_kv_groups"]:
            k = b.repeat(k, repeats=cfg["n_heads"] // cfg["n_kv_groups"], axis=1)
            v = b.repeat(v, repeats=cfg["n_heads"] // cfg["n_kv_groups"], axis=1)

        # Scale & Attention
        q = b.mul(q, b.const([cfg["query_pre_attn_scalar"] ** -0.5], DType.FP32))
        scores = b.dot(q, b.permute(k, [0, 1, 3, 2]))
        probs = b.softmax(b.add(scores, mask))

        # Output projection
        context = b.reshape(
            b.permute(b.dot(probs, v), [0, 2, 1, 3]), shapes["flat_shape"]
        )
        return b.dot(
            context,
            b.permute(
                b.param(
                    f"{prefix}.o_proj.weight",
                    (cfg["emb_dim"], cfg["n_heads"] * cfg["head_dim"]),
                ),
                [1, 0],
            ),
        )

    def _mlp(self, x, layer_idx):
        cfg = self.cfg
        b = self.builder
        prefix = f"model.layers.{layer_idx}.mlp"

        gate = b.gelu(
            b.dot(
                x,
                b.permute(
                    b.param(
                        f"{prefix}.gate_proj.weight",
                        (cfg["hidden_dim"], cfg["emb_dim"]),
                    ),
                    [1, 0],
                ),
            )
        )
        up = b.dot(
            x,
            b.permute(
                b.param(
                    f"{prefix}.up_proj.weight", (cfg["hidden_dim"], cfg["emb_dim"])
                ),
                [1, 0],
            ),
        )
        down_weight = b.param(
            f"{prefix}.down_proj.weight", (cfg["emb_dim"], cfg["hidden_dim"])
        )

        return b.dot(b.mul(gate, up), b.permute(down_weight, [1, 0]))


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


# Define timer for debug timing that can use with __enter__ and __exit__. Only enabled when DEBUG_EXECUTION is set.
class Timer:
    def __init__(self, name="Elapsed"):
        if not DEBUG_EXECUTION:
            return
        self.name = name
        self.start = time.perf_counter()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not DEBUG_EXECUTION:
            return
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start
        print(f"{self.name}: {self.elapsed:.2f} seconds")


def main():
    print("Initializing Gemma 3 (270M) on tensor_graphs...")

    # 1. Download/Load Weights
    tokenizer_path = "resources/tokenizer.json"
    weights_path = "resources/model.safetensors"

    if not os.path.exists(weights_path):
        print(f"Weights not found at {weights_path}. Please ensure the file exists.")
        return

    print(f"Loading weights from {weights_path}...")
    with Timer("Loading weights"):
        state_dict = load_file(weights_path)

    # Convert weights to FP32 numpy
    weights_np = {}
    print("Converting weights to FP32 numpy...")
    with Timer("Converting weights to FP32 numpy"):
        for k, v in state_dict.items():
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
    with Timer("Building graph"):
        logits_node = model.forward(in_node, seq_len_node, MAX_SEQ_LEN, shapes)

    # Initialize Session
    session = GraphSession(logits_node)

    # Base inputs
    base_inputs = model.weights
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
        with Timer("Running session"):
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

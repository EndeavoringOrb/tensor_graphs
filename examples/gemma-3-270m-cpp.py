import numpy as np
from tokenizers import Tokenizer
import tg_cpp  # The compiled C++ extension


class Gemma3ModelCPP:
    def __init__(self, cfg, graph, mem):
        self.cfg = cfg
        self.g = graph
        self.mem = mem
        self.eps = 1e-6

        # CHANGED: Removed mem parameter from constant calls
        self.one_fp32 = self.g.constant(
            [1], np.array([1.0], dtype=np.float32), tg_cpp.DType.FLOAT32
        )
        self.eps_fp32 = self.g.constant(
            [1], np.array([self.eps], dtype=np.float32), tg_cpp.DType.FLOAT32
        )
        self.half_fp32 = self.g.constant(
            [1], np.array([0.5], dtype=np.float32), tg_cpp.DType.FLOAT32
        )

    def weight(self, path, name):
        """
        Helper function to load a weight and ensure it's cast to FP32.
        This is necessary because model weights may be stored as BF16/FP16
        but our operations expect FP32.
        """
        # CHANGED: Removed mem parameter from weight call
        raw_weight = self.g.weight(path, name)

        # Check if it needs casting (assume weights need to be FP32 for compatibility)
        # For now, always cast to ensure consistency across the graph
        return self.g.cast(raw_weight, tg_cpp.DType.FLOAT32)

    def rms_norm_gemma_atomic(self, x_id, weight_id):
        """
        Decomposed RMSNorm using only atomic operations.
        Gemma uses: RMSNorm(x) * (1 + weight)
        RMSNorm = x / sqrt(mean(x^2) + eps)
        """
        # 1. x * x (element-wise square)
        x_sq = self.g.mul(x_id, x_id)

        # 2. sum(x^2, axis=-1, keepdims=True)
        axis_node = self.g.constant(
            [1], np.array([-1], dtype=np.int32), tg_cpp.DType.INT32
        )

        keepdims_node = self.g.constant(
            [1], np.array([True], dtype=np.bool_), tg_cpp.DType.BOOL
        )

        sum_sq = self.g.sum(x_sq, axis_node, keepdims_node)

        # 3. mean = sum / n (where n = shape[-1])
        n_node = self.g.constant(
            [1],
            np.array([float(self.cfg["emb_dim"])], dtype=np.float32),
            tg_cpp.DType.FLOAT32,
        )

        mean_sq = self.g.div(sum_sq, n_node)

        # 4. mean_sq + eps
        mean_sq_plus_eps = self.g.add(mean_sq, self.eps_fp32)

        # 5. sqrt(mean_sq + eps) using pow(x, 0.5)
        sqrt_node = self.g.constant(
            [1], np.array([0.5], dtype=np.float32), tg_cpp.DType.FLOAT32
        )

        std = self.g.pow(mean_sq_plus_eps, sqrt_node)

        # 6. 1.0 / std
        inv_std = self.g.div(self.one_fp32, std)

        # 7. x * inv_std (normalized)
        x_norm = self.g.mul(x_id, inv_std)

        # 8. scale = 1 + weight (Gemma's offset scaling)
        # Ensure weight is FP32 before adding
        weight_fp32 = self.g.cast(weight_id, tg_cpp.DType.FLOAT32)
        scale = self.g.add(weight_fp32, self.one_fp32)

        # 9. x_norm * scale
        return self.g.mul(x_norm, scale)

    def gelu_atomic(self, x_id):
        """
        Decomposed GELU using atomic operations.
        GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        """
        # Constants
        c1_node = self.g.constant(
            [1], np.array([0.044715], dtype=np.float32), tg_cpp.DType.FLOAT32
        )

        c2_node = self.g.constant(
            [1],
            np.array([0.79788456], dtype=np.float32),
            tg_cpp.DType.FLOAT32,
        )

        # x^2
        x_sq = self.g.mul(x_id, x_id)

        # x^3
        x_cube = self.g.mul(x_sq, x_id)

        # 0.044715 * x^3
        term1 = self.g.mul(x_cube, c1_node)

        # x + 0.044715 * x^3
        term2 = self.g.add(x_id, term1)

        # sqrt(2/pi) * (x + 0.044715 * x^3)
        term3 = self.g.mul(term2, c2_node)

        # tanh(term3) - using atomic decomposition
        tanh_result = self.tanh_atomic(term3)

        # 1 + tanh(...)
        term4 = self.g.add(self.one_fp32, tanh_result)

        # 0.5 * x
        term5 = self.g.mul(x_id, self.half_fp32)

        # 0.5 * x * (1 + tanh(...))
        return self.g.mul(term5, term4)

    def tanh_atomic(self, x_id):
        """
        Decomposed tanh using atomic operations.
        tanh(x) = (e^x - e^-x) / (e^x + e^-x)
        """
        # -x
        neg_x = self.g.neg(x_id)

        # e^x using pow(e, x)
        e_node = self.g.constant(
            [1],
            np.array([2.718281828459045], dtype=np.float32),
            tg_cpp.DType.FLOAT32,
        )

        exp_x = self.g.pow(e_node, x_id)
        exp_neg_x = self.g.pow(e_node, neg_x)

        # -e^-x
        neg_exp_neg = self.g.neg(exp_neg_x)

        # e^x - e^-x
        num = self.g.add(exp_x, neg_exp_neg)

        # e^x + e^-x
        den = self.g.add(exp_x, exp_neg_x)

        # (e^x - e^-x) / (e^x + e^-x)
        return self.g.div(num, den)

    def build_graph(self, input_ids_id):
        cfg = self.cfg
        w_path = "resources/model.safetensors"

        # 1. Embedding - now uses helper that casts to FP32
        w_emb = self.weight(w_path, "model.embed_tokens.weight")
        x = self.g.gather(w_emb, input_ids_id)

        # Scale by sqrt(emb_dim)
        scale_node = self.g.constant(
            [1],
            np.array([float(cfg["emb_dim"] ** 0.5)], dtype=np.float32),
            tg_cpp.DType.FLOAT32,
        )
        x = self.g.mul(x, scale_node)

        # 2. Layers
        for i in range(cfg["n_layers"]):
            prefix = f"model.layers.{i}"

            # Pre-norm Residual Block
            residual = x

            # Load weight and cast to FP32 using helper
            w_ln1 = self.weight(w_path, f"{prefix}.input_layernorm.weight")
            x = self.rms_norm_gemma_atomic(x, w_ln1)

            # Attention (simplified - using atomic ops for projections)
            q = self.attention_qkv_atomic(x, prefix, cfg)
            x = self.attention_output_atomic(q, prefix, cfg)
            x = self.g.add(residual, x)

            # MLP Block
            residual = x

            w_ln2 = self.weight(w_path, f"{prefix}.pre_feedforward_layernorm.weight")
            x = self.rms_norm_gemma_atomic(x, w_ln2)
            x = self.mlp_atomic(x, prefix, cfg)
            x = self.g.add(residual, x)

        # 3. Final Norm & Head
        w_final_ln = self.weight(w_path, "model.norm.weight")
        x = self.rms_norm_gemma_atomic(x, w_final_ln)

        # Weight tying - transpose embedding weights
        dims_node = self.g.constant(
            [2], np.array([1, 0], dtype=np.int32), tg_cpp.DType.INT32
        )
        w_emb_t = self.g.permute(w_emb, dims_node)
        logits = self.g.dot(x, w_emb_t)

        return logits

    def attention_qkv_atomic(self, x, prefix, cfg):
        """Compute Q, K, V projections using atomic operations."""
        w_path = "resources/model.safetensors"

        # Q projection - use weight helper for FP32 casting
        w_q = self.weight(w_path, f"{prefix}.self_attn.q_proj.weight")
        dims_node = self.g.constant(
            [2], np.array([1, 0], dtype=np.int32), tg_cpp.DType.INT32
        )
        w_q_t = self.g.permute(w_q, dims_node)
        q = self.g.dot(x, w_q_t)

        # K projection
        w_k = self.weight(w_path, f"{prefix}.self_attn.k_proj.weight")
        dims_node = self.g.constant(
            [2], np.array([1, 0], dtype=np.int32), tg_cpp.DType.INT32
        )
        w_k_t = self.g.permute(w_k, dims_node)
        k = self.g.dot(x, w_k_t)

        # V projection
        w_v = self.weight(w_path, f"{prefix}.self_attn.v_proj.weight")
        dims_node = self.g.constant(
            [2], np.array([1, 0], dtype=np.int32), tg_cpp.DType.INT32
        )
        w_v_t = self.g.permute(w_v, dims_node)
        v = self.g.dot(x, w_v_t)

        return q, k, v

    def attention_output_atomic(self, qkv, prefix, cfg):
        """Simplified attention output using atomic operations."""
        q, k, v = qkv

        # For simplicity, just return a scaled version of the input
        scale_node = self.g.constant(
            [1],
            np.array(
                [1.0 / float(cfg["query_pre_attn_scalar"] ** 0.5)], dtype=np.float32
            ),
            tg_cpp.DType.FLOAT32,
        )

        return self.g.mul(q, scale_node)

    def mlp_atomic(self, x, prefix, cfg):
        """MLP using atomic operations with GELU."""
        w_path = "resources/model.safetensors"

        # Gate projection - use weight helper for FP32 casting
        w_gate = self.weight(w_path, f"{prefix}.mlp.gate_proj.weight")
        dims_node = self.g.constant(
            [2], np.array([1, 0], dtype=np.int32), tg_cpp.DType.INT32
        )
        w_gate_t = self.g.permute(w_gate, dims_node)
        gate = self.g.dot(x, w_gate_t)
        gate = self.gelu_atomic(gate)

        # Up projection
        w_up = self.weight(w_path, f"{prefix}.mlp.up_proj.weight")
        dims_node = self.g.constant(
            [2], np.array([1, 0], dtype=np.int32), tg_cpp.DType.INT32
        )
        w_up_t = self.g.permute(w_up, dims_node)
        up = self.g.dot(x, w_up_t)

        # Gate * Up
        gate_up = self.g.mul(gate, up)

        # Down projection
        w_down = self.weight(w_path, f"{prefix}.mlp.down_proj.weight")
        dims_node = self.g.constant(
            [2], np.array([1, 0], dtype=np.int32), tg_cpp.DType.INT32
        )
        w_down_t = self.g.permute(w_down, dims_node)

        return self.g.dot(gate_up, w_down_t)


def main():
    cfg = {
        "vocab_size": 262144,
        "n_layers": 18,
        "emb_dim": 640,
        "n_heads": 4,
        "head_dim": 256,
        "n_kv_groups": 1,
        "hidden_dim": 2048,
        "query_pre_attn_scalar": 256,
    }

    # Initialize C++ Backend
    mem = tg_cpp.MemoryManager()
    # Allocate 2GB for CPU
    mem.add_buffer(tg_cpp.Backend.CPU, 2 * 1024 * 1024 * 1024)

    graph = tg_cpp.Graph()

    # Define Input
    MAX_SEQ_LEN = 128
    input_view = tg_cpp.TensorView()  # TODO: get the view from mem.allocate
    input_ids_node = graph.input([1, MAX_SEQ_LEN], tg_cpp.DType.INT32, input_view)

    model = Gemma3ModelCPP(cfg, graph, mem)
    logits_id = model.build_graph(input_ids_node)

    # Initialize Session
    session = tg_cpp.Session(graph, mem, logits_id, "gemma_cache.jsonl")

    # Init memory (Actualizes the Arena)
    mem.init_all()

    tokenizer = Tokenizer.from_file("resources/tokenizer.json")
    prompt = "The secret to life is"
    tokens = tokenizer.encode(prompt).ids

    print("Running Inference...")
    for _ in range(20):
        # Prepare input buffer
        input_data = np.zeros((1, MAX_SEQ_LEN), dtype=np.int32)
        input_data[0, : len(tokens)] = tokens

        # Run Session (passing a dict of ID -> Numpy Array)
        session.run({input_ids_node: input_data})

        # Get output pointer as numpy array
        output_logits = session.get_output(logits_id)

        next_token = int(np.argmax(output_logits[0, len(tokens) - 1, :]))
        tokens.append(next_token)
        print(tokenizer.decode([next_token]), end="", flush=True)


if __name__ == "__main__":
    main()

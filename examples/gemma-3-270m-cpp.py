import numpy as np
from tokenizers import Tokenizer
import tg_cpp  # The compiled C++ extension


class Gemma3ModelCPP:
    def __init__(self, cfg, seq_len, graph, mem):
        self.cfg = cfg
        self.g = graph
        self.mem = mem
        self.eps = 1e-6
        self.seq_len = seq_len
        self.w_path = "resources/model.safetensors"

        # Constants
        self.one_fp32 = self.g.constant(
            [1], np.array([1.0], dtype=np.float32), tg_cpp.DType.FLOAT32
        )
        self.eps_fp32 = self.g.constant(
            [1], np.array([self.eps], dtype=np.float32), tg_cpp.DType.FLOAT32
        )
        self.half_fp32 = self.g.constant(
            [1], np.array([0.5], dtype=np.float32), tg_cpp.DType.FLOAT32
        )

    def weight(self, name):
        """Helper to load a weight and cast to FP32."""
        raw_weight = self.g.weight(self.w_path, name)
        return self.g.cast(raw_weight, tg_cpp.DType.FLOAT32)

    def repeat_3d_axis(self, tensor_id, repeats, axis):
        if repeats <= 1:
            return tensor_id
        rep_node = self.g.constant(
            [1], np.array([repeats], dtype=np.int32), tg_cpp.DType.INT32
        )
        ax_node = self.g.constant(
            [1], np.array([axis], dtype=np.int32), tg_cpp.DType.INT32
        )
        return self.g.repeat(tensor_id, rep_node, ax_node)

    def expand_scalar_to_3d(self, scalar_id, dim0, dim1, dim2):
        shape_node = self.g.constant(
            [3], np.array([1, 1, 1], dtype=np.int32), tg_cpp.DType.INT32
        )
        out = self.g.reshape(scalar_id, shape_node)
        if dim0 > 1:
            out = self.repeat_3d_axis(out, dim0, 0)
        if dim1 > 1:
            out = self.repeat_3d_axis(out, dim1, 1)
        if dim2 > 1:
            out = self.repeat_3d_axis(out, dim2, 2)
        return out

    def expand_1d_to_3d(self, vec_id, vec_len, dim0, dim1):
        shape_node = self.g.constant(
            [3], np.array([1, 1, vec_len], dtype=np.int32), tg_cpp.DType.INT32
        )
        out = self.g.reshape(vec_id, shape_node)
        if dim0 > 1:
            out = self.repeat_3d_axis(out, dim0, 0)
        if dim1 > 1:
            out = self.repeat_3d_axis(out, dim1, 1)
        return out

    def rms_norm_gemma_atomic(self, x_id, weight_id, dim0, dim_size):
        x_sq = self.g.mul(x_id, x_id)
        axis_node = self.g.constant(
            [1], np.array([-1], dtype=np.int32), tg_cpp.DType.INT32
        )
        sum_sq = self.g.sum(x_sq, axis_node)

        n_node = self.expand_scalar_to_3d(
            self.g.constant(
                [1], np.array([float(dim_size)], dtype=np.float32), tg_cpp.DType.FLOAT32
            ),
            dim0,
            self.seq_len,
            1,
        )
        mean_sq = self.g.div(sum_sq, n_node)

        eps_exp = self.expand_scalar_to_3d(self.eps_fp32, dim0, self.seq_len, 1)
        std = self.g.pow(
            self.g.add(mean_sq, eps_exp),
            self.expand_scalar_to_3d(self.half_fp32, dim0, self.seq_len, 1),
        )

        one_exp = self.expand_scalar_to_3d(self.one_fp32, dim0, self.seq_len, 1)
        inv_std = self.repeat_3d_axis(self.g.div(one_exp, std), dim_size, 2)
        x_norm = self.g.mul(x_id, inv_std)

        weight_exp = self.expand_1d_to_3d(weight_id, dim_size, dim0, self.seq_len)
        scale = self.g.add(
            weight_exp,
            self.expand_scalar_to_3d(self.one_fp32, dim0, self.seq_len, dim_size),
        )
        return self.g.mul(x_norm, scale)

    def build_graph(self, input_ids_id):
        cfg = self.cfg
        w_emb = self.weight("model.embed_tokens.weight")
        x = self.g.gather(w_emb, input_ids_id)

        # Scale embedding
        scale_val = float(cfg["emb_dim"] ** 0.5)
        scale_node = self.expand_scalar_to_3d(
            self.g.constant(
                [1], np.array([scale_val], dtype=np.float32), tg_cpp.DType.FLOAT32
            ),
            1,
            self.seq_len,
            cfg["emb_dim"],
        )
        x = self.g.mul(x, scale_node)

        for i in range(cfg["n_layers"]):
            prefix = f"model.layers.{i}"
            residual = x

            # Pre-norm
            w_ln1 = self.weight(f"{prefix}.input_layernorm.weight")
            x = self.rms_norm_gemma_atomic(x, w_ln1, 1, cfg["emb_dim"])

            # Attention (Simplified projection logic matching C++ impl requirements)
            x = self.attention_block(x, prefix)

            w_post_attn = self.weight(f"{prefix}.post_attention_layernorm.weight")
            x = self.rms_norm_gemma_atomic(x, w_post_attn, 1, cfg["emb_dim"])
            x = self.g.add(residual, x)

            # MLP
            residual = x
            w_ln2 = self.weight(f"{prefix}.pre_feedforward_layernorm.weight")
            x = self.rms_norm_gemma_atomic(x, w_ln2, 1, cfg["emb_dim"])
            x = self.mlp_block(x, prefix)

            w_post_ff = self.weight(f"{prefix}.post_feedforward_layernorm.weight")
            x = self.rms_norm_gemma_atomic(x, w_post_ff, 1, cfg["emb_dim"])
            x = self.g.add(residual, x)

        w_final_ln = self.weight("model.norm.weight")
        x = self.rms_norm_gemma_atomic(x, w_final_ln, 1, cfg["emb_dim"])

        # Head
        perm_dims = self.g.constant(
            [2], np.array([1, 0], dtype=np.int32), tg_cpp.DType.INT32
        )
        w_emb_t = self.g.permute(w_emb, perm_dims)
        s3 = self.g.constant(
            [3],
            np.array([1, cfg["emb_dim"], cfg["vocab_size"]], dtype=np.int32),
            tg_cpp.DType.INT32,
        )
        return self.g.dot(x, self.g.reshape(w_emb_t, s3))

    def attention_block(self, x, prefix):
        # Mirroring the simplified projection used for the small 270m example
        w_q = self.weight(f"{prefix}.self_attn.q_proj.weight")
        perm_dims = self.g.constant(
            [2], np.array([1, 0], dtype=np.int32), tg_cpp.DType.INT32
        )
        w_q_t = self.g.permute(w_q, perm_dims)
        s3 = self.g.constant(
            [3],
            np.array(
                [1, self.cfg["emb_dim"], self.cfg["n_heads"] * self.cfg["head_dim"]],
                dtype=np.int32,
            ),
            tg_cpp.DType.INT32,
        )
        q = self.g.dot(x, self.g.reshape(w_q_t, s3))

        # Scaling matching the query_pre_attn_scalar
        scale = 1.0 / (self.cfg["query_pre_attn_scalar"] ** 0.5)
        scale_node = self.expand_scalar_to_3d(
            self.g.constant(
                [1], np.array([scale], dtype=np.float32), tg_cpp.DType.FLOAT32
            ),
            1,
            self.seq_len,
            self.cfg["n_heads"] * self.cfg["head_dim"],
        )
        return self.g.mul(q, scale_node)

    def mlp_block(self, x, prefix):
        w_gate = self.weight(f"{prefix}.mlp.gate_proj.weight")
        perm_dims = self.g.constant(
            [2], np.array([1, 0], dtype=np.int32), tg_cpp.DType.INT32
        )
        w_gate_t = self.g.permute(w_gate, perm_dims)
        s3 = self.g.constant(
            [3],
            np.array([1, self.cfg["emb_dim"], self.cfg["hidden_dim"]], dtype=np.int32),
            tg_cpp.DType.INT32,
        )
        return self.g.dot(x, self.g.reshape(w_gate_t, s3))


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

    mem = tg_cpp.MemoryManager()
    mem.add_buffer(tg_cpp.Backend.CPU, 2 * 1024 * 1024 * 1024)
    graph = tg_cpp.Graph()
    MAX_SEQ_LEN = 128

    # Input Definition
    input_ids_id = graph.allocateId()
    mem.allocate(
        tg_cpp.Backend.CPU, input_ids_id, MAX_SEQ_LEN * 4, tg_cpp.StorageType.PERSISTENT
    )

    input_view = tg_cpp.TensorView()
    input_view.shape, input_view.strides, input_view.dtype = (
        [1, MAX_SEQ_LEN],
        [MAX_SEQ_LEN, 1],
        tg_cpp.DType.INT32,
    )
    input_ids_node = graph.inputWithId(
        input_ids_id,
        [1, MAX_SEQ_LEN],
        tg_cpp.DType.INT32,
        input_view,
        tg_cpp.StorageType.PERSISTENT,
    )

    model = Gemma3ModelCPP(cfg, MAX_SEQ_LEN, graph, mem)
    logits_id = model.build_graph(input_ids_node)

    # Use a unique cache path for Python to avoid Node ID collisions with the C++ binary
    session = tg_cpp.Session(
        graph, mem, logits_id, "dirty_region_caches/gemma-3-270m-python.jsonl"
    )
    mem.init_all()

    tokenizer = Tokenizer.from_file("resources/tokenizer.json")
    tokens = tokenizer.encode("The secret to life is").ids

    print("Running Inference...")
    for _ in range(10):
        input_data = np.zeros((1, MAX_SEQ_LEN), dtype=np.int32)
        input_data[0, : len(tokens)] = tokens
        session.run({input_ids_node: input_data})

        output_logits = session.get_output(logits_id)
        # Reshape flat output back to [Batch, Seq, Vocab] for argmax
        logits_reshaped = output_logits.reshape(1, MAX_SEQ_LEN, cfg["vocab_size"])
        next_token = int(np.argmax(logits_reshaped[0, len(tokens) - 1, :]))
        tokens.append(next_token)
        print(tokenizer.decode([next_token]), end="", flush=True)


if __name__ == "__main__":
    main()

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

        # Global Constants used for expansion
        self.one_fp32 = self.g.constant(
            [1], np.array([1.0], dtype=np.float32), tg_cpp.DType.FLOAT32
        )
        self.eps_fp32 = self.g.constant(
            [1], np.array([self.eps], dtype=np.float32), tg_cpp.DType.FLOAT32
        )
        self.half_fp32 = self.g.constant(
            [1], np.array([0.5], dtype=np.float32), tg_cpp.DType.FLOAT32
        )

        self.neg_one_fp32 = self.g.constant(
            [1], np.array([-1.0], dtype=np.float32), tg_cpp.DType.FLOAT32
        )
        self.two_fp32 = self.g.constant(
            [1], np.array([2.0], dtype=np.float32), tg_cpp.DType.FLOAT32
        )
        self.e_fp32 = self.g.constant(
            [1], np.array([2.718281828459], dtype=np.float32), tg_cpp.DType.FLOAT32
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
        sqrt_exp = self.expand_scalar_to_3d(self.half_fp32, dim0, self.seq_len, 1)
        std = self.g.pow(self.g.add(mean_sq, eps_exp), sqrt_exp)

        one_exp = self.expand_scalar_to_3d(self.one_fp32, dim0, self.seq_len, 1)
        inv_std = self.repeat_3d_axis(self.g.div(one_exp, std), dim_size, 2)
        x_norm = self.g.mul(x_id, inv_std)

        weight_exp = self.expand_1d_to_3d(weight_id, dim_size, dim0, self.seq_len)
        scale = self.g.add(
            weight_exp,
            self.expand_scalar_to_3d(self.one_fp32, dim0, self.seq_len, dim_size),
        )
        return self.g.mul(x_norm, scale)

    def tanh_atomic(self, x_id, last_dim):
        # tanh(x) = 2 / (1 + e^(-2x)) - 1
        neg_two = self.expand_scalar_to_3d(
            self.g.constant(
                [1], np.array([-2.0], dtype=np.float32), tg_cpp.DType.FLOAT32
            ),
            1,
            self.seq_len,
            last_dim,
        )
        two = self.expand_scalar_to_3d(self.two_fp32, 1, self.seq_len, last_dim)
        e_node = self.expand_scalar_to_3d(self.e_fp32, 1, self.seq_len, last_dim)
        one_node = self.expand_scalar_to_3d(self.one_fp32, 1, self.seq_len, last_dim)

        neg_2x = self.g.mul(x_id, neg_two)
        exp_neg_2x = self.g.pow(e_node, neg_2x)
        den = self.g.add(one_node, exp_neg_2x)
        return self.g.add(self.g.div(two, den), self.g.neg(one_node))

    def gelu_atomic(self, x_id, last_dim):
        # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        c1 = self.expand_scalar_to_3d(
            self.g.constant(
                [1], np.array([0.044715], dtype=np.float32), tg_cpp.DType.FLOAT32
            ),
            1,
            self.seq_len,
            last_dim,
        )
        c2 = self.expand_scalar_to_3d(
            self.g.constant(
                [1], np.array([0.79788456], dtype=np.float32), tg_cpp.DType.FLOAT32
            ),
            1,
            self.seq_len,
            last_dim,
        )

        x_cube = self.g.mul(self.g.mul(x_id, x_id), x_id)
        inner = self.g.mul(self.g.add(x_id, self.g.mul(x_cube, c1)), c2)
        tanh_out = self.tanh_atomic(inner, last_dim)

        one_node = self.expand_scalar_to_3d(self.one_fp32, 1, self.seq_len, last_dim)
        half_node = self.expand_scalar_to_3d(self.half_fp32, 1, self.seq_len, last_dim)
        return self.g.mul(self.g.mul(x_id, half_node), self.g.add(one_node, tanh_out))

    def compute_rope(self):
        # Implementation mirrors main.cpp RoPE generation
        zero_int = self.g.constant(
            [1], np.array([0], dtype=np.int32), tg_cpp.DType.INT32
        )
        h_dim_half = self.cfg["head_dim"] // 2

        # inv_freq = 1.0 / (10000 ^ (arange(0, head_dim, 2) / head_dim))
        indices = self.g.cast(
            self.g.arange(
                zero_int,
                self.g.constant(
                    [1],
                    np.array([self.cfg["head_dim"]], dtype=np.int32),
                    tg_cpp.DType.INT32,
                ),
                self.g.constant([1], np.array([2], dtype=np.int32), tg_cpp.DType.INT32),
            ),
            tg_cpp.DType.FLOAT32,
        )
        h_dim_fp = self.g.repeat(
            self.g.constant(
                [1],
                np.array([float(self.cfg["head_dim"])], dtype=np.float32),
                tg_cpp.DType.FLOAT32,
            ),
            self.g.constant(
                [1], np.array([h_dim_half], dtype=np.int32), tg_cpp.DType.INT32
            ),
            zero_int,
        )
        exponent = self.g.div(indices, h_dim_fp)
        base = self.g.repeat(
            self.g.constant(
                [1], np.array([10000.0], dtype=np.float32), tg_cpp.DType.FLOAT32
            ),
            self.g.constant(
                [1], np.array([h_dim_half], dtype=np.int32), tg_cpp.DType.INT32
            ),
            zero_int,
        )
        inv_freq = self.g.div(
            self.g.repeat(
                self.one_fp32,
                self.g.constant(
                    [1], np.array([h_dim_half], dtype=np.int32), tg_cpp.DType.INT32
                ),
                zero_int,
            ),
            self.g.pow(base, exponent),
        )

        # outer product pos * inv_freq
        pos = self.g.cast(
            self.g.arange(
                zero_int,
                self.g.constant(
                    [1], np.array([self.seq_len], dtype=np.int32), tg_cpp.DType.INT32
                ),
                self.g.constant([1], np.array([1], dtype=np.int32), tg_cpp.DType.INT32),
            ),
            tg_cpp.DType.FLOAT32,
        )
        pos_col = self.g.reshape(
            pos,
            self.g.constant(
                [2], np.array([self.seq_len, 1], dtype=np.int32), tg_cpp.DType.INT32
            ),
        )
        freq_row = self.g.reshape(
            inv_freq,
            self.g.constant(
                [2], np.array([1, h_dim_half], dtype=np.int32), tg_cpp.DType.INT32
            ),
        )

        angles_half = self.g.mul(
            self.repeat_3d_axis(pos_col, h_dim_half, 1),
            self.repeat_3d_axis(freq_row, self.seq_len, 0),
        )
        angles = self.g.concat(
            [angles_half, angles_half],
            self.g.constant([1], np.array([1], dtype=np.int32), tg_cpp.DType.INT32),
        )

        s3 = self.g.constant(
            [3],
            np.array([1, self.seq_len, self.cfg["head_dim"]], dtype=np.int32),
            tg_cpp.DType.INT32,
        )
        return self.g.reshape(self.g.cos(angles), s3), self.g.reshape(
            self.g.sin(angles), s3
        )

    def apply_rope(self, x_id, cos_id, sin_id, n_groups):
        # [groups, seq, head_dim]
        h_dim = self.cfg["head_dim"]
        half = h_dim // 2

        starts1 = self.g.constant(
            [3], np.array([0, 0, 0], dtype=np.int32), tg_cpp.DType.INT32
        )
        ends1 = self.g.constant(
            [3],
            np.array([n_groups, self.seq_len, half], dtype=np.int32),
            tg_cpp.DType.INT32,
        )
        steps = self.g.constant(
            [3], np.array([1, 1, 1], dtype=np.int32), tg_cpp.DType.INT32
        )
        x1 = self.g.slice(x_id, starts1, ends1, steps)

        starts2 = self.g.constant(
            [3], np.array([0, 0, half], dtype=np.int32), tg_cpp.DType.INT32
        )
        ends2 = self.g.constant(
            [3],
            np.array([n_groups, self.seq_len, h_dim], dtype=np.int32),
            tg_cpp.DType.INT32,
        )
        x2 = self.g.slice(x_id, starts2, ends2, steps)

        rotated = self.g.concat(
            [self.g.neg(x2), x1],
            self.g.constant([1], np.array([2], dtype=np.int32), tg_cpp.DType.INT32),
        )
        cos_exp = self.repeat_3d_axis(cos_id, n_groups, 0)
        sin_exp = self.repeat_3d_axis(sin_id, n_groups, 0)

        return self.g.add(self.g.mul(x_id, cos_exp), self.g.mul(rotated, sin_exp))

    def build_graph(self, input_ids_id):
        w_emb = self.weight("model.embed_tokens.weight")
        x = self.g.gather(w_emb, input_ids_id)

        # Scaling
        scale_val = self.cfg["emb_dim"] ** 0.5
        x = self.g.mul(
            x,
            self.expand_scalar_to_3d(
                self.g.constant(
                    [1], np.array([scale_val], dtype=np.float32), tg_cpp.DType.FLOAT32
                ),
                1,
                self.seq_len,
                self.cfg["emb_dim"],
            ),
        )

        rope_cos, rope_sin = self.compute_rope()

        # Causal Mask (simplified triu logic)
        mask_shape = self.g.constant(
            [2],
            np.array([self.seq_len, self.seq_len], dtype=np.int32),
            tg_cpp.DType.INT32,
        )
        ones = self.g.fill(self.one_fp32, mask_shape)
        triu_mask = self.g.triu(
            ones,
            self.g.constant([1], np.array([1], dtype=np.int32), tg_cpp.DType.INT32),
        )
        neg_inf = self.expand_scalar_to_3d(
            self.g.constant(
                [1], np.array([-1e9], dtype=np.float32), tg_cpp.DType.FLOAT32
            ),
            1,
            self.seq_len,
            self.seq_len,
        )
        mask = self.g.reshape(
            self.g.mul(triu_mask, self.g.reshape(neg_inf, mask_shape)),
            self.g.constant(
                [3],
                np.array([1, self.seq_len, self.seq_len], dtype=np.int32),
                tg_cpp.DType.INT32,
            ),
        )

        for i in range(self.cfg["n_layers"]):
            prefix = f"model.layers.{i}"
            residual = x

            # Attention Branch
            x = self.rms_norm_gemma_atomic(
                x,
                self.weight(f"{prefix}.input_layernorm.weight"),
                1,
                self.cfg["emb_dim"],
            )
            x = self.attention_block(x, prefix, rope_cos, rope_sin, mask)

            x = self.rms_norm_gemma_atomic(
                x,
                self.weight(f"{prefix}.post_attention_layernorm.weight"),
                1,
                self.cfg["emb_dim"],
            )
            x = self.g.add(residual, x)

            # MLP Branch
            residual = x
            x = self.rms_norm_gemma_atomic(
                x,
                self.weight(f"{prefix}.pre_feedforward_layernorm.weight"),
                1,
                self.cfg["emb_dim"],
            )
            x = self.mlp_block(x, prefix)

            x = self.rms_norm_gemma_atomic(
                x,
                self.weight(f"{prefix}.post_feedforward_layernorm.weight"),
                1,
                self.cfg["emb_dim"],
            )
            x = self.g.add(residual, x)

        x = self.rms_norm_gemma_atomic(
            x, self.weight("model.norm.weight"), 1, self.cfg["emb_dim"]
        )

        # Output Head (Transposed embedding dot)
        perm = self.g.constant(
            [2], np.array([1, 0], dtype=np.int32), tg_cpp.DType.INT32
        )
        w_emb_t = self.g.permute(w_emb, perm)
        s3 = self.g.constant(
            [3],
            np.array([1, self.cfg["emb_dim"], self.cfg["vocab_size"]], dtype=np.int32),
            tg_cpp.DType.INT32,
        )
        return self.g.dot(x, self.g.reshape(w_emb_t, s3))

    def attention_block(self, x, prefix, cos, sin, mask):
        def project(name, out_d):
            w = self.weight(f"{prefix}.self_attn.{name}_proj.weight")
            w_t = self.g.permute(
                w,
                self.g.constant(
                    [2], np.array([1, 0], dtype=np.int32), tg_cpp.DType.INT32
                ),
            )
            s3 = self.g.constant(
                [3],
                np.array([1, self.cfg["emb_dim"], out_d], dtype=np.int32),
                tg_cpp.DType.INT32,
            )
            return self.g.dot(x, self.g.reshape(w_t, s3))

        q = project("q", self.cfg["n_heads"] * self.cfg["head_dim"])
        k = project("k", self.cfg["n_kv_groups"] * self.cfg["head_dim"])
        v = project("v", self.cfg["n_kv_groups"] * self.cfg["head_dim"])

        # Reshape for multi-head [Batch, Seq, Heads, Dim] -> [Heads, Seq, Dim]
        perm4 = self.g.constant(
            [4], np.array([0, 2, 1, 3], dtype=np.int32), tg_cpp.DType.INT32
        )

        def split_heads(tensor, n_heads):
            s4 = self.g.constant(
                [4],
                np.array(
                    [1, self.seq_len, n_heads, self.cfg["head_dim"]], dtype=np.int32
                ),
                tg_cpp.DType.INT32,
            )
            s3 = self.g.constant(
                [3],
                np.array([n_heads, self.seq_len, self.cfg["head_dim"]], dtype=np.int32),
                tg_cpp.DType.INT32,
            )
            return self.g.reshape(self.g.permute(self.g.reshape(tensor, s4), perm4), s3)

        q, k, v = (
            split_heads(q, self.cfg["n_heads"]),
            split_heads(k, self.cfg["n_kv_groups"]),
            split_heads(v, self.cfg["n_kv_groups"]),
        )

        # Per-head Norm
        q = self.rms_norm_gemma_atomic(
            q,
            self.weight(f"{prefix}.self_attn.q_norm.weight"),
            self.cfg["n_heads"],
            self.cfg["head_dim"],
        )
        k = self.rms_norm_gemma_atomic(
            k,
            self.weight(f"{prefix}.self_attn.k_norm.weight"),
            self.cfg["n_kv_groups"],
            self.cfg["head_dim"],
        )

        q, k = self.apply_rope(q, cos, sin, self.cfg["n_heads"]), self.apply_rope(
            k, cos, sin, self.cfg["n_kv_groups"]
        )

        # GQA Repeat
        if self.cfg["n_heads"] != self.cfg["n_kv_groups"]:
            reps = self.g.constant(
                [1],
                np.array(
                    [self.cfg["n_heads"] // self.cfg["n_kv_groups"]], dtype=np.int32
                ),
                tg_cpp.DType.INT32,
            )
            zero = self.g.constant(
                [1], np.array([0], dtype=np.int32), tg_cpp.DType.INT32
            )
            k, v = self.g.repeat(k, reps, zero), self.g.repeat(v, reps, zero)

        # Scaled Dot Product
        scale = 1.0 / (self.cfg["query_pre_attn_scalar"] ** 0.5)
        q_scaled = self.g.mul(
            q,
            self.expand_scalar_to_3d(
                self.g.constant(
                    [1], np.array([scale], dtype=np.float32), tg_cpp.DType.FLOAT32
                ),
                self.cfg["n_heads"],
                self.seq_len,
                self.cfg["head_dim"],
            ),
        )

        k_t = self.g.permute(
            k,
            self.g.constant(
                [3], np.array([0, 2, 1], dtype=np.int32), tg_cpp.DType.INT32
            ),
        )
        scores = self.g.add(
            self.g.dot(q_scaled, k_t), self.repeat_3d_axis(mask, self.cfg["n_heads"], 0)
        )

        # Softmax
        axis = self.g.constant([1], np.array([-1], dtype=np.int32), tg_cpp.DType.INT32)
        max_s = self.repeat_3d_axis(self.g.max(scores, axis), self.seq_len, 2)
        exp_s = self.g.pow(
            self.expand_scalar_to_3d(
                self.e_fp32, self.cfg["n_heads"], self.seq_len, self.seq_len
            ),
            self.g.add(scores, self.g.neg(max_s)),
        )
        probs = self.g.div(
            exp_s, self.repeat_3d_axis(self.g.sum(exp_s, axis), self.seq_len, 2)
        )

        # Out projection
        context = self.g.dot(probs, v)
        s3_flat = self.g.constant(
            [3],
            np.array(
                [1, self.seq_len, self.cfg["n_heads"] * self.cfg["head_dim"]],
                dtype=np.int32,
            ),
            tg_cpp.DType.INT32,
        )
        context_flat = self.g.reshape(
            self.g.permute(
                self.g.reshape(
                    context,
                    self.g.constant(
                        [4],
                        np.array(
                            [
                                self.cfg["n_heads"],
                                1,
                                self.seq_len,
                                self.cfg["head_dim"],
                            ],
                            dtype=np.int32,
                        ),
                        tg_cpp.DType.INT32,
                    ),
                ),
                perm4,
            ),
            s3_flat,
        )

        w_o = self.weight(f"{prefix}.self_attn.o_proj.weight")
        w_o_t = self.g.permute(
            w_o,
            self.g.constant([2], np.array([1, 0], dtype=np.int32), tg_cpp.DType.INT32),
        )
        return self.g.dot(
            context_flat,
            self.g.reshape(
                w_o_t,
                self.g.constant(
                    [3],
                    np.array(
                        [
                            1,
                            self.cfg["n_heads"] * self.cfg["head_dim"],
                            self.cfg["emb_dim"],
                        ],
                        dtype=np.int32,
                    ),
                    tg_cpp.DType.INT32,
                ),
            ),
        )

    def mlp_block(self, x, prefix):
        def project(name, out_d):
            w = self.weight(f"{prefix}.mlp.{name}_proj.weight")
            w_t = self.g.permute(
                w,
                self.g.constant(
                    [2], np.array([1, 0], dtype=np.int32), tg_cpp.DType.INT32
                ),
            )
            s3 = self.g.constant(
                [3],
                np.array([1, self.cfg["emb_dim"], out_d], dtype=np.int32),
                tg_cpp.DType.INT32,
            )
            return self.g.dot(x, self.g.reshape(w_t, s3))

        gate = self.gelu_atomic(
            project("gate", self.cfg["hidden_dim"]), self.cfg["hidden_dim"]
        )
        up = project("up", self.cfg["hidden_dim"])
        gate_up = self.g.mul(gate, up)

        w_down = self.weight(f"{prefix}.mlp.down_proj.weight")
        w_down_t = self.g.permute(
            w_down,
            self.g.constant([2], np.array([1, 0], dtype=np.int32), tg_cpp.DType.INT32),
        )
        return self.g.dot(
            gate_up,
            self.g.reshape(
                w_down_t,
                self.g.constant(
                    [3],
                    np.array(
                        [1, self.cfg["hidden_dim"], self.cfg["emb_dim"]], dtype=np.int32
                    ),
                    tg_cpp.DType.INT32,
                ),
            ),
        )


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

    # Initialize Memory and Graph
    mem = tg_cpp.MemoryManager({tg_cpp.Backend.CPU: 2 * 1024 * 1024 * 1024})
    # If CUDA is available, you could add: mem.add_buffer(tg_cpp.Backend.CUDA, 2 * 1024 * 1024 * 1024)
    graph = tg_cpp.Graph()
    MAX_SEQ_LEN = 128

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
    input_node = graph.inputWithId(
        input_ids_id,
        [1, MAX_SEQ_LEN],
        tg_cpp.DType.INT32,
        input_view,
        tg_cpp.StorageType.PERSISTENT,
    )

    model = Gemma3ModelCPP(cfg, MAX_SEQ_LEN, graph, mem)
    logits_id = model.build_graph(input_node)

    session = tg_cpp.Session(
        graph, mem, logits_id, "dirty_region_caches/gemma-3-270m-python.jsonl"
    )
    mem.init()

    tokenizer = Tokenizer.from_file("resources/tokenizer.json")
    prompt = "The secret to life is"
    tokens = tokenizer.encode(prompt).ids

    print(f"Running Inference for prompt: '{prompt}'")
    for _ in range(10):
        input_data = np.zeros((1, MAX_SEQ_LEN), dtype=np.int32)
        input_data[0, : len(tokens)] = tokens
        session.run({input_node: input_data})

        # get_output returns a NumPy array view
        output_logits = session.get_output(logits_id, graph)
        # Reshape to [Batch, Seq, Vocab]
        logits_reshaped = output_logits.reshape(1, MAX_SEQ_LEN, cfg["vocab_size"])

        # Greedy sample
        next_token = int(np.argmax(logits_reshaped[0, len(tokens) - 1, :]))
        tokens.append(next_token)
        print(tokenizer.decode([next_token]), end="", flush=True)
    print()


if __name__ == "__main__":
    main()

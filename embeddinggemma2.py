"""Python graph definition and runtime wrapper for EmbeddingGemma 2."""

from __future__ import annotations

import math
from pathlib import Path

import tensor_graphs as tg


def _weight(graph, model_path: str, name: str):
    return graph.cast(graph.weight(model_path, name), tg.DType.FLOAT32)


def _matrix(graph, x, weight, in_dim: int, out_dim: int):
    weight = graph.contiguous(graph.permute_axes(weight, [1, 0]))
    weight = graph.reshape(weight, [1, in_dim, out_dim])
    return graph.dot(x, weight)


def _rms_norm(graph, x, scale, rows: int, dim: int, eps: float = 1e-6, with_scale: bool = True):
    square_sum = graph.sum_axis(graph.mul(x, x), -1)
    mean = graph.div(square_sum, graph.fill(float(dim), [1, rows, 1]))
    inverse = graph.pow(
        graph.add(mean, graph.fill(eps, [1, rows, 1])),
        graph.fill(-0.5, [1, rows, 1]),
    )
    normalized = graph.mul(x, graph.repeat(inverse, dim, 2))
    if with_scale:
        scale = graph.reshape(scale, [1, 1, dim])
        normalized = graph.mul(normalized, graph.repeat(scale, rows, 1))
    return normalized


def _gelu_tanh(graph, x, rows: int, dim: int):
    shape = [1, rows, dim]
    x3 = graph.mul(graph.mul(x, x), x)
    z = graph.mul(
        graph.add(x, graph.mul(graph.fill(0.044715, shape), x3)),
        graph.fill(0.7978845608028654, shape),
    )
    exp = graph.pow(graph.fill(math.e, shape), graph.mul(graph.fill(-2.0, shape), z))
    tanh = graph.add(graph.div(graph.fill(2.0, shape), graph.add(graph.fill(1.0, shape), exp)), graph.fill(-1.0, shape))
    return graph.mul(graph.mul(x, graph.fill(0.5, shape)), graph.add(graph.fill(1.0, shape), tanh))


def _l2_normalize(graph, x, dim: int):
    inverse = graph.pow(graph.sum_axis(graph.mul(x, x), -1), graph.fill(-0.5, [1, 1, 1]))
    return graph.mul(x, graph.repeat(inverse, dim, 2))


class _TextGraph:
    vocab_size = 262144
    hidden_size = 512
    intermediate_size = 2048
    embedding_dim = 768
    num_layers = 24
    num_heads = 4
    num_kv_heads = 2
    head_dim = 256
    layer_input_dim = 512
    sliding_window = 512
    eps = 1e-6

    def __init__(self, graph, model_path: str, seq_len: int):
        self.g = graph
        self.path = model_path
        self.seq_len = seq_len

    def _rope(self, dim: int, theta: float, sine: bool):
        values = []
        for pos in range(self.seq_len):
            for d in range(dim):
                half_idx = d if d < dim // 2 else d - dim // 2
                angle = pos * theta ** (-2.0 * half_idx / dim)
                values.append(math.sin(angle) if sine else math.cos(angle))
        return self.g.constant_float32([1, self.seq_len, dim], values)

    def _apply_rope(self, x, cosine, sine, heads: int, dim: int):
        g = self.g
        cosine = g.repeat(g.reshape(cosine, [1, self.seq_len, dim]), heads, 0)
        sine = g.repeat(g.reshape(sine, [1, self.seq_len, dim]), heads, 0)
        first = g.contiguous(g.slice_dims(x, [0, 0, 0], [heads, self.seq_len, dim // 2], [1, 1, 1]))
        second = g.contiguous(g.slice_dims(x, [0, 0, dim // 2], [heads, self.seq_len, dim], [1, 1, 1]))
        rotated = g.concat([g.neg(second), first], 2)
        return g.add(g.mul(x, cosine), g.mul(rotated, sine))

    def _project(self, x, prefix: str, name: str, out_dim: int):
        weight = _weight(self.g, self.path, f"{prefix}.self_attn.{name}.weight")
        return _matrix(self.g, x, weight, self.hidden_size, out_dim)

    def _attention(self, x, attention_mask, prefix: str, layer_idx: int):
        g = self.g
        global_attention = layer_idx % 6 == 5
        dim = 512 if global_attention else self.head_dim
        kv_heads = 1 if global_attention else self.num_kv_heads
        q_width = self.num_heads * dim
        kv_width = kv_heads * dim
        q = self._project(x, prefix, "q_proj", q_width)
        k = self._project(x, prefix, "k_proj", kv_width)
        v = self._project(x, prefix, "v_proj", kv_width)

        def to_heads(value, heads):
            value = g.reshape(value, [1, self.seq_len, heads, dim])
            value = g.permute_axes(value, [0, 2, 1, 3])
            return g.reshape(value, [heads, self.seq_len, dim])

        q, k, v = to_heads(q, self.num_heads), to_heads(k, kv_heads), to_heads(v, kv_heads)
        q_norm = _weight(g, self.path, f"{prefix}.self_attn.q_norm.weight")
        k_norm = _weight(g, self.path, f"{prefix}.self_attn.k_norm.weight")
        q = g.reshape(_rms_norm(g, g.reshape(q, [1, self.num_heads * self.seq_len, dim]), q_norm,
                                self.num_heads * self.seq_len, dim, self.eps),
                      [self.num_heads, self.seq_len, dim])
        k = g.reshape(_rms_norm(g, g.reshape(k, [1, kv_heads * self.seq_len, dim]), k_norm,
                                kv_heads * self.seq_len, dim, self.eps),
                      [kv_heads, self.seq_len, dim])
        v = g.reshape(_rms_norm(g, g.reshape(v, [1, kv_heads * self.seq_len, dim]), k_norm,
                                kv_heads * self.seq_len, dim, self.eps, False),
                      [kv_heads, self.seq_len, dim])
        theta = 1_000_000.0 if global_attention else 10_000.0
        q = self._apply_rope(q, self._rope(dim, theta, False), self._rope(dim, theta, True), self.num_heads, dim)
        k = self._apply_rope(k, self._rope(dim, theta, False), self._rope(dim, theta, True), kv_heads, dim)
        if kv_heads != self.num_heads:
            repeats_per_kv_head = self.num_heads // kv_heads
            k = g.reshape(k, [kv_heads, 1, self.seq_len, dim])
            v = g.reshape(v, [kv_heads, 1, self.seq_len, dim])
            k = g.contiguous(g.repeat(k, repeats_per_kv_head, 1))
            v = g.contiguous(g.repeat(v, repeats_per_kv_head, 1))
            k = g.reshape(k, [self.num_heads, self.seq_len, dim])
            v = g.reshape(v, [self.num_heads, self.seq_len, dim])
        scores = g.dot(q, g.contiguous(g.permute_axes(k, [0, 2, 1])))
        mask = []
        for _head in range(self.num_heads):
            for i in range(self.seq_len):
                for j in range(self.seq_len):
                    blocked = not global_attention and abs(i - j) > self.sliding_window
                    mask.append(-1.0e9 if blocked else 0.0)
        scores = g.add(scores, g.constant_float32([self.num_heads, self.seq_len, self.seq_len], mask))
        key_mask = g.repeat(g.repeat(g.reshape(attention_mask, [1, 1, self.seq_len]), self.seq_len, 1), self.num_heads, 0)
        padding = g.mul(g.add(g.fill(1.0, [self.num_heads, self.seq_len, self.seq_len]), g.neg(key_mask)),
                        g.fill(-1.0e9, [self.num_heads, self.seq_len, self.seq_len]))
        scores = g.add(scores, padding)
        maximum = g.repeat(g.max_axis(scores, -1), self.seq_len, 2)
        exps = g.pow(g.fill(math.e, [self.num_heads, self.seq_len, self.seq_len]), g.add(scores, g.neg(maximum)))
        probs = g.div(exps, g.repeat(g.sum_axis(exps, -1), self.seq_len, 2))
        context = g.dot(probs, v)
        context = g.contiguous(g.permute_axes(g.reshape(context, [1, self.num_heads, self.seq_len, dim]), [0, 2, 1, 3]))
        context = g.reshape(context, [1, self.seq_len, q_width])
        weight = _weight(g, self.path, f"{prefix}.self_attn.o_proj.weight")
        return _matrix(g, context, weight, q_width, self.hidden_size)

    def _mlp(self, x, prefix: str):
        g = self.g
        def project(name, out_dim):
            return _matrix(g, x, _weight(g, self.path, f"{prefix}.mlp.{name}.weight"), self.hidden_size, out_dim)
        gate = _gelu_tanh(g, project("gate_proj", self.intermediate_size), self.seq_len, self.intermediate_size)
        up = project("up_proj", self.intermediate_size)
        return _matrix(g, g.mul(gate, up), _weight(g, self.path, f"{prefix}.mlp.down_proj.weight"),
                       self.intermediate_size, self.hidden_size)

    def build(self, input_ids_or_embeddings, attention_mask, inputs_embeds: bool = False):
        g = self.g
        x = input_ids_or_embeddings
        if not inputs_embeds:
            embeddings = g.gather(_weight(g, self.path, "language_model.embed_tokens.weight"), x)
            x = g.mul(embeddings, g.fill(math.sqrt(self.hidden_size), [1, self.seq_len, self.hidden_size]))
        ple = _matrix(g, x, _weight(g, self.path, "language_model.ple.per_layer_model_projection.weight"),
                      self.hidden_size, self.num_layers * self.layer_input_dim)
        ple = g.mul(ple, g.fill(1.0 / math.sqrt(self.hidden_size), [1, self.seq_len, self.num_layers * self.layer_input_dim]))
        ple = g.reshape(ple, [1, self.seq_len, self.num_layers, self.layer_input_dim])
        ple_norm = _weight(g, self.path, "language_model.ple.per_layer_projection_norm.weight")
        ple = g.reshape(_rms_norm(g, g.reshape(ple, [1, self.seq_len * self.num_layers, self.layer_input_dim]),
                                  ple_norm, self.seq_len * self.num_layers, self.layer_input_dim, self.eps),
                        [1, self.seq_len, self.num_layers, self.layer_input_dim])
        for layer_idx in range(self.num_layers):
            prefix = f"language_model.layers.{layer_idx}"
            residual = x
            norm = _weight(g, self.path, f"{prefix}.input_layernorm.weight")
            attn_input = _rms_norm(g, x, norm, self.seq_len, self.hidden_size, self.eps)
            attn_out = self._attention(attn_input, attention_mask, prefix, layer_idx)
            norm = _weight(g, self.path, f"{prefix}.post_attention_layernorm.weight")
            x = g.add(residual, _rms_norm(g, attn_out, norm, self.seq_len, self.hidden_size, self.eps))
            residual = x
            norm = _weight(g, self.path, f"{prefix}.pre_feedforward_layernorm.weight")
            ff = self._mlp(_rms_norm(g, x, norm, self.seq_len, self.hidden_size, self.eps), prefix)
            norm = _weight(g, self.path, f"{prefix}.post_feedforward_layernorm.weight")
            x = g.add(residual, _rms_norm(g, ff, norm, self.seq_len, self.hidden_size, self.eps))
            ple_i = g.contiguous(g.slice_dims(ple, [0, 0, layer_idx, 0],
                                              [1, self.seq_len, layer_idx + 1, self.layer_input_dim], [1, 1, 1, 1]))
            ple_i = g.reshape(ple_i, [1, self.seq_len, self.layer_input_dim])
            gate_w = _weight(g, self.path, f"{prefix}.ple_block.per_layer_input_gate.weight")
            gate = _matrix(g, x, gate_w, self.hidden_size, self.layer_input_dim)
            gate = g.mul(_gelu_tanh(g, gate, self.seq_len, self.layer_input_dim), ple_i)
            proj_w = _weight(g, self.path, f"{prefix}.ple_block.per_layer_projection.weight")
            projected = _matrix(g, gate, proj_w, self.layer_input_dim, self.hidden_size)
            norm = _weight(g, self.path, f"{prefix}.ple_block.post_per_layer_input_norm.weight")
            value = g.add(x, _rms_norm(g, projected, norm, self.seq_len, self.hidden_size, self.eps))
            scalar = g.fill_from(_weight(g, self.path, f"{prefix}.layer_scalar"), [1, self.seq_len, self.hidden_size])
            x = g.mul(value, scalar)
        norm = _weight(g, self.path, "language_model.norm.weight")
        x = _rms_norm(g, x, norm, self.seq_len, self.hidden_size, self.eps)
        projection = _weight(g, self.path, "language_model.embedding_projection.weight")
        token_embeddings = _matrix(g, x, projection, self.hidden_size, self.embedding_dim)
        mask = g.repeat(g.reshape(attention_mask, [1, self.seq_len, 1]), self.embedding_dim, 2)
        pooled = g.sum_axis(g.mul(token_embeddings, mask), 1)
        count = g.sum_axis(attention_mask, 1)
        count = g.repeat(g.reshape(count, [1, 1, 1]), self.embedding_dim, 2)
        return _l2_normalize(g, g.div(pooled, count), self.embedding_dim)


class _VisionGraph:
    hidden_size = 768
    intermediate_size = 3072
    num_layers = 16
    heads = 12
    head_dim = 64
    patch_size = 16
    pool_size = 3
    position_size = 10240
    eps = 1e-6
    rope_theta = 100.0

    def __init__(self, graph, model_path: str, patches: int, grid_width: int):
        self.g, self.path = graph, model_path
        self.patches, self.grid_width = patches, grid_width
        self.output_length = patches // (self.pool_size * self.pool_size)
        self.seq_len = patches

    def _project(self, x, name: str, in_dim: int, out_dim: int):
        return _matrix(self.g, x, _weight(self.g, self.path, name), in_dim, out_dim)

    def _norm(self, x, name: str, rows: int, with_scale=True):
        scale = _weight(self.g, self.path, name) if with_scale else self.g.fill(1.0, [self.hidden_size])
        return _rms_norm(self.g, x, scale, rows, self.hidden_size, self.eps, with_scale)

    def _rotate(self, x):
        cos_values, sin_values = [], []
        axis_dim = self.head_dim // 4
        for row in range(self.seq_len):
            for d in range(self.head_dim):
                coord = row % self.grid_width if d < self.head_dim // 2 else row // self.grid_width
                idx = (d % (self.head_dim // 2)) % axis_dim
                angle = coord * self.rope_theta ** (-2.0 * idx / (2.0 * axis_dim))
                cos_values.append(math.cos(angle))
                sin_values.append(math.sin(angle))
        g = self.g
        cosine = g.repeat(g.constant_float32([1, self.seq_len, self.head_dim], cos_values), self.heads, 0)
        sine = g.repeat(g.constant_float32([1, self.seq_len, self.head_dim], sin_values), self.heads, 0)
        first = g.contiguous(g.slice_dims(x, [0, 0, 0], [self.heads, self.seq_len, self.head_dim // 2], [1, 1, 1]))
        second = g.contiguous(g.slice_dims(x, [0, 0, self.head_dim // 2], [self.heads, self.seq_len, self.head_dim], [1, 1, 1]))
        rotated = g.concat([g.neg(second), first], 2)
        return g.add(g.mul(x, cosine), g.mul(rotated, sine))

    def _attention(self, x, prefix: str):
        g, h, d, s = self.g, self.heads, self.head_dim, self.seq_len
        def project(name):
            return self._project(x, f"{prefix}.self_attn.{name}.linear.weight", self.hidden_size, h * d)
        def heads(value):
            value = g.reshape(value, [1, s, h, d])
            return g.reshape(g.permute_axes(value, [0, 2, 1, 3]), [h, s, d])
        q, k, v = map(heads, (project("q_proj"), project("k_proj"), project("v_proj")))
        q = g.reshape(self._norm(g.reshape(q, [1, h * s, d]), f"{prefix}.self_attn.q_norm.weight", h * s), [h, s, d])
        k = g.reshape(self._norm(g.reshape(k, [1, h * s, d]), f"{prefix}.self_attn.k_norm.weight", h * s), [h, s, d])
        v = g.reshape(self._norm(g.reshape(v, [1, h * s, d]), "", h * s, False), [h, s, d])
        q, k = self._rotate(q), self._rotate(k)
        scores = g.dot(q, g.contiguous(g.permute_axes(k, [0, 2, 1])))
        exps = g.pow(g.fill(math.e, [h, s, s]), g.add(scores, g.neg(g.repeat(g.max_axis(scores, -1), s, 2))))
        probs = g.div(exps, g.repeat(g.sum_axis(exps, -1), s, 2))
        context = g.dot(probs, v)
        context = g.contiguous(g.permute_axes(g.reshape(context, [1, h, s, d]), [0, 2, 1, 3]))
        context = g.reshape(context, [1, s, self.hidden_size])
        return self._project(context, f"{prefix}.self_attn.o_proj.linear.weight", self.hidden_size, self.hidden_size)

    def _mlp(self, x, prefix: str):
        gate = self._project(x, f"{prefix}.mlp.gate_proj.linear.weight", self.hidden_size, self.intermediate_size)
        up = self._project(x, f"{prefix}.mlp.up_proj.linear.weight", self.hidden_size, self.intermediate_size)
        activated = self.g.mul(_gelu_tanh(self.g, gate, self.seq_len, self.intermediate_size), up)
        return self._project(activated, f"{prefix}.mlp.down_proj.linear.weight", self.intermediate_size, self.hidden_size)

    def _position_embeddings(self):
        g = self.g
        table = _weight(g, self.path, "vision_tower.patch_embedder.position_embedding_table")
        x_table = g.reshape(g.slice_dims(table, [0, 0, 0], [1, self.position_size, self.hidden_size], [1, 1, 1]),
                            [self.position_size, self.hidden_size])
        y_table = g.reshape(g.slice_dims(table, [1, 0, 0], [2, self.position_size, self.hidden_size], [1, 1, 1]),
                            [self.position_size, self.hidden_size])
        xs = [i % self.grid_width for i in range(self.patches)]
        ys = [i // self.grid_width for i in range(self.patches)]
        x_idx = g.constant_int32([self.patches], xs)
        y_idx = g.constant_int32([self.patches], ys)
        x_emb = g.reshape(g.gather(x_table, x_idx), [1, self.patches, self.hidden_size])
        y_emb = g.reshape(g.gather(y_table, y_idx), [1, self.patches, self.hidden_size])
        return g.add(x_emb, y_emb)

    def build(self, pixel_values):
        g = self.g
        pixel_width = 3 * self.patch_size * self.patch_size
        x = g.mul(g.add(pixel_values, g.fill(-0.5, [1, self.patches, pixel_width])),
                  g.fill(2.0, [1, self.patches, pixel_width]))
        x = self._project(x, "vision_tower.patch_embedder.input_proj.weight", pixel_width, self.hidden_size)
        x = g.add(x, self._position_embeddings())
        for i in range(self.num_layers):
            prefix = f"vision_tower.encoder.layers.{i}"
            residual = x
            x = self._norm(x, f"{prefix}.input_layernorm.weight", self.seq_len)
            x = g.add(residual, self._norm(self._attention(x, prefix), f"{prefix}.post_attention_layernorm.weight", self.seq_len))
            residual = x
            x = self._norm(x, f"{prefix}.pre_feedforward_layernorm.weight", self.seq_len)
            x = g.add(residual, self._norm(self._mlp(x, prefix), f"{prefix}.post_feedforward_layernorm.weight", self.seq_len))
        pool_weights = [0.0] * (self.output_length * self.patches)
        grid_height = self.patches // self.grid_width
        pooled_width = self.grid_width // self.pool_size
        pool_scale = 1.0 / (self.pool_size * self.pool_size)
        for y in range(grid_height):
            for x_idx in range(self.grid_width):
                patch = y * self.grid_width + x_idx
                pooled = (y // self.pool_size) * pooled_width + x_idx // self.pool_size
                pool_weights[pooled * self.patches + patch] = pool_scale
        pool_matrix = g.constant_float32([1, self.output_length, self.patches], pool_weights)
        x = g.dot(pool_matrix, x)
        x = g.mul(x, g.fill(math.sqrt(self.hidden_size), [1, self.output_length, self.hidden_size]))
        x = _rms_norm(g, x, g.fill(1.0, [self.hidden_size]), self.output_length, self.hidden_size, self.eps, False)
        projection = _weight(g, self.path, "embed_vision.embedding_projection.weight")
        return _matrix(g, x, projection, self.hidden_size, 512)


class _AudioGraph:
    input_size = 128
    hidden_size = 1024
    num_layers = 12
    heads = 8
    chunk_size = 12
    context_left = 13
    context_right = 0
    conv_kernel = 5
    output_size = 1536
    eps = 1e-6
    logit_cap = 50.0
    residual_weight = 0.5

    def __init__(self, graph, model_path: str, mel_frames: int):
        if mel_frames < 4:
            raise ValueError("EmbeddingGemma 2 audio requires at least four mel frames")
        self.g, self.path = graph, model_path
        self.feature_frames = mel_frames
        self.seq_len = ((mel_frames + 1) // 2 + 1) // 2

    def _weight(self, name: str):
        return _weight(self.g, self.path, name)

    def _clamp(self, x, prefix: str, output=False):
        g = self.g
        suffix = "output" if output else "input"
        lo, hi = self._weight(f"{prefix}.{suffix}_min"), self._weight(f"{prefix}.{suffix}_max")
        shape = list(g.getNode(x).shape)
        lower = g.add(lo, g.relu(g.add(x, g.neg(lo)), shape))
        return g.add(lower, g.neg(g.relu(g.add(lower, g.neg(hi)), shape)))

    def _linear(self, x, name: str, in_dim: int, out_dim: int, rows: int, clippable=False):
        g = self.g
        if clippable:
            x = self._clamp(x, name)
        result = _matrix(g, x, self._weight(f"{name}.linear.weight"), in_dim, out_dim)
        return self._clamp(result, name, True) if clippable else result

    def _norm(self, x, name: str, rows: int, dim: int, with_scale=True):
        scale = self._weight(f"{name}.weight") if with_scale else self.g.fill(1.0, [dim])
        return _rms_norm(self.g, x, scale, rows, dim, self.eps, with_scale)

    def _layer_norm_last(self, x, name: str, rows: int, dim: int):
        g = self.g
        mean = g.div(g.sum_axis(x, -1), g.fill(float(dim), [1, rows, 1]))
        centered = g.add(x, g.neg(g.repeat(mean, dim, 2)))
        variance = g.div(g.sum_axis(g.mul(centered, centered), -1), g.fill(float(dim), [1, rows, 1]))
        inverse = g.pow(g.add(variance, g.fill(self.eps, [1, rows, 1])), g.fill(-0.5, [1, rows, 1]))
        normalized = g.mul(centered, g.repeat(inverse, dim, 2))
        return g.mul(normalized, g.reshape(self._weight(f"{name}.weight"), [1, 1, dim]))

    def _conv_subsample(self, features, in_channels: int, out_channels: int, prefix: str,
                        input_h: int, input_w: int):
        g = self.g
        cols = g.im2col(features, 3, 2, 1)
        out_h, out_w = (input_h + 1) // 2, (input_w + 1) // 2
        weight = self._weight(f"{prefix}.conv.weight")
        weight = g.reshape(weight, [out_channels, in_channels * 9])
        weight = g.reshape(g.contiguous(g.permute_axes(weight, [1, 0])), [1, in_channels * 9, out_channels])
        out = g.dot(cols, weight)
        out = g.reshape(out, [1, out_channels, out_h, out_w])
        out = g.contiguous(g.permute_axes(out, [0, 2, 3, 1]))
        out = g.reshape(out, [1, out_h * out_w, out_channels])
        out = self._layer_norm_last(out, f"{prefix}.norm", out_h * out_w, out_channels)
        return g.relu(out, [1, out_h * out_w, out_channels]), out_h, out_w

    def _softplus(self, x, rows: int, dim: int):
        shape = list(self.g.getNode(x).shape)
        exp_x = self.g.pow(self.g.fill(math.e, shape), x)
        return self.g.log(self.g.add(self.g.fill(1.0, shape), exp_x))

    def _silu(self, x, rows: int, dim: int):
        g = self.g
        shape = [1, rows, dim]
        exp_x = g.pow(g.fill(math.e, shape), g.neg(x))
        return g.div(x, g.add(g.fill(1.0, shape), exp_x))

    def _feed_forward(self, x, prefix: str):
        g = self.g
        residual = x
        x = self._norm(x, f"{prefix}.pre_layer_norm", self.seq_len, self.hidden_size)
        x = self._linear(x, f"{prefix}.ffw_layer_1", self.hidden_size, self.hidden_size * 4, self.seq_len, True)
        x = self._silu(x, self.seq_len, self.hidden_size * 4)
        x = self._linear(x, f"{prefix}.ffw_layer_2", self.hidden_size * 4, self.hidden_size, self.seq_len, True)
        x = self._norm(x, f"{prefix}.post_layer_norm", self.seq_len, self.hidden_size)
        return g.add(residual, g.mul(x, g.fill(self.residual_weight, [1, self.seq_len, self.hidden_size])))

    def _causal_conv(self, x, prefix: str):
        g = self.g
        conv_input = self._norm(x, f"{prefix}.pre_layer_norm", self.seq_len, self.hidden_size)
        start = self._linear(conv_input, f"{prefix}.linear_start", self.hidden_size, self.hidden_size * 2, self.seq_len, True)
        a = g.contiguous(g.slice_dims(start, [0, 0, 0], [1, self.seq_len, self.hidden_size], [1, 1, 1]))
        b = g.contiguous(g.slice_dims(start, [0, 0, self.hidden_size], [1, self.seq_len, self.hidden_size * 2], [1, 1, 1]))
        conv_input = g.mul(a, b)
        kernel = self._weight(f"{prefix}.depthwise_conv1d.weight")
        convolved = g.fill(0.0, [1, self.seq_len, self.hidden_size])
        for offset in range(self.conv_kernel):
            left = self.conv_kernel - 1 - offset
            sample = g.contiguous(g.slice_dims(conv_input, [0, left, 0],
                                               [1, self.seq_len, self.hidden_size], [1, 1, 1]))
            if left:
                sample = g.concat([g.fill(0.0, [1, left, self.hidden_size]), sample], 1)
            wk = g.reshape(g.slice_dims(kernel, [0, 0, offset],
                                        [self.hidden_size, 1, offset + 1], [1, 1, 1]),
                           [1, 1, self.hidden_size])
            convolved = g.add(convolved, g.mul(sample, g.repeat(wk, self.seq_len, 1)))
        convolved = self._norm(convolved, f"{prefix}.conv_norm", self.seq_len, self.hidden_size)
        convolved = self._silu(convolved, self.seq_len, self.hidden_size)
        convolved = self._linear(convolved, f"{prefix}.linear_end", self.hidden_size, self.hidden_size, self.seq_len, True)
        return g.add(x, convolved)

    def _attention(self, x, prefix: str):
        g = self.g
        heads, dim, seq_len = self.heads, self.hidden_size // self.heads, self.seq_len
        def project(name):
            return self._linear(x, f"{prefix}.{name}", self.hidden_size, self.hidden_size, seq_len, True)
        q = g.reshape(project("q_proj"), [1, seq_len, heads, dim])
        k = g.reshape(project("k_proj"), [1, seq_len, heads, dim])
        v = g.reshape(project("v_proj"), [1, seq_len, heads, dim])
        q = g.mul(q, g.fill(1.0 / math.sqrt(dim) / math.log(2.0), [1, seq_len, heads, dim]))
        k = g.mul(k, g.fill(math.log1p(math.e) / math.log(2.0), [1, seq_len, heads, dim]))
        dim_scale = g.reshape(self._weight(f"{prefix}.per_dim_scale"), [1, 1, 1, dim])
        q = g.mul(q, self._softplus(dim_scale, 1, dim))
        q = g.reshape(g.contiguous(g.permute_axes(q, [0, 2, 1, 3])), [heads, seq_len, dim])
        k = g.reshape(g.contiguous(g.permute_axes(k, [0, 2, 1, 3])), [heads, seq_len, dim])
        v = g.reshape(g.contiguous(g.permute_axes(v, [0, 2, 1, 3])), [heads, seq_len, dim])
        scores = g.dot(q, g.contiguous(g.permute_axes(k, [0, 2, 1])))
        mask = [-1.0e9] * (heads * seq_len * seq_len)
        for i in range(seq_len):
            chunk_start = (i // self.chunk_size) * self.chunk_size
            lo = max(0, chunk_start - (self.context_left - 1))
            hi = min(seq_len, chunk_start + self.chunk_size + self.context_right)
            for j in range(lo, hi):
                for h in range(heads):
                    mask[(h * seq_len + i) * seq_len + j] = 0.0
        scores = g.add(scores, g.constant_float32([heads, seq_len, seq_len], mask))
        shape = [heads, seq_len, seq_len]
        maximum = g.repeat(g.max_axis(scores, -1), seq_len, 2)
        exp_neg = g.pow(g.fill(math.e, shape), g.mul(g.fill(-2.0 / self.logit_cap, shape), scores))
        tanh_scores = g.add(g.div(g.fill(2.0, shape), g.add(g.fill(1.0, shape), exp_neg)), g.fill(-1.0, shape))
        scores = g.mul(g.fill(self.logit_cap, shape), tanh_scores)
        shifted = g.add(scores, g.neg(g.repeat(g.max_axis(scores, -1), seq_len, 2)))
        exp_scores = g.pow(g.fill(math.e, shape), shifted)
        weights = g.div(exp_scores, g.repeat(g.sum_axis(exp_scores, -1), seq_len, 2))
        context = g.dot(weights, v)
        context = g.contiguous(g.permute_axes(g.reshape(context, [1, heads, seq_len, dim]), [0, 2, 1, 3]))
        context = g.reshape(context, [1, seq_len, self.hidden_size])
        return self._linear(context, f"{prefix}.post", self.hidden_size, self.hidden_size, seq_len, True)

    def build(self, input_features):
        g = self.g
        x = g.reshape(input_features, [1, 1, self.feature_frames, self.input_size])
        x, h0, w0 = self._conv_subsample(x, 1, 128, "audio_tower.subsample_conv_projection.layer0",
                                        self.feature_frames, self.input_size)
        x = g.reshape(g.permute_axes(g.reshape(x, [1, h0, w0, 128]), [0, 3, 1, 2]), [1, 128, h0, w0])
        x, h1, w1 = self._conv_subsample(x, 128, 32, "audio_tower.subsample_conv_projection.layer1", h0, w0)
        x = g.reshape(x, [1, h1, w1 * 32])
        x = self._linear(x, "audio_tower.subsample_conv_projection.input_proj_linear", w1 * 32, self.hidden_size, h1)
        for i in range(self.num_layers):
            prefix = f"audio_tower.layers.{i}"
            x = self._feed_forward(x, f"{prefix}.feed_forward1")
            residual = x
            x = self._norm(x, f"{prefix}.norm_pre_attn", self.seq_len, self.hidden_size)
            x = self._attention(x, f"{prefix}.self_attn")
            x = self._norm(x, f"{prefix}.norm_post_attn", self.seq_len, self.hidden_size)
            x = g.add(residual, x)
            x = self._causal_conv(x, f"{prefix}.lconv1d")
            x = self._feed_forward(x, f"{prefix}.feed_forward2")
            x = self._norm(x, f"{prefix}.norm_out", self.seq_len, self.hidden_size)
        output_w = self._weight("audio_tower.output_proj.weight")
        output_b = self._weight("audio_tower.output_proj.bias")
        x = g.add(_matrix(g, x, output_w, self.hidden_size, self.output_size),
                  g.reshape(output_b, [1, 1, self.output_size]))
        x = _rms_norm(g, x, g.fill(1.0, [self.output_size]), self.seq_len, self.output_size, self.eps, False)
        return _matrix(g, x, self._weight("embed_audio.embedding_projection.weight"), self.output_size, 512)


class EmbeddingGemma2:
    """Compile and run an EmbeddingGemma 2 graph defined in Python."""

    def __init__(self, model_path, modality, sequence_length=32, patch_count=2520,
                 patch_grid_width=60, mel_frames=280, video_frames=1,
                 media_placeholder_positions=(), cache_file="", compile_no_weights_bucket=False,
                 compile_dirty_input_bucket=False, disable_node_caching=False,
                 disable_compilation_caching=False, min_compile_seconds=1.0):
        if modality not in ("text", "image", "video", "audio"):
            raise ValueError("EmbeddingGemma2 modality must be text, image, video, or audio")
        if modality == "video" and video_frames < 1:
            raise ValueError("EmbeddingGemma2 video requires at least one frame")
        if modality in ("image", "video") and (
            not patch_count or not patch_grid_width or patch_count % 9
            or (patch_count // patch_grid_width) % 3 or patch_grid_width % 3
        ):
            raise ValueError("EmbeddingGemma2 vision patch grid must divide evenly into 3x3 pooling windows")
        self.modality = modality
        self.model_path = str(model_path)
        self.sequence_length = sequence_length
        self.input_shape = None
        self.token_ids_id = None
        self.graph = tg.Graph()
        self.mem = tg.MemoryManager()
        self.media_placeholder_positions = list(media_placeholder_positions)
        self.token_shape = [1, sequence_length]
        if modality == "text":
            self.input_shape = [1, sequence_length]
            self.input_id = self.graph.input(self.input_shape, tg.DType.INT32)
            self.token_ids_id = self.input_id
            text_embeddings = None
        else:
            self.token_ids_id = self.graph.input(self.token_shape, tg.DType.INT32)
            if modality in ("image", "video"):
                frame_count = video_frames if modality == "video" else 1
                self.input_shape = [1, frame_count, patch_count, 768]
                self.input_id = self.graph.input(self.input_shape, tg.DType.FLOAT32)
                vision = _VisionGraph(self.graph, self.model_path, patch_count, patch_grid_width)
                frames = []
                for frame in range(frame_count):
                    frame_value = self.graph.slice_dims(
                        self.input_id, [0, frame, 0, 0], [1, frame + 1, patch_count, 768], [1, 1, 1, 1]
                    )
                    frames.append(vision.build(self.graph.reshape(frame_value, [1, patch_count, 768])))
                text_embeddings = frames[0] if len(frames) == 1 else self.graph.concat(frames, 1)
            else:
                self.input_shape = [1, mel_frames, 128]
                self.input_id = self.graph.input(self.input_shape, tg.DType.FLOAT32)
                text_embeddings = _AudioGraph(self.graph, self.model_path, mel_frames).build(self.input_id)
            text_embeddings = self._merge_media(text_embeddings, sequence_length)
        attention_mask = self.graph.fill(1.0, [1, sequence_length])
        self.root_id = _TextGraph(self.graph, self.model_path, sequence_length).build(
            self.input_id if modality == "text" else text_embeddings,
            attention_mask,
            inputs_embeds=modality != "text",
        )
        self.session = tg.Session(self.graph, self.mem, self.root_id, cache_file,
                                  disable_node_caching, False, False, False,
                                  disable_compilation_caching, min_compile_seconds)
        self.session.ensure_full_bucket()
        if compile_no_weights_bucket or compile_dirty_input_bucket:
            dirty = {}
            if compile_no_weights_bucket:
                dirty[self.token_ids_id] = tg.make_full_regions(self.token_shape)
            dirty[self.input_id] = tg.make_full_regions(self.input_shape)
            if compile_dirty_input_bucket:
                dirty[self.token_ids_id] = tg.make_full_regions(self.token_shape)
            out_regions = tg.make_full_regions(self.graph.getNode(self.root_id).shape)
            self.session.add_bucket(dirty, out_regions)
        self.session.compile()

    def _merge_media(self, media_embeddings, seq_len: int):
        positions = self.media_placeholder_positions
        expected_media = (
            (self.input_shape[2] // 9) * self.input_shape[1]
            if self.modality in ("image", "video")
            else ((self.input_shape[1] + 1) // 2 + 1) // 2
        )
        if len(positions) != expected_media:
            raise ValueError("media placeholder count does not match encoder soft-token count")
        g = self.graph
        text = g.mul(g.gather(_weight(g, self.model_path, "language_model.embed_tokens.weight"), self.token_ids_id),
                     g.fill(math.sqrt(512.0), [1, seq_len, 512]))
        segments, text_cursor, media_cursor = [], 0, 0
        position_cursor = 0
        while position_cursor < len(positions):
            begin = positions[position_cursor]
            if begin < text_cursor or begin >= seq_len:
                raise ValueError("media placeholder positions must be sorted and within input_ids")
            if begin > text_cursor:
                segments.append(g.contiguous(g.slice_dims(text, [0, text_cursor, 0], [1, begin, 512], [1, 1, 1])))
            end_cursor = position_cursor + 1
            while end_cursor < len(positions) and positions[end_cursor] == positions[end_cursor - 1] + 1:
                end_cursor += 1
            run = end_cursor - position_cursor
            segments.append(g.contiguous(g.slice_dims(media_embeddings, [0, media_cursor, 0],
                                                       [1, media_cursor + run, 512], [1, 1, 1])))
            text_cursor, media_cursor = begin + run, media_cursor + run
            position_cursor = end_cursor
        if text_cursor < seq_len:
            segments.append(g.contiguous(g.slice_dims(text, [0, text_cursor, 0], [1, seq_len, 512], [1, 1, 1])))
        return g.concat(segments, 1)

    def embed_text(self, token_ids):
        if self.modality != "text":
            raise RuntimeError("embed_text is available only for text sessions")
        self.session.write_input_int32(self.input_id, token_ids)
        return self.session.run_float32(self.root_id)

    def embed_media(self, values, token_ids):
        if self.modality == "text":
            raise RuntimeError("embed_media is unavailable for text sessions")
        if len(token_ids) != self.sequence_length:
            raise ValueError("token count does not match compiled media sequence length")
        text_ids = list(token_ids)
        for position in self.media_placeholder_positions:
            if not 0 <= position < len(text_ids):
                raise ValueError("media placeholder position is outside token_ids")
            text_ids[position] = 0
        self.session.write_input_int32(self.token_ids_id, text_ids)
        self.session.write_input_float32(self.input_id, values)
        return self.session.run_float32(self.root_id)


__all__ = ["EmbeddingGemma2"]

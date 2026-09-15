import random
from typing import Protocol

import numpy as np
import tensor_graphs

from .config import TrainConfig

DEFAULT_MODEL_PATHS = {
    "gemma-3-270m": "models/google/gemma-3-270m",
    "qwen-3.6-35b-a3b": "models/Qwen/Qwen3.6-35B-A3B",
    "krea": "models/krea/Krea-2-Turbo",
    "krea-2-turbo": "models/krea/Krea-2-Turbo",
    "krea-2-turbo-vae": "models/krea/Krea-2-Turbo/qwen_image_vae.safetensors",
    "vae": "models/krea/Krea-2-Turbo/qwen_image_vae.safetensors",
    "qwen-image-vae": "models/krea/Krea-2-Turbo/qwen_image_vae.safetensors",
    "qwen3-vl": "models/krea/Krea-2-Turbo/qwen3vl_4b_bf16.safetensors",
    "qwen3-vl-bf16": "models/krea/Krea-2-Turbo/qwen3vl_4b_bf16.safetensors",
    "qwen3vl": "models/krea/Krea-2-Turbo/qwen3vl_4b_bf16.safetensors",
    "qwen3vl_4b_bf16": "models/krea/Krea-2-Turbo/qwen3vl_4b_bf16.safetensors",
    "deepseek-v4": "models/deepseek-ai/DeepSeek-V4",
}


def get_default_model_path(model_name: str) -> str:
    norm = model_name.lower().replace("_", "-")
    for k, v in DEFAULT_MODEL_PATHS.items():
        if k.lower().replace("_", "-") == norm:
            return v
    return f"models/{model_name}"


def generate_random_graph(
    num_nodes: int = 20,
    hidden_dim: int = 128,
    seq_len: int = 64,
    seed: int | None = None,
) -> tuple[tensor_graphs.Graph, tensor_graphs.LogicalId, list]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    g = tensor_graphs.Graph()
    available_nodes = []

    in0 = g.input([1, seq_len, hidden_dim], tensor_graphs.DType.FLOAT32)
    in1 = g.input([1, seq_len, hidden_dim], tensor_graphs.DType.FLOAT32)
    in_w = g.input([1, hidden_dim, hidden_dim], tensor_graphs.DType.FLOAT32)

    available_nodes.append((in0, [1, seq_len, hidden_dim]))
    available_nodes.append((in1, [1, seq_len, hidden_dim]))
    available_nodes.append((in_w, [1, hidden_dim, hidden_dim]))

    runtime_inputs = {in0, in1, in_w}
    op_choices = ["unary", "binary", "dot", "reduce", "reshape", "permute"]

    for _ in range(num_nodes):
        op = random.choice(op_choices)
        if op == "unary":
            src, shape = random.choice(available_nodes)
            u_op = random.choice(["sin", "cos", "neg", "log"])
            out = getattr(g, u_op)(src)
            available_nodes.append((out, list(shape)))

        elif op == "binary":
            src1, shape1 = random.choice(available_nodes)
            compat_nodes = [n for n in available_nodes if n[1] == shape1]
            if compat_nodes:
                src2, _ = random.choice(compat_nodes)
                b_op = random.choice(["add", "mul", "div"])
                out = getattr(g, b_op)(src1, src2)
                available_nodes.append((out, list(shape1)))

        elif op == "dot":
            src1, shape1 = random.choice(available_nodes)
            if len(shape1) >= 2:
                K = shape1[-1]
                compat_nodes = [
                    n
                    for n in available_nodes
                    if len(n[1]) == len(shape1)
                    and n[1][:-2] == shape1[:-2]
                    and n[1][-2] == K
                ]
                if compat_nodes:
                    src2, shape2 = random.choice(compat_nodes)
                    out = g.dot(src1, src2)
                    out_shape = list(shape1[:-1]) + [shape2[-1]]
                    available_nodes.append((out, out_shape))

        elif op == "reduce":
            src, shape = random.choice(available_nodes)
            if len(shape) > 0:
                axis = random.randint(0, len(shape) - 1)
                ax_id = g.constant([axis])
                r_op = random.choice(["sum", "max"])
                out = getattr(g, r_op)(src, ax_id)
                out_shape = list(shape)
                out_shape[axis] = 1
                available_nodes.append((out, out_shape))

        elif op == "reshape":
            src, shape = random.choice(available_nodes)
            if len(shape) == 3:
                out_shape = [shape[0], shape[2], shape[1]]
                out = g.reshape(src, out_shape)
                available_nodes.append((out, out_shape))

        elif op == "permute":
            src, shape = random.choice(available_nodes)
            if len(shape) == 3:
                dims = [0, 2, 1]
                dims_id = g.constant(dims)
                out = g.permute(src, dims_id)
                out_shape = [shape[dims[0]], shape[dims[1]], shape[dims[2]]]
                available_nodes.append((out, out_shape))

    root, root_shape = available_nodes[-1]
    full_bucket = tensor_graphs.Bucket()

    reachable_nodes = set()
    stack = [root]
    while stack:
        curr = stack.pop()
        if curr in reachable_nodes:
            continue
        reachable_nodes.add(curr)
        node = g.getNode(curr)
        for child in node.child_ids:
            stack.append(child)

    dirty_map = {}
    for node_id in reachable_nodes:
        if node_id in runtime_inputs:
            node = g.getNode(node_id)
            r = tensor_graphs.Region()
            r.region = [tensor_graphs.Dim(0, d) for d in node.shape]
            dirty_map[node_id] = [r]

    full_bucket.inputDirtyRegions = dirty_map
    r_out = tensor_graphs.Region()
    r_out.region = [tensor_graphs.Dim(0, d) for d in root_shape]
    full_bucket.outputNeededRegion = [r_out]

    return g, root, [full_bucket]


def compute_graph_memory_metrics(
    graph: tensor_graphs.Graph, root_id: tensor_graphs.LogicalId
) -> dict[str, int]:
    """Computes exact memory requirements across all reachable nodes in the graph."""
    reachable_nodes = set()
    stack = [root_id]
    while stack:
        curr = stack.pop()
        if curr in reachable_nodes:
            continue
        reachable_nodes.add(curr)
        node = graph.getNode(curr)
        for child in node.child_ids:
            stack.append(child)

    def node_bytes(n):
        num_elems = 1
        for d in n.shape:
            num_elems *= d
        return num_elems * 4

    max_node_bytes = 0
    total_node_bytes = 0
    max_op_peak = 0

    for node_id in reachable_nodes:
        node = graph.getNode(node_id)
        out_sz = node_bytes(node)
        in_sz = sum(
            node_bytes(graph.getNode(c)) for c in node.child_ids if graph.hasNode(c)
        )

        max_node_bytes = max(max_node_bytes, out_sz)
        total_node_bytes += out_sz
        max_op_peak = max(max_op_peak, out_sz + in_sz)

    return {
        "max_node_bytes": max_node_bytes,
        "max_op_peak": max(max_op_peak, max_node_bytes),
        "total_node_bytes": max(total_node_bytes, max_node_bytes),
    }


def sample_calibrated_mem_cap(metrics: dict[str, int], rng: random.Random) -> int:
    """Samples memory capacity across impossible, tight, moderate, and generous regimes."""
    max_node = metrics["max_node_bytes"]
    max_op_peak = metrics["max_op_peak"]
    total_bytes = metrics["total_node_bytes"]

    mode = rng.random()

    if mode < 0.15:
        # Impossible / Infeasible (15%): Cap is smaller than required for a single op or tensor
        factor = rng.uniform(0.1, 0.8)
        cap = int(max_node * factor)
        return max(512, cap)

    elif mode < 0.50:
        # Hard / Very Tight (35%): Calibrated right at the minimum theoretical peak for 1 op
        factor = rng.uniform(0.9, 1.3)
        return max(max_node, int(max_op_peak * factor))

    elif mode < 0.80:
        # Moderate / Constrained (30%): Requires smart buffer reuse across multiple steps
        factor = rng.uniform(1.4, 2.5)
        return max(max_op_peak, int(max_op_peak * factor))

    else:
        # Generous / Relaxed (20%): Large budget focusing primarily on execution latency
        factor = rng.uniform(1.2, 3.0)
        return min(256 * 1024 * 1024, max(int(total_bytes * factor), max_op_peak * 4))


class BaseGraphProvider(Protocol):
    def get_context(
        self, config: TrainConfig, episode: int = 0
    ) -> tensor_graphs.SaturatedEGraphContext: ...


class ModelGraphProvider:
    def __init__(self, model_name: str, model_path: str):
        self.model_name = model_name
        self.model_path = model_path
        self._cached_context = None

    def get_context(
        self, config: TrainConfig, episode: int = 0
    ) -> tensor_graphs.SaturatedEGraphContext:
        if self._cached_context is None:
            model_path = self.model_path or get_default_model_path(self.model_name)
            self._cached_context = tensor_graphs.build_and_saturate_egraph(
                self.model_name,
                model_path,
                config.log_cost_calls,
                True,
                config.seq_len,
            )
            if config.bucket_weights:
                self._cached_context.bucket_weights = config.bucket_weights
        return self._cached_context


class RandomGraphProvider:
    def __init__(self, worker_rank: int = 0):
        self.worker_rank = worker_rank
        self._cached_context = None
        self._last_sampled_episode = -1

    def get_context(
        self, config: TrainConfig, episode: int = 0
    ) -> tensor_graphs.SaturatedEGraphContext:
        need_resample = False
        if self._cached_context is None or (
            config.resample_graph_every > 0
            and (episode - self._last_sampled_episode) >= config.resample_graph_every
        ):
            need_resample = True

        if need_resample:
            seed = (
                (config.random_seed or 42) + self.worker_rank * 10007 + episode
            ) & 0x7FFFFFFF
            rng = random.Random(seed)

            num_nodes = rng.randint(config.random_min_nodes, config.random_max_nodes)
            graph, root, buckets = generate_random_graph(
                num_nodes=num_nodes,
                hidden_dim=config.random_dim,
                seq_len=config.random_seq_len,
                seed=seed,
            )

            metrics = compute_graph_memory_metrics(graph, root)
            mem_cap = sample_calibrated_mem_cap(metrics, rng)

            self._cached_context = tensor_graphs.build_and_saturate_egraph_from_graph(
                graph, root, buckets, config.log_cost_calls, mem_cap
            )
            if config.bucket_weights:
                self._cached_context.bucket_weights = config.bucket_weights
            self._last_sampled_episode = episode

        return self._cached_context


def get_graph_provider(config: TrainConfig, worker_rank: int = 0) -> BaseGraphProvider:
    if config.graph_source.lower() == "random":
        return RandomGraphProvider(worker_rank=worker_rank)
    return ModelGraphProvider(config.model_name, config.model_path)

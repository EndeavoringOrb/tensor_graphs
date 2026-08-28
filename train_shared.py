import dataclasses
import json
import math
import pickle
import random
import socket
import struct
import zlib
from pathlib import Path
from typing import Protocol

import numpy as np
import tensor_graphs
import torch

from train_models import PolicyValueRNN

torch.set_float32_matmul_precision("high")


@dataclasses.dataclass
class TrainConfig:
    algo: str = "gumbel_alphazero"  # "gumbel_alphazero", "alphazero", "reinforce"
    model_type: str = "rnn"  # "rnn" or "transformer"
    run_dir: str = "runs"
    model_name: str = "gemma-3-270m"
    model_path: str = "models/google/gemma-3-270m"
    seq_len: int = 128  # LLM model sequence length (e.g. gemma-3-270m)
    num_simulations: int = 4
    level_simulations: list = dataclasses.field(default_factory=lambda: [1, 1, 1, 1])
    replay_buffer_size: int = 100_000
    batch_size: int = 64
    save_interval: int = 100

    # Graph Source & Generation Config
    graph_source: str = "model"  # "model" or "random"
    random_min_nodes: int = 10
    random_max_nodes: int = 300
    random_hidden_dim: int = 128
    random_seq_len: int = 64
    random_seed: int | None = None
    resample_graph_every: int = 1

    # Transformer / Model Architecture Config
    d_model: int = 32
    nhead: int = 2
    num_layers: int = 2
    max_feat_dim: int = 8

    lr: float = 1e-3
    log_cost_calls: bool = False
    bucket_idx: int = -1
    compile_decode_buckets: bool = False
    workers: int = 4
    cpp_threads: int = 1

    # PUCT & Gumbel Noise Annealing Config
    c_puct: float = 1.25
    base_noise: float = 0.25
    min_noise: float = 0.01
    decay_episodes: int = 500
    depth_gamma: float = 0.99
    c_scale: float = 1.0
    c_visit: float = 50.0
    gumbel_max_actions: int = 2  # top-m candidate actions for sequential halving. half of num_simulations

    # REINFORCE / PPO hyperparameters
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    temperature: float = 1.0

    # Networking Config
    host: str = "127.0.0.1"
    port: int = 5000
    use_bluetooth: bool = False
    bt_host_address: str = "AC:F2:3C:A7:F7:EC"
    bt_port: int = 4

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TrainConfig":
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=4), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TrainConfig":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls.from_dict(data)


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


# ==============================================================================
# GRAPH PROVIDER & GENERATOR
# ==============================================================================
def generate_random_graph(
    num_nodes: int = 20,
    hidden_dim: int = 128,
    seq_len: int = 64,
    seed: int | None = None,
) -> tuple[tensor_graphs.Graph, tensor_graphs.LogicalId, list]:
    """Generates an extensive, shape-compatible random computation graph."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    g = tensor_graphs.Graph()

    available_nodes = []

    # Establish base inputs
    in0 = g.input([1, seq_len, hidden_dim], tensor_graphs.DType.FLOAT32)
    in1 = g.input([1, seq_len, hidden_dim], tensor_graphs.DType.FLOAT32)
    in_w = g.input([1, hidden_dim, hidden_dim], tensor_graphs.DType.FLOAT32)

    available_nodes.append((in0, [1, seq_len, hidden_dim]))
    available_nodes.append((in1, [1, seq_len, hidden_dim]))
    available_nodes.append((in_w, [1, hidden_dim, hidden_dim]))

    # Keep track of specifically the runtime inputs for dirtiness mapping
    runtime_inputs = {in0, in1, in_w}

    op_choices = [
        "unary",
        "binary",
        "dot",
        "reduce",
        "reshape",
        "permute",
        "concat",
        "repeat",
        "slice",
        "triu",
        "fill",
    ]

    for _ in range(num_nodes):
        op = random.choice(op_choices)

        if op == "unary":
            src, shape = random.choice(available_nodes)
            u_op = random.choice(["sin", "cos", "neg", "log", "relu"])
            if u_op == "sin":
                out = g.sin(src)
            elif u_op == "cos":
                out = g.cos(src)
            elif u_op == "neg":
                out = g.neg(src)
            elif u_op == "log":
                out = g.log(src)
            elif u_op == "relu":
                out = g.relu(src, shape)
            available_nodes.append((out, list(shape)))

        elif op == "binary":
            src1, shape1 = random.choice(available_nodes)
            # Binary element-wise requires exact shape match
            compat_nodes = [n for n in available_nodes if n[1] == shape1]
            if compat_nodes:
                src2, _ = random.choice(compat_nodes)
                b_op = random.choice(["add", "mul", "div", "pow"])
                if b_op == "add":
                    out = g.add(src1, src2)
                elif b_op == "mul":
                    out = g.mul(src1, src2)
                elif b_op == "div":
                    out = g.div(src1, src2)
                elif b_op == "pow":
                    out = g.pow(src1, src2)
                available_nodes.append((out, list(shape1)))

        elif op == "dot":
            src1, shape1 = random.choice(available_nodes)
            if len(shape1) >= 2:
                K = shape1[-1]
                # Dot: requires identical prefix ranks and compatible inner K dimension
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
                r_op = random.choice(["sum", "max"])
                ax_id = g.constant([axis])  # Reduction axis requires LogicalId
                if r_op == "sum":
                    out = g.sum(src, ax_id)
                elif r_op == "max":
                    out = g.max(src, ax_id)
                out_shape = list(shape)
                out_shape[axis] = 1
                available_nodes.append((out, out_shape))

        elif op == "reshape":
            src, shape = random.choice(available_nodes)
            if len(shape) == 3:
                out_shape = [shape[0], shape[2], shape[1]]
                out = g.reshape(
                    src, out_shape
                )  # Reshape python binding takes list[int]
                available_nodes.append((out, out_shape))
            elif len(shape) == 2:
                out_shape = [shape[1], shape[0]]
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
            elif len(shape) == 2:
                dims = [1, 0]
                dims_id = g.constant(dims)
                out = g.permute(src, dims_id)
                out_shape = [shape[dims[0]], shape[dims[1]]]
                available_nodes.append((out, out_shape))

        elif op == "concat":
            src1, shape1 = random.choice(available_nodes)
            if len(shape1) > 0:
                axis = random.randint(0, len(shape1) - 1)
                compat_nodes = []
                for n in available_nodes:
                    s2 = n[1]
                    if len(s2) == len(shape1):
                        match = True
                        for d in range(len(shape1)):
                            if d != axis and shape1[d] != s2[d]:
                                match = False
                                break
                        if match:
                            compat_nodes.append(n)
                if compat_nodes:
                    src2, shape2 = random.choice(compat_nodes)
                    out = g.concat([src1, src2], axis)
                    out_shape = list(shape1)
                    out_shape[axis] = shape1[axis] + shape2[axis]
                    available_nodes.append((out, out_shape))

        elif op == "repeat":
            src, shape = random.choice(available_nodes)
            if len(shape) > 0:
                # Find an axis with size 1 to satisfy native striding constraints
                ones = [i for i, d in enumerate(shape) if d == 1]
                if ones:
                    axis = random.choice(ones)
                    repeats = random.choice([2, 3])
                    out = g.repeat(src, repeats, axis)
                    out_shape = list(shape)
                    out_shape[axis] *= repeats
                    available_nodes.append((out, out_shape))

        elif op == "slice":
            src, shape = random.choice(available_nodes)
            if len(shape) > 0:
                axis = random.randint(0, len(shape) - 1)
                if shape[axis] > 1:
                    st = random.randint(0, shape[axis] - 1)
                    en = random.randint(st + 1, shape[axis])

                    starts = [0] * len(shape)
                    ends = list(shape)
                    steps = [1] * len(shape)

                    starts[axis] = st
                    ends[axis] = en

                    st_id = g.constant(starts)
                    en_id = g.constant(ends)
                    stps_id = g.constant(steps)

                    out = g.slice(src, st_id, en_id, stps_id)
                    out_shape = list(shape)
                    out_shape[axis] = en - st
                    available_nodes.append((out, out_shape))

        elif op == "triu":
            src, shape = random.choice(available_nodes)
            if len(shape) >= 2 and shape[-1] == shape[-2]:
                k_id = g.constant([0])
                out = g.triu(src, k_id)
                available_nodes.append((out, list(shape)))

        elif op == "fill":
            f_shape = [1, seq_len, hidden_dim]
            out = g.fill(1.0, f_shape)
            available_nodes.append((out, f_shape))

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
            model_path = self.model_path
            if not model_path or (
                model_path == "models/google/gemma-3-270m"
                and self.model_name != "gemma-3-270m"
            ):
                model_path = get_default_model_path(self.model_name)

            self._cached_context = tensor_graphs.build_and_saturate_egraph(
                self.model_name,
                model_path,
                config.log_cost_calls,
                config.compile_decode_buckets,
                config.seq_len,
            )
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
            num_nodes = random.Random(seed).randint(
                config.random_min_nodes, config.random_max_nodes
            )

            graph, root, buckets = generate_random_graph(
                num_nodes=num_nodes,
                hidden_dim=config.random_hidden_dim,
                seq_len=config.random_seq_len,
                seed=seed,
            )

            rng = random.Random(seed)
            mem_cap = rng.randint(1 * 1024, 256 * 1024 * 1024)  # 1KB to 256MB

            self._cached_context = tensor_graphs.build_and_saturate_egraph_from_graph(
                graph, root, buckets, config.log_cost_calls, mem_cap
            )
            self._last_sampled_episode = episode

        return self._cached_context


def get_graph_provider(config: TrainConfig, worker_rank: int = 0):
    if config.graph_source.lower() == "random":
        return RandomGraphProvider(worker_rank=worker_rank)
    return ModelGraphProvider(config.model_name, config.model_path)


# ==============================================================================
# SPARSE PREFIX DEDUPLICATION & TRAJECTORY CODEC
# ==============================================================================
@dataclasses.dataclass
class PrefixData:
    global_feature: np.ndarray  # (1, 8) float32
    node_features: np.ndarray  # (N, 8) float32
    edge_index: np.ndarray  # (2, E) int64
    phase_id: int


class TrajectoryCodec:
    @staticmethod
    def compute_prefix(
        phase_id: int,
        node_features: np.ndarray,
        edge_src: np.ndarray,
        edge_dst: np.ndarray,
    ) -> tuple[int, PrefixData]:
        N = len(node_features)

        global_feature = np.zeros((1, 8), dtype=np.float32)
        global_feature[0, 0] = phase_id

        nodes = np.zeros((N, 8), dtype=np.float32)
        if N > 0:
            nodes[:, 0] = np.arange(N)
            dim_feat = min(7, node_features.shape[1])
            nodes[:, 1 : 1 + dim_feat] = node_features[:, :dim_feat]

        if len(edge_src) > 0:
            edge_index = np.stack([edge_src, edge_dst], axis=0).astype(np.int64)
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)

        prefix_key = hash(
            global_feature.tobytes()
            + nodes.tobytes()
            + edge_index.tobytes()
            + bytes([phase_id])
        )
        return prefix_key, PrefixData(
            global_feature=global_feature,
            node_features=nodes,
            edge_index=edge_index,
            phase_id=phase_id,
        )

    @staticmethod
    def pack_episode(
        mcts_tree: dict,
        best_Z: float,
        prefix_registry: dict[int, PrefixData],
        tau: float = 1.25,
        blend_k: float = 2.0,
        algo: str = "gumbel_alphazero",
        model_type: str = "rnn",
    ) -> dict:
        referenced_prefixes = {}
        transitions = []

        for _, node_data in mcts_tree.items():
            pkey = node_data.get("prefix_key", 0)
            if model_type == "transformer":
                if pkey in prefix_registry and pkey not in referenced_prefixes:
                    referenced_prefixes[pkey] = prefix_registry[pkey]

            counts = node_data["N"]
            total_counts = float(counts.sum())
            prior_p = node_data["P"]
            is_gumbel = node_data.get("is_gumbel", False) or (
                algo == "gumbel_alphazero"
            )

            if is_gumbel and "logits" in node_data:
                logits = node_data["logits"]
                W_sa = node_data["W"]
                v_s = node_data.get("v", 0.0)
                Q_sa = np.where(counts > 0, W_sa / np.maximum(counts, 1.0), v_s)
                visited_mask = counts > 0
                if np.any(visited_mask):
                    min_q = min(float(v_s), float(np.min(Q_sa[visited_mask])))
                    max_q = max(float(v_s), float(np.max(Q_sa[visited_mask])))
                    if max_q > min_q:
                        Q_norm = (Q_sa - min_q) / (max_q - min_q)
                    else:
                        Q_norm = np.full_like(Q_sa, 0.5)
                else:
                    Q_norm = np.zeros_like(Q_sa)

                max_visit = float(np.max(counts)) if len(counts) > 0 else 0.0
                c_visit = 50.0
                c_scale = 1.0
                sigma_q = (c_visit + max_visit) * c_scale * Q_norm
                improved_logits = logits + sigma_q
                exp_l = np.exp(improved_logits - np.max(improved_logits))
                pi = exp_l / np.maximum(1e-8, exp_l.sum())
            elif total_counts > 0:
                # Target temperature softening to avoid extreme one-hot targets on low sim counts
                smoothed_counts = np.power(counts, 1.0 / max(1e-4, tau))
                sum_smoothed = smoothed_counts.sum()
                smoothed_pi = (
                    smoothed_counts / sum_smoothed
                    if sum_smoothed > 0
                    else prior_p.copy()
                )

                # Prior blending: smoothly interpolate between prior and MCTS policy
                blend_weight = total_counts / (total_counts + blend_k)
                pi = blend_weight * smoothed_pi + (1.0 - blend_weight) * prior_p
            else:
                pi = prior_p.copy()

            tr = {
                "prefix_key": pkey,
                "action_features": node_data["action_features"],
                "phase_id": node_data.get("phase_id", 0),
                "pis": pi.astype(np.float32),
                "z": best_Z,
            }
            if "hidden" in node_data and node_data["hidden"] is not None:
                tr["hidden"] = node_data["hidden"]
            if "global_feat" in node_data and node_data["global_feat"] is not None:
                tr["global_feat"] = node_data["global_feat"]

            transitions.append(tr)

        return {
            "model_type": model_type,
            "prefixes": referenced_prefixes,
            "transitions": transitions,
        }


# ==============================================================================
# NETWORK SOCKET UTILITIES
# ==============================================================================
def create_client_socket(config: TrainConfig):
    if config.use_bluetooth:
        if not hasattr(socket, "AF_BLUETOOTH"):
            raise RuntimeError("Bluetooth is not supported on this platform.")
        sock = socket.socket(
            socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM
        )
        sock.connect((config.bt_host_address, config.bt_port))
    else:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((config.host, config.port))
    return sock


def create_server_socket(host: str, port: int, use_bluetooth: bool = False):
    is_bt = use_bluetooth or (
        isinstance(host, str) and ":" in host and len(host.split(":")) == 6
    )
    if is_bt:
        if not hasattr(socket, "AF_BLUETOOTH"):
            raise RuntimeError("Bluetooth is not supported on this platform.")
        sock = socket.socket(
            socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM
        )
        sock.bind((host, port))
    else:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
    sock.listen(5)
    return sock


def send_msg(sock, msg):
    data = zlib.compress(pickle.dumps(msg))
    sock.sendall(struct.pack(">I", len(data)) + data)


def recvall(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data


def recv_msg(sock):
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack(">I", raw_msglen)[0]
    data = recvall(sock, msglen)
    if data is None:
        return None
    return pickle.loads(zlib.decompress(data))


class ActorDelegate(tensor_graphs.SearchDelegate):
    PHASE_MAP = {"cache": 0, "extract": 1, "dispatch": 2, "bufferize": 3, "malloc": 4}

    def __init__(
        self,
        agent=None,
        model: PolicyValueRNN | None = None,
        req_queue=None,
        resp_queue=None,
        worker_id: int = 0,
        mcts_tree: dict | None = None,
        c_puct: float = 1.25,
        exploration_noise=None,
        episode: int = 0,
        decay_episodes: int = 500,
        base_noise: float = 0.25,
        min_noise: float = 0.01,
        depth_gamma: float = 0.7,
        version: int = 0,
        shared_action_feats: torch.Tensor | None = None,
        shared_logits: torch.Tensor | None = None,
        shared_v: torch.Tensor | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.agent = agent
        self.model = model
        self.req_queue = req_queue
        self.resp_queue = resp_queue
        self.worker_id = worker_id
        self.mcts_tree = mcts_tree if mcts_tree is not None else {}
        self.c_puct = c_puct
        self.episode = episode
        self.decay_episodes = decay_episodes
        self.base_noise = base_noise if exploration_noise is None else exploration_noise
        self.min_noise = min_noise
        self.depth_gamma = depth_gamma
        self.version = version

        if device is not None:
            self.device = device
        elif model is not None:
            self.device = next(model.parameters()).device
        elif agent is not None:
            self.device = next(agent.parameters()).device
        else:
            self.device = torch.device("cpu")

        if self.model is not None:
            self.root_hidden = self.model.init_hidden(batch_size=1, device=self.device)
            self.current_hidden = self.root_hidden.clone()
            self.hidden_stack: list[torch.Tensor] = []
            self.log_mem_cpp = 0.5
            self.log_mem_cuda = 0.5
            self.log_mem_opencl = 0.5

        # Shared memory references
        self.shared_action_feats = shared_action_feats
        self.shared_logits = shared_logits
        self.shared_v = shared_v

        if exploration_noise is not None:
            self.episode_noise = (
                0.0
                if exploration_noise == 0.0
                else max(
                    min_noise,
                    exploration_noise * (1.0 - episode / max(1, decay_episodes)),
                )
            )
        else:
            self.episode_noise = max(
                min_noise, base_noise * (1.0 - episode / max(1, decay_episodes))
            )

        self.active_stack = []
        self.prefix_registry: dict[int, PrefixData] = {}
        self.current_prefix_keys: dict[str, int] = {}
        self.phase_values = {}
        self.prefix_cache_ctx = {}
        self.prefix_cache_v = {}
        self.worker_registered_prefixes = set()

    def reset_for_episode(self, mem_caps: dict[int, int] | None = None):
        if self.model is not None:
            if mem_caps is not None:
                self.log_mem_cpp = (
                    math.log1p(
                        max(0.0, float(mem_caps.get(1, 16 * 1024 * 1024 * 1024)))
                    )
                    / 25.0
                )
                self.log_mem_cuda = (
                    math.log1p(
                        max(0.0, float(mem_caps.get(2, 24 * 1024 * 1024 * 1024)))
                    )
                    / 25.0
                )
                self.log_mem_opencl = (
                    math.log1p(max(0.0, float(mem_caps.get(3, 1024 * 1024 * 1024))))
                    / 25.0
                )
            self.current_hidden = self.model.init_hidden(
                batch_size=1, device=self.device
            )
            self.hidden_stack.clear()

    def fast_fail(self) -> bool:
        return True

    def push_state(self):
        if self.model is not None:
            self.hidden_stack.append(self.current_hidden.clone())

    def pop_state(self):
        if self.active_stack:
            self.active_stack.pop()
        if self.model is not None and self.hidden_stack:
            self.current_hidden = self.hidden_stack.pop()

    def on_leaf_evaluated(self, cost: float):
        cost_val = float(cost)
        if 0.0 <= cost_val < float("inf") and not math.isnan(cost_val):
            # Positive cost -> Valid plan
            z = 1000.0 / (cost_val + 1.0)
        elif cost_val < 0.0:
            # Negative cost is our shaped hierarchical failure reward in [-1.0, 0.0)
            z = cost_val
        else:
            z = -1.0

        for state_key, act in self.active_stack:
            if state_key in self.mcts_tree:
                self.mcts_tree[state_key]["N"][act] += 1.0
                self.mcts_tree[state_key]["W"][act] += z

    def _get_global_feature(self, phase_id: int) -> np.ndarray:
        depth = float(len(self.active_stack)) / 100.0
        return np.array(
            [
                getattr(self, "log_mem_cpp", 0.5),
                getattr(self, "log_mem_cuda", 0.5),
                getattr(self, "log_mem_opencl", 0.5),
                depth,
                float(phase_id),
                0.0,
                0.0,
                1.0,
            ],
            dtype=np.float32,
        )

    def _store_raw_graph(self, phase_name, node_features, edge_src, edge_dst):
        phase_id = self.PHASE_MAP[phase_name]
        if self.model is not None:
            # RNN mode: full graph structure is not processed by RNN; no graph serialization or RPC needed!
            self.current_prefix_keys[phase_name] = phase_id
            return

        dim_map = {
            "cache": 5,
            "extract": 5,
            "dispatch": 5,
            "bufferize": 5,
            "malloc": 3,
        }
        dim = dim_map.get(phase_name, 5)
        if not node_features:
            nf = np.zeros((0, dim), dtype=np.float32)
            src = np.zeros(0, dtype=np.int64)
            dst = np.zeros(0, dtype=np.int64)
        else:
            nf = np.nan_to_num(
                np.array(node_features, dtype=np.float32), posinf=1e9, neginf=-1e9
            )
            nf = (
                nf.reshape(-1, dim)
                if len(nf) > 0
                else np.zeros((0, dim), dtype=np.float32)
            )
            src = np.array(edge_src, dtype=np.int64)
            dst = np.array(edge_dst, dtype=np.int64)

        prefix_key, prefix_data = TrajectoryCodec.compute_prefix(phase_id, nf, src, dst)
        self.prefix_registry[prefix_key] = prefix_data
        self.current_prefix_keys[phase_name] = prefix_key

        cache_key = (self.version, prefix_key)
        if self.agent is not None:
            if cache_key not in self.prefix_cache_ctx:
                device = next(self.agent.parameters()).device
                gf_t = torch.tensor(
                    prefix_data.global_feature, dtype=torch.float32, device=device
                ).unsqueeze(0)
                nf_t = torch.tensor(
                    prefix_data.node_features, dtype=torch.float32, device=device
                ).unsqueeze(0)
                e_t = torch.tensor(
                    prefix_data.edge_index, dtype=torch.int64, device=device
                )
                pid_t = torch.tensor(
                    [prefix_data.phase_id], dtype=torch.int64, device=device
                )

                with (
                    torch.inference_mode(),
                    torch.autocast(device_type=device.type, dtype=torch.bfloat16),
                ):
                    v, ctx = self.agent.encode_prefix(gf_t, nf_t, e_t, pid_t)

                self.prefix_cache_ctx[cache_key] = ctx
                self.prefix_cache_v[cache_key] = v.item() if v is not None else 0.0
            self.phase_values[phase_name] = self.prefix_cache_v[cache_key]
        else:
            if self.req_queue is not None:
                if cache_key not in self.worker_registered_prefixes:
                    self.req_queue.put(
                        ("register_prefix", self.version, prefix_key, prefix_data)
                    )
                    self.worker_registered_prefixes.add(cache_key)

    def init_cache_graph(self, node_features, edge_src, edge_dst):
        self._store_raw_graph("cache", node_features, edge_src, edge_dst)

    def init_egraph(self, node_features, edge_src, edge_dst):
        self._store_raw_graph("extract", node_features, edge_src, edge_dst)

    def init_dispatch_graph(self, node_features, edge_src, edge_dst):
        self._store_raw_graph("dispatch", node_features, edge_src, edge_dst)

    def init_bufferize_graph(self, node_features, edge_src, edge_dst):
        self._store_raw_graph("bufferize", node_features, edge_src, edge_dst)

    def init_malloc_graph(self, node_features, edge_src, edge_dst):
        self._store_raw_graph("malloc", node_features, edge_src, edge_dst)

    def _evaluate_node(
        self, phase_name: str, action_feats_np: np.ndarray, prefix_key: int
    ):
        phase_id = self.PHASE_MAP[phase_name]

        if self.model is not None:
            # In-Process PolicyValueRNN Evaluation
            global_feat = self._get_global_feature(phase_id)
            g_t = torch.tensor(
                global_feat, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            a_t = torch.tensor(
                action_feats_np, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            with torch.inference_mode():
                logits, val = self.model.evaluate_candidates(
                    self.current_hidden, g_t, a_t, phase_id
                )
            scores = logits[0].cpu().numpy()
            v = val[0, 0].item()
            self.phase_values[phase_name] = v
            hidden_np = self.current_hidden[0].detach().cpu().numpy()
            return scores, v, phase_id, global_feat, hidden_np

        cache_key = (self.version, prefix_key)
        if self.agent is not None:
            # Local In-Process AlphaZeroTransformer Evaluation
            device = next(self.agent.parameters()).device
            ctx = self.prefix_cache_ctx[cache_key]
            v = self.prefix_cache_v[cache_key]

            A_len = action_feats_np.shape[0]
            padded_actions = torch.zeros(
                (1, A_len, 8), dtype=torch.float32, device=device
            )
            padded_pid = torch.full(
                (1, A_len), phase_id, dtype=torch.int64, device=device
            )

            dim_feat = min(7, action_feats_np.shape[1])
            padded_actions[0, :A_len, 1 : 1 + dim_feat] = torch.tensor(
                action_feats_np[:, :dim_feat], dtype=torch.float32, device=device
            )
            padded_actions[0, :A_len, 0] = torch.arange(
                A_len, dtype=torch.float32, device=device
            )

            with (
                torch.inference_mode(),
                torch.autocast(device_type=device.type, dtype=torch.bfloat16),
            ):
                logits = self.agent.evaluate_actions(
                    padded_actions, padded_pid, context=ctx
                )
            scores = logits[0, :A_len].cpu().float().numpy()
        else:
            if self.shared_action_feats is not None:
                # Shared Memory Fast-Path (Remote Evaluator)
                A_len = action_feats_np.shape[0]
                max_actions = self.shared_action_feats.shape[1]

                if A_len > max_actions:
                    A_len = max_actions
                    action_feats_np = action_feats_np[:A_len]

                dim_feat = min(7, action_feats_np.shape[1])

                # Zero out the active buffer slice first
                self.shared_action_feats[self.worker_id, :A_len, :].zero_()
                self.shared_action_feats[self.worker_id, :A_len, :dim_feat].copy_(
                    torch.from_numpy(action_feats_np[:, :dim_feat])
                )

                self.req_queue.put(
                    (
                        "evaluate_shm",
                        self.version,
                        prefix_key,
                        A_len,
                        phase_id,
                        self.worker_id,
                    )
                )
                status, *data = self.resp_queue.get()

                if status == "error" and data[0] == "missing_prefix":
                    pdata = self.prefix_registry[prefix_key]
                    self.req_queue.put(
                        ("register_prefix", self.version, prefix_key, pdata)
                    )
                    self.worker_registered_prefixes.add(cache_key)
                    self.req_queue.put(
                        (
                            "evaluate_shm",
                            self.version,
                            prefix_key,
                            A_len,
                            phase_id,
                            self.worker_id,
                        )
                    )
                    status, *data = self.resp_queue.get()

                scores = self.shared_logits[self.worker_id, :A_len].clone().numpy()
                v = self.shared_v[self.worker_id].item()
                self.phase_values[phase_name] = v
            else:
                # Queue Fallback
                self.req_queue.put(
                    (
                        "evaluate",
                        self.version,
                        prefix_key,
                        action_feats_np,
                        phase_id,
                        self.worker_id,
                    )
                )
                status, *data = self.resp_queue.get()

                if status == "error" and data[0] == "missing_prefix":
                    pdata = self.prefix_registry[prefix_key]
                    self.req_queue.put(
                        ("register_prefix", self.version, prefix_key, pdata)
                    )
                    self.worker_registered_prefixes.add(cache_key)
                    self.req_queue.put(
                        (
                            "evaluate",
                            self.version,
                            prefix_key,
                            action_feats_np,
                            phase_id,
                            self.worker_id,
                        )
                    )
                    status, *data = self.resp_queue.get()

                scores, v = data
                self.phase_values[phase_name] = v

        return scores, v, phase_id, None, None

    @torch.inference_mode()
    def _order_items(self, items, phase_name, extract_fn):
        num_actions = len(items)
        if num_actions <= 1:
            return list(range(num_actions))

        action_feats = extract_fn(items)
        if isinstance(action_feats, torch.Tensor):
            action_feats_np = action_feats.cpu().numpy()
        else:
            action_feats_np = np.array(action_feats, dtype=np.float32)

        prefix_key = self.current_prefix_keys[phase_name]
        state_key = hash((self.version, prefix_key, action_feats_np.tobytes()))

        if state_key not in self.mcts_tree:
            scores, v, phase_id, global_feat, hidden = self._evaluate_node(
                phase_name, action_feats_np, prefix_key
            )

            P = torch.softmax(torch.tensor(scores, dtype=torch.float32), dim=0).numpy()
            v = self.phase_values.get(phase_name, 0.0)

            current_depth = len(self.active_stack)
            effective_noise = self.episode_noise * (self.depth_gamma**current_depth)

            if effective_noise > 0.001:
                num_a = len(scores)
                dirichlet_alpha = max(0.01, 1.0 / math.sqrt(max(1, num_a)))
                noise = (
                    torch.distributions.Dirichlet(torch.full((num_a,), dirichlet_alpha))
                    .sample()
                    .numpy()
                )
                P_perturbed = (1.0 - effective_noise) * P + effective_noise * noise
            else:
                P_perturbed = P.copy()

            self.mcts_tree[state_key] = {
                "N": np.zeros(num_actions, dtype=np.float32),
                "W": np.zeros(num_actions, dtype=np.float32),
                "P": P_perturbed,
                "v": v,
                "prefix_key": prefix_key,
                "phase_id": phase_id,
                "action_features": action_feats_np,
                "global_feat": global_feat,
                "hidden": hidden,
            }

        node_data = self.mcts_tree[state_key]
        N_sa = node_data["N"]
        W_sa = node_data["W"]
        P_sa = node_data["P"]
        v_s = node_data["v"]

        N_s = N_sa.sum()
        Q_sa = np.where(N_sa > 0, W_sa / np.maximum(N_sa, 1.0), v_s)

        visited_mask = N_sa > 0
        if np.any(visited_mask):
            min_q = min(float(v_s), float(np.min(Q_sa[visited_mask])))
            max_q = max(float(v_s), float(np.max(Q_sa[visited_mask])))
            if max_q > min_q:
                Q_norm = (Q_sa - min_q) / (max_q - min_q)
            else:
                Q_norm = np.full_like(Q_sa, 0.5)
        else:
            Q_norm = np.zeros_like(Q_sa)

        U_sa = self.c_puct * P_sa * (math.sqrt(max(1.0, float(N_s))) / (1.0 + N_sa))

        puct_scores = Q_norm + U_sa
        order = np.argsort(-puct_scores).tolist()
        chosen_idx = order[0]

        if self.model is not None:
            phase_id = self.PHASE_MAP[phase_name]
            g_feat = self._get_global_feature(phase_id)
            g_t = torch.tensor(
                g_feat, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            a_t = torch.tensor(
                action_feats_np, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            chosen_a_t = a_t[:, chosen_idx : chosen_idx + 1, :].squeeze(1)
            with torch.inference_mode():
                _, self.current_hidden = self.model.step(
                    self.current_hidden, g_t, chosen_a_t, phase_id
                )

        self.active_stack.append((state_key, chosen_idx))
        return order

    def order_cache(self, choices):
        return self._order_items(choices, "cache", self._extract_cache_features)

    def order_enodes(self, enodes):
        return self._order_items(enodes, "extract", self._extract_dispatch_features)

    def order_dispatch(self, ready_nodes):
        return self._order_items(
            ready_nodes, "dispatch", self._extract_dispatch_features
        )

    def order_bufferize(self, choices):
        return self._order_items(choices, "bufferize", self._extract_bufferize_features)

    def order_malloc(self, avail_buffers):
        return self._order_items(avail_buffers, "malloc", self._extract_malloc_features)

    def _extract_cache_features(self, items):
        feats = []
        for f in items:
            mem_type = float(f.mem_space.type) if hasattr(f, "mem_space") else 0.0
            feats.append(
                [
                    float(f.is_cached),
                    math.log1p(max(0.0, float(f.size))),
                    mem_type,
                    float(f.op_type),
                    float(f.num_users),
                ]
            )
        return torch.nan_to_num(
            torch.tensor(feats, dtype=torch.float32), posinf=1e9, neginf=-1e9
        )

    def _extract_dispatch_features(self, items):
        feats = []
        for f in items:
            num_nodes = len(f.graph.nodes) if hasattr(f, "graph") and f.graph else 0
            num_edges = (
                sum(len(n.child_ids) for n in f.graph.nodes.values())
                if num_nodes
                else 0
            )
            mem_type = float(f.mem_space.type) if hasattr(f, "mem_space") else 0.0
            eng_len = float(len(f.engine_idxs)) if hasattr(f, "engine_idxs") else 0.0
            feats.append(
                [
                    math.log1p(max(0.0, float(f.cost))),
                    math.log1p(max(0.0, float(getattr(f, "dp_cost", 0.0)))),
                    math.log1p(max(0.0, float(f.size))),
                    mem_type,
                    eng_len,
                    float(num_nodes),
                    float(num_edges),
                ]
            )
        return torch.nan_to_num(
            torch.tensor(feats, dtype=torch.float32), posinf=1e9, neginf=-1e9
        )

    def _extract_bufferize_features(self, items):
        feats = [
            [
                float(f.is_new_buffer),
                math.log1p(max(0.0, float(f.size))),
                math.log1p(max(0.0, float(f.parent_size))),
                float(f.parent_birth_time),
            ]
            for f in items
        ]
        return torch.nan_to_num(
            torch.tensor(feats, dtype=torch.float32), posinf=1e9, neginf=-1e9
        )

    def _extract_malloc_features(self, items):
        feats = [
            [
                math.log1p(max(0.0, float(f.size))),
                float(f.start),
                float(f.end),
                math.log1p(max(0.0, float(getattr(f, "mem_cap", 0.0)))),
            ]
            for f in items
        ]
        return torch.nan_to_num(
            torch.tensor(feats, dtype=torch.float32), posinf=1e9, neginf=-1e9
        )


class GumbelActorDelegate(ActorDelegate):
    """
    Sequential Halving Gumbel AlphaZero Delegate (Danihelka et al., 2022).
    Enforces exact simulation budget distribution across phases at root decision points.
    """
    def __init__(
        self,
        *args,
        num_simulations: int = 8,
        max_actions: int = 4,
        c_scale: float = 1.0,
        c_visit: float = 50.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.num_simulations = num_simulations
        self.max_actions = max_actions
        self.c_scale = c_scale
        self.c_visit = c_visit
        
        # Sequential Halving budget schedule calculation
        # For N=8, m=4 -> K=2 rounds:
        # Phase 0: 4 candidates, 1 visit each (4 total)
        # Phase 1: 2 candidates, 2 visits each (4 total)
        self.K = max(1, int(math.ceil(math.log2(self.max_actions))))
        self.root_halving_state = {}  # state_key -> dict tracking SH rounds and candidate sets

    def reset_for_episode(self, mem_caps: dict[int, int] | None = None):
        super().reset_for_episode(mem_caps=mem_caps)
        self.root_halving_state.clear()

    @torch.inference_mode()
    def _order_items(self, items, phase_name: str, extract_fn):
        num_actions = len(items)
        if num_actions <= 1:
            return list(range(num_actions))

        action_feats = extract_fn(items)
        action_feats_np = (
            action_feats.cpu().numpy()
            if isinstance(action_feats, torch.Tensor)
            else np.array(action_feats, dtype=np.float32)
        )

        prefix_key = self.current_prefix_keys[phase_name]
        state_key = hash((self.version, prefix_key, action_feats_np.tobytes()))
        is_root = (len(self.active_stack) == 0)

        # ---------------------------------------------------------------------
        # 1. First time visiting this state: Initialize logits & Gumbel noise
        # ---------------------------------------------------------------------
        if state_key not in self.mcts_tree:
            scores, v, phase_id, global_feat, hidden = self._evaluate_node(
                phase_name, action_feats_np, prefix_key
            )
            num_a = len(scores)

            if is_root:
                # Sample standard Gumbel(0, 1) noise at root
                u = np.random.uniform(1e-7, 1.0 - 1e-7, size=num_a).astype(np.float32)
                gumbel_noise = -np.log(-np.log(u))
                
                # Top-m candidate selection
                m = min(self.max_actions, num_a)
                gumbel_prior_scores = scores + gumbel_noise
                top_m_candidates = np.argsort(-gumbel_prior_scores)[:m].tolist()

                self.root_halving_state[state_key] = {
                    "m": m,
                    "K": self.K,
                    "current_phase": 0,
                    "active_candidates": top_m_candidates,
                    "phase_sims_completed": 0,
                    "target_visits_per_cand": max(1, self.num_simulations // (self.K * m)),
                    "candidate_cursor": 0,
                }
            else:
                gumbel_noise = np.zeros(num_a, dtype=np.float32)

            P = torch.softmax(torch.tensor(scores, dtype=torch.float32), dim=0).numpy()

            self.mcts_tree[state_key] = {
                "N": np.zeros(num_actions, dtype=np.float32),
                "W": np.zeros(num_actions, dtype=np.float32),
                "P": P,
                "logits": scores.astype(np.float32),
                "gumbel_noise": gumbel_noise,
                "v": float(v),
                "prefix_key": prefix_key,
                "phase_id": phase_id,
                "action_features": action_feats_np,
                "is_gumbel": True,
                "is_root": is_root,
                "global_feat": global_feat,
                "hidden": hidden,
            }

        node_data = self.mcts_tree[state_key]
        N_sa = node_data["N"]
        W_sa = node_data["W"]
        logits = node_data["logits"]
        gumbel = node_data["gumbel_noise"]
        v_s = node_data["v"]

        # Completed Q-values
        Q_sa = np.where(N_sa > 0, W_sa / np.maximum(N_sa, 1.0), v_s)
        visited_mask = N_sa > 0
        if np.any(visited_mask):
            min_q = min(float(v_s), float(np.min(Q_sa[visited_mask])))
            max_q = max(float(v_s), float(np.max(Q_sa[visited_mask])))
            Q_norm = (Q_sa - min_q) / (max_q - min_q) if max_q > min_q else np.full_like(Q_sa, 0.5)
        else:
            Q_norm = np.zeros_like(Q_sa)

        max_visit = float(np.max(N_sa)) if len(N_sa) > 0 else 0.0
        sigma_q = (self.c_visit + max_visit) * self.c_scale * Q_norm

        # ---------------------------------------------------------------------
        # 2. Sequential Halving Action Selection (Root) vs Deterministic Greedy (Internal)
        # ---------------------------------------------------------------------
        if is_root and state_key in self.root_halving_state:
            sh = self.root_halving_state[state_key]
            active_cands = sh["active_candidates"]

            # Check if current phase budget is exhausted
            if sh["phase_sims_completed"] >= len(active_cands) * sh["target_visits_per_cand"]:
                if sh["current_phase"] < sh["K"] - 1:
                    # Score active candidates and eliminate bottom half
                    cand_scores = [logits[a] + gumbel[a] + sigma_q[a] for a in active_cands]
                    sorted_indices = np.argsort(-np.array(cand_scores))
                    next_size = max(1, len(active_cands) // 2)
                    sh["active_candidates"] = [active_cands[i] for i in sorted_indices[:next_size]]
                    
                    sh["current_phase"] += 1
                    sh["phase_sims_completed"] = 0
                    sh["target_visits_per_cand"] = max(1, self.num_simulations // (sh["K"] * len(sh["active_candidates"])))
                    sh["candidate_cursor"] = 0
                    active_cands = sh["active_candidates"]

            # Pick next candidate in active set round-robin
            chosen_idx = active_cands[sh["candidate_cursor"] % len(active_cands)]
            sh["candidate_cursor"] += 1
            sh["phase_sims_completed"] += 1

            # Order: chosen action at index 0, followed by others sorted by score
            all_scores = logits + gumbel + sigma_q
            remaining = [a for a in np.argsort(-all_scores) if a != chosen_idx]
            order = [chosen_idx] + remaining
        else:
            # Internal node: deterministic greedy on logits + sigma(q)
            internal_scores = logits + sigma_q
            order = np.argsort(-internal_scores).tolist()
            chosen_idx = order[0]

        # RNN hidden state update
        if self.model is not None:
            phase_id = self.PHASE_MAP[phase_name]
            g_feat = self._get_global_feature(phase_id)
            g_t = torch.tensor(g_feat, dtype=torch.float32, device=self.device).unsqueeze(0)
            a_t = torch.tensor(action_feats_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            chosen_a_t = a_t[:, chosen_idx : chosen_idx + 1, :].squeeze(1)
            with torch.inference_mode():
                _, self.current_hidden = self.model.step(self.current_hidden, g_t, chosen_a_t, phase_id)

        self.active_stack.append((state_key, chosen_idx))
        return order


HeuristicDelegate = tensor_graphs.HeuristicSearchDelegate


@dataclasses.dataclass
class RNNTransition:
    hidden: np.ndarray  # [hidden_dim]
    global_feat: np.ndarray  # [global_dim]
    action_feats: np.ndarray  # [A, feat_dim]
    phase_id: int
    chosen_idx: int
    log_prob: float
    value: float


@dataclasses.dataclass
class RNNEpisode:
    transitions: list[RNNTransition]
    cost: float
    reward: float


class RNNREINFORCEDelegate(tensor_graphs.SearchDelegate):
    """
    Search Delegate for REINFORCE.
    Maintains the RNN hidden state on a push/pop stack as the C++ planner
    explores and backtracks through the search space tree.
    """

    PHASE_MAP = {
        "cache": 0,
        "extract": 1,
        "dispatch": 2,
        "bufferize": 3,
        "malloc": 4,
        "frontier": 5,
    }

    def __init__(
        self,
        model: PolicyValueRNN,
        mem_caps: dict[int, int] | None = None,
        temperature: float = 1.0,
        is_training: bool = True,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.model = model
        self.device = device if device is not None else next(model.parameters()).device
        self.temperature = max(1e-4, float(temperature))
        self.is_training = is_training
        self.mem_caps = mem_caps or {}

        # Precompute normalized memory caps vector: [cpp, cuda, opencl, ...]
        self.log_mem_cpp = (
            math.log1p(max(0.0, float(self.mem_caps.get(1, 16 * 1024 * 1024 * 1024))))
            / 25.0
        )
        self.log_mem_cuda = (
            math.log1p(max(0.0, float(self.mem_caps.get(2, 24 * 1024 * 1024 * 1024))))
            / 25.0
        )
        self.log_mem_opencl = (
            math.log1p(max(0.0, float(self.mem_caps.get(3, 1024 * 1024 * 1024)))) / 25.0
        )

        # State management
        self.root_hidden = self.model.init_hidden(batch_size=1, device=self.device)
        self.current_hidden = self.root_hidden.clone()
        self.hidden_stack: list[torch.Tensor] = []
        self.active_path: list[RNNTransition] = []
        self.completed_episodes: list[RNNEpisode] = []

    def reset_for_episode(self, mem_caps: dict[int, int] | None = None):
        if mem_caps is not None:
            self.mem_caps = mem_caps
            self.log_mem_cpp = (
                math.log1p(
                    max(0.0, float(self.mem_caps.get(1, 16 * 1024 * 1024 * 1024)))
                )
                / 25.0
            )
            self.log_mem_cuda = (
                math.log1p(
                    max(0.0, float(self.mem_caps.get(2, 24 * 1024 * 1024 * 1024)))
                )
                / 25.0
            )
            self.log_mem_opencl = (
                math.log1p(max(0.0, float(self.mem_caps.get(3, 1024 * 1024 * 1024))))
                / 25.0
            )

        self.current_hidden = self.model.init_hidden(batch_size=1, device=self.device)
        self.hidden_stack.clear()
        self.active_path.clear()
        self.completed_episodes.clear()

    def fast_fail(self) -> bool:
        return False

    def push_state(self):
        self.hidden_stack.append(self.current_hidden.clone())

    def pop_state(self):
        if self.hidden_stack:
            self.current_hidden = self.hidden_stack.pop()
        if self.active_path:
            self.active_path.pop()

    def init_cache_graph(self, node_features, edge_src, edge_dst):
        pass

    def init_egraph(self, node_features, edge_src, edge_dst):
        pass

    def init_dispatch_graph(self, node_features, edge_src, edge_dst):
        pass

    def init_bufferize_graph(self, node_features, edge_src, edge_dst):
        pass

    def init_malloc_graph(self, node_features, edge_src, edge_dst):
        pass

    def on_leaf_evaluated(self, cost: float):
        cost_val = float(cost)
        if 0.0 <= cost_val < float("inf") and not math.isnan(cost_val):
            reward = 1000.0 / (cost_val + 1.0)
        elif cost_val < 0.0:
            reward = cost_val
        else:
            reward = -1.0

        if self.active_path:
            self.completed_episodes.append(
                RNNEpisode(
                    transitions=list(self.active_path),
                    cost=cost_val if cost_val >= 0.0 else float("inf"),
                    reward=reward,
                )
            )

    def _get_global_feature(self, phase_id: int) -> np.ndarray:
        depth = float(len(self.active_path)) / 100.0
        return np.array(
            [
                self.log_mem_cpp,
                self.log_mem_cuda,
                self.log_mem_opencl,
                depth,
                float(phase_id),
                0.0,
                0.0,
                1.0,
            ],
            dtype=np.float32,
        )

    @torch.inference_mode()
    def _order_items(self, items, phase_name: str, extract_fn) -> list[int]:
        num_actions = len(items)
        if num_actions <= 1:
            return list(range(num_actions))

        phase_id = self.PHASE_MAP[phase_name]
        action_feats = extract_fn(items)
        if isinstance(action_feats, torch.Tensor):
            action_feats_np = action_feats.cpu().numpy()
        else:
            action_feats_np = np.array(action_feats, dtype=np.float32)

        global_feat_np = self._get_global_feature(phase_id)

        g_t = torch.tensor(
            global_feat_np, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        a_t = torch.tensor(
            action_feats_np, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        logits, value = self.model.evaluate_candidates(
            self.current_hidden, g_t, a_t, phase_id
        )
        scores = logits[0].cpu().numpy()
        val_item = value[0, 0].item()

        if self.is_training:
            probs = torch.softmax(logits[0] / self.temperature, dim=-1).cpu().numpy()
            probs = np.nan_to_num(probs, nan=1.0 / num_actions)
            probs = probs / probs.sum()

            chosen_idx = int(np.random.choice(num_actions, p=probs))
            log_prob = float(np.log(max(1e-8, probs[chosen_idx])))

            # Order: chosen action first, followed by others sorted by probability
            remaining = [i for i in range(num_actions) if i != chosen_idx]
            remaining.sort(key=lambda idx: -probs[idx])
            order = [chosen_idx] + remaining
        else:
            order = np.argsort(-scores).tolist()
            chosen_idx = order[0]
            probs = torch.softmax(logits[0], dim=-1).cpu().numpy()
            log_prob = float(np.log(max(1e-8, probs[chosen_idx])))

        # Record step in active path
        self.active_path.append(
            RNNTransition(
                hidden=self.current_hidden[0].detach().cpu().numpy(),
                global_feat=global_feat_np,
                action_feats=action_feats_np,
                phase_id=phase_id,
                chosen_idx=chosen_idx,
                log_prob=log_prob,
                value=val_item,
            )
        )

        # Transition the hidden state for the selected branch
        chosen_a_t = a_t[:, chosen_idx : chosen_idx + 1, :].squeeze(1)
        _, self.current_hidden = self.model.step(
            self.current_hidden, g_t, chosen_a_t, phase_id
        )

        return order

    def order_cache(self, choices):
        return self._order_items(choices, "cache", self._extract_cache_features)

    def order_enodes(self, enodes):
        return self._order_items(enodes, "extract", self._extract_dispatch_features)

    def order_dispatch(self, ready_nodes):
        return self._order_items(
            ready_nodes, "dispatch", self._extract_dispatch_features
        )

    def order_bufferize(self, choices):
        return self._order_items(choices, "bufferize", self._extract_bufferize_features)

    def order_malloc(self, avail_buffers):
        return self._order_items(avail_buffers, "malloc", self._extract_malloc_features)

    def order_frontier(self, frontier):
        return self._order_items(frontier, "frontier", self._extract_frontier_features)

    def _extract_cache_features(self, items):
        feats = []
        for f in items:
            mem_type = float(f.mem_space.type) if hasattr(f, "mem_space") else 0.0
            feats.append(
                [
                    float(f.is_cached),
                    math.log1p(max(0.0, float(f.size))),
                    mem_type,
                    float(f.op_type),
                    float(f.num_users),
                ]
            )
        return torch.nan_to_num(
            torch.tensor(feats, dtype=torch.float32), posinf=1e9, neginf=-1e9
        )

    def _extract_dispatch_features(self, items):
        feats = []
        for f in items:
            num_nodes = len(f.graph.nodes) if hasattr(f, "graph") and f.graph else 0
            num_edges = (
                sum(len(n.child_ids) for n in f.graph.nodes.values())
                if num_nodes
                else 0
            )
            mem_type = float(f.mem_space.type) if hasattr(f, "mem_space") else 0.0
            eng_len = float(len(f.engine_idxs)) if hasattr(f, "engine_idxs") else 0.0
            feats.append(
                [
                    math.log1p(max(0.0, float(f.cost))),
                    math.log1p(max(0.0, float(getattr(f, "dp_cost", 0.0)))),
                    math.log1p(max(0.0, float(f.size))),
                    mem_type,
                    eng_len,
                    float(num_nodes),
                    float(num_edges),
                ]
            )
        return torch.nan_to_num(
            torch.tensor(feats, dtype=torch.float32), posinf=1e9, neginf=-1e9
        )

    def _extract_bufferize_features(self, items):
        feats = [
            [
                float(f.is_new_buffer),
                math.log1p(max(0.0, float(f.size))),
                math.log1p(max(0.0, float(f.parent_size))),
                float(f.parent_birth_time),
            ]
            for f in items
        ]
        return torch.nan_to_num(
            torch.tensor(feats, dtype=torch.float32), posinf=1e9, neginf=-1e9
        )

    def _extract_malloc_features(self, items):
        feats = [
            [
                math.log1p(max(0.0, float(f.size))),
                float(f.start),
                float(f.end),
                math.log1p(max(0.0, float(getattr(f, "mem_cap", 0.0)))),
            ]
            for f in items
        ]
        return torch.nan_to_num(
            torch.tensor(feats, dtype=torch.float32), posinf=1e9, neginf=-1e9
        )

    def _extract_frontier_features(self, items):
        feats = [
            [
                float(f.eclass_id),
                float(f.num_enodes),
                math.log1p(max(0.0, float(f.min_dp_cp_cost))),
                math.log1p(max(0.0, float(f.min_dp_cost))),
                math.log1p(max(0.0, float(f.size))),
                float(f.dtype),
                float(f.mem_space.type) if hasattr(f, "mem_space") else 0.0,
            ]
            for f in items
        ]
        return torch.nan_to_num(
            torch.tensor(feats, dtype=torch.float32), posinf=1e9, neginf=-1e9
        )


def create_search_delegate(
    config: TrainConfig,
    agent=None,
    model: PolicyValueRNN | None = None,
    req_queue=None,
    resp_queue=None,
    worker_id: int = 0,
    mcts_tree: dict | None = None,
    episode: int = 0,
    version: int = 0,
    shared_action_feats: torch.Tensor | None = None,
    shared_logits: torch.Tensor | None = None,
    shared_v: torch.Tensor | None = None,
    temperature: float = 1.0,
    is_training: bool = True,
    device: torch.device | None = None,
):
    algo = getattr(config, "algo", "gumbel_alphazero").lower().replace("-", "_")
    if algo in ["reinforce", "ppo", "rnn"]:
        return RNNREINFORCEDelegate(
            model=model,
            temperature=temperature,
            is_training=is_training,
            device=device,
        )
    elif algo in ["alphazero", "az", "puct"]:
        return ActorDelegate(
            agent=agent,
            model=model,
            req_queue=req_queue,
            resp_queue=resp_queue,
            worker_id=worker_id,
            mcts_tree=mcts_tree,
            c_puct=config.c_puct,
            episode=episode,
            decay_episodes=config.decay_episodes,
            base_noise=config.base_noise,
            min_noise=config.min_noise,
            depth_gamma=config.depth_gamma,
            version=version,
            shared_action_feats=shared_action_feats,
            shared_logits=shared_logits,
            shared_v=shared_v,
            device=device,
        )
    else:
        return GumbelActorDelegate(
            agent=agent,
            model=model,
            req_queue=req_queue,
            resp_queue=resp_queue,
            worker_id=worker_id,
            mcts_tree=mcts_tree,
            num_simulations=config.num_simulations,
            max_actions=min(getattr(config, "gumbel_max_actions", 4), config.num_simulations),
            c_scale=getattr(config, "c_scale", 1.0),
            c_visit=getattr(config, "c_visit", 50.0),
            c_puct=config.c_puct,
            episode=episode,
            decay_episodes=config.decay_episodes,
            base_noise=config.base_noise,
            min_noise=config.min_noise,
            depth_gamma=config.depth_gamma,
            version=version,
            shared_action_feats=shared_action_feats,
            shared_logits=shared_logits,
            shared_v=shared_v,
            device=device,
        )
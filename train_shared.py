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

torch.set_float32_matmul_precision("high")


@dataclasses.dataclass
class TrainConfig:
    run_dir: str = "runs"
    model_name: str = "gemma-3-270m"
    model_path: str = "models/google/gemma-3-270m"
    num_simulations: int = 10
    level_simulations: list = dataclasses.field(default_factory=lambda: [1, 1, 1, 1])
    replay_buffer_size: int = 100_000
    batch_size: int = 64
    save_interval: int = 100

    # Graph Source & Generation Config
    graph_source: str = "model"  # "model" or "random"
    random_min_nodes: int = 10
    random_max_nodes: int = 30
    random_hidden_dim: int = 128
    random_seq_len: int = 64
    random_seed: int | None = None
    resample_graph_every: int = 0

    # Transformer Architecture Config
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

    # PUCT & Noise Annealing Config
    c_puct: float = 1.25
    base_noise: float = 0.25
    min_noise: float = 0.01
    decay_episodes: int = 500
    depth_gamma: float = 0.99

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
    "krea": "models/krea/Krea-2-Turbo/krea.safetensors",
    "krea-2-turbo": "models/krea/Krea-2-Turbo/krea.safetensors",
    "krea-2-turbo-vae": "models/krea/Krea-2-Turbo/qwen_image_vae.safetensors",
    "vae": "models/krea/Krea-2-Turbo/qwen_image_vae.safetensors",
    "qwen-image-vae": "models/krea/Krea-2-Turbo/qwen_image_vae.safetensors",
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
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    g = tensor_graphs.Graph()
    shape_standard = [1, seq_len, hidden_dim]
    shape_proj = [1, hidden_dim, hidden_dim]

    in0 = g.input(shape_standard, tensor_graphs.DType.FLOAT32)
    in1 = g.input(shape_standard, tensor_graphs.DType.FLOAT32)
    in_w = g.input(shape_proj, tensor_graphs.DType.FLOAT32)

    available_nodes = [in0, in1]
    weights = [in_w]

    for _ in range(num_nodes):
        op_choice = random.choice(
            ["elementwise", "unary", "dot", "reduce", "reshape_cycle"]
        )

        if op_choice == "unary":
            src = random.choice(available_nodes)
            u_op = random.choice(["sin", "cos", "neg", "relu"])
            if u_op == "sin":
                node = g.sin(src)
            elif u_op == "cos":
                node = g.cos(src)
            elif u_op == "neg":
                node = g.neg(src)
            else:
                node = g.relu(src, shape_standard)
            available_nodes.append(node)

        elif op_choice == "elementwise":
            src1 = random.choice(available_nodes)
            src2 = random.choice(available_nodes)
            b_op = random.choice(["add", "mul"])
            if b_op == "add":
                node = g.add(src1, src2)
            else:
                node = g.mul(src1, src2)
            available_nodes.append(node)

        elif op_choice == "dot":
            src = random.choice(available_nodes)
            w = random.choice(weights)
            node = g.dot(src, w)
            available_nodes.append(node)

        elif op_choice == "reduce":
            src = random.choice(available_nodes)
            axis_const = g.constant([-1])
            node = g.sum(src, axis_const)
            node = g.repeat(node, hidden_dim, 2)
            available_nodes.append(node)

        elif op_choice == "reshape_cycle":
            src = random.choice(available_nodes)
            sh_flat = [1, seq_len * hidden_dim]
            sh_rec = [1, seq_len, hidden_dim]
            node = g.reshape(src, sh_flat)
            node = g.reshape(node, sh_rec)
            available_nodes.append(node)

    root = available_nodes[-1]

    full_bucket = tensor_graphs.Bucket()
    dim_b = tensor_graphs.Dim(0, 1)
    dim_s = tensor_graphs.Dim(0, seq_len)
    dim_h = tensor_graphs.Dim(0, hidden_dim)

    r_in = tensor_graphs.Region()
    r_in.region = [dim_b, dim_s, dim_h]

    r_w = tensor_graphs.Region()
    r_w.region = [dim_b, dim_h, dim_h]

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
    if in0 in reachable_nodes:
        dirty_map[in0] = [r_in]
    if in1 in reachable_nodes:
        dirty_map[in1] = [r_in]
    if in_w in reachable_nodes:
        dirty_map[in_w] = [r_w]

    full_bucket.inputDirtyRegions = dirty_map
    full_bucket.outputNeededRegion = [r_in]

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
            if (
                not model_path
                or (
                    model_path == "models/google/gemma-3-270m"
                    and self.model_name != "gemma-3-270m"
                )
            ):
                model_path = get_default_model_path(self.model_name)

            self._cached_context = tensor_graphs.build_and_saturate_egraph(
                self.model_name,
                model_path,
                config.log_cost_calls,
                config.compile_decode_buckets,
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

            self._cached_context = tensor_graphs.build_and_saturate_egraph_from_graph(
                graph, root, buckets, config.log_cost_calls
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
    ) -> dict:
        referenced_prefixes = {}
        transitions = []

        for _, node_data in mcts_tree.items():
            pkey = node_data["prefix_key"]
            if pkey in prefix_registry and pkey not in referenced_prefixes:
                referenced_prefixes[pkey] = prefix_registry[pkey]

            counts = node_data["N"]
            total_counts = float(counts.sum())
            prior_p = node_data["P"]

            if total_counts > 0:
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

            transitions.append(
                {
                    "prefix_key": pkey,
                    "action_features": node_data["action_features"],
                    "phase_id": node_data.get("phase_id", 0),
                    "pis": pi.astype(np.float32),
                    "z": best_Z,
                }
            )

        return {
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


# ==============================================================================
# ACTOR DELEGATE
# ==============================================================================
class ActorDelegate(tensor_graphs.SearchDelegate):
    PHASE_MAP = {"cache": 0, "extract": 1, "dispatch": 2, "bufferize": 3, "malloc": 4}

    def __init__(
        self,
        agent=None,
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
    ):
        super().__init__()
        self.agent = agent
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

    def push_state(self):
        pass

    def pop_state(self):
        if self.active_stack:
            self.active_stack.pop()

    def on_leaf_evaluated(self, cost: float):
        cost_val = float(cost)
        z = 1000.0 / (cost_val + 1.0) if cost_val < float("inf") else -1.0

        for state_key, act in self.active_stack:
            if state_key in self.mcts_tree:
                self.mcts_tree[state_key]["N"][act] += 1.0
                self.mcts_tree[state_key]["W"][act] += z

    def _store_raw_graph(self, phase_name, node_features, edge_src, edge_dst):
        phase_id = self.PHASE_MAP[phase_name]
        dim = (
            5
            if phase_name in ["cache", "bufferize"]
            else (4 if phase_name in ["extract", "dispatch"] else 3)
        )
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
            phase_id = self.PHASE_MAP[phase_name]
            cache_key = (self.version, prefix_key)

            if self.agent is not None:
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

            P = torch.softmax(torch.tensor(scores, dtype=torch.float32), dim=0).numpy()
            v = self.phase_values.get(phase_name, 0.0)

            current_depth = len(self.active_stack)
            effective_noise = self.episode_noise * (self.depth_gamma**current_depth)

            if effective_noise > 0.001:
                # Dynamically scale alpha inversely with sqrt(|A|)
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
            }

        node_data = self.mcts_tree[state_key]
        N_sa = node_data["N"]
        W_sa = node_data["W"]
        P_sa = node_data["P"]
        v_s = node_data["v"]

        N_s = N_sa.sum()
        Q_sa = np.where(N_sa > 0, W_sa / np.maximum(N_sa, 1.0), v_s)

        # Min-Max Q-value normalization so Q matches the U exploration scale
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

        self.active_stack.append((state_key, order[0]))
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
            [math.log1p(max(0.0, float(f.size))), float(f.start), float(f.end)]
            for f in items
        ]
        return torch.nan_to_num(
            torch.tensor(feats, dtype=torch.float32), posinf=1e9, neginf=-1e9
        )

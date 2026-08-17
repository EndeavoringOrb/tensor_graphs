import dataclasses
import math
import pickle
import random
import socket
import struct
import zlib
from typing import Protocol

import numpy as np
import tensor_graphs
import torch
from torch import nn

torch.set_float32_matmul_precision("high")


@dataclasses.dataclass
class TrainConfig:
    run_dir: str = "runs/server_train"
    model_name: str = "gemma-3-270m"
    model_path: str = "models/google/gemma-3-270m"
    num_simulations: int = 10
    level_simulations: list = dataclasses.field(default_factory=lambda: [2, 7, 10, 3])
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
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 3
    max_feat_dim: int = 8

    lr: float = 1e-3
    log_cost_calls: bool = False
    bucket_idx: int = -1
    compile_decode_buckets: bool = False
    workers: int = 4
    cpp_threads: int = (
        1  # Prevents C++ thread oversubscription in multi-process workers
    )

    # PUCT & Noise Annealing Config
    c_puct: float = 1.25
    base_noise: float = 0.25
    min_noise: float = 0.01
    decay_episodes: int = 500
    depth_gamma: float = 0.7

    # Networking Config
    host: str = "127.0.0.1"
    port: int = 5000
    use_bluetooth: bool = False
    bt_host_address: str = "AC:F2:3C:A7:F7:EC"
    bt_port: int = 4


# ==============================================================================
# GRAPH PROVIDER INTERFACE & RANDOM GENERATOR
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
            self._cached_context = tensor_graphs.build_and_saturate_egraph(
                self.model_name,
                self.model_path,
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
# PREFIX DEDUPLICATION & TRAJECTORY CODEC
# ==============================================================================
@dataclasses.dataclass
class PrefixData:
    features: np.ndarray  # (1 + N + E, 8) float32
    token_types: np.ndarray  # (1 + N + E,) int64
    phase_ids: np.ndarray  # (1 + N + E,) int64
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
        E = len(edge_src)
        L = 1 + N + E

        features = np.zeros((L, 8), dtype=np.float32)
        token_types = np.zeros(L, dtype=np.int64)
        phase_ids = np.full(L, phase_id, dtype=np.int64)

        features[0, 0] = phase_id
        token_types[0] = 0

        if N > 0:
            features[1 : N + 1, 0] = np.arange(N)
            dim_feat = min(7, node_features.shape[1])
            features[1 : N + 1, 1 : 1 + dim_feat] = node_features[:, :dim_feat]
        token_types[1 : N + 1] = 1

        if E > 0:
            features[N + 1 : N + E + 1, 0] = edge_src
            features[N + 1 : N + E + 1, 1] = edge_dst
        token_types[N + 1 : N + E + 1] = 2

        prefix_key = hash(
            features.tobytes() + token_types.tobytes() + phase_ids.tobytes()
        )
        return prefix_key, PrefixData(
            features=features,
            token_types=token_types,
            phase_ids=phase_ids,
            phase_id=phase_id,
        )

    @staticmethod
    def pack_episode(
        mcts_tree: dict,
        best_Z: float,
        prefix_registry: dict[int, PrefixData],
    ) -> dict:
        referenced_prefixes = {}
        transitions = []

        for _, node_data in mcts_tree.items():
            pkey = node_data["prefix_key"]
            if pkey in prefix_registry and pkey not in referenced_prefixes:
                referenced_prefixes[pkey] = prefix_registry[pkey]

            counts = node_data["N"]
            total_counts = counts.sum()
            pi = counts / total_counts if total_counts > 0 else node_data["P"]

            transitions.append(
                {
                    "prefix_key": pkey,
                    "action_features": node_data["action_features"],
                    "phase_id": node_data.get("phase_id", 0),
                    "pis": pi,
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
# TRANSFORMER MODEL
# ==============================================================================
class CustomSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv = (k, v)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(out), new_kv


class CustomTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CustomSelfAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)
        )

    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_out, new_kv = self.attn(self.norm1(x), past_kv=past_kv)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, new_kv


class AlphaZeroTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=3, max_feat_dim=8):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_feat_dim = max_feat_dim

        self.feat_proj = nn.Linear(max_feat_dim, d_model)
        self.type_emb = nn.Embedding(4, d_model)
        self.phase_emb = nn.Embedding(5, d_model)

        self.layers = nn.ModuleList(
            [CustomTransformerBlock(d_model, nhead) for _ in range(num_layers)]
        )

        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )

        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )

    def encode_prefix(self, features, token_types, phase_ids):
        x = (
            self.feat_proj(features)
            + self.type_emb(token_types)
            + self.phase_emb(phase_ids)
        )

        prefix_kvs = []
        for layer in self.layers:
            x, kv = layer(x, past_kv=None)
            prefix_kvs.append(kv)

        v = self.value_head(x[:, 0]).squeeze(-1)
        return v, prefix_kvs

    def evaluate_actions(self, action_features, phase_ids, past_kv):
        B, A, _ = action_features.shape
        token_types = torch.full(
            (B, A), 3, dtype=torch.int64, device=action_features.device
        )
        x = (
            self.feat_proj(action_features)
            + self.type_emb(token_types)
            + self.phase_emb(phase_ids)
        )

        for i, layer in enumerate(self.layers):
            x, _ = layer(x, past_kv=past_kv[i])

        logits = self.policy_head(x).squeeze(-1)
        return logits


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
        self.prefix_cache_kv = {}
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
            if cache_key not in self.prefix_cache_kv:
                device = next(self.agent.parameters()).device
                f_t = torch.tensor(
                    prefix_data.features, dtype=torch.float32, device=device
                ).unsqueeze(0)
                tt_t = torch.tensor(
                    prefix_data.token_types, dtype=torch.int64, device=device
                ).unsqueeze(0)
                p_t = torch.tensor(
                    prefix_data.phase_ids, dtype=torch.int64, device=device
                ).unsqueeze(0)

                with (
                    torch.inference_mode(),
                    torch.autocast(device_type=device.type, dtype=torch.bfloat16),
                ):
                    v, kv = self.agent.encode_prefix(f_t, tt_t, p_t)

                self.prefix_cache_kv[cache_key] = kv
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
                kv = self.prefix_cache_kv[cache_key]
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
                        padded_actions, padded_pid, past_kv=kv
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
                noise = (
                    torch.distributions.Dirichlet(torch.full((len(scores),), 0.3))
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
        U_sa = self.c_puct * P_sa * (math.sqrt(max(1.0, float(N_s))) / (1.0 + N_sa))

        puct_scores = Q_sa + U_sa
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

import dataclasses
import math
import pickle
import socket
import struct
import zlib

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclasses.dataclass
class TrainConfig:
    run_dir: str = "runs/server_train"
    model_name: str = "gemma-3-270m"
    model_path: str = "models/google/gemma-3-270m"
    num_simulations: int = 10
    level_simulations: list = dataclasses.field(default_factory=lambda: [2, 7, 10, 3])
    replay_buffer_size: int = 1_000_000
    batch_size: int = 1024
    save_interval: int = 100
    hidden_dim: int = 64
    lr: float = 1e-3
    log_cost_calls: bool = False
    bucket_idx: int = -1
    compile_decode_buckets: bool = False
    workers: int = 4
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
# NETWORK SOCKET UTILITIES
# ==============================================================================
def create_client_socket(config: TrainConfig):
    """Creates and connects a client socket for TCP or Bluetooth RFCOMM."""
    if config.use_bluetooth:
        if not hasattr(socket, "AF_BLUETOOTH"):
            raise RuntimeError(
                "Bluetooth (AF_BLUETOOTH) is not supported on this platform."
            )
        sock = socket.socket(
            socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM
        )
        sock.connect((config.bt_host_address, config.bt_port))
    else:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((config.host, config.port))
    return sock


def create_server_socket(host: str, port: int, use_bluetooth: bool = False):
    """Creates, binds, and listens on a server socket for TCP or Bluetooth RFCOMM."""
    is_bt = use_bluetooth or (
        isinstance(host, str) and ":" in host and len(host.split(":")) == 6
    )
    if is_bt:
        if not hasattr(socket, "AF_BLUETOOTH"):
            raise RuntimeError(
                "Bluetooth (AF_BLUETOOTH) is not supported on this platform."
            )
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
    """Compresses and frames the message to handle slow/fragmented streams."""
    data = zlib.compress(pickle.dumps(msg))
    sock.sendall(struct.pack(">I", len(data)) + data)


def recvall(sock, n):
    """Helper to receive exactly n bytes over TCP/RFCOMM."""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data


def recv_msg(sock):
    """Reads the frame length and decompresses the payload."""
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack(">I", raw_msglen)[0]
    data = recvall(sock, msglen)
    if data is None:
        return None
    return pickle.loads(zlib.decompress(data))


# ==============================================================================
# MODELS
# ==============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True) * (x.size(-1) ** -0.5)
        return (x / (norm + self.eps)) * self.weight


class GNNModel(nn.Module):
    def __init__(self, in_features, hidden_dim=64):
        super().__init__()
        self.in_norm = RMSNorm(in_features)
        self.node_emb = nn.Linear(in_features, hidden_dim)
        self.msg_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, node_features, edge_src, edge_dst):
        x = self.in_norm(node_features)
        x = F.gelu(self.node_emb(x))

        if len(edge_dst) > 0:
            src_x = x[edge_src]
            dst_x = x[edge_dst]
            msg = self.msg_net(torch.cat([src_x, dst_x], dim=-1))
            aggr_msg = torch.zeros_like(x)
            aggr_msg.index_add_(0, edge_dst, msg)
            x = x + self.update_net(torch.cat([x, aggr_msg], dim=-1))

        return x.mean(dim=0)


class DecisionModel(nn.Module):
    def __init__(self, global_dim, feature_dim, hidden_dim=64):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(global_dim + feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.value = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_state, options_features):
        N = options_features.size(0)
        global_expanded = global_state.unsqueeze(0).expand(N, -1)

        policy_in = torch.cat([global_expanded, options_features], dim=1)
        scores = self.policy(policy_in).squeeze(1)
        val = self.value(global_state.unsqueeze(0)).squeeze(0)
        return scores, val


class AlphaZeroAgent(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.cache_gnn = GNNModel(in_features=5, hidden_dim=hidden_dim)
        self.cache_dec = DecisionModel(
            global_dim=hidden_dim, feature_dim=5, hidden_dim=hidden_dim
        )

        self.extract_gnn = GNNModel(in_features=4, hidden_dim=hidden_dim)
        self.extract_dec = DecisionModel(
            global_dim=hidden_dim, feature_dim=6, hidden_dim=hidden_dim
        )

        self.dispatch_gnn = GNNModel(in_features=4, hidden_dim=hidden_dim)
        self.dispatch_dec = DecisionModel(
            global_dim=hidden_dim, feature_dim=6, hidden_dim=hidden_dim
        )

        self.bufferize_gnn = GNNModel(in_features=5, hidden_dim=hidden_dim)
        self.bufferize_dec = DecisionModel(
            global_dim=hidden_dim, feature_dim=4, hidden_dim=hidden_dim
        )

        self.malloc_gnn = GNNModel(in_features=3, hidden_dim=hidden_dim)
        self.malloc_dec = DecisionModel(
            global_dim=hidden_dim, feature_dim=3, hidden_dim=hidden_dim
        )


import tensor_graphs


class ActorDelegate(tensor_graphs.SearchDelegate):
    def __init__(
        self,
        agent,
        mcts_tree: dict | None = None,
        c_puct: float = 1.25,
        exploration_noise=None,
        episode: int = 0,
        decay_episodes: int = 500,
        base_noise: float = 0.25,
        min_noise: float = 0.01,
        depth_gamma: float = 0.7,
    ):
        super().__init__()
        self.agent = agent
        self.mcts_tree = mcts_tree if mcts_tree is not None else {}
        self.c_puct = c_puct
        self.episode = episode
        self.decay_episodes = decay_episodes
        self.base_noise = base_noise if exploration_noise is None else exploration_noise
        self.min_noise = min_noise
        self.depth_gamma = depth_gamma

        if exploration_noise is not None:
            if exploration_noise == 0.0:
                self.episode_noise = 0.0
            else:
                self.episode_noise = max(
                    min_noise,
                    exploration_noise * (1.0 - episode / max(1, decay_episodes)),
                )
        else:
            self.episode_noise = max(
                min_noise,
                base_noise * (1.0 - episode / max(1, decay_episodes)),
            )

        self.active_stack = []
        self.globals = {}

    def push_state(self):
        pass

    def pop_state(self):
        if self.active_stack:
            self.active_stack.pop()

    def on_leaf_evaluated(self, cost: float):
        cost_val = float(cost)
        if cost_val < float("inf"):
            z = 1000.0 / (cost_val + 1.0)
        else:
            z = -1.0

        for state_key, act in self.active_stack:
            if state_key in self.mcts_tree:
                self.mcts_tree[state_key]["N"][act] += 1.0
                self.mcts_tree[state_key]["W"][act] += z

    def _prepare_graphs(self, node_features, edge_src, edge_dst, feat_dim):
        if not node_features:
            return None, None, None
        nf = torch.tensor(node_features, dtype=torch.float32).view(-1, feat_dim)
        nf = torch.nan_to_num(nf, posinf=1e9, neginf=-1e9)
        src = torch.tensor(edge_src, dtype=torch.int64)
        dst = torch.tensor(edge_dst, dtype=torch.int64)
        return nf, src, dst

    def init_cache_graph(self, node_features, edge_src, edge_dst):
        nf, src, dst = self._prepare_graphs(node_features, edge_src, edge_dst, 5)
        if nf is not None:
            self.globals["cache_dec"] = self.agent.cache_gnn(nf, src, dst)

    def init_egraph(self, node_features, edge_src, edge_dst):
        nf, src, dst = self._prepare_graphs(node_features, edge_src, edge_dst, 4)
        if nf is not None:
            self.globals["extract_dec"] = self.agent.extract_gnn(nf, src, dst)

    def init_dispatch_graph(self, node_features, edge_src, edge_dst):
        nf, src, dst = self._prepare_graphs(node_features, edge_src, edge_dst, 4)
        if nf is not None:
            self.globals["dispatch_dec"] = self.agent.dispatch_gnn(nf, src, dst)

    def init_bufferize_graph(self, node_features, edge_src, edge_dst):
        nf, src, dst = self._prepare_graphs(node_features, edge_src, edge_dst, 5)
        if nf is not None:
            self.globals["bufferize_dec"] = self.agent.bufferize_gnn(nf, src, dst)

    def init_malloc_graph(self, node_features, edge_src, edge_dst):
        nf, src, dst = self._prepare_graphs(node_features, edge_src, edge_dst, 3)
        if nf is not None:
            self.globals["malloc_dec"] = self.agent.malloc_gnn(nf, src, dst)

    @torch.inference_mode()
    def _order_items(self, items, dec_type, extract_fn):
        if len(items) <= 1:
            if len(items) == 1:
                features = extract_fn(items)
                global_state = self.globals.get(
                    dec_type, torch.zeros(self.agent.hidden_dim)
                )
                state_key = hash(
                    (
                        global_state.cpu().numpy().tobytes(),
                        features.cpu().numpy().tobytes(),
                    )
                )
                if state_key not in self.mcts_tree:
                    self.mcts_tree[state_key] = {
                        "N": np.zeros(1, dtype=np.float32),
                        "W": np.zeros(1, dtype=np.float32),
                        "P": np.ones(1, dtype=np.float32),
                        "v": 0.0,
                        "type": dec_type,
                        "global_state": global_state.cpu().numpy(),
                        "features": features.cpu().numpy(),
                    }
                self.active_stack.append((state_key, 0))
            return list(range(len(items)))

        features = extract_fn(items)
        global_state = self.globals.get(dec_type, torch.zeros(self.agent.hidden_dim))

        state_key = hash(
            (global_state.cpu().numpy().tobytes(), features.cpu().numpy().tobytes())
        )
        num_actions = len(items)

        if state_key not in self.mcts_tree:
            with torch.no_grad():
                dec_model = getattr(self.agent, dec_type)
                scores, val = dec_model(global_state, features)

            P = torch.softmax(scores, dim=0).cpu().numpy()
            v = val.item() if hasattr(val, "item") else float(val)

            current_depth = len(self.active_stack)
            effective_noise = self.episode_noise * (self.depth_gamma**current_depth)

            if effective_noise > 0.001:
                noise = (
                    torch.distributions.Dirichlet(torch.full_like(scores, 0.3))
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
                "type": dec_type,
                "global_state": global_state.cpu().numpy(),
                "features": features.cpu().numpy(),
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
        return self._order_items(choices, "cache_dec", self._extract_cache_features)

    def order_enodes(self, enodes):
        return self._order_items(enodes, "extract_dec", self._extract_dispatch_features)

    def order_dispatch(self, ready_nodes):
        return self._order_items(
            ready_nodes, "dispatch_dec", self._extract_dispatch_features
        )

    def order_bufferize(self, choices):
        return self._order_items(
            choices, "bufferize_dec", self._extract_bufferize_features
        )

    def order_malloc(self, avail_buffers):
        return self._order_items(
            avail_buffers, "malloc_dec", self._extract_malloc_features
        )

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

            log_cost = math.log1p(max(0.0, float(f.cost)))
            log_size = math.log1p(max(0.0, float(f.size)))

            feats.append(
                [
                    log_cost,
                    log_size,
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

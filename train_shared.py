# File: train_shared.py
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
# UNIFIED TRANSFORMER MODEL
# ==============================================================================
class AlphaZeroTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=3, max_feat_dim=8):
        super().__init__()
        self.d_model = d_model
        self.max_feat_dim = max_feat_dim

        self.feat_proj = nn.Linear(max_feat_dim, d_model)
        self.type_emb = nn.Embedding(4, d_model)  # 0=Global, 1=Node, 2=Edge, 3=Action
        self.phase_emb = nn.Embedding(
            5, d_model
        )  # 0=cache, 1=extract, 2=dispatch, 3=bufferize, 4=malloc

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )

        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )

    def forward(self, features, token_types, phase_ids, key_padding_mask=None):
        # features shape: (B, L, max_feat_dim)
        x = (
            self.feat_proj(features)
            + self.type_emb(token_types)
            + self.phase_emb(phase_ids)
        )

        # TODO: Consider SDPA / FlashAttention context managers here if sequences get extremely long.
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)

        # Global token is always at index 0. It predicts the value (cost).
        v = self.value_head(x[:, 0]).squeeze(-1)

        # Compute logits for all tokens (we will filter out actions externally)
        logits = self.policy_head(x).squeeze(-1)  # (B, L)

        return logits, v


import tensor_graphs


class ActorDelegate(tensor_graphs.SearchDelegate):
    PHASE_MAP = {"cache": 0, "extract": 1, "dispatch": 2, "bufferize": 3, "malloc": 4}

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
        self.raw_graphs = {}

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

        self.raw_graphs[phase_name] = {
            "node_features": nf,
            "edge_src": src,
            "edge_dst": dst,
        }

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

    def _build_sequence(self, phase_name, action_features):
        phase_id = self.PHASE_MAP[phase_name]
        dim = (
            5
            if phase_name in ["cache", "bufferize"]
            else (4 if phase_name in ["extract", "dispatch"] else 3)
        )
        raw_graph = self.raw_graphs.get(
            phase_name,
            {
                "node_features": np.zeros((0, dim), dtype=np.float32),
                "edge_src": [],
                "edge_dst": [],
            },
        )
        node_feats = raw_graph["node_features"]
        edge_src = raw_graph["edge_src"]
        edge_dst = raw_graph["edge_dst"]

        N = len(node_feats)
        E = len(edge_src)
        A = len(action_features)
        L = 1 + N + E + A

        features = np.zeros((L, 8), dtype=np.float32)
        token_types = np.zeros(L, dtype=np.int64)
        phase_ids = np.full(L, phase_id, dtype=np.int64)

        # 0. Global Token
        features[0, 0] = phase_id
        token_types[0] = 0

        # 1. Node Tokens [ID + Features]
        if N > 0:
            features[1 : N + 1, 0] = np.arange(N)
            dim = min(7, node_feats.shape[1])
            features[1 : N + 1, 1 : 1 + dim] = node_feats[:, :dim]
        token_types[1 : N + 1] = 1

        # 2. Edge Tokens [Src_ID + Dst_ID + padding]
        if E > 0:
            features[N + 1 : N + E + 1, 0] = edge_src
            features[N + 1 : N + E + 1, 1] = edge_dst
        token_types[N + 1 : N + E + 1] = 2

        # 3. Action Tokens [Action_ID + Features]
        if A > 0:
            features[N + E + 1 :, 0] = np.arange(A)
            if isinstance(action_features, torch.Tensor):
                action_features = action_features.cpu().numpy()
            dim = min(7, action_features.shape[1])
            features[N + E + 1 :, 1 : 1 + dim] = action_features[:, :dim]
        token_types[N + E + 1 :] = 3

        return features, token_types, phase_ids

    @torch.inference_mode()
    def _order_items(self, items, phase_name, extract_fn):
        action_feats = extract_fn(items)
        features, token_types, phase_ids = self._build_sequence(
            phase_name, action_feats
        )

        state_key = hash(
            features.tobytes() + token_types.tobytes() + phase_ids.tobytes()
        )
        num_actions = len(items)

        if num_actions <= 1:
            if num_actions == 1 and state_key not in self.mcts_tree:
                self.mcts_tree[state_key] = {
                    "N": np.zeros(1, dtype=np.float32),
                    "W": np.zeros(1, dtype=np.float32),
                    "P": np.ones(1, dtype=np.float32),
                    "v": 0.0,
                    "features": features,
                    "token_types": token_types,
                    "phase_ids": phase_ids,
                }
                self.active_stack.append((state_key, 0))
            return list(range(num_actions))

        if state_key not in self.mcts_tree:
            f_t = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            tt_t = torch.tensor(token_types, dtype=torch.int64).unsqueeze(0)
            p_t = torch.tensor(phase_ids, dtype=torch.int64).unsqueeze(0)

            action_mask = tt_t == 3  # shape (1, L)

            logits, val = self.agent(f_t, tt_t, p_t)

            # Mask non-actions and slice directly with the 2D boolean mask
            logits = logits.masked_fill(~action_mask, -float("inf"))
            scores = logits[action_mask]  # Fixed 1D extraction: shape (num_actions,)

            P = torch.softmax(scores, dim=0).cpu().numpy()
            v = val.item()

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
                "features": features,
                "token_types": token_types,
                "phase_ids": phase_ids,
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

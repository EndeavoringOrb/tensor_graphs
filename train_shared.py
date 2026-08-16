import dataclasses
import math
import pickle
import socket
import struct
import zlib
from typing import List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
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
    hidden_dim: int = 128
    lr: float = 1e-3
    log_cost_calls: bool = False
    bucket_idx: int = -1
    compile_decode_buckets: bool = False
    workers: int = 4
    c_puct: float = 1.25
    base_noise: float = 0.25
    min_noise: float = 0.01
    decay_episodes: int = 500
    depth_gamma: float = 0.7
    host: str = "127.0.0.1"
    port: int = 5000
    use_bluetooth: bool = False
    bt_host_address: str = "AC:F2:3C:A7:F7:EC"
    bt_port: int = 4
    transformer_layers: int = 2
    transformer_heads: int = 4
    transformer_dropout: float = 0.1


# ----------------------------------------------------------------------
# Constants for unified MDP
# ----------------------------------------------------------------------
DEC_TYPES = ["cache_dec", "extract_dec", "dispatch_dec", "bufferize_dec", "malloc_dec"]
PHASE_MAP = {name: i for i, name in enumerate(DEC_TYPES)}
ID_TO_PHASE = {v: k for k, v in PHASE_MAP.items()}

MAX_GNN_DIM = 5
MAX_OPT_DIM = 6
NUM_PHASES = len(DEC_TYPES)

GNN_FEAT_DIMS = {
    "cache_dec": 5,
    "extract_dec": 4,
    "dispatch_dec": 4,
    "bufferize_dec": 5,
    "malloc_dec": 3,
}
FEAT_DIMS = {
    "cache_dec": 5,
    "extract_dec": 6,
    "dispatch_dec": 6,
    "bufferize_dec": 4,
    "malloc_dec": 3,
}


# ----------------------------------------------------------------------
# Networking
# ----------------------------------------------------------------------
def create_client_socket(config: TrainConfig):
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
    data = zlib.compress(pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL))
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


# ----------------------------------------------------------------------
# Model building blocks
# ----------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True) * (x.size(-1) ** -0.5)
        return (x / (norm + self.eps)) * self.weight


class UnifiedStateEncoder(nn.Module):
    """
    Single encoder for all 5 phases.
    - phase-specific linear stems for nodes and options (input padded to MAX dims)
    - shared transformer over nodes
    - mean pool -> global state
    """

    def __init__(self, hidden_dim=128, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.phase_emb = nn.Embedding(NUM_PHASES, hidden_dim)

        self.node_stems = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(MAX_GNN_DIM, hidden_dim),
                    nn.GELU(),
                )
                for _ in range(NUM_PHASES)
            ]
        )
        self.opt_stems = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(MAX_OPT_DIM, hidden_dim),
                    nn.GELU(),
                )
                for _ in range(NUM_PHASES)
            ]
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = RMSNorm(hidden_dim)

    def project_nodes(self, nodes_concat, node_phase):
        total_nodes = nodes_concat.size(0)
        device = nodes_concat.device
        hidden = self.hidden_dim
        out = torch.zeros(total_nodes, hidden, device=device, dtype=torch.float32)
        if total_nodes == 0:
            return out
        # ensure float
        if not nodes_concat.dtype.is_floating_point:
            nodes_concat = nodes_concat.float()
        else:
            nodes_concat = nodes_concat.float()
        for pid in range(NUM_PHASES):
            mask = node_phase == pid
            if mask.any():
                out[mask] = self.node_stems[pid](nodes_concat[mask])
        out = out + self.phase_emb(node_phase)
        return out

    def project_options(self, feats_concat, opt_phase):
        total_opts = feats_concat.size(0)
        device = feats_concat.device
        hidden = self.hidden_dim
        out = torch.zeros(total_opts, hidden, device=device, dtype=torch.float32)
        if total_opts == 0:
            return out
        if not feats_concat.dtype.is_floating_point:
            feats_concat = feats_concat.float()
        else:
            feats_concat = feats_concat.float()
        for pid in range(NUM_PHASES):
            mask = opt_phase == pid
            if mask.any():
                out[mask] = self.opt_stems[pid](feats_concat[mask])
        out = out + self.phase_emb(opt_phase)
        return out

    def encode_nodes_batch(self, phase_ids, nodes_concat, n_lengths):
        B = phase_ids.size(0)
        device = phase_ids.device
        hidden = self.hidden_dim
        if B == 0:
            return torch.zeros(0, hidden, device=device)
        total_nodes = nodes_concat.size(0)
        if total_nodes == 0:
            return torch.zeros(B, hidden, device=device)

        node_to_graph = torch.repeat_interleave(
            torch.arange(B, device=device), n_lengths
        )
        node_phase = phase_ids[node_to_graph]

        node_emb = self.project_nodes(nodes_concat, node_phase)

        N_max = int(n_lengths.max().item())
        padded = torch.zeros(B, N_max, hidden, device=device, dtype=node_emb.dtype)

        offsets = torch.zeros(B, dtype=torch.long, device=device)
        if B > 1:
            offsets[1:] = torch.cumsum(n_lengths[:-1], dim=0)
        arange_nodes = torch.arange(total_nodes, device=device)
        idx_within = arange_nodes - offsets[node_to_graph]

        padded[node_to_graph, idx_within] = node_emb

        mask = torch.arange(N_max, device=device).unsqueeze(0).expand(
            B, N_max
        ) >= n_lengths.unsqueeze(1)

        transformed = self.transformer(padded, src_key_padding_mask=mask)

        valid = (~mask).unsqueeze(-1).float()
        summed = (transformed * valid).sum(dim=1)
        counts = valid.sum(dim=1).clamp(min=1.0)
        g_states = summed / counts
        g_states = self.norm(g_states)
        return g_states

    @torch.no_grad()
    def encode_single(self, phase_id: int, node_features: torch.Tensor):
        device = node_features.device
        if node_features.size(0) == 0:
            return torch.zeros(self.hidden_dim, device=device)
        nf = node_features
        if nf.size(1) < MAX_GNN_DIM:
            pad = torch.zeros(
                nf.size(0), MAX_GNN_DIM - nf.size(1), device=device, dtype=nf.dtype
            )
            nf = torch.cat([nf, pad], dim=1)
        elif nf.size(1) > MAX_GNN_DIM:
            nf = nf[:, :MAX_GNN_DIM]
        emb = self.node_stems[phase_id](nf) + self.phase_emb.weight[phase_id]
        padded = emb.unsqueeze(0)
        N = padded.size(1)
        mask = torch.zeros(1, N, dtype=torch.bool, device=device)
        transformed = self.transformer(padded, src_key_padding_mask=mask)
        g = transformed.mean(dim=1).squeeze(0)
        g = self.norm(g)
        return g


class UnifiedDecisionModel(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, 1),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, 1),
        )


class AlphaZeroAgent(nn.Module):
    def __init__(
        self, hidden_dim=128, transformer_layers=2, transformer_heads=4, dropout=0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = UnifiedStateEncoder(
            hidden_dim=hidden_dim,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            dropout=dropout,
        )
        self.decision = UnifiedDecisionModel(hidden_dim=hidden_dim)

    def forward_single(
        self,
        phase_id: int,
        node_features: torch.Tensor,
        edge_src,
        edge_dst,
        options_features: torch.Tensor,
    ):
        device = (
            options_features.device
            if options_features.numel() > 0
            else node_features.device
        )
        if node_features.dim() == 1:
            node_features = node_features.unsqueeze(0)
        if options_features.dim() == 1:
            options_features = options_features.unsqueeze(0)

        global_state = self.encoder.encode_single(phase_id, node_features)

        M = options_features.size(0)
        if M == 0:
            return (
                torch.zeros(0, device=device),
                self.decision.value(global_state.unsqueeze(0)).squeeze(0),
                global_state,
            )

        if options_features.size(1) < MAX_OPT_DIM:
            pad = torch.zeros(
                M,
                MAX_OPT_DIM - options_features.size(1),
                device=device,
                dtype=options_features.dtype,
            )
            opt_padded = torch.cat([options_features, pad], dim=1)
        else:
            opt_padded = options_features[:, :MAX_OPT_DIM]

        opt_emb = (
            self.encoder.opt_stems[phase_id](opt_padded)
            + self.encoder.phase_emb.weight[phase_id]
        )

        global_expanded = global_state.unsqueeze(0).expand(M, -1)
        policy_in = torch.cat([global_expanded, opt_emb], dim=1)
        scores = self.decision.policy(policy_in).squeeze(1)
        val = self.decision.value(global_state.unsqueeze(0)).squeeze(0)
        return scores, val, global_state

    def forward_batch(self, phase_ids, nodes_concat, n_lengths, feats_concat, N_list):
        device = phase_ids.device
        g_states = self.encoder.encode_nodes_batch(phase_ids, nodes_concat, n_lengths)

        total_opts = feats_concat.size(0)
        B = phase_ids.size(0)
        if total_opts == 0:
            return (
                g_states,
                torch.zeros(0, device=device),
                self.decision.value(g_states),
            )

        if not isinstance(N_list, torch.Tensor):
            N_tensor = torch.tensor(N_list, device=device, dtype=torch.long)
        else:
            N_tensor = N_list.to(device)

        opt_to_graph = torch.repeat_interleave(torch.arange(B, device=device), N_tensor)
        opt_phase = phase_ids[opt_to_graph]
        opt_emb = self.encoder.project_options(feats_concat, opt_phase)

        g_repeated = torch.repeat_interleave(g_states, N_tensor, dim=0)

        policy_in = torch.cat([g_repeated, opt_emb], dim=1)
        all_scores = self.decision.policy(policy_in).squeeze(1)
        vals = self.decision.value(g_states)
        return g_states, all_scores, vals


# ----------------------------------------------------------------------
# ActorDelegate - unified
# ----------------------------------------------------------------------
import tensor_graphs


class ActorDelegate(tensor_graphs.SearchDelegate):
    def __init__(
        self,
        agent: AlphaZeroAgent,
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
                min_noise, base_noise * (1.0 - episode / max(1, decay_episodes))
            )

        self.active_stack = []
        self.globals: Dict[str, torch.Tensor] = {}
        self.raw_graphs: Dict[str, dict] = {}

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

    def _prepare_graphs(self, node_features, edge_src, edge_dst, feat_dim):
        if not node_features:
            nf = torch.zeros((0, MAX_GNN_DIM), dtype=torch.float32)
            src = torch.zeros(0, dtype=torch.int64)
            dst = torch.zeros(0, dtype=torch.int64)
            return nf, src, dst
        arr = np.array(node_features, dtype=np.float32)
        if feat_dim == 0:
            N = 0
        else:
            N = len(arr) // feat_dim
        if N == 0:
            nf = torch.zeros((0, MAX_GNN_DIM), dtype=torch.float32)
            src = torch.zeros(0, dtype=torch.int64)
            dst = torch.zeros(0, dtype=torch.int64)
            return nf, src, dst
        raw = torch.tensor(arr, dtype=torch.float32).view(N, feat_dim)
        raw = torch.nan_to_num(raw, posinf=1e9, neginf=-1e9)
        if feat_dim < MAX_GNN_DIM:
            pad = torch.zeros(N, MAX_GNN_DIM - feat_dim, dtype=torch.float32)
            nf = torch.cat([raw, pad], dim=1)
        else:
            nf = raw[:, :MAX_GNN_DIM]
        src = (
            torch.tensor(edge_src, dtype=torch.int64)
            if len(edge_src) > 0
            else torch.zeros(0, dtype=torch.int64)
        )
        dst = (
            torch.tensor(edge_dst, dtype=torch.int64)
            if len(edge_dst) > 0
            else torch.zeros(0, dtype=torch.int64)
        )
        return nf, src, dst

    def _process_gnn(
        self, dec_type, gnn_model, node_features, edge_src, edge_dst, feat_dim
    ):
        nf, src, dst = self._prepare_graphs(node_features, edge_src, edge_dst, feat_dim)
        self.raw_graphs[dec_type] = {
            "node_features": nf.cpu().numpy(),
            "edge_src": src.cpu().numpy(),
            "edge_dst": dst.cpu().numpy(),
        }
        phase_id = PHASE_MAP.get(dec_type, 0)
        if len(nf) > 0:
            with torch.no_grad():
                global_state = self.agent.encoder.encode_single(phase_id, nf)
            self.globals[dec_type] = global_state
        else:
            self.globals[dec_type] = torch.zeros(self.agent.hidden_dim)

    def init_cache_graph(self, node_features, edge_src, edge_dst):
        self._process_gnn("cache_dec", None, node_features, edge_src, edge_dst, 5)

    def init_egraph(self, node_features, edge_src, edge_dst):
        self._process_gnn("extract_dec", None, node_features, edge_src, edge_dst, 4)

    def init_dispatch_graph(self, node_features, edge_src, edge_dst):
        self._process_gnn("dispatch_dec", None, node_features, edge_src, edge_dst, 4)

    def init_bufferize_graph(self, node_features, edge_src, edge_dst):
        self._process_gnn("bufferize_dec", None, node_features, edge_src, edge_dst, 5)

    def init_malloc_graph(self, node_features, edge_src, edge_dst):
        self._process_gnn("malloc_dec", None, node_features, edge_src, edge_dst, 3)

    @torch.inference_mode()
    def _order_items(self, items, dec_type, extract_fn):
        features_raw = extract_fn(items)
        if not isinstance(features_raw, torch.Tensor):
            features_raw = torch.tensor(features_raw, dtype=torch.float32)
        global_state = self.globals.get(dec_type, torch.zeros(self.agent.hidden_dim))
        raw_graph = self.raw_graphs.get(
            dec_type,
            {
                "node_features": np.zeros((0, MAX_GNN_DIM), dtype=np.float32),
                "edge_src": np.zeros(0, dtype=np.int64),
                "edge_dst": np.zeros(0, dtype=np.int64),
            },
        )
        phase_id = PHASE_MAP.get(dec_type, 0)
        state_key = hash(
            (
                phase_id,
                global_state.cpu().numpy().tobytes(),
                features_raw.cpu().numpy().tobytes(),
            )
        )
        num_actions = len(items)

        if len(items) <= 1:
            if len(items) == 1 and state_key not in self.mcts_tree:
                M = features_raw.size(0)
                feat_dim = features_raw.size(1) if features_raw.dim() > 1 else 0
                if features_raw.dim() == 1:
                    features_raw = features_raw.unsqueeze(0)
                if feat_dim < MAX_OPT_DIM:
                    pad = torch.zeros(M, MAX_OPT_DIM - feat_dim)
                    feats_padded = torch.cat([features_raw, pad], dim=1)
                else:
                    feats_padded = features_raw[:, :MAX_OPT_DIM]
                self.mcts_tree[state_key] = {
                    "N": np.zeros(1, dtype=np.float32),
                    "W": np.zeros(1, dtype=np.float32),
                    "P": np.ones(1, dtype=np.float32),
                    "v": 0.0,
                    "type": dec_type,
                    "phase": phase_id,
                    "features": feats_padded.cpu().numpy(),
                    "node_features": raw_graph["node_features"],
                    "edge_src": raw_graph["edge_src"],
                    "edge_dst": raw_graph["edge_dst"],
                }
                self.active_stack.append((state_key, 0))
            return list(range(len(items)))

        if state_key not in self.mcts_tree:
            with torch.no_grad():
                scores, val, _ = self.agent.forward_single(
                    phase_id,
                    torch.tensor(raw_graph["node_features"], dtype=torch.float32),
                    torch.tensor(raw_graph["edge_src"], dtype=torch.int64),
                    torch.tensor(raw_graph["edge_dst"], dtype=torch.int64),
                    features_raw,
                )
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

            M = features_raw.size(0)
            feat_dim = features_raw.size(1)
            if feat_dim < MAX_OPT_DIM:
                pad = torch.zeros(M, MAX_OPT_DIM - feat_dim)
                feats_padded = torch.cat([features_raw, pad], dim=1)
            else:
                feats_padded = features_raw[:, :MAX_OPT_DIM]

            self.mcts_tree[state_key] = {
                "N": np.zeros(num_actions, dtype=np.float32),
                "W": np.zeros(num_actions, dtype=np.float32),
                "P": P_perturbed,
                "v": v,
                "type": dec_type,
                "phase": phase_id,
                "features": feats_padded.cpu().numpy(),
                "node_features": raw_graph["node_features"],
                "edge_src": raw_graph["edge_src"],
                "edge_dst": raw_graph["edge_dst"],
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

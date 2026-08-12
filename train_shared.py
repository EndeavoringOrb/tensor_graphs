import dataclasses
import math
import pickle
import socket
import struct
import zlib

import torch
import torch.nn.functional as F
from torch import nn


@dataclasses.dataclass
class TrainConfig:
    run_dir: str = "runs/server_train"
    model_name: str = "gemma-3-270m"
    model_path: str = "models/google/gemma-3-270m"
    num_simulations: int = 30
    replay_buffer_size: int = 50000
    batch_size: int = 1024
    save_interval: int = 20
    hidden_dim: int = 64
    lr: float = 1e-3
    log_cost_calls: bool = False
    workers: int = 4
    # Networking Config
    host: str = "127.0.0.1"
    port: int = 5000
    use_bluetooth: bool = False
    bt_host_address: str = "AC:F2:3C:A7:F7:EC"  # Kept for backward compatibility
    bt_port: int = 4  # Kept for backward compatibility


# ==============================================================================
# NETWORK SOCKET UTILITIES
# ==============================================================================
def create_client_socket(host: str, port: int, use_bluetooth: bool = False):
    """Creates and connects a client socket for TCP or Bluetooth RFCOMM."""
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
        sock.connect((host, port))
    else:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
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
    # Prefix each message with a 4-byte length (network byte order)
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
# MODELS (With RMSNorm for scaling raw features)
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


# ==============================================================================
# ACTOR DELEGATE
# ==============================================================================
import tensor_graphs


class ActorDelegate(tensor_graphs.SearchDelegate):
    def __init__(self, agent, exploration_noise=0.25):
        super().__init__()
        self.agent = agent
        self.exploration_noise = exploration_noise
        self.trajectory = []
        self.globals = {}

    def _prepare_graphs(self, node_features, edge_src, edge_dst, feat_dim):
        if not node_features:
            return None, None, None
        nf = torch.tensor(node_features, dtype=torch.float32).view(-1, feat_dim)
        nf = torch.nan_to_num(nf, posinf=1e9, neginf=-1e9)
        src = torch.tensor(edge_src, dtype=torch.int64)
        dst = torch.tensor(edge_dst, dtype=torch.int64)
        return nf, src, dst

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

    def _order_items(self, items, dec_type, extract_fn):
        if len(items) <= 1:
            return list(range(len(items)))

        features = extract_fn(items)
        global_state = self.globals.get(dec_type, torch.zeros(self.agent.hidden_dim))

        with torch.no_grad():
            dec_model = getattr(self.agent, dec_type)
            scores, _ = dec_model(global_state, features)

        P = torch.softmax(scores, dim=0).cpu().numpy()

        if self.exploration_noise > 0:
            noise = (
                torch.distributions.Dirichlet(torch.full_like(scores, 0.3))
                .sample()
                .numpy()
            )
            P = (1 - self.exploration_noise) * P + self.exploration_noise * noise

        order = torch.argsort(torch.tensor(P), descending=True).tolist()

        self.trajectory.append(
            {
                "type": dec_type,
                "global_state": global_state.cpu().numpy(),
                "features": features.cpu().numpy(),
                "P": P,
                "top_action": order[0],
            }
        )

        return order

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

            # Log-transform large numerical scales
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

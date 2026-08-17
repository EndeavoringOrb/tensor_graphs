import ctypes
import dataclasses
import math
import pickle
import random
import socket
import struct
import zlib
from multiprocessing import shared_memory
from typing import Protocol, Any

import numpy as np
import tensor_graphs
import torch
from torch import nn

try:
    import flashinfer

    HAS_FLASHINFER = True
except ImportError:
    HAS_FLASHINFER = False

torch.set_float32_matmul_precision("high")

# Maximum parameters for shared memory pre-allocated slots
MAX_ACTIONS = 1024
MAX_FEATS = 8
MAX_GRAPH_TOKENS = 65536


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
    cpp_threads: int = 1

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
# LOCK-FREE SHARED MEMORY SPSC RING BUFFER
# ==============================================================================
class ShmRequestSlot(ctypes.Structure):
    _fields_ = [
        ("msg_type", ctypes.c_uint32),
        ("version", ctypes.c_int64),
        ("prefix_key", ctypes.c_int64),
        ("phase_id", ctypes.c_int32),
        ("num_actions", ctypes.c_int32),
        ("action_features", ctypes.c_float * (MAX_ACTIONS * MAX_FEATS)),
    ]


class ShmResponseSlot(ctypes.Structure):
    _fields_ = [
        ("ready", ctypes.c_uint32),
        ("num_actions", ctypes.c_int32),
        ("value", ctypes.c_float),
        ("logits", ctypes.c_float * MAX_ACTIONS),
    ]


class ShmSPSCQueue:
    """Zero-copy single-producer single-consumer lock-free shared-memory circular queue."""

    def __init__(
        self,
        name: str,
        slot_cls,
        capacity: int = 32,
        create: bool = False,
    ):
        self.capacity = capacity
        self.slot_size = ctypes.sizeof(slot_cls)
        self.header_size = 16  # head (uint32), tail (uint32)
        self.total_size = self.header_size + self.capacity * self.slot_size
        self.slot_cls = slot_cls

        if create:
            try:
                self.shm = shared_memory.SharedMemory(
                    name=name, create=True, size=self.total_size
                )
            except FileExistsError:
                temp = shared_memory.SharedMemory(name=name)
                temp.close()
                temp.unlink()
                self.shm = shared_memory.SharedMemory(
                    name=name, create=True, size=self.total_size
                )
            self.shm.buf[: self.header_size] = b"\x00" * self.header_size
        else:
            self.shm = shared_memory.SharedMemory(name=name, create=False)

        self._head_ptr = (ctypes.c_uint32).from_buffer(self.shm.buf, 0)
        self._tail_ptr = (ctypes.c_uint32).from_buffer(self.shm.buf, 4)

    def write_slot(self) -> tuple[int, Any]:
        head = self._head_ptr.value
        tail = self._tail_ptr.value
        if head - tail >= self.capacity:
            return -1, None

        slot_idx = head % self.capacity
        offset = self.header_size + slot_idx * self.slot_size
        slot = self.slot_cls.from_buffer(self.shm.buf, offset)
        return slot_idx, slot

    def commit_write(self):
        self._head_ptr.value += 1

    def read_slot(self) -> tuple[int, Any]:
        head = self._head_ptr.value
        tail = self._tail_ptr.value
        if tail >= head:
            return -1, None

        slot_idx = tail % self.capacity
        offset = self.header_size + slot_idx * self.slot_size
        slot = self.slot_cls.from_buffer(self.shm.buf, offset)
        return slot_idx, slot

    def commit_read(self):
        self._tail_ptr.value += 1

    def is_empty(self) -> bool:
        return self._tail_ptr.value >= self._head_ptr.value

    def close(self):
        self.shm.close()

    def unlink(self):
        try:
            self.shm.unlink()
        except FileNotFoundError:
            pass


# ==============================================================================
# RADIX KV CACHE (SGLANG-STYLE PAGED KV TREE)
# ==============================================================================
class RadixNode:
    def __init__(
        self,
        prefix_key: int,
        page_indices: torch.Tensor,
        num_tokens: int,
        value: float = 0.0,
    ):
        self.prefix_key = prefix_key
        self.page_indices = page_indices
        self.num_tokens = num_tokens
        self.value = value


class RadixTreeKVCache:
    """Manages physical paged KV cache allocation and Radix tree structure."""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_pages: int = 250000,
        device="cuda",
    ):
        self.device = device
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = 1  # Page size 1 for granular token-level radix reuse

        # Layout: (num_layers, max_pages, 2, page_size, num_heads, head_dim)
        self.paged_kv_data = torch.zeros(
            (num_layers, max_pages, 2, self.page_size, num_heads, head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        self.nodes: dict[int, RadixNode] = {}
        self.free_page_ptr = 0

    def insert(
        self,
        prefix_key: int,
        num_tokens: int,
        layer_k: list[torch.Tensor],  # list of (L, num_heads, head_dim)
        layer_v: list[torch.Tensor],  # list of (L, num_heads, head_dim)
        value: float,
    ) -> RadixNode:
        if prefix_key in self.nodes:
            return self.nodes[prefix_key]

        start_page = self.free_page_ptr
        end_page = start_page + num_tokens
        self.free_page_ptr = end_page

        for l in range(self.num_layers):
            self.paged_kv_data[l, start_page:end_page, 0, 0, :, :] = layer_k[l]
            self.paged_kv_data[l, start_page:end_page, 1, 0, :, :] = layer_v[l]

        page_indices = torch.arange(
            start_page, end_page, dtype=torch.int32, device=self.device
        )
        node = RadixNode(
            prefix_key=prefix_key,
            page_indices=page_indices,
            num_tokens=num_tokens,
            value=value,
        )
        self.nodes[prefix_key] = node
        return node

    def get(self, prefix_key: int) -> RadixNode | None:
        return self.nodes.get(prefix_key, None)

    def contains(self, prefix_key: int) -> bool:
        return prefix_key in self.nodes


# ==============================================================================
# TRANSFORMER ARCHITECTURE WITH FLASHINFER PAGED RADIX ATTENTION
# ==============================================================================
class FlashInferSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward_prefix_layer(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encodes full prefix and outputs K and V to be cached in the Radix Tree."""
        L = x.shape[0]
        q = (
            self.q_proj(x)
            .view(L, self.num_heads, self.head_dim)
            .to(dtype=torch.bfloat16)
        )
        k = (
            self.k_proj(x)
            .view(L, self.num_heads, self.head_dim)
            .to(dtype=torch.bfloat16)
        )
        v = (
            self.v_proj(x)
            .view(L, self.num_heads, self.head_dim)
            .to(dtype=torch.bfloat16)
        )

        if HAS_FLASHINFER and x.is_cuda:
            attn_out = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=False)
        else:
            q_t = q.unsqueeze(0).transpose(1, 2)
            k_t = k.unsqueeze(0).transpose(1, 2)
            v_t = v.unsqueeze(0).transpose(1, 2)
            attn_out = torch.nn.functional.scaled_dot_product_attention(
                q_t, k_t, v_t, is_causal=False
            )
            attn_out = attn_out.transpose(1, 2).squeeze(0)

        out = self.out_proj(attn_out.view(L, self.d_model))
        return out, k, v

    def forward_paged_actions(
        self,
        q_ragged: torch.Tensor,  # (total_A, num_heads, head_dim)
        paged_kv_data_layer: (
            torch.Tensor
        ),  # (max_pages, 2, page_size, num_heads, head_dim)
        prefill_wrapper: (flashinfer.BatchPrefillWithPagedKVCacheWrapper | None) = None,
    ) -> torch.Tensor:
        """Runs SGLang-style FlashInfer paged prefill attention."""
        out = prefill_wrapper.run(q_ragged, paged_kv_data_layer)
        out = self.out_proj(out.view(-1, self.d_model))
        return out


class FlashInferTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = FlashInferSelfAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward_prefix(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        norm_x = self.norm1(x)
        attn_out, k, v = self.attn.forward_prefix_layer(norm_x)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, k, v

    def forward_paged_actions(
        self,
        x_ragged: torch.Tensor,
        paged_kv_data_layer: torch.Tensor,
        prefill_wrapper: flashinfer.BatchPrefillWithPagedKVCacheWrapper,
    ) -> torch.Tensor:
        norm_x = self.norm1(x_ragged)
        total_A = norm_x.shape[0]
        q = (
            self.attn.q_proj(norm_x)
            .view(total_A, self.attn.num_heads, self.attn.head_dim)
            .to(dtype=torch.bfloat16)
        )
        attn_out = self.attn.forward_paged_actions(
            q, paged_kv_data_layer, prefill_wrapper=prefill_wrapper
        )
        x_ragged = x_ragged + attn_out
        x_ragged = x_ragged + self.mlp(self.norm2(x_ragged))
        return x_ragged


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
            [FlashInferTransformerBlock(d_model, nhead) for _ in range(num_layers)]
        )

        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )

        self.wrapper = None
        self.workspace_buffer = None

    def _init_flashinfer_wrapper(self, device):
        if self.wrapper is None and HAS_FLASHINFER and device.type == "cuda":
            self.workspace_buffer = torch.empty(
                128 * 1024 * 1024, dtype=torch.uint8, device=device
            )
            self.wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                self.workspace_buffer, kv_layout="NHD"
            )

    def encode_prefix(
        self,
        features: torch.Tensor,  # (L, 8)
        token_types: torch.Tensor,  # (L,)
        phase_ids: torch.Tensor,  # (L,)
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        x = (
            self.feat_proj(features)
            + self.type_emb(token_types)
            + self.phase_emb(phase_ids)
        )

        k_layers, v_layers = [], []
        for layer in self.layers:
            x, k, v = layer.forward_prefix(x)
            k_layers.append(k)
            v_layers.append(v)

        v_val = self.value_head(x[0]).squeeze(-1)
        return v_val, k_layers, v_layers

    def evaluate_actions_paged(
        self,
        ragged_action_features: torch.Tensor,  # (total_A, 8)
        ragged_phase_ids: torch.Tensor,  # (total_A,)
        paged_kv_data: (
            torch.Tensor
        ),  # (num_layers, max_pages, 2, 1, num_heads, head_dim)
        paged_kv_indices: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        q_indptr: torch.Tensor,
    ) -> torch.Tensor:
        total_A = ragged_action_features.shape[0]
        token_types = torch.full(
            (total_A,), 3, dtype=torch.int64, device=ragged_action_features.device
        )

        x = (
            self.feat_proj(ragged_action_features)
            + self.type_emb(token_types)
            + self.phase_emb(ragged_phase_ids)
        )

        self._init_flashinfer_wrapper(ragged_action_features.device)

        # Plan the fused batched FlashInfer kernel across all heterogeneous requests
        self.wrapper.plan(
            qo_indptr=q_indptr,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
            num_qo_heads=self.nhead,
            num_kv_heads=self.nhead,
            head_dim_qk=self.d_model // self.nhead,
            page_size=1,
            causal=False,
        )

        for l, layer in enumerate(self.layers):
            x = layer.forward_paged_actions(
                x, paged_kv_data[l], prefill_wrapper=self.wrapper
            )

        logits = self.policy_head(x).squeeze(-1)
        return logits


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
    features: np.ndarray
    token_types: np.ndarray
    phase_ids: np.ndarray
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
# ACTOR DELEGATE (SEARCH INTEGRATION)
# ==============================================================================
class ActorDelegate(tensor_graphs.SearchDelegate):
    PHASE_MAP = {"cache": 0, "extract": 1, "dispatch": 2, "bufferize": 3, "malloc": 4}

    def __init__(
        self,
        agent=None,
        shm_req_queue: ShmSPSCQueue | None = None,
        shm_resp_queue: ShmSPSCQueue | None = None,
        prefix_dict: dict | None = None,
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
        self.shm_req = shm_req_queue
        self.shm_resp = shm_resp_queue
        self.shared_prefix_dict = prefix_dict
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

        if self.shared_prefix_dict is not None:
            if prefix_key not in self.shared_prefix_dict:
                self.shared_prefix_dict[prefix_key] = prefix_data

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

            if self.agent is not None:
                # Direct local evaluation
                device = next(self.agent.parameters()).device
                pdata = self.prefix_registry[prefix_key]
                f_t = torch.tensor(pdata.features, dtype=torch.float32, device=device)
                tt_t = torch.tensor(pdata.token_types, dtype=torch.int64, device=device)
                p_t = torch.tensor(pdata.phase_ids, dtype=torch.int64, device=device)

                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    v_pred, k_layers, v_layers = self.agent.encode_prefix(
                        f_t, tt_t, p_t
                    )

                A_len = action_feats_np.shape[0]
                dim_feat = min(MAX_FEATS - 1, action_feats_np.shape[1])
                actions_t = torch.zeros(
                    (A_len, MAX_FEATS), dtype=torch.float32, device=device
                )
                actions_t[:, 0] = torch.arange(
                    A_len, dtype=torch.float32, device=device
                )
                actions_t[:, 1 : 1 + dim_feat] = torch.tensor(
                    action_feats_np[:, :dim_feat], dtype=torch.float32, device=device
                )
                pids_t = torch.full(
                    (A_len,), phase_id, dtype=torch.int64, device=device
                )

                # Execute local evaluation
                L = pdata.features.shape[0]
                q_indptr = torch.tensor([0, A_len], dtype=torch.int32, device=device)
                paged_kv_indptr = torch.tensor([0, L], dtype=torch.int32, device=device)
                paged_kv_indices = torch.arange(L, dtype=torch.int32, device=device)
                paged_kv_last_page_len = torch.tensor(
                    [1], dtype=torch.int32, device=device
                )

                paged_kv_data = torch.zeros(
                    (
                        len(k_layers),
                        L,
                        2,
                        1,
                        self.agent.nhead,
                        self.agent.d_model // self.agent.nhead,
                    ),
                    dtype=torch.bfloat16,
                    device=device,
                )
                for l in range(len(k_layers)):
                    paged_kv_data[l, :, 0, 0, :, :] = k_layers[l]
                    paged_kv_data[l, :, 1, 0, :, :] = v_layers[l]

                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    logits = self.agent.evaluate_actions_paged(
                        actions_t,
                        pids_t,
                        paged_kv_data,
                        paged_kv_indices,
                        paged_kv_indptr,
                        paged_kv_last_page_len,
                        q_indptr,
                    )
                scores = logits[:A_len].cpu().float().numpy()
                v = float(v_pred.item())
            else:
                # Lock-free Shared-Memory IPC to centralized inference worker
                A_len = min(MAX_ACTIONS, action_feats_np.shape[0])
                dim_feat = min(MAX_FEATS - 1, action_feats_np.shape[1])

                while True:
                    s_idx, slot = self.shm_req.write_slot()
                    if slot is not None:
                        slot.msg_type = 1
                        slot.version = self.version
                        slot.prefix_key = prefix_key
                        slot.phase_id = phase_id
                        slot.num_actions = A_len

                        flat_view = np.frombuffer(
                            slot.action_features, dtype=np.float32
                        ).reshape(MAX_ACTIONS, MAX_FEATS)
                        flat_view.fill(0.0)
                        flat_view[:A_len, 0] = np.arange(A_len, dtype=np.float32)
                        flat_view[:A_len, 1 : 1 + dim_feat] = action_feats_np[
                            :A_len, :dim_feat
                        ]
                        self.shm_req.commit_write()
                        break

                while True:
                    r_idx, r_slot = self.shm_resp.read_slot()
                    if r_slot is not None:
                        status = r_slot.ready
                        if status == 1:
                            v = float(r_slot.value)
                            logits_view = np.frombuffer(
                                r_slot.logits, dtype=np.float32
                            )[:A_len]
                            scores = logits_view.copy()
                            self.shm_resp.commit_read()
                            break
                        elif status == 2:
                            self.shm_resp.commit_read()
                            if self.shared_prefix_dict is not None:
                                self.shared_prefix_dict[prefix_key] = (
                                    self.prefix_registry[prefix_key]
                                )
                            while True:
                                _, s2 = self.shm_req.write_slot()
                                if s2 is not None:
                                    s2.msg_type = 1
                                    s2.version = self.version
                                    s2.prefix_key = prefix_key
                                    s2.phase_id = phase_id
                                    s2.num_actions = A_len
                                    self.shm_req.commit_write()
                                    break

            P = torch.softmax(torch.tensor(scores, dtype=torch.float32), dim=0).numpy()
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

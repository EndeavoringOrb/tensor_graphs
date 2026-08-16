import argparse
import dataclasses
import json
import os
import queue
import random
import struct
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import optim

from train_shared import (
    AlphaZeroAgent,
    TrainConfig,
    create_server_socket,
    recv_msg,
    send_msg,
    MAX_GNN_DIM,
    MAX_OPT_DIM,
    NUM_PHASES,
    PHASE_MAP,
    DEC_TYPES,
)

global_weights = {}
weights_lock = threading.Lock()
weights_ready_event = threading.Event()


class ReplayBuffer:
    """
    Single unified buffer optimized for thousands of transitions per episode.
    Stores phase ids + variable length node/edge/option pools.
    """

    def __init__(
        self,
        maxlen: int,
        gnn_feat_dim: int = MAX_GNN_DIM,
        feat_dim: int = MAX_OPT_DIM,
        pool_multiplier: int = 32,  # increased for high throughput
        pin_memory: bool = True,
    ):
        self.maxlen = maxlen
        self.gnn_feat_dim = gnn_feat_dim
        self.feat_dim = feat_dim

        # for thousands per episode, increase pool size
        self.max_pool_size = maxlen * pool_multiplier
        self.max_node_pool_size = maxlen * pool_multiplier * 8
        self.max_edge_pool_size = maxlen * pool_multiplier * 16

        self.state_ptr = 0
        self.size = 0

        self.pool_ptr = 0
        self.node_ptr = 0
        self.edge_ptr = 0

        # metadata per transition
        self.zs = torch.zeros((maxlen, 1), dtype=torch.float32, pin_memory=pin_memory)
        self.phase_ids = torch.zeros(maxlen, dtype=torch.int64, pin_memory=pin_memory)

        self.num_opts = torch.zeros(maxlen, dtype=torch.int64)
        self.opts_offset = torch.zeros(maxlen, dtype=torch.int64)

        self.num_nodes = torch.zeros(maxlen, dtype=torch.int64)
        self.nodes_offset = torch.zeros(maxlen, dtype=torch.int64)

        self.num_edges = torch.zeros(maxlen, dtype=torch.int64)
        self.edges_offset = torch.zeros(maxlen, dtype=torch.int64)

        # pools
        self.feats_pool = torch.zeros(
            (self.max_pool_size, feat_dim), dtype=torch.float32, pin_memory=pin_memory
        )
        self.pis_pool = torch.zeros(
            self.max_pool_size, dtype=torch.float32, pin_memory=pin_memory
        )

        self.node_feats_pool = torch.zeros(
            (self.max_node_pool_size, gnn_feat_dim),
            dtype=torch.float32,
            pin_memory=pin_memory,
        )
        self.edges_src_pool = torch.zeros(
            self.max_edge_pool_size, dtype=torch.int64, pin_memory=pin_memory
        )
        self.edges_dst_pool = torch.zeros(
            self.max_edge_pool_size, dtype=torch.int64, pin_memory=pin_memory
        )

    def _concat_list(self, lst, dtype, dim=0):
        # lst: list of np arrays, possibly empty, variable length
        if len(lst) == 0:
            return torch.zeros(0, dtype=dtype)
        # filter empty
        total = sum(x.shape[0] for x in lst if x is not None and len(x) > 0)
        if total == 0:
            # need shape for 2D case
            if lst and hasattr(lst[0], "ndim") and lst[0].ndim > 1:
                return torch.zeros((0, lst[0].shape[1]), dtype=dtype)
            return torch.zeros(0, dtype=dtype)
        # concatenate via numpy first for speed
        # convert each to numpy if tensor
        np_list = []
        for x in lst:
            if x is None:
                continue
            if isinstance(x, torch.Tensor):
                x = x.cpu().numpy()
            if x.shape[0] == 0:
                continue
            np_list.append(x)
        if len(np_list) == 0:
            if lst and hasattr(lst[0], "ndim") and lst[0].ndim > 1:
                return torch.zeros((0, lst[0].shape[1]), dtype=dtype)
            return torch.zeros(0, dtype=dtype)
        concat = np.concatenate(np_list, axis=0)
        return torch.from_numpy(concat).to(dtype)

    def extend(
        self, phase_list, nf_list, esrc_list, edst_list, feats_list, pis_list, zs_list
    ):
        """
        phase_list: list[int] len n
        nf_list: list[np array (num_nodes, gnn_dim)]
        esrc_list, edst_list: list[np array]
        feats_list: list[np array (num_opts, feat_dim)]
        pis_list: list[np array (num_opts)]
        zs_list: list[float] or np array
        Optimized for n up to thousands.
        """
        n = len(feats_list)
        if n == 0:
            return

        # Convert
        phase_tensor = (
            torch.tensor(phase_list, dtype=torch.int64)
            if not isinstance(phase_list, torch.Tensor)
            else phase_list
        )
        zs_tensor = (
            torch.tensor(zs_list, dtype=torch.float32).unsqueeze(1)
            if not isinstance(zs_list, torch.Tensor)
            else zs_list.unsqueeze(1)
        )

        opts_lengths = np.array(
            [f.shape[0] if hasattr(f, "shape") else len(f) for f in feats_list],
            dtype=np.int64,
        )
        opts_lengths_tensor = torch.from_numpy(opts_lengths)

        nf_lengths = np.array(
            [f.shape[0] if hasattr(f, "shape") else 0 for f in nf_list], dtype=np.int64
        )
        nf_lengths_tensor = torch.from_numpy(nf_lengths)

        edge_lengths = np.array(
            [e.shape[0] if hasattr(e, "shape") else len(e) for e in esrc_list],
            dtype=np.int64,
        )
        edge_lengths_tensor = torch.from_numpy(edge_lengths)

        feats_tensor = self._concat_list(feats_list, torch.float32)
        pis_tensor = self._concat_list(pis_list, torch.float32)
        nf_tensor = self._concat_list(nf_list, torch.float32)
        esrc_tensor = self._concat_list(esrc_list, torch.int64)
        edst_tensor = self._concat_list(edst_list, torch.int64)

        def write_to_pool(pool, ptr, max_size, data):
            data_len = data.size(0) if hasattr(data, "size") else len(data)
            if data_len == 0:
                return ptr, ptr
            if ptr + data_len > max_size:
                # wrap around
                ptr = 0
                if data_len > max_size:
                    data = data[:max_size]
                    data_len = max_size
            start = ptr
            pool[start : start + data_len] = data
            return start, ptr + data_len

        start_opts, self.pool_ptr = write_to_pool(
            self.feats_pool, self.pool_ptr, self.max_pool_size, feats_tensor
        )
        self.pis_pool[start_opts : start_opts + pis_tensor.size(0)] = pis_tensor

        start_nodes, self.node_ptr = write_to_pool(
            self.node_feats_pool, self.node_ptr, self.max_node_pool_size, nf_tensor
        )

        start_edges, self.edge_ptr = write_to_pool(
            self.edges_src_pool, self.edge_ptr, self.max_edge_pool_size, esrc_tensor
        )
        self.edges_dst_pool[start_edges : start_edges + edst_tensor.size(0)] = (
            edst_tensor
        )

        def get_item_offsets(start, lengths):
            item_offsets = torch.zeros(n, dtype=torch.int64)
            if n == 0:
                return item_offsets
            item_offsets[0] = start
            if n > 1:
                item_offsets[1:] = start + torch.cumsum(lengths[:-1], dim=0)
            return item_offsets

        opts_offsets = get_item_offsets(start_opts, opts_lengths_tensor)
        nodes_offsets = get_item_offsets(start_nodes, nf_lengths_tensor)
        edges_offsets = get_item_offsets(start_edges, edge_lengths_tensor)

        # write metadata circularly
        if self.state_ptr + n <= self.maxlen:
            slc = slice(self.state_ptr, self.state_ptr + n)
            self.zs[slc] = zs_tensor
            self.phase_ids[slc] = phase_tensor
            self.num_opts[slc] = opts_lengths_tensor
            self.opts_offset[slc] = opts_offsets
            self.num_nodes[slc] = nf_lengths_tensor
            self.nodes_offset[slc] = nodes_offsets
            self.num_edges[slc] = edge_lengths_tensor
            self.edges_offset[slc] = edges_offsets
            self.state_ptr = (self.state_ptr + n) % self.maxlen
        else:
            p1 = self.maxlen - self.state_ptr
            p2 = n - p1
            slc1 = slice(self.state_ptr, self.maxlen)
            self.zs[slc1] = zs_tensor[:p1]
            self.phase_ids[slc1] = phase_tensor[:p1]
            self.num_opts[slc1] = opts_lengths_tensor[:p1]
            self.opts_offset[slc1] = opts_offsets[:p1]
            self.num_nodes[slc1] = nf_lengths_tensor[:p1]
            self.nodes_offset[slc1] = nodes_offsets[:p1]
            self.num_edges[slc1] = edge_lengths_tensor[:p1]
            self.edges_offset[slc1] = edges_offsets[:p1]

            slc2 = slice(0, p2)
            self.zs[slc2] = zs_tensor[p1:]
            self.phase_ids[slc2] = phase_tensor[p1:]
            self.num_opts[slc2] = opts_lengths_tensor[p1:]
            self.opts_offset[slc2] = opts_offsets[p1:]
            self.num_nodes[slc2] = nf_lengths_tensor[p1:]
            self.nodes_offset[slc2] = nodes_offsets[p1:]
            self.num_edges[slc2] = edge_lengths_tensor[p1:]
            self.edges_offset[slc2] = edges_offsets[p1:]

            self.state_ptr = p2

        self.size = min(self.maxlen, self.size + n)

    def sample_batch(self, batch_size: int):
        batch_idxs = torch.randint(0, self.size, (batch_size,))

        z_targets = self.zs[batch_idxs]
        phase_batch = self.phase_ids[batch_idxs]

        lengths = self.num_opts[batch_idxs]
        offsets = self.opts_offset[batch_idxs]

        n_lengths = self.num_nodes[batch_idxs]
        n_offsets = self.nodes_offset[batch_idxs]

        e_lengths = self.num_edges[batch_idxs]
        e_offsets = self.edges_offset[batch_idxs]

        def gather_pool(pool, offsets_tensor, lengths_tensor):
            total_N = int(lengths_tensor.sum().item())
            if total_N == 0:
                if pool.dim() > 1:
                    return torch.zeros((0, pool.size(1)), dtype=pool.dtype)
                return torch.zeros(0, dtype=pool.dtype)
            offsets_cum = torch.zeros(batch_size, dtype=torch.int64)
            if batch_size > 1:
                offsets_cum[1:] = torch.cumsum(lengths_tensor[:-1], dim=0)
            idx_within = torch.arange(total_N) - torch.repeat_interleave(
                offsets_cum, lengths_tensor
            )
            flat_idx = (
                torch.repeat_interleave(offsets_tensor, lengths_tensor) + idx_within
            )
            return pool[flat_idx]

        feats_concat = gather_pool(self.feats_pool, offsets, lengths)
        pi_concat = gather_pool(self.pis_pool, offsets, lengths)

        nodes_concat = gather_pool(self.node_feats_pool, n_offsets, n_lengths)
        src_concat = gather_pool(self.edges_src_pool, e_offsets, e_lengths)
        dst_concat = gather_pool(self.edges_dst_pool, e_offsets, e_lengths)

        # shift edges for block-diagonal batching (kept for compatibility, though transformer ignores)
        if len(src_concat) > 0:
            node_offsets_cum = torch.zeros(batch_size, dtype=torch.int64)
            if batch_size > 1:
                node_offsets_cum[1:] = torch.cumsum(n_lengths[:-1], dim=0)
            edge_shifts = torch.repeat_interleave(node_offsets_cum, e_lengths)
            shifted_src = src_concat + edge_shifts
            shifted_dst = dst_concat + edge_shifts
        else:
            shifted_src = src_concat
            shifted_dst = dst_concat

        N_list = lengths.tolist()

        return (
            phase_batch,
            nodes_concat,
            shifted_src,
            shifted_dst,
            n_lengths,
            feats_concat,
            pi_concat,
            z_targets,
            N_list,
        )

    def __len__(self):
        return self.size


def setup_run_dir(base_dir="runs", run_dir=None, resume_latest=False) -> str:
    runs_dir = Path(base_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    if run_dir:
        target_dir = Path(run_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir.as_posix()

    if resume_latest:
        existing = [int(d) for d in os.listdir(runs_dir) if d.isdigit()]
        if existing:
            target_dir = runs_dir / str(max(existing))
            return target_dir.as_posix()

    existing = [int(d) for d in os.listdir(runs_dir) if d.isdigit()]
    run_idx = max(existing) + 1 if existing else 1
    target_dir = runs_dir / str(run_idx)
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir.as_posix()


def client_handler(client_sock, client_info, replay_queue):
    print(f"[Server] Worker connected from {client_info}")
    try:
        while True:
            msg = recv_msg(client_sock)
            if not msg:
                break

            msg_type = msg.get("type")
            if msg_type == "req_weights":
                weights_ready_event.wait(timeout=60.0)
                with weights_lock:
                    send_msg(client_sock, {"type": "weights", "data": global_weights})

            elif msg_type == "trajectory":
                cost = msg.get("cost", float("inf"))
                costs = msg.get("costs", [])
                if not costs and cost < float("inf"):
                    costs = [cost]

                for c in costs:
                    replay_queue.put({"type": "cost_metric", "cost": float(c)})

                trajectory_data = msg.get("data", {})
                total_transitions = 0

                # New unified format: dict with keys phase, node_features, etc.
                if isinstance(trajectory_data, dict) and "phase" in trajectory_data:
                    total_transitions = len(trajectory_data.get("Zs", []))
                    if total_transitions > 0:
                        replay_queue.put(
                            {"type": "trajectory_data", "data": trajectory_data}
                        )
                else:
                    # Legacy format: dict of dec_type -> {node_features, edge_src, ...}
                    # Convert to unified on the fly
                    phase_list = []
                    nf_list = []
                    esrc_list = []
                    edst_list = []
                    feats_list = []
                    pis_list = []
                    zs_list = []
                    for dt, d in trajectory_data.items():
                        pid = PHASE_MAP.get(dt, 0)
                        n = len(d.get("Zs", []))
                        phase_list.extend([pid] * n)
                        nf_list.extend(d.get("node_features", []))
                        esrc_list.extend(d.get("edge_src", []))
                        edst_list.extend(d.get("edge_dst", []))
                        feats_list.extend(d.get("features", []))
                        pis_list.extend(d.get("pis", []))
                        zs_list.extend(d.get("Zs", []))
                    total_transitions = len(zs_list)
                    if total_transitions > 0:
                        unified = {
                            "phase": phase_list,
                            "node_features": nf_list,
                            "edge_src": esrc_list,
                            "edge_dst": edst_list,
                            "features": feats_list,
                            "pis": pis_list,
                            "Zs": zs_list,
                        }
                        replay_queue.put({"type": "trajectory_data", "data": unified})

            elif msg_type == "cost_metric":
                cost = msg.get("cost")
                if cost is not None and cost < float("inf"):
                    replay_queue.put({"type": "cost_metric", "cost": float(cost)})

    except Exception as e:
        print(f"[Server] Worker {client_info} connection error: {e}")
    finally:
        print(f"[Server] Worker disconnected from {client_info}")
        client_sock.close()


def accept_loop(server_sock, conn_type_label, replay_queue):
    try:
        while True:
            client_sock, client_info = server_sock.accept()
            client_thread = threading.Thread(
                target=client_handler,
                args=(client_sock, client_info, replay_queue),
                daemon=True,
            )
            client_thread.start()
    except Exception as e:
        print(f"[Server] {conn_type_label} accept loop ended: {e}")


def batch_generator_worker(buffer, batch_queue, config, buffer_lock):
    while True:
        can_train = False
        with buffer_lock:
            can_train = len(buffer) >= config.batch_size

        if not can_train:
            time.sleep(0.02)
            continue

        with buffer_lock:
            batch = buffer.sample_batch(config.batch_size)

        batch_queue.put(batch)


def learner_process(config: TrainConfig, replay_queue: queue.Queue):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Learner] Using device: {device}")

    agent = AlphaZeroAgent(
        hidden_dim=config.hidden_dim,
        transformer_layers=config.transformer_layers,
        transformer_heads=config.transformer_heads,
        dropout=config.transformer_dropout,
    ).to(device)

    os.makedirs(config.run_dir, exist_ok=True)
    losses_bin_path = os.path.join(config.run_dir, "losses.bin")
    costs_bin_path = os.path.join(config.run_dir, "costs.bin")
    model_filepath = Path(config.run_dir) / "model.safetensors"
    pack_fmt = "<If"

    if model_filepath.exists():
        try:
            state_dict = load_file(model_filepath)
            agent.load_state_dict(state_dict, strict=False)
            agent.to(device)
            print(f"[Learner] Loaded existing model weights from {model_filepath}")
        except Exception as e:
            print(f"[Learner] Warning: Failed to load {model_filepath}: {e}")

    cpu_state_dict = {k: v.cpu() for k, v in agent.state_dict().items()}
    with weights_lock:
        global_weights.update(cpu_state_dict)
    weights_ready_event.set()
    save_file(cpu_state_dict, model_filepath)

    # torch.compile for high GPU utilization
    try:
        agent_opt = torch.compile(agent, dynamic=True)
        print("[Learner] Using torch.compile with dynamic=True")
    except Exception as e:
        print(f"[Learner] torch.compile failed: {e}, using eager")
        agent_opt = agent

    optimizer = optim.Adam(agent.parameters(), lr=config.lr)

    buffer = ReplayBuffer(
        maxlen=config.replay_buffer_size,
        gnn_feat_dim=MAX_GNN_DIM,
        feat_dim=MAX_OPT_DIM,
        pool_multiplier=32,
        pin_memory=True,
    )
    buffer_lock = threading.Lock()

    batch_queue = queue.Queue(maxsize=16)
    for _ in range(2):
        threading.Thread(
            target=batch_generator_worker,
            args=(buffer, batch_queue, config, buffer_lock),
            daemon=True,
        ).start()

    batches_processed = 0
    if os.path.exists(losses_bin_path) and os.path.getsize(losses_bin_path) >= 8:
        try:
            with open(losses_bin_path, "rb") as f_bin:
                f_bin.seek(-8, os.SEEK_END)
                last_idx, _ = struct.unpack(pack_fmt, f_bin.read(8))
                batches_processed = int(last_idx)
            print(f"[Learner] Resuming loss logging at batch {batches_processed}")
        except Exception as e:
            print(
                f"[Learner] Warning: Could not read last entry of {losses_bin_path}: {e}"
            )

    cost_count = 0
    if os.path.exists(costs_bin_path) and os.path.getsize(costs_bin_path) >= 8:
        try:
            with open(costs_bin_path, "rb") as f_bin:
                f_bin.seek(-8, os.SEEK_END)
                last_idx, _ = struct.unpack(pack_fmt, f_bin.read(8))
                cost_count = int(last_idx)
            print(f"[Learner] Resuming cost logging at count {cost_count}")
        except Exception as e:
            print(
                f"[Learner] Warning: Could not read last entry of {costs_bin_path}: {e}"
            )

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    while True:
        incoming_data = []
        while not replay_queue.empty():
            try:
                item = replay_queue.get_nowait()
                if isinstance(item, dict) and item.get("type") == "cost_metric":
                    cost_count += 1
                    cost_val = float(item["cost"])
                    with open(costs_bin_path, "ab") as f_bin:
                        f_bin.write(struct.pack(pack_fmt, cost_count, cost_val))
                        f_bin.flush()
                elif isinstance(item, dict) and item.get("type") == "trajectory_data":
                    incoming_data.append(item["data"])
            except queue.Empty:
                break

        if incoming_data:
            with buffer_lock:
                for data_dict in incoming_data:
                    # data_dict is unified
                    buffer.extend(
                        data_dict.get("phase", []),
                        data_dict.get("node_features", []),
                        data_dict.get("edge_src", []),
                        data_dict.get("edge_dst", []),
                        data_dict.get("features", []),
                        data_dict.get("pis", []),
                        data_dict.get("Zs", []),
                    )

        try:
            batch = batch_queue.get(timeout=0.05)
        except queue.Empty:
            continue

        (
            phase_batch,
            nodes_concat,
            shifted_src,
            shifted_dst,
            num_nodes_tensor,
            feats_concat,
            pi_concat,
            z_targets,
            N_list,
        ) = batch

        agent.train()
        optimizer.zero_grad()

        # Move to device
        phase_batch = phase_batch.to(device, non_blocking=True)
        nodes_concat = nodes_concat.to(device, non_blocking=True)
        num_nodes_tensor = num_nodes_tensor.to(device, non_blocking=True)
        feats_concat = feats_concat.to(device, non_blocking=True)
        pi_concat = pi_concat.to(device, non_blocking=True)
        z_targets = z_targets.to(device, non_blocking=True)

        # Use autocast for high GPU utilization
        if device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                g_states, all_scores, vals = agent_opt.forward_batch(
                    phase_batch, nodes_concat, num_nodes_tensor, feats_concat, N_list
                )
                # Value loss
                value_loss = F.mse_loss(vals, z_targets, reduction="mean")

                # Policy loss with per-graph softmax
                B = len(N_list)
                N_tensor = torch.tensor(N_list, device=device)
                batch_idx = torch.repeat_interleave(
                    torch.arange(B, device=device), N_tensor
                )

                max_scores = torch.full(
                    (B,), -float("inf"), device=device, dtype=all_scores.dtype
                )
                max_scores.scatter_reduce_(
                    0, batch_idx, all_scores, reduce="amax", include_self=False
                )

                shifted = all_scores - max_scores[batch_idx]
                exp_scores = torch.exp(shifted)
                sum_exp = torch.zeros(
                    B, device=device, dtype=all_scores.dtype
                ).scatter_add_(0, batch_idx, exp_scores)

                log_p = shifted - torch.log(sum_exp)[batch_idx]
                policy_loss = -(pi_concat * log_p).sum() / B

                total_loss = policy_loss + value_loss
        else:
            g_states, all_scores, vals = agent_opt.forward_batch(
                phase_batch, nodes_concat, num_nodes_tensor, feats_concat, N_list
            )
            value_loss = F.mse_loss(vals, z_targets, reduction="mean")
            B = len(N_list)
            N_tensor = torch.tensor(N_list, device=device)
            batch_idx = torch.repeat_interleave(
                torch.arange(B, device=device), N_tensor
            )
            max_scores = torch.full(
                (B,), -float("inf"), device=device, dtype=all_scores.dtype
            )
            max_scores.scatter_reduce_(
                0, batch_idx, all_scores, reduce="amax", include_self=False
            )
            shifted = all_scores - max_scores[batch_idx]
            exp_scores = torch.exp(shifted)
            sum_exp = torch.zeros(
                B, device=device, dtype=all_scores.dtype
            ).scatter_add_(0, batch_idx, exp_scores)
            log_p = shifted - torch.log(sum_exp)[batch_idx]
            policy_loss = -(pi_concat * log_p).sum() / B
            total_loss = policy_loss + value_loss

        if device.type == "cuda" and scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()

        batches_processed += 1
        loss_val = float(total_loss.detach().item())
        total_buf_size = len(buffer)
        print(
            f"[Learner] Batch {batches_processed:04d} | BufSize: {total_buf_size} | Loss: {loss_val:.4f} | Pol: {policy_loss.item():.4f} Val: {value_loss.item():.4f}"
        )

        with open(losses_bin_path, "ab") as f_bin:
            f_bin.write(struct.pack(pack_fmt, batches_processed, loss_val))
            f_bin.flush()

        if batches_processed % config.save_interval == 0:
            cpu_state_dict = {k: v.cpu() for k, v in agent.state_dict().items()}
            with weights_lock:
                global_weights.update(cpu_state_dict)
            save_file(cpu_state_dict, model_filepath)


def main():
    parser = argparse.ArgumentParser(
        description="AlphaZero TensorGraph Server / Learner - Unified MDP"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="TCP listen address"
    )
    parser.add_argument("--port", type=int, default=5000, help="TCP listen port")
    parser.add_argument(
        "-bt",
        "--enable-bluetooth",
        action="store_true",
        help="Also listen on Bluetooth RFCOMM",
    )
    parser.add_argument(
        "--bt-address", type=str, default="AC:F2:3C:A7:F7:EC", help="Bluetooth host MAC"
    )
    parser.add_argument(
        "--bt-port", type=int, default=4, help="Bluetooth RFCOMM channel"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1024, help="Replay buffer batch size"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dim")
    parser.add_argument(
        "--transformer-layers", type=int, default=2, help="Transformer layers"
    )
    parser.add_argument(
        "--transformer-heads", type=int, default=4, help="Transformer heads"
    )
    parser.add_argument(
        "--run-dir", type=str, default=None, help="Path to specific run directory"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from latest run dir"
    )
    parser.add_argument("--c-puct", type=float, default=1.25, help="PUCT constant")
    parser.add_argument("--base-noise", type=float, default=0.25, help="Initial noise")
    parser.add_argument("--min-noise", type=float, default=0.01, help="Min noise")
    parser.add_argument(
        "--decay-episodes", type=int, default=500, help="Decay episodes"
    )
    parser.add_argument("--depth-gamma", type=float, default=0.7, help="Depth gamma")

    args = parser.parse_args()

    run_dir = setup_run_dir(run_dir=args.run_dir, resume_latest=args.resume)

    config_file = Path(run_dir) / "config.json"
    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                saved_config = json.load(f)
            config = TrainConfig(**saved_config)
            print(f"[Server] Loaded existing run configuration from {config_file}")
        except Exception as e:
            print(f"[Server] Could not load existing config ({e}), creating default.")
            config = TrainConfig()
    else:
        config = TrainConfig()

    config.run_dir = run_dir
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.host = args.host
    config.port = args.port
    config.c_puct = args.c_puct
    config.base_noise = args.base_noise
    config.min_noise = args.min_noise
    config.decay_episodes = args.decay_episodes
    config.depth_gamma = args.depth_gamma
    config.hidden_dim = args.hidden_dim
    config.transformer_layers = args.transformer_layers
    config.transformer_heads = args.transformer_heads

    with open(os.path.join(config.run_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(config), f, indent=4)

    replay_queue = queue.Queue()

    learner_thread = threading.Thread(
        target=learner_process, args=(config, replay_queue), daemon=True
    )
    learner_thread.start()

    server_sockets = []
    tcp_sock = create_server_socket(config.host, config.port, use_bluetooth=False)
    server_sockets.append(tcp_sock)
    tcp_thread = threading.Thread(
        target=accept_loop, args=(tcp_sock, "TCP/IP", replay_queue), daemon=True
    )
    tcp_thread.start()

    print("=========================================================")
    print(f" Server Listening on TCP: {config.host}:{config.port}")
    print(
        f" Unified MDP - Transformer {config.transformer_layers}L {config.transformer_heads}H hidden={config.hidden_dim}"
    )

    if args.enable_bluetooth:
        try:
            bt_sock = create_server_socket(
                args.bt_address, args.bt_port, use_bluetooth=True
            )
            server_sockets.append(bt_sock)
            bt_thread = threading.Thread(
                target=accept_loop,
                args=(bt_sock, "Bluetooth RFCOMM", replay_queue),
                daemon=True,
            )
            bt_thread.start()
            print(
                f" Server Listening on Bluetooth: {args.bt_address} (Channel {args.bt_port})"
            )
        except Exception as e:
            print(f" Could not bind Bluetooth socket: {e}")

    print(f" Run Directory: {config.run_dir}")
    print("=========================================================")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down server.")
    finally:
        for s in server_sockets:
            s.close()


if __name__ == "__main__":
    main()

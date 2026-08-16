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
)

DEC_TYPES = ["cache_dec", "extract_dec", "dispatch_dec", "bufferize_dec", "malloc_dec"]
FEAT_DIMS = {
    "cache_dec": 5,
    "extract_dec": 6,
    "dispatch_dec": 6,
    "bufferize_dec": 4,
    "malloc_dec": 3,
}


global_weights = {}
weights_lock = threading.Lock()


class ReplayBuffer:
    def __init__(
        self,
        maxlen: int,
        global_dim: int,
        feat_dim: int,
        pool_multiplier: int = 16,
        pin_memory: bool = True,
    ):
        self.maxlen = maxlen
        self.max_pool_size = maxlen * pool_multiplier
        self.global_dim = global_dim
        self.feat_dim = feat_dim

        self.state_ptr = 0
        self.pool_ptr = 0
        self.size = 0

        # Pre-allocated fixed-size CPU tensors (pinned for non-blocking GPU transfer)
        self.global_states = torch.zeros(
            (maxlen, global_dim), dtype=torch.float32, pin_memory=pin_memory
        )
        self.zs = torch.zeros((maxlen, 1), dtype=torch.float32, pin_memory=pin_memory)
        self.offsets = torch.zeros(maxlen, dtype=torch.int64)
        self.lengths = torch.zeros(maxlen, dtype=torch.int64)

        # Pre-allocated variable-size feature/pi pools
        self.feats_pool = torch.zeros(
            (self.max_pool_size, feat_dim), dtype=torch.float32, pin_memory=pin_memory
        )
        self.pis_pool = torch.zeros(
            self.max_pool_size, dtype=torch.float32, pin_memory=pin_memory
        )

    def extend(self, gs_list, feats_list, pis_list, zs_list):
        n = len(gs_list)
        if n == 0:
            return

        # 1. Stack incoming trajectory batch into CPU tensors
        gs_tensor = torch.from_numpy(np.stack(gs_list)).float()
        zs_tensor = torch.tensor(zs_list, dtype=torch.float32).unsqueeze(1)

        lengths_arr = np.array([f.shape[0] for f in feats_list], dtype=np.int64)
        lengths_tensor = torch.from_numpy(lengths_arr)
        total_feats = int(lengths_arr.sum())

        feats_tensor = torch.from_numpy(np.concatenate(feats_list, axis=0)).float()
        pis_tensor = torch.from_numpy(np.concatenate(pis_list, axis=0)).float()

        # 2. Write features into flat pool ring
        if self.pool_ptr + total_feats > self.max_pool_size:
            self.pool_ptr = 0  # Wrap around

        start_pool = self.pool_ptr
        self.feats_pool[start_pool : start_pool + total_feats] = feats_tensor
        self.pis_pool[start_pool : start_pool + total_feats] = pis_tensor

        # Item-wise offsets in pool
        item_offsets = torch.zeros(n, dtype=torch.int64)
        item_offsets[0] = start_pool
        if n > 1:
            item_offsets[1:] = start_pool + torch.cumsum(lengths_tensor[:-1], dim=0)

        self.pool_ptr += total_feats

        # 3. Write states into ring buffer
        if self.state_ptr + n <= self.maxlen:
            self.global_states[self.state_ptr : self.state_ptr + n] = gs_tensor
            self.zs[self.state_ptr : self.state_ptr + n] = zs_tensor
            self.offsets[self.state_ptr : self.state_ptr + n] = item_offsets
            self.lengths[self.state_ptr : self.state_ptr + n] = lengths_tensor
            self.state_ptr = (self.state_ptr + n) % self.maxlen
        else:
            first_part = self.maxlen - self.state_ptr
            second_part = n - first_part

            self.global_states[self.state_ptr : self.maxlen] = gs_tensor[:first_part]
            self.zs[self.state_ptr : self.maxlen] = zs_tensor[:first_part]
            self.offsets[self.state_ptr : self.maxlen] = item_offsets[:first_part]
            self.lengths[self.state_ptr : self.maxlen] = lengths_tensor[:first_part]

            self.global_states[:second_part] = gs_tensor[first_part:]
            self.zs[:second_part] = zs_tensor[first_part:]
            self.offsets[:second_part] = item_offsets[first_part:]
            self.lengths[:second_part] = lengths_tensor[first_part:]
            self.state_ptr = second_part

        self.size = min(self.maxlen, self.size + n)

    def sample_batch(self, batch_size: int):
        """Batched indexing with 0 python loops."""
        # 1. Random sample batch indices
        batch_idxs = torch.randint(0, self.size, (batch_size,))

        # 2. Native tensor indexing for fixed-size components
        g_states = self.global_states[batch_idxs]
        z_targets = self.zs[batch_idxs]
        offsets = self.offsets[batch_idxs]
        lengths = self.lengths[batch_idxs]

        # 3. Vectorized ragged slice indexing for variable-length features & pis
        total_N = int(lengths.sum().item())
        offsets_cum = torch.zeros(batch_size, dtype=torch.int64)
        if batch_size > 1:
            offsets_cum[1:] = torch.cumsum(lengths[:-1], dim=0)

        idx_within_segment = torch.arange(total_N) - torch.repeat_interleave(
            offsets_cum, lengths
        )
        flat_indices = torch.repeat_interleave(offsets, lengths) + idx_within_segment

        feats_concat = self.feats_pool[flat_indices]
        pi_concat = self.pis_pool[flat_indices]
        N_list = lengths.tolist()

        return g_states, feats_concat, pi_concat, z_targets, N_list

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
                total_transitions = sum(len(d["Zs"]) for d in trajectory_data.values())

                if total_transitions > 0:
                    replay_queue.put(
                        {"type": "trajectory_data", "data": trajectory_data}
                    )

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
    """Continuously fetches batches using native tensor indexing."""
    while True:
        with buffer_lock:
            can_train = any(len(buffer[dt]) >= config.batch_size for dt in DEC_TYPES)
            if not can_train:
                time.sleep(0.02)
                continue

            prepared = {}
            for dt in DEC_TYPES:
                if len(buffer[dt]) >= config.batch_size:
                    # Native tensor indexing runs in ~0.3 ms
                    prepared[dt] = buffer[dt].sample_batch(config.batch_size)

        batch_queue.put(prepared)


def learner_process(config: TrainConfig, replay_queue: queue.Queue):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Learner] Using device: {device}")

    agent = AlphaZeroAgent(hidden_dim=config.hidden_dim).to(device)
    agent_opt = torch.compile(agent)
    optimizer = optim.Adam(agent.parameters(), lr=config.lr)

    # Pre-allocate pinned CPU tensor replay buffers
    buffer = {
        dt: ReplayBuffer(
            maxlen=config.replay_buffer_size,
            global_dim=config.hidden_dim,
            feat_dim=FEAT_DIMS[dt],
            pool_multiplier=16,
            pin_memory=True,
        )
        for dt in DEC_TYPES
    }
    buffer_lock = threading.Lock()

    # Prefetch queue
    batch_queue = queue.Queue(maxsize=8)
    for _ in range(2):  # 2 threads are plenty with tensor indexing
        threading.Thread(
            target=batch_generator_worker,
            args=(buffer, batch_queue, config, buffer_lock),
            daemon=True,
        ).start()

    os.makedirs(config.run_dir, exist_ok=True)
    losses_bin_path = os.path.join(config.run_dir, "losses.bin")
    costs_bin_path = os.path.join(config.run_dir, "costs.bin")
    model_filepath = Path(config.run_dir) / "model.safetensors"
    pack_fmt = "<If"

    if model_filepath.exists():
        try:
            state_dict = load_file(model_filepath)
            agent.load_state_dict(state_dict)
            agent.to(device)
            print(f"[Learner] Loaded existing model weights from {model_filepath}")
        except Exception as e:
            print(f"[Learner] Warning: Failed to load {model_filepath}: {e}")

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

    cpu_state_dict = {k: v.cpu() for k, v in agent.state_dict().items()}
    with weights_lock:
        global_weights.update(cpu_state_dict)
    save_file(cpu_state_dict, model_filepath)

    while True:
        # Drain network queue
        while not replay_queue.empty():
            try:
                item = replay_queue.get_nowait()
                if isinstance(item, dict) and item.get("type") == "cost_metric":
                    cost_count += 1
                    cost_val = float(item["cost"])
                    with open(costs_bin_path, "ab") as f_bin:
                        f_bin.write(struct.pack(pack_fmt, cost_count, cost_val))
                        f_bin.flush()
                elif item["type"] == "trajectory_data":
                    data_dict = item["data"]
                    with buffer_lock:
                        for dt, d in data_dict.items():
                            buffer[dt].extend(
                                d["global_states"], d["features"], d["pis"], d["Zs"]
                            )
            except queue.Empty:
                break

        try:
            prepared_batches = batch_queue.get(timeout=0.05)
        except queue.Empty:
            continue

        agent.train()
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)
        types_trained = 0

        for dt, (
            g_states,
            feats_concat,
            pi_concat,
            z_targets,
            N_list,
        ) in prepared_batches.items():
            dec_model = getattr(agent_opt, dt)

            # non_blocking=True allows the transfer to happen concurrently with GPU execution
            g_states = g_states.to(device, non_blocking=True)
            feats_concat = feats_concat.to(device, non_blocking=True)
            pi_concat = pi_concat.to(device, non_blocking=True)
            z_targets = z_targets.to(device, non_blocking=True)

            # B: Value Loss
            vals = dec_model.value(g_states)
            value_loss = F.mse_loss(vals, z_targets, reduction="mean")

            # C: Policy Loss
            g_repeated = torch.repeat_interleave(
                g_states, torch.tensor(N_list, device=device), dim=0
            )
            policy_in = torch.cat([g_repeated, feats_concat], dim=1)
            all_scores = dec_model.policy(policy_in).squeeze(1)

            B = len(N_list)
            N_tensor = torch.tensor(N_list, device=device)
            batch_idx = torch.repeat_interleave(
                torch.arange(B, device=device), N_tensor
            )

            # 1. Segmented Max for numerical stability
            max_scores = torch.full(
                (B,), -float("inf"), device=device, dtype=all_scores.dtype
            )
            max_scores.scatter_reduce_(0, batch_idx, all_scores, reduce="amax")

            # 2. Segmented Log-Sum-Exp
            shifted = all_scores - max_scores[batch_idx]
            exp_scores = torch.exp(shifted)
            sum_exp = torch.zeros(
                B, device=device, dtype=all_scores.dtype
            ).scatter_add_(0, batch_idx, exp_scores)

            # 3. Log Softmax & Cross Entropy Loss
            log_p = shifted - torch.log(sum_exp)[batch_idx]
            policy_loss = -(pi_concat * log_p).sum() / B

            type_loss = policy_loss + value_loss
            total_loss += type_loss
            types_trained += 1

        if types_trained > 0:
            total_loss = total_loss / types_trained
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()

            batches_processed += 1
            loss_val = float(total_loss.detach().item())
            total_buf_size = sum(len(b) for b in buffer.values())
            print(
                f"[Learner] Batch {batches_processed:04d} | Total BufSize: {total_buf_size} | Loss: {loss_val:.4f}"
            )

            with open(losses_bin_path, "ab") as f_bin:
                f_bin.write(struct.pack(pack_fmt, batches_processed, loss_val))
                f_bin.flush()

            if batches_processed % config.save_interval == 0:
                cpu_state_dict = {k: v.cpu() for k, v in agent.state_dict().items()}
                with weights_lock:
                    global_weights.clear()
                    global_weights.update(cpu_state_dict)
                save_file(cpu_state_dict, model_filepath)


def main():
    parser = argparse.ArgumentParser(
        description="AlphaZero TensorGraph Server / Learner"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="TCP listen address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="TCP listen port (default: 5000)"
    )
    parser.add_argument(
        "-bt",
        "--enable-bluetooth",
        action="store_true",
        help="Also listen on Bluetooth RFCOMM simultaneously",
    )
    parser.add_argument(
        "--bt-address",
        type=str,
        default="AC:F2:3C:A7:F7:EC",
        help="Bluetooth host MAC address",
    )
    parser.add_argument(
        "--bt-port", type=int, default=4, help="Bluetooth RFCOMM channel"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1024, help="Replay buffer batch size"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to specific run directory to resume or save to (e.g., runs/8)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest existing run directory if --run-dir is not specified",
    )
    # PUCT & Noise Annealing Options
    parser.add_argument(
        "--c-puct", type=float, default=1.25, help="PUCT exploration constant"
    )
    parser.add_argument(
        "--base-noise", type=float, default=0.25, help="Initial exploration noise"
    )
    parser.add_argument(
        "--min-noise", type=float, default=0.01, help="Minimum exploration noise floor"
    )
    parser.add_argument(
        "--decay-episodes",
        type=int,
        default=500,
        help="Number of episodes over which to decay noise",
    )
    parser.add_argument(
        "--depth-gamma", type=float, default=0.7, help="Per-depth noise decay factor"
    )

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
            print(
                f"[Server] Could not load existing config ({e}), creating default config."
            )
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

import argparse
import dataclasses
import json
import os
import queue
import random
import struct
import threading
import time
from collections import deque
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

global_weights = {}
weights_lock = threading.Lock()


class ReplayBuffer:
    def __init__(self, maxlen):
        self.buffer = []
        self.maxlen = maxlen
        self.ptr = 0

    def append(self, item):
        if len(self.buffer) < self.maxlen:
            self.buffer.append(item)
        else:
            self.buffer[self.ptr] = item
            self.ptr = (self.ptr + 1) % self.maxlen

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


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

                print(
                    f"[Server] Received trajectory from {client_info} "
                    f"(Best Cost: {cost:.4f} ms, Extractions: {len(costs)}, Transitions: {total_transitions})"
                )

                # Forward bulk payload to learner queue
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


def batch_generator_thread(buffer, batch_queue, config, buffer_lock):
    """Background thread to prepare batches on the CPU and pin memory."""
    while True:
        with buffer_lock:
            # 1. Wait until we have enough samples
            can_train = any(len(buffer[dt]) >= config.batch_size for dt in DEC_TYPES)
            if not can_train:
                time.sleep(0.1)
                continue

            # 2. Fast sampling inside the lock
            batches = {}
            for dt in DEC_TYPES:
                if len(buffer[dt]) >= config.batch_size:
                    batches[dt] = buffer[dt].sample(config.batch_size)

        # 3. Heavy lifting (np.stack, np.concatenate) OUTSIDE the lock!
        prepared_batches = {}
        for dt, batch in batches.items():
            # pin_memory() speeds up CPU -> GPU transfers and enables non_blocking=True
            g_states = torch.from_numpy(np.stack([s[0] for s in batch])).pin_memory()
            feats_concat = torch.from_numpy(
                np.concatenate([s[1] for s in batch], axis=0)
            ).pin_memory()
            pi_concat = torch.from_numpy(
                np.concatenate([s[2] for s in batch], axis=0)
            ).pin_memory()
            z_targets = (
                torch.tensor([s[3] for s in batch], dtype=torch.float32)
                .unsqueeze(1)
                .pin_memory()
            )
            N_list = [s[1].shape[0] for s in batch]

            prepared_batches[dt] = (
                g_states,
                feats_concat,
                pi_concat,
                z_targets,
                N_list,
            )

        # Put into queue (blocks if queue is full, naturally preventing CPU from running away)
        batch_queue.put(prepared_batches)


def learner_process(config: TrainConfig, replay_queue: queue.Queue):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Learner] Using device: {device}")

    agent = AlphaZeroAgent(hidden_dim=config.hidden_dim).to(device)
    original_state_names = set(agent.state_dict().keys())
    optimizer = optim.Adam(agent.parameters(), lr=config.lr)

    # Use our new fast ReplayBuffer
    buffer = {dt: ReplayBuffer(config.replay_buffer_size) for dt in DEC_TYPES}
    buffer_lock = threading.Lock()

    # Keep up to 4 fully collated batches ready for the GPU
    batch_queue = queue.Queue(maxsize=4)

    # Start the background batch generator thread
    bg_thread = threading.Thread(
        target=batch_generator_thread,
        args=(buffer, batch_queue, config, buffer_lock),
        daemon=True,
    )
    bg_thread.start()

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

    cpu_state_dict = {k: v.cpu() for k, v in agent.state_dict().items() if k in original_state_names}
    with weights_lock:
        global_weights.update(cpu_state_dict)
    save_file(cpu_state_dict, model_filepath)

    agent = torch.compile(agent)

    while True:
        # 1. Drain incoming network data into the replay buffer quickly
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
                            for gs, f, p, z in zip(
                                d["global_states"], d["features"], d["pis"], d["Zs"]
                            ):
                                buffer[dt].append((gs, f, p, z))
            except queue.Empty:
                break

        # 2. Try to grab a prepared batch (timeout so we can go back to draining the network queue)
        try:
            prepared_batches = batch_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        # 3. GPU Training Block
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
            dec_model = getattr(agent, dt)

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
            batch_idx = torch.repeat_interleave(torch.arange(B, device=device), N_tensor)

            # 1. Segmented Max for numerical stability
            max_scores = torch.full((B,), -float("inf"), device=device, dtype=all_scores.dtype)
            max_scores.scatter_reduce_(0, batch_idx, all_scores, reduce="amax")

            # 2. Segmented Log-Sum-Exp
            shifted = all_scores - max_scores[batch_idx]
            exp_scores = torch.exp(shifted)
            sum_exp = torch.zeros(B, device=device, dtype=all_scores.dtype).scatter_add_(0, batch_idx, exp_scores)

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
                cpu_state_dict = {k: v.cpu() for k, v in agent.state_dict().items() if k in original_state_names}
                with weights_lock:
                    global_weights.clear()
                    global_weights.update(cpu_state_dict)
                save_file(cpu_state_dict, model_filepath)
    else:
        time.sleep(1)


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

# File: train_server.py
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
    AlphaZeroTransformer,
    TrainConfig,
    create_server_socket,
    recv_msg,
    send_msg,
)

global_weights = {}
weights_lock = threading.Lock()
weights_ready_event = threading.Event()


class UnifiedReplayBuffer:
    def __init__(self, maxlen: int):
        self.buffer = deque(maxlen=maxlen)

    def extend(self, transitions):
        self.buffer.extend(transitions)

    def sample_batch(self, batch_size: int):
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

                trajectory_data = msg.get("data", [])
                if trajectory_data:
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
            threading.Thread(
                target=client_handler,
                args=(client_sock, client_info, replay_queue),
                daemon=True,
            ).start()
    except Exception as e:
        print(f"[Server] {conn_type_label} accept loop ended: {e}")


def batch_generator_worker(buffer, batch_queue, config, buffer_lock):
    while True:
        with buffer_lock:
            can_train = len(buffer) >= config.batch_size
            if can_train:
                batch = buffer.sample_batch(config.batch_size)

        if not can_train:
            time.sleep(0.02)
            continue

        # Convert to Tensors and Collate (Padding)
        features_list = [
            torch.tensor(t["features"], dtype=torch.float32) for t in batch
        ]
        token_types_list = [
            torch.tensor(t["token_types"], dtype=torch.int64) for t in batch
        ]
        phase_ids_list = [
            torch.tensor(t["phase_ids"], dtype=torch.int64) for t in batch
        ]
        zs = torch.tensor([t["z"] for t in batch], dtype=torch.float32)

        # Pad to max length in the batch
        features = torch.nn.utils.rnn.pad_sequence(
            features_list, batch_first=True, padding_value=0.0
        )
        token_types = torch.nn.utils.rnn.pad_sequence(
            token_types_list, batch_first=True, padding_value=0
        )
        phase_ids = torch.nn.utils.rnn.pad_sequence(
            phase_ids_list, batch_first=True, padding_value=0
        )

        # Generate Attention Padding Mask (True = Ignore)
        B, L_max = token_types.shape
        lengths = torch.tensor([len(t["token_types"]) for t in batch])
        key_padding_mask = torch.arange(L_max).expand(B, L_max) >= lengths.unsqueeze(1)

        # Align policy targets with the dynamically padded sequence
        padded_pis = torch.zeros((B, L_max), dtype=torch.float32)
        for i, t in enumerate(batch):
            tt = t["token_types"]
            # Action tokens have token_type == 3
            action_indices = np.where(tt == 3)[0]
            padded_pis[i, action_indices] = torch.tensor(t["pis"], dtype=torch.float32)

        batch_queue.put(
            {
                "features": features,
                "token_types": token_types,
                "phase_ids": phase_ids,
                "key_padding_mask": key_padding_mask,
                "padded_pis": padded_pis,
                "zs": zs,
            }
        )


def learner_process(config: TrainConfig, replay_queue: queue.Queue):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Learner] Using device: {device}")

    agent = AlphaZeroTransformer(
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        max_feat_dim=config.max_feat_dim,
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

    # Immediately populate global_weights and release client requests
    cpu_state_dict = {k: v.cpu() for k, v in agent.state_dict().items()}
    with weights_lock:
        global_weights.update(cpu_state_dict)
    weights_ready_event.set()
    save_file(cpu_state_dict, model_filepath)

    agent_opt = torch.compile(agent, dynamic=True)
    optimizer = optim.Adam(agent.parameters(), lr=config.lr)

    buffer = UnifiedReplayBuffer(maxlen=config.replay_buffer_size)
    buffer_lock = threading.Lock()

    batch_queue = queue.Queue(maxsize=8)
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
        except Exception:
            pass

    cost_count = 0
    if os.path.exists(costs_bin_path) and os.path.getsize(costs_bin_path) >= 8:
        try:
            with open(costs_bin_path, "rb") as f_bin:
                f_bin.seek(-8, os.SEEK_END)
                last_idx, _ = struct.unpack(pack_fmt, f_bin.read(8))
                cost_count = int(last_idx)
            print(f"[Learner] Resuming cost logging at count {cost_count}")
        except Exception:
            pass

    while True:
        # Drain network queue
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
                elif item["type"] == "trajectory_data":
                    incoming_data.extend(item["data"])
            except queue.Empty:
                break

        if incoming_data:
            with buffer_lock:
                buffer.extend(incoming_data)

        try:
            batch = batch_queue.get(timeout=0.05)
        except queue.Empty:
            continue

        agent_opt.train()
        optimizer.zero_grad()

        features = batch["features"].to(device, non_blocking=True)
        token_types = batch["token_types"].to(device, non_blocking=True)
        phase_ids = batch["phase_ids"].to(device, non_blocking=True)
        key_padding_mask = batch["key_padding_mask"].to(device, non_blocking=True)
        padded_pis = batch["padded_pis"].to(device, non_blocking=True)
        zs = batch["zs"].to(device, non_blocking=True)

        logits, v = agent_opt(
            features, token_types, phase_ids, key_padding_mask=key_padding_mask
        )

        # 1. Value Loss
        value_loss = F.mse_loss(v, zs, reduction="mean")

        # 2. Policy Loss
        action_mask = token_types == 3
        logits = logits.masked_fill(~action_mask, -float("inf"))
        log_probs = F.log_softmax(logits, dim=1)

        # Prevent 0.0 * -inf from generating NaN values
        loss_matrix = torch.where(
            padded_pis > 0, padded_pis * log_probs, torch.zeros_like(log_probs)
        )
        policy_loss = -loss_matrix.sum(dim=1).mean()

        total_loss = policy_loss + value_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent_opt.parameters(), 1.0)
        optimizer.step()

        batches_processed += 1
        loss_val = float(total_loss.detach().item())
        print(
            f"[Learner] Batch {batches_processed:04d} | Total BufSize: {len(buffer)} | Loss: {loss_val:.4f}"
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
        "--run-dir", type=str, default=None, help="Path to specific run directory"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest existing run directory",
    )
    # Transformer Architecture Config
    parser.add_argument(
        "--d-model", type=int, default=128, help="Transformer dimension"
    )
    parser.add_argument(
        "--nhead", type=int, default=4, help="Transformer attention heads"
    )
    parser.add_argument("--num-layers", type=int, default=3, help="Transformer layers")
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
    config.d_model = args.d_model
    config.nhead = args.nhead
    config.num_layers = args.num_layers

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

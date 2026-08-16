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

global_weights = {}
weights_lock = threading.Lock()


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

                print(
                    f"[Server] Received trajectory from {client_info} (Best Cost: {cost:.4f} ms, Extractions: {len(costs)})"
                )
                for transition in msg.get("data", []):
                    replay_queue.put(transition)

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


def learner_process(config: TrainConfig, replay_queue: queue.Queue):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Learner] Using device: {device}")

    agent = AlphaZeroAgent(hidden_dim=config.hidden_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=config.lr)
    buffer = []

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
        while not replay_queue.empty():
            try:
                item = replay_queue.get_nowait()
                if isinstance(item, dict) and item.get("type") == "cost_metric":
                    cost_count += 1
                    cost_val = float(item["cost"])
                    with open(costs_bin_path, "ab") as f_bin:
                        f_bin.write(struct.pack(pack_fmt, cost_count, cost_val))
                        f_bin.flush()
                else:
                    buffer.append(item)
                    if len(buffer) > config.replay_buffer_size:
                        buffer.pop(0)
            except queue.Empty:
                break

        if len(buffer) >= config.batch_size:
            agent.train()
            batch = random.sample(buffer, config.batch_size)
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=device)

            for dec_type in [
                "cache_dec",
                "extract_dec",
                "dispatch_dec",
                "bufferize_dec",
                "malloc_dec",
            ]:
                sub_batch = [b for b in batch if b["type"] == dec_type]
                if not sub_batch:
                    continue

                type_loss = torch.tensor(0.0, device=device)
                dec_model = getattr(agent, dec_type)

                for sample in sub_batch:
                    g_state = torch.from_numpy(sample["global_state"]).to(device)
                    feats = torch.from_numpy(sample["features"]).to(device)
                    pi_target = torch.from_numpy(sample["pi"]).to(device)
                    z_target = torch.tensor(
                        [sample["Z"]], dtype=torch.float32, device=device
                    )

                    scores, val = dec_model(g_state, feats)

                    log_p = F.log_softmax(scores, dim=0)
                    policy_loss = -(pi_target * log_p).sum()
                    value_loss = F.mse_loss(val, z_target)

                    type_loss = type_loss + policy_loss + value_loss

                total_loss = total_loss + type_loss

            total_loss = total_loss / len(batch)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()

            batches_processed += 1
            loss_val = float(total_loss.detach().item())
            print(
                f"[Learner] Batch {batches_processed:04d} | BufSize: {len(buffer)} | Loss: {loss_val:.4f}"
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
    parser.add_argument(
        "--replay-buffer-size",
        type=int,
        default=1_000_000,
        help="Max size of replay buffer",
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
    config.replay_buffer_size = args.replay_buffer_size

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

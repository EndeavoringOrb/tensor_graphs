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
from torch import optim

from train_shared import (
    AlphaZeroAgent,
    TrainConfig,
    create_server_socket,
    recv_msg,
    send_msg,
)

# Thread-safe global weights storage for the server
global_weights = {}
weights_lock = threading.Lock()


def setup_run_dir(base_dir="runs") -> str:
    runs_dir = Path(base_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    existing = [int(d) for d in os.listdir(runs_dir) if d.isdigit()]
    run_idx = max(existing) + 1 if existing else 1
    run_dir = runs_dir / str(run_idx)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir.as_posix()


def client_handler(client_sock, client_info, replay_queue):
    """Dedicated thread for handling a worker client."""
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

                # Queue extraction costs to be logged to costs.bin
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
    """Accepts incoming connections on a specific server socket."""
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
    """Central gradient update and metrics logging loop running locally on the Server."""
    agent = AlphaZeroAgent(hidden_dim=config.hidden_dim)
    optimizer = optim.Adam(agent.parameters(), lr=config.lr)
    buffer = []

    # Ensure run directory exists
    os.makedirs(config.run_dir, exist_ok=True)
    losses_bin_path = os.path.join(config.run_dir, "losses.bin")
    costs_bin_path = os.path.join(config.run_dir, "costs.bin")
    pack_fmt = "<If"
    cost_count = 0

    # Store initial weights safely
    with weights_lock:
        global_weights.update({k: v.cpu() for k, v in agent.state_dict().items()})

    batches_processed = 0

    while True:
        # Drain incoming queue from all workers into Replay Buffer and metrics files
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

        # Gradient Updates
        if len(buffer) >= config.batch_size:
            agent.train()
            batch = random.sample(buffer, config.batch_size)
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0)

            for dec_type in [
                "extract_dec",
                "dispatch_dec",
                "bufferize_dec",
                "malloc_dec",
            ]:
                sub_batch = [b for b in batch if b["type"] == dec_type]
                if not sub_batch:
                    continue

                type_loss = torch.tensor(0.0)
                dec_model = getattr(agent, dec_type)

                for sample in sub_batch:
                    g_state = torch.from_numpy(sample["global_state"])
                    feats = torch.from_numpy(sample["features"])
                    pi_target = torch.from_numpy(sample["pi"])
                    z_target = torch.tensor([sample["Z"]], dtype=torch.float32)

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

            # Log loss metric to binary file
            with open(losses_bin_path, "ab") as f_bin:
                f_bin.write(struct.pack(pack_fmt, batches_processed, loss_val))
                f_bin.flush()

            # Sync updated weights for worker threads
            if batches_processed % config.save_interval == 0:
                with weights_lock:
                    global_weights.clear()
                    global_weights.update(
                        {k: v.cpu() for k, v in agent.state_dict().items()}
                    )
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

    args = parser.parse_args()

    config = TrainConfig()
    config.run_dir = setup_run_dir()
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.host = args.host
    config.port = args.port

    # Save run config
    with open(os.path.join(config.run_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(config), f, indent=4)

    replay_queue = queue.Queue()

    # 1. Start Learner Thread
    learner_thread = threading.Thread(
        target=learner_process, args=(config, replay_queue), daemon=True
    )
    learner_thread.start()

    server_sockets = []

    # 2. Setup TCP Server Socket
    tcp_sock = create_server_socket(config.host, config.port, use_bluetooth=False)
    server_sockets.append(tcp_sock)
    tcp_thread = threading.Thread(
        target=accept_loop, args=(tcp_sock, "TCP/IP", replay_queue), daemon=True
    )
    tcp_thread.start()

    print("=========================================================")
    print(f" Server Listening on TCP: {config.host}:{config.port}")

    # 3. Optionally Setup Bluetooth Server Socket in Parallel
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

import argparse
import queue
import random
import struct
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import tensor_graphs
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import optim

from train_models import AlphaZeroTransformer
from train_shared import (
    PrefixData,
    TrainConfig,
    create_server_socket,
    recv_msg,
    send_msg,
)

torch.set_float32_matmul_precision("high")

global_weights = {}
global_version = 0
weights_lock = threading.Lock()
weights_ready_event = threading.Event()


class UnifiedReplayBuffer:
    def __init__(self, maxlen: int):
        self.buffer = deque(maxlen=maxlen)
        self.prefix_table: dict[int, PrefixData] = {}

    def extend_payload(self, payload: dict):
        if "prefixes" in payload:
            self.prefix_table.update(payload["prefixes"])
        if "transitions" in payload:
            self.buffer.extend(payload["transitions"])

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

    existing = [
        int(d.name) for d in runs_dir.iterdir() if d.is_dir() and d.name.isdigit()
    ]
    if resume_latest and existing:
        target_dir = runs_dir / str(max(existing))
        return target_dir.as_posix()

    run_idx = max(existing) + 1 if existing else 1
    target_dir = runs_dir / str(run_idx)
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir.as_posix()


def client_handler(client_sock, client_info, replay_queue, config: TrainConfig):
    print(f"[Server] Worker connected from {client_info}")
    try:
        while True:
            msg = recv_msg(client_sock)
            if not msg:
                break

            msg_type = msg.get("type")
            if msg_type == "req_config":
                send_msg(
                    client_sock,
                    {"type": "config", "config": config.to_dict()},
                )

            elif msg_type == "req_version":
                weights_ready_event.wait(timeout=60.0)
                with weights_lock:
                    send_msg(
                        client_sock,
                        {"type": "version", "version": global_version},
                    )

            elif msg_type == "req_weights":
                weights_ready_event.wait(timeout=60.0)
                with weights_lock:
                    send_msg(
                        client_sock,
                        {
                            "type": "weights",
                            "version": global_version,
                            "data": global_weights,
                        },
                    )

            elif msg_type == "trajectory":
                cost = msg.get("cost", float("inf"))
                costs = msg.get("costs", [])
                if not costs and cost < float("inf"):
                    costs = [cost]

                for c in costs:
                    replay_queue.put({"type": "cost_metric", "cost": float(c)})

                payload = msg.get("payload", {})
                if payload:
                    replay_queue.put({"type": "trajectory_payload", "payload": payload})

            elif msg_type == "cost_metric":
                cost = msg.get("cost")
                if cost is not None and cost < float("inf"):
                    replay_queue.put({"type": "cost_metric", "cost": float(cost)})

    except Exception as e:
        print(f"[Server] Worker {client_info} connection error: {e}")
    finally:
        print(f"[Server] Worker disconnected from {client_info}")
        client_sock.close()


def accept_loop(server_sock, conn_type_label, replay_queue, config: TrainConfig):
    try:
        while True:
            client_sock, client_info = server_sock.accept()
            threading.Thread(
                target=client_handler,
                args=(client_sock, client_info, replay_queue, config),
                daemon=True,
            ).start()
    except Exception as e:
        print(f"[Server] {conn_type_label} accept loop ended: {e}")


def learner_process(config: TrainConfig, replay_queue: queue.Queue):
    global global_version
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Learner] Using device: {device}")

    agent = AlphaZeroTransformer(
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        max_feat_dim=config.max_feat_dim,
    ).to(device)

    run_dir_path = Path(config.run_dir)
    run_dir_path.mkdir(parents=True, exist_ok=True)
    losses_bin_path = run_dir_path / "losses.bin"
    costs_bin_path = run_dir_path / "costs.bin"
    model_filepath = run_dir_path / "model.safetensors"
    pack_fmt = "<If"

    if model_filepath.exists():
        try:
            state_dict = load_file(model_filepath)
            agent.load_state_dict(state_dict, strict=False)
            agent.to(device)
            print(f"[Learner] Loaded existing model weights from {model_filepath}")
        except Exception as e:
            print(f"[Learner] Warning: Failed to load {model_filepath}: {e}")

    optimizer = optim.Adam(agent.parameters(), lr=config.lr)
    buffer = UnifiedReplayBuffer(maxlen=config.replay_buffer_size)

    batches_processed = 0
    if losses_bin_path.exists() and losses_bin_path.stat().st_size >= 8:
        try:
            with open(losses_bin_path, "rb") as f_bin:
                f_bin.seek(-8, 2)
                last_idx, _ = struct.unpack(pack_fmt, f_bin.read(8))
                batches_processed = int(last_idx)
            print(f"[Learner] Resuming loss logging at batch {batches_processed}")
        except Exception:
            pass

    cpu_state_dict = {k: v.cpu() for k, v in agent.state_dict().items()}
    with weights_lock:
        global_weights.update(cpu_state_dict)
        global_version = batches_processed
    weights_ready_event.set()
    save_file(cpu_state_dict, model_filepath)

    cost_count = 0
    if costs_bin_path.exists() and costs_bin_path.stat().st_size >= 8:
        try:
            with open(costs_bin_path, "rb") as f_bin:
                f_bin.seek(-8, 2)
                last_idx, _ = struct.unpack(pack_fmt, f_bin.read(8))
                cost_count = int(last_idx)
            print(f"[Learner] Resuming cost logging at count {cost_count}")
        except Exception:
            pass

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
                elif item["type"] == "trajectory_payload":
                    buffer.extend_payload(item["payload"])
            except queue.Empty:
                break

        if len(buffer) < config.batch_size:
            time.sleep(0.05)
            continue

        raw_batch = buffer.sample_batch(config.batch_size)
        groups = defaultdict(list)
        for item in raw_batch:
            groups[item["prefix_key"]].append(item)

        agent.train()
        optimizer.zero_grad()
        total_loss = 0.0
        n_transitions = 0

        for pkey, items in groups.items():
            prefix = buffer.prefix_table.get(pkey)
            if prefix is None:
                continue

            gf_t = torch.tensor(
                prefix.global_feature, dtype=torch.float32, device=device
            ).unsqueeze(0)
            nf_t = torch.tensor(
                prefix.node_features, dtype=torch.float32, device=device
            ).unsqueeze(0)
            e_t = torch.tensor(prefix.edge_index, dtype=torch.int64, device=device)
            pid_t = torch.tensor([prefix.phase_id], dtype=torch.int64, device=device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                v_pred, ctx = agent.encode_prefix(gf_t, nf_t, e_t, pid_t)

            B_g = len(items)
            max_A = max(len(it["action_features"]) for it in items)
            padded_actions = torch.zeros(
                (B_g, max_A, 8), dtype=torch.float32, device=device
            )
            padded_pid = torch.zeros((B_g, max_A), dtype=torch.int64, device=device)
            padded_pis = torch.zeros((B_g, max_A), dtype=torch.float32, device=device)
            action_mask = torch.zeros((B_g, max_A), dtype=torch.bool, device=device)

            for i, it in enumerate(items):
                A_len = len(it["action_features"])
                dim_feat = min(7, it["action_features"].shape[1])
                padded_actions[i, :A_len, 1 : 1 + dim_feat] = torch.tensor(
                    it["action_features"][:, :dim_feat],
                    dtype=torch.float32,
                    device=device,
                )
                padded_actions[i, :A_len, 0] = torch.arange(
                    A_len, dtype=torch.float32, device=device
                )
                padded_pid[i, :A_len] = it.get("phase_id", prefix.phase_id)
                padded_pis[i, :A_len] = torch.tensor(
                    it["pis"], dtype=torch.float32, device=device
                )
                action_mask[i, :A_len] = True

            batched_ctx = ctx.expand(B_g, -1, -1)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = agent.evaluate_actions(
                    padded_actions, padded_pid, context=batched_ctx
                )

            logits = logits.masked_fill(~action_mask, -float("inf"))
            log_probs = F.log_softmax(logits.float(), dim=1)

            loss_matrix = torch.where(
                padded_pis > 0, padded_pis * log_probs, torch.zeros_like(log_probs)
            )
            p_loss = -loss_matrix.sum(dim=1).mean()

            zs = torch.tensor(
                [it["z"] for it in items], dtype=torch.float32, device=device
            )
            v_loss = F.mse_loss(v_pred.float().expand_as(zs), zs, reduction="mean")

            group_loss = p_loss + v_loss
            group_loss.backward()

            total_loss += float(group_loss.detach().item()) * B_g
            n_transitions += B_g

        if n_transitions > 0:
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()

            batches_processed += 1
            avg_loss = total_loss / n_transitions
            print(
                f"[Learner] Batch {batches_processed:04d} | Total BufSize: {len(buffer)} (Prefixes: {len(buffer.prefix_table)}) | Loss: {avg_loss:.4f}"
            )

            with open(losses_bin_path, "ab") as f_bin:
                f_bin.write(struct.pack(pack_fmt, batches_processed, avg_loss))
                f_bin.flush()

            if batches_processed % config.save_interval == 0:
                cpu_state_dict = {k: v.cpu() for k, v in agent.state_dict().items()}
                with weights_lock:
                    global_weights.update(cpu_state_dict)
                    global_version = batches_processed
                weights_ready_event.set()
                save_file(cpu_state_dict, model_filepath)


def main():
    parser = argparse.ArgumentParser(
        description="AlphaZero TensorGraph Server / Learner"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="TCP listen address"
    )
    parser.add_argument("--port", type=int, default=5000, help="TCP listen port")
    parser.add_argument(
        "-bt",
        "--enable-bluetooth",
        action="store_true",
        help="Listen on Bluetooth RFCOMM",
    )
    parser.add_argument(
        "--bt-address", type=str, default="AC:F2:3C:A7:F7:EC", help="Bluetooth host MAC"
    )
    parser.add_argument(
        "--bt-port", type=int, default=4, help="Bluetooth RFCOMM channel"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Replay buffer batch size"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--run-dir", type=str, default=None, help="Path to run directory"
    )
    parser.add_argument("--resume", action="store_true", help="Resume from latest run")
    parser.add_argument(
        "--graph-source", type=str, default="model", choices=["model", "random"]
    )
    parser.add_argument("--random-min-nodes", type=int, default=10)
    parser.add_argument("--random-max-nodes", type=int, default=30)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--nhead", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--c-puct", type=float, default=1.25)
    parser.add_argument("--base-noise", type=float, default=0.25)
    parser.add_argument("--min-noise", type=float, default=0.01)
    parser.add_argument("--decay-episodes", type=int, default=500)
    parser.add_argument("--depth-gamma", type=float, default=0.7)
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="C++ threads for graph operations (default: auto)",
    )

    args = parser.parse_args()
    if args.threads > 0:
        tensor_graphs.set_num_threads(args.threads)

    run_dir = setup_run_dir(run_dir=args.run_dir, resume_latest=args.resume)
    config_file = Path(run_dir) / "config.json"

    try:
        config = TrainConfig.load(config_file)
        print(f"[Server] Loaded existing run configuration from {config_file}")
    except FileNotFoundError as e:
        print(f"[Server] Could not load existing config ({e}), creating default.")
        config = TrainConfig()

    config.run_dir = run_dir
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.host = args.host
    config.port = args.port
    config.graph_source = args.graph_source
    config.random_min_nodes = args.random_min_nodes
    config.random_max_nodes = args.random_max_nodes
    config.c_puct = args.c_puct
    config.base_noise = args.base_noise
    config.min_noise = args.min_noise
    config.decay_episodes = args.decay_episodes
    config.depth_gamma = args.depth_gamma
    config.d_model = args.d_model
    config.nhead = args.nhead
    config.num_layers = args.num_layers

    config.save(config_file)

    replay_queue = queue.Queue()
    learner_thread = threading.Thread(
        target=learner_process, args=(config, replay_queue), daemon=True
    )
    learner_thread.start()

    server_sockets = []
    tcp_sock = create_server_socket(config.host, config.port, use_bluetooth=False)
    server_sockets.append(tcp_sock)
    tcp_thread = threading.Thread(
        target=accept_loop, args=(tcp_sock, "TCP/IP", replay_queue, config), daemon=True
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
                args=(bt_sock, "Bluetooth RFCOMM", replay_queue, config),
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

import argparse
import math
import queue
import socket
import struct
import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import optim

from .buffer import TrajectoryReplayBuffer, get_eviction_strategy
from .config import TrainConfig
from .graph_provider import get_default_model_path
from .model import CostPredictorRNN
from .net import create_server_socket, recv_msg, send_msg

torch.set_float32_matmul_precision("high")

global_weights = {}
global_version = 0
weights_lock = threading.Lock()
weights_ready_event = threading.Event()


def setup_run_dir(
    base_dir: str = "runs",
    run_dir: str | None = None,
    resume_latest: bool = False,
) -> str:
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

    run_idx = max(existing) + 1 if existing else 0
    target_dir = runs_dir / str(run_idx)
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir.as_posix()


def client_handler(
    client_sock,
    client_info,
    replay_queue: queue.Queue,
    config: TrainConfig,
    shutdown_event: threading.Event | None = None,
):
    print(f"[Server] Client connected: {client_info}")
    try:
        while shutdown_event is None or not shutdown_event.is_set():
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

            elif msg_type in ("trajectories_batch", "transitions_batch"):
                data = msg.get("data")
                if data:
                    replay_queue.put(data)

    except Exception as e:
        if shutdown_event is None or not shutdown_event.is_set():
            print(f"[Server] Client {client_info} error: {e}")
    finally:
        print(f"[Server] Client disconnected: {client_info}")
        client_sock.close()


def learner_thread_fn(
    config: TrainConfig,
    replay_queue: queue.Queue,
    shutdown_event: threading.Event | None = None,
):
    global global_version
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Learner] Cost-Predictor Optimizer running on device: {device}")

    model = CostPredictorRNN(hidden_dim=config.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    losses_bin_path = run_dir / "losses.bin"
    costs_bin_path = run_dir / "costs.bin"
    model_filepath = run_dir / "model.safetensors"
    pack_fmt = "<If"

    if model_filepath.exists():
        try:
            state_dict = load_file(model_filepath)
            model.load_state_dict(state_dict, strict=False)
            print(f"[Learner] Loaded existing model from {model_filepath}")
        except Exception as e:
            print(f"[Learner] Warning: Failed to load {model_filepath}: {e}")

    batches_processed = 0
    if losses_bin_path.exists() and losses_bin_path.stat().st_size >= 8:
        try:
            with open(losses_bin_path, "rb") as f_bin:
                f_bin.seek(-8, 2)
                last_idx, _ = struct.unpack(pack_fmt, f_bin.read(8))
                batches_processed = int(last_idx)
        except Exception:
            pass

    cost_count = 0
    if costs_bin_path.exists() and costs_bin_path.stat().st_size >= 8:
        try:
            with open(costs_bin_path, "rb") as f_bin:
                f_bin.seek(-8, 2)
                last_idx, _ = struct.unpack(pack_fmt, f_bin.read(8))
                cost_count = int(last_idx)
        except Exception:
            pass

    # Publish initial weights
    cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
    with weights_lock:
        global_weights.update(cpu_state)
        global_version = batches_processed
    weights_ready_event.set()
    save_file(cpu_state, model_filepath)

    # Eviction strategy for trajectory replay buffer
    strategy = get_eviction_strategy(config.buffer_strategy)
    buffer = TrajectoryReplayBuffer(
        maxlen=config.buffer_size,
        strategy=strategy,
    )
    print(f"[Learner] Trajectory Replay Buffer Strategy: {strategy.__class__.__name__}")

    last_warmup_log = 0.0

    while shutdown_event is None or not shutdown_event.is_set():
        while not replay_queue.empty():
            try:
                data = replay_queue.get_nowait()
                leaf_costs = data.get("leaf_costs", [])
                if leaf_costs:
                    with open(costs_bin_path, "ab") as f_bin:
                        buf = bytearray()
                        for c in leaf_costs:
                            cost_val = float(c)
                            if math.isfinite(cost_val):
                                cost_count += 1
                                buf.extend(struct.pack(pack_fmt, cost_count, cost_val))
                        if buf:
                            f_bin.write(buf)

                trajectories = data.get("trajectories", [])
                if trajectories:
                    buffer.add_trajectories(trajectories)
            except queue.Empty:
                break

        if len(buffer) < config.batch_size:
            if time.time() - last_warmup_log > 3.0:
                print(
                    f"[Learner] Warming up replay buffer: {len(buffer)}/{config.batch_size} trajectories..."
                )
                last_warmup_log = time.time()
            time.sleep(0.05)
            continue

        # Sample a batch of full trajectories
        sampled = buffer.sample_batch(config.batch_size)
        if not sampled:
            time.sleep(0.01)
            continue

        batch_trajs, batch_indices = sampled

        model.train()
        optimizer.zero_grad()

        # Dynamic rollout across current parameters
        pred_costs, targets, lengths = model.unroll_trajectories(batch_trajs, device)

        per_step_loss = F.smooth_l1_loss(pred_costs, targets, reduction="none")
        total_loss = per_step_loss.mean()

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("[Learner] Warning: NaN/Inf loss detected, skipping backward step.")
            continue

        total_loss.backward()

        # Compute trajectory-level mean losses for replay buffer priority updates
        per_step_loss_np = per_step_loss.detach().cpu().numpy()
        traj_losses = []
        offset = 0
        for L in lengths:
            traj_l = float(per_step_loss_np[offset : offset + L].mean())
            traj_losses.append(traj_l)
            offset += L

        buffer.update_losses(batch_indices, traj_losses)

        has_nan_grad = False
        for p in model.parameters():
            if p.grad is not None and (
                torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
            ):
                has_nan_grad = True
                break

        if has_nan_grad:
            print(
                "[Learner] Warning: NaN/Inf gradient detected, skipping optimizer step."
            )
            optimizer.zero_grad()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batches_processed += 1
            loss_val = total_loss.item()

            print(
                f"[Learner] Batch {batches_processed:04d} | TrajBuf: {len(buffer):05d} | Loss: {loss_val:.4e}"
            )

            if math.isfinite(loss_val):
                with open(losses_bin_path, "ab") as f_bin:
                    f_bin.write(struct.pack(pack_fmt, batches_processed, loss_val))
                    f_bin.flush()

            if batches_processed % config.save_interval == 0:
                cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
                with weights_lock:
                    global_weights.update(cpu_state)
                    global_version = batches_processed
                weights_ready_event.set()
                save_file(cpu_state, model_filepath)


def main():
    parser = argparse.ArgumentParser(
        description="TensorGraph RNN Cost Predictor Training Server"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="TCP listen host")
    parser.add_argument("--port", type=int, default=5000, help="TCP listen port")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Output run directory (default: auto runs/0, runs/1, ...)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest run directory in runs/",
    )
    parser.add_argument(
        "--buffer-strategy",
        type=str,
        default="lowest_loss",
        choices=["fifo", "lowest_loss", "highest_cost"],
        help="Replay buffer eviction strategy",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=1000,
        help="Replay buffer capacity in full trajectories",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Optimization batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Optimizer learning rate"
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=64, help="RNN hidden dimension"
    )
    parser.add_argument(
        "--graph-source",
        type=str,
        default="model",
        choices=["model", "random"],
        help="Graph workload source",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemma-3-270m",
        help="Target model architecture name",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Target model weights directory",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Exploration epsilon for client workers",
    )
    args = parser.parse_args()

    run_dir = setup_run_dir(run_dir=args.run_dir, resume_latest=args.resume)
    config_file = Path(run_dir) / "config.json"

    try:
        config = TrainConfig.load(config_file)
        print(f"[Server] Loaded existing run configuration from {config_file}")
    except FileNotFoundError:
        config = TrainConfig()

    config.run_dir = run_dir
    config.host = args.host
    config.port = args.port
    config.buffer_strategy = args.buffer_strategy
    config.buffer_size = args.buffer_size
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.hidden_dim = args.hidden_dim
    config.graph_source = args.graph_source
    config.model_name = args.model
    config.model_path = (
        args.model_path
        if args.model_path is not None
        else get_default_model_path(args.model)
    )
    config.epsilon = args.epsilon

    config.save(config_file)

    replay_queue = queue.Queue()
    shutdown_event = threading.Event()

    learner_thread = threading.Thread(
        target=learner_thread_fn,
        args=(config, replay_queue, shutdown_event),
        daemon=True,
    )
    learner_thread.start()

    server_sock = create_server_socket(config.host, config.port)
    server_sock.settimeout(1.0)

    print("=========================================================")
    print(" TensorGraph RNN Cost-Predictor Training Server")
    print(f" Listening on TCP: {config.host}:{config.port}")
    print(f" Buffer Strategy: {config.buffer_strategy.upper()}")
    print(f" Run Directory: {config.run_dir}")
    print("=========================================================")

    try:
        while not shutdown_event.is_set():
            try:
                client_sock, client_info = server_sock.accept()
                threading.Thread(
                    target=client_handler,
                    args=(
                        client_sock,
                        client_info,
                        replay_queue,
                        config,
                        shutdown_event,
                    ),
                    daemon=True,
                ).start()
            except socket.timeout:
                continue
            except OSError:
                break
    except (KeyboardInterrupt, SystemExit):
        print("\n[Server] Ctrl+C received. Shutting down...")
    finally:
        shutdown_event.set()
        try:
            server_sock.close()
        except Exception:
            pass
        print("[Server] Server stopped cleanly.")


if __name__ == "__main__":
    main()

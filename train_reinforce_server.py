#!/usr/bin/env python3
# File: train_reinforce_server.py
import argparse
import math
import queue
import struct
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensor_graphs
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import optim
from tqdm import tqdm

from train_models import PolicyValueRNN
from train_shared import (
    RNNEpisode,
    TrainConfig,
    create_server_socket,
    get_default_model_path,
    recv_msg,
    send_msg,
)

torch.set_float32_matmul_precision("high")

global_weights = {}
global_version = 0
weights_lock = threading.Lock()
weights_ready_event = threading.Event()


def setup_run_dir(base_dir="runs", run_dir=None, resume_latest=False) -> str:
    runs_dir = Path(base_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    if run_dir:
        target_dir = Path(run_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir.as_posix()

    existing = [int(d.name) for d in runs_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    if resume_latest and existing:
        target_dir = runs_dir / str(max(existing))
        return target_dir.as_posix()

    run_idx = max(existing) + 1 if existing else 1
    target_dir = runs_dir / str(run_idx)
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir.as_posix()


def client_handler(client_sock, client_info, trajectory_queue, config: TrainConfig):
    print(f"[Server] Worker connected: {client_info}")
    try:
        while True:
            msg = recv_msg(client_sock)
            if not msg:
                break

            msg_type = msg.get("type")
            if msg_type == "req_config":
                send_msg(client_sock, {"type": "config", "config": config.to_dict()})

            elif msg_type == "req_version":
                weights_ready_event.wait(timeout=60.0)
                with weights_lock:
                    send_msg(client_sock, {"type": "version", "version": global_version})

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

            elif msg_type == "episodes":
                episodes = msg.get("episodes", [])
                for ep in episodes:
                    trajectory_queue.put(ep)

    except Exception as e:
        print(f"[Server] Worker {client_info} error: {e}")
    finally:
        print(f"[Server] Worker disconnected: {client_info}")
        client_sock.close()


def accept_loop(server_sock, conn_type, trajectory_queue, config: TrainConfig):
    try:
        while True:
            sock, info = server_sock.accept()
            threading.Thread(
                target=client_handler,
                args=(sock, info, trajectory_queue, config),
                daemon=True,
            ).start()
    except Exception as e:
        print(f"[Server] {conn_type} listener stopped: {e}")


def reinforce_learner_thread(
    config: TrainConfig,
    trajectory_queue: queue.Queue,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
):
    global global_version
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Learner] REINFORCE Optimizer running on: {device}")

    model = PolicyValueRNN(hidden_dim=config.d_model, global_dim=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    run_dir = Path(config.run_dir)
    losses_bin = run_dir / "losses.bin"
    costs_bin = run_dir / "costs.bin"
    model_file = run_dir / "model.safetensors"
    pack_fmt = "<If"

    if model_file.exists():
        try:
            state_dict = load_file(model_file)
            model.load_state_dict(state_dict, strict=False)
            print(f"[Learner] Resumed model weights from {model_file}")
        except Exception as e:
            print(f"[Learner] Warning: Could not load {model_file}: {e}")

    cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    with weights_lock:
        global_weights.update(cpu_state_dict)
        global_version = 0
    weights_ready_event.set()
    save_file(cpu_state_dict, model_file)

    batch_steps = 0
    episode_count = 0
    episodes_buffer: list[RNNEpisode] = []

    while True:
        try:
            ep = trajectory_queue.get(timeout=0.1)
            episodes_buffer.append(ep)
            episode_count += 1
            if ep.cost < float("inf"):
                with open(costs_bin, "ab") as f:
                    f.write(struct.pack(pack_fmt, episode_count, ep.cost))
                    f.flush()
        except queue.Empty:
            pass

        if len(episodes_buffer) < config.batch_size:
            time.sleep(0.05)
            continue

        batch = episodes_buffer[: config.batch_size]
        episodes_buffer = episodes_buffer[config.batch_size :]

        total_batch_transitions = sum(len(ep.transitions) for ep in batch)
        if total_batch_transitions == 0:
            continue

        # 1. Flatten all transitions and group by phase_id for batching
        phase_groups = defaultdict(list)
        total_costs = []

        for ep in batch:
            if ep.cost < float("inf"):
                total_costs.append(ep.cost)
            # Terminal reward target
            target_return = float(ep.reward)
            for tr in ep.transitions:
                phase_groups[tr.phase_id].append((tr, target_return))

        model.train()
        optimizer.zero_grad()

        total_loss = torch.tensor(0.0, device=device)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_transitions = 0

        with tqdm(
            total=total_batch_transitions,
            desc=f"Batch {batch_steps + 1:04d}",
            unit="step",
            leave=False,
        ) as pbar:
            for phase_id, items in phase_groups.items():
                B_p = len(items)
                if B_p == 0:
                    continue

                max_A = max(len(tr.action_feats) for tr, _ in items)
                if max_A == 0:
                    pbar.update(B_p)
                    continue

                # Prepare batched tensors
                feat_dim = items[0][0].action_feats.shape[-1]
                h_batch = torch.tensor(
                    np.stack([tr.hidden for tr, _ in items]),
                    dtype=torch.float32,
                    device=device,
                )
                g_batch = torch.tensor(
                    np.stack([tr.global_feat for tr, _ in items]),
                    dtype=torch.float32,
                    device=device,
                )
                returns_batch = torch.tensor(
                    [ret for _, ret in items],
                    dtype=torch.float32,
                    device=device,
                )
                chosen_indices = torch.tensor(
                    [tr.chosen_idx for tr, _ in items],
                    dtype=torch.int64,
                    device=device,
                )

                padded_actions = torch.zeros((B_p, max_A, feat_dim), dtype=torch.float32, device=device)
                action_mask = torch.zeros((B_p, max_A), dtype=torch.bool, device=device)

                for i, (tr, _) in enumerate(items):
                    A_len = len(tr.action_feats)
                    padded_actions[i, :A_len] = torch.tensor(
                        tr.action_feats, dtype=torch.float32, device=device
                    )
                    action_mask[i, :A_len] = True

                # Vectorized forward pass
                logits, values = model.evaluate_candidates(h_batch, g_batch, padded_actions, phase_id)
                values = values.squeeze(-1)  # [B_p]

                # Mask invalid action logits
                masked_logits = logits.masked_fill(~action_mask, -1e9)
                log_probs = F.log_softmax(masked_logits, dim=-1)
                probs = F.softmax(masked_logits, dim=-1)

                selected_log_probs = log_probs.gather(1, chosen_indices.unsqueeze(-1)).squeeze(-1)

                # Normalized Advantage Calculation
                advantages = (returns_batch - values.detach())
                if B_p > 1:
                    adv_std = advantages.std()
                    if not torch.isnan(adv_std) and adv_std > 1e-6:
                        advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)
                advantages = torch.clamp(advantages, -5.0, 5.0)

                # Losses
                policy_loss = -(selected_log_probs * advantages).sum()
                value_loss = F.smooth_l1_loss(values, returns_batch, reduction="sum")
                
                # Entropy
                safe_probs = torch.where(action_mask, probs, torch.zeros_like(probs))
                safe_log_probs = torch.where(action_mask, log_probs, torch.zeros_like(log_probs))
                entropy_loss = -(safe_probs * safe_log_probs).sum()

                phase_total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss

                phase_total_loss.backward()

                total_loss = total_loss + phase_total_loss.detach()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_transitions += B_p
                pbar.update(B_p)

        if total_transitions > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_steps += 1
            avg_c = np.mean(total_costs) if total_costs else float("inf")
            min_c = np.min(total_costs) if total_costs else float("inf")
            avg_loss = total_loss.item() / total_transitions
            avg_p_loss = total_policy_loss / total_transitions
            avg_v_loss = total_value_loss / total_transitions

            tqdm.write(
                f"[Learner] Batch {batch_steps:04d} | Ep: {episode_count:04d} | Steps: {total_transitions:05d} "
                f"| Loss: {avg_loss:.4f} (P: {avg_p_loss:.3f}, V: {avg_v_loss:.3f}) | Avg Cost: {avg_c:.3f} ms | Best: {min_c:.3f} ms"
            )

            with open(losses_bin, "ab") as f_bin:
                f_bin.write(struct.pack(pack_fmt, batch_steps, avg_loss))
                f_bin.flush()

            if batch_steps % config.save_interval == 0:
                cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
                with weights_lock:
                    global_weights.update(cpu_state)
                    global_version = batch_steps
                weights_ready_event.set()
                save_file(cpu_state, model_file)


def main():
    parser = argparse.ArgumentParser(description="REINFORCE TensorGraph Optimizer Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("-bt", "--enable-bluetooth", action="store_true")
    parser.add_argument("--bt-address", type=str, default="AC:F2:3C:A7:F7:EC")
    parser.add_argument("--bt-port", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16, help="Episodes per update batch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model", type=str, default="gemma-3-270m")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--graph-source", type=str, default="model", choices=["model", "random"])
    parser.add_argument("--random-min-nodes", type=int, default=10)
    parser.add_argument("--random-max-nodes", type=int, default=30)
    parser.add_argument("--threads", type=int, default=0)

    args = parser.parse_args()
    if args.threads > 0:
        tensor_graphs.set_num_threads(args.threads)

    run_dir = setup_run_dir(run_dir=args.run_dir, resume_latest=args.resume)
    config_file = Path(run_dir) / "config.json"

    try:
        config = TrainConfig.load(config_file)
    except FileNotFoundError:
        config = TrainConfig()

    config.run_dir = run_dir
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.d_model = args.hidden_dim
    config.host = args.host
    config.port = args.port
    config.graph_source = args.graph_source
    config.random_min_nodes = args.random_min_nodes
    config.random_max_nodes = args.random_max_nodes
    if args.model is not None:
        config.model_name = args.model
        if args.model_path is None:
            config.model_path = get_default_model_path(args.model)
    if args.model_path is not None:
        config.model_path = args.model_path

    config.save(config_file)

    trajectory_queue = queue.Queue()
    threading.Thread(
        target=reinforce_learner_thread,
        args=(config, trajectory_queue),
        daemon=True,
    ).start()

    server_sockets = []
    tcp_sock = create_server_socket(config.host, config.port, use_bluetooth=False)
    server_sockets.append(tcp_sock)
    threading.Thread(
        target=accept_loop,
        args=(tcp_sock, "TCP/IP", trajectory_queue, config),
        daemon=True,
    ).start()

    print("=========================================================")
    print(f" REINFORCE Server listening on {config.host}:{config.port}")
    print(f" Model: {config.model_name} | Run Dir: {config.run_dir}")
    print("=========================================================")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping REINFORCE Server.")
    finally:
        for s in server_sockets:
            s.close()


if __name__ == "__main__":
    main()
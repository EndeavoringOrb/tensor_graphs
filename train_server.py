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

from train_models import AlphaZeroTransformer, PolicyValueRNN
from train_shared import (
    PrefixData,
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


class UnifiedReplayBuffer:
    def __init__(self, maxlen: int, alpha: float = 0.6, eps: float = 1e-4):
        self.maxlen = maxlen
        self.alpha = alpha
        self.eps = eps
        self.buffer: list[dict] = []
        self.prefix_table: dict[int, PrefixData] = {}
        self.max_priority = 1.0

    def extend_payload(self, payload: dict):
        if "prefixes" in payload:
            self.prefix_table.update(payload["prefixes"])
        if "transitions" in payload:
            for item in payload["transitions"]:
                if "loss" not in item:
                    item["loss"] = None
                self.buffer.append(item)
            self._evict_excess()

    def _evict_excess(self):
        if len(self.buffer) > self.maxlen:
            num_to_evict = len(self.buffer) - self.maxlen
            # Sorting key: (has_no_loss, loss_value)
            # (0, loss_value): items with loss, sorted ascending by lowest loss.
            # (1, 0.0): items with no loss (None), preserved over items with loss.
            self.buffer.sort(
                key=lambda x: (
                    (1, 0.0) if x.get("loss") is None else (0, float(x["loss"]))
                )
            )
            # Evict the N items with the lowest loss
            self.buffer = self.buffer[num_to_evict:]

            # Clean up unreferenced prefixes
            used_keys = {item["prefix_key"] for item in self.buffer}
            self.prefix_table = {
                k: v for k, v in self.prefix_table.items() if k in used_keys
            }

    def sample_batch(self, batch_size: int):
        n = len(self.buffer)
        if n == 0:
            return []
        batch_size = min(batch_size, n)

        # Compute PER sampling probabilities
        priorities = np.empty(n, dtype=np.float64)
        for idx, item in enumerate(self.buffer):
            loss = item.get("loss")
            if loss is None:
                priorities[idx] = self.max_priority
            else:
                priorities[idx] = float(loss) + self.eps

        scaled_priorities = priorities**self.alpha
        total_p = scaled_priorities.sum()
        if total_p <= 0 or np.isnan(total_p):
            probs = np.ones(n, dtype=np.float64) / n
        else:
            probs = scaled_priorities / total_p

        indices = np.random.choice(n, size=batch_size, replace=False, p=probs)
        return [self.buffer[i] for i in indices]

    def update_priority(self, loss_val: float):
        if loss_val > self.max_priority:
            self.max_priority = float(loss_val)

    def __len__(self):
        return len(self.buffer)


class RNNReplayBuffer:
    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self.buffer: list[dict] = []

    def extend(self, transitions: list[dict]):
        for t in transitions:
            self.buffer.append(t)
        if len(self.buffer) > self.maxlen:
            self.buffer = self.buffer[-self.maxlen :]

    def sample_batch(self, batch_size: int):
        if not self.buffer:
            return []
        batch_size = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[i] for i in indices]

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

            elif msg_type == "episodes":
                cost = msg.get("cost", float("inf"))
                costs = msg.get("costs", [])
                if not costs and cost < float("inf"):
                    costs = [cost]

                for c in costs:
                    replay_queue.put({"type": "cost_metric", "cost": float(c)})

                episodes = msg.get("episodes", [])
                if episodes:
                    replay_queue.put({"type": "episodes", "episodes": episodes})

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


def rnn_mcts_learner_process(config: TrainConfig, replay_queue: queue.Queue):
    global global_version
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[Learner] RNN Search Optimizer ({config.algo.upper()}) running on device: {device}"
    )

    model = PolicyValueRNN(hidden_dim=config.d_model, global_dim=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    run_dir_path = Path(config.run_dir)
    run_dir_path.mkdir(parents=True, exist_ok=True)
    losses_bin_path = run_dir_path / "losses.bin"
    costs_bin_path = run_dir_path / "costs.bin"
    model_filepath = run_dir_path / "model.safetensors"
    pack_fmt = "<If"

    if model_filepath.exists():
        try:
            state_dict = load_file(model_filepath)
            model.load_state_dict(state_dict, strict=False)
            model.to(device)
            print(f"[Learner] Loaded existing model weights from {model_filepath}")
        except Exception as e:
            print(f"[Learner] Warning: Failed to load {model_filepath}: {e}")

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

    cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    with weights_lock:
        global_weights.update(cpu_state_dict)
        global_version = batches_processed
    weights_ready_event.set()
    save_file(cpu_state_dict, model_filepath)

    buffer = RNNReplayBuffer(maxlen=config.replay_buffer_size)

    while True:
        while not replay_queue.empty():
            try:
                item = replay_queue.get_nowait()
                if isinstance(item, dict):
                    if item.get("type") == "cost_metric":
                        cost_count += 1
                        cost_val = float(item["cost"])
                        with open(costs_bin_path, "ab") as f_bin:
                            f_bin.write(struct.pack(pack_fmt, cost_count, cost_val))
                            f_bin.flush()
                    elif item.get("type") == "trajectory_payload":
                        payload = item.get("payload", {})
                        if "transitions" in payload:
                            buffer.extend(payload["transitions"])
            except queue.Empty:
                break

        if len(buffer) < config.batch_size:
            time.sleep(0.05)
            continue

        raw_batch = buffer.sample_batch(config.batch_size)
        phase_groups = defaultdict(list)
        for tr in raw_batch:
            phase_groups[tr.get("phase_id", 0)].append(tr)

        model.train()
        optimizer.zero_grad()

        total_loss = torch.tensor(0.0, device=device)
        total_p_loss = 0.0
        total_v_loss = 0.0
        total_transitions = 0

        for phase_id, items in phase_groups.items():
            B_p = len(items)
            if B_p == 0:
                continue

            max_A = max(len(it["action_features"]) for it in items)
            if max_A == 0:
                continue

            feat_dim = items[0]["action_features"].shape[-1]
            default_h = np.zeros(config.d_model, dtype=np.float32)
            default_g = np.zeros(8, dtype=np.float32)

            h_batch = torch.tensor(
                np.stack([it.get("hidden", default_h) for it in items]),
                dtype=torch.float32,
                device=device,
            )
            g_batch = torch.tensor(
                np.stack([it.get("global_feat", default_g) for it in items]),
                dtype=torch.float32,
                device=device,
            )
            z_batch = torch.tensor(
                [it.get("z", 0.0) for it in items],
                dtype=torch.float32,
                device=device,
            )

            padded_actions = torch.zeros(
                (B_p, max_A, feat_dim), dtype=torch.float32, device=device
            )
            padded_pis = torch.zeros((B_p, max_A), dtype=torch.float32, device=device)
            action_mask = torch.zeros((B_p, max_A), dtype=torch.bool, device=device)

            for i, it in enumerate(items):
                A_len = len(it["action_features"])
                padded_actions[i, :A_len] = torch.tensor(
                    it["action_features"], dtype=torch.float32, device=device
                )
                padded_pis[i, :A_len] = torch.tensor(
                    it["pis"], dtype=torch.float32, device=device
                )
                action_mask[i, :A_len] = True

            logits, values = model.evaluate_candidates(
                h_batch, g_batch, padded_actions, phase_id
            )
            values = values.squeeze(-1)

            masked_logits = logits.masked_fill(~action_mask, -1e9)
            log_probs = F.log_softmax(masked_logits, dim=-1)

            safe_log_probs = torch.where(
                action_mask, log_probs, torch.zeros_like(log_probs)
            )
            policy_loss = -(padded_pis * safe_log_probs).sum(dim=-1).mean()
            value_loss = F.smooth_l1_loss(values, z_batch)

            phase_loss = policy_loss + getattr(config, "value_coef", 0.5) * value_loss
            phase_loss.backward()

            total_loss = total_loss + phase_loss.detach() * B_p
            total_p_loss += policy_loss.item() * B_p
            total_v_loss += value_loss.item() * B_p
            total_transitions += B_p

        if total_transitions > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batches_processed += 1
            avg_loss = total_loss.item() / total_transitions
            avg_p = total_p_loss / total_transitions
            avg_v = total_v_loss / total_transitions

            print(
                f"[Learner] Batch {batches_processed:04d} | BufSize: {len(buffer):05d} "
                f"| Loss: {avg_loss:.4f} (Policy: {avg_p:.3f}, Value: {avg_v:.3f})"
            )

            with open(losses_bin_path, "ab") as f_bin:
                f_bin.write(struct.pack(pack_fmt, batches_processed, avg_loss))
                f_bin.flush()

            if batches_processed % config.save_interval == 0:
                cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
                with weights_lock:
                    global_weights.update(cpu_state)
                    global_version = batches_processed
                weights_ready_event.set()
                save_file(cpu_state, model_filepath)


def rnn_reinforce_learner_process(config: TrainConfig, replay_queue: queue.Queue):
    global global_version
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Learner] RNN REINFORCE/PPO Optimizer running on device: {device}")

    model = PolicyValueRNN(hidden_dim=config.d_model, global_dim=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    run_dir_path = Path(config.run_dir)
    run_dir_path.mkdir(parents=True, exist_ok=True)
    losses_bin_path = run_dir_path / "losses.bin"
    costs_bin_path = run_dir_path / "costs.bin"
    model_filepath = run_dir_path / "model.safetensors"
    pack_fmt = "<If"

    if model_filepath.exists():
        try:
            state_dict = load_file(model_filepath)
            model.load_state_dict(state_dict, strict=False)
            model.to(device)
            print(f"[Learner] Loaded existing model weights from {model_filepath}")
        except Exception as e:
            print(f"[Learner] Warning: Failed to load {model_filepath}: {e}")

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

    cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    with weights_lock:
        global_weights.update(cpu_state_dict)
        global_version = batches_processed
    weights_ready_event.set()
    save_file(cpu_state_dict, model_filepath)

    episodes_buffer: list[RNNEpisode] = []
    episode_count = 0

    while True:
        while not replay_queue.empty():
            try:
                item = replay_queue.get_nowait()
                if isinstance(item, dict):
                    if item.get("type") == "cost_metric":
                        cost_count += 1
                        cost_val = float(item["cost"])
                        with open(costs_bin_path, "ab") as f_bin:
                            f_bin.write(struct.pack(pack_fmt, cost_count, cost_val))
                            f_bin.flush()
                    elif item.get("type") == "episodes":
                        for ep in item.get("episodes", []):
                            episodes_buffer.append(ep)
                            episode_count += 1
                            if ep.cost < float("inf"):
                                cost_count += 1
                                with open(costs_bin_path, "ab") as f_bin:
                                    f_bin.write(
                                        struct.pack(
                                            pack_fmt, cost_count, float(ep.cost)
                                        )
                                    )
                                    f_bin.flush()
            except queue.Empty:
                break

        if len(episodes_buffer) < config.batch_size:
            time.sleep(0.05)
            continue

        batch = episodes_buffer[: config.batch_size]
        episodes_buffer = episodes_buffer[config.batch_size :]

        total_batch_transitions = sum(len(ep.transitions) for ep in batch)
        if total_batch_transitions == 0:
            continue

        phase_groups = defaultdict(list)
        total_costs = []

        for ep in batch:
            if ep.cost < float("inf"):
                total_costs.append(ep.cost)
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

        for phase_id, items in phase_groups.items():
            B_p = len(items)
            if B_p == 0:
                continue

            max_A = max(len(tr.action_feats) for tr, _ in items)
            if max_A == 0:
                continue

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
            old_log_probs = torch.tensor(
                [tr.log_prob for tr, _ in items],
                dtype=torch.float32,
                device=device,
            )

            padded_actions = torch.zeros(
                (B_p, max_A, feat_dim), dtype=torch.float32, device=device
            )
            action_mask = torch.zeros((B_p, max_A), dtype=torch.bool, device=device)

            for i, (tr, _) in enumerate(items):
                A_len = len(tr.action_feats)
                padded_actions[i, :A_len] = torch.tensor(
                    tr.action_feats, dtype=torch.float32, device=device
                )
                action_mask[i, :A_len] = True

            logits, values = model.evaluate_candidates(
                h_batch, g_batch, padded_actions, phase_id
            )
            values = values.squeeze(-1)

            masked_logits = logits.masked_fill(~action_mask, -1e9)
            log_probs = F.log_softmax(masked_logits, dim=-1)
            probs = F.softmax(masked_logits, dim=-1)

            selected_log_probs = log_probs.gather(
                1, chosen_indices.unsqueeze(-1)
            ).squeeze(-1)

            # Normalized Advantage Calculation
            advantages = returns_batch - values.detach()
            if B_p > 1:
                adv_std = advantages.std()
                if not torch.isnan(adv_std) and adv_std > 1e-6:
                    advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)
            advantages = torch.clamp(advantages, -5.0, 5.0)

            # PPO Clipped Surrogate Objective
            ratio = torch.exp(selected_log_probs - old_log_probs)
            clip_eps = getattr(config, "clip_eps", 0.2)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).sum()

            value_loss = F.smooth_l1_loss(values, returns_batch, reduction="sum")

            # Entropy Regularization
            safe_probs = torch.where(action_mask, probs, torch.zeros_like(probs))
            safe_log_probs = torch.where(
                action_mask, log_probs, torch.zeros_like(log_probs)
            )
            entropy_loss = -(safe_probs * safe_log_probs).sum()

            phase_total_loss = (
                policy_loss
                + getattr(config, "value_coef", 0.5) * value_loss
                - getattr(config, "entropy_coef", 0.01) * entropy_loss
            )

            phase_total_loss.backward()

            total_loss = total_loss + phase_total_loss.detach()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()
            total_transitions += B_p

        if total_transitions > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batches_processed += 1
            avg_c = np.mean(total_costs) if total_costs else float("inf")
            min_c = np.min(total_costs) if total_costs else float("inf")
            avg_loss = total_loss.item() / total_transitions
            avg_p_loss = total_policy_loss / total_transitions
            avg_v_loss = total_value_loss / total_transitions

            print(
                f"[Learner] Batch {batches_processed:04d} | Ep: {episode_count:04d} | Steps: {total_transitions:05d} "
                f"| Loss: {avg_loss:.4f} (P: {avg_p_loss:.3f}, V: {avg_v_loss:.3f}) | Avg Cost: {avg_c:.3f} ms | Best: {min_c:.3f} ms"
            )

            with open(losses_bin_path, "ab") as f_bin:
                f_bin.write(struct.pack(pack_fmt, batches_processed, avg_loss))
                f_bin.flush()

            if batches_processed % config.save_interval == 0:
                cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
                with weights_lock:
                    global_weights.update(cpu_state)
                    global_version = batches_processed
                weights_ready_event.set()
                save_file(cpu_state, model_filepath)


def transformer_mcts_learner_process(config: TrainConfig, replay_queue: queue.Queue):
    global global_version
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[Learner] AlphaZero Transformer Search Optimizer ({config.algo.upper()}) running on device: {device}"
    )

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
                elif item.get("type") == "trajectory_payload":
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

            if len(prefix.node_features) == 0:
                dummy_nf = np.zeros((1, 8), dtype=np.float32)
                dummy_e = np.zeros((2, 0), dtype=np.int64)
            else:
                dummy_nf = prefix.node_features
                dummy_e = prefix.edge_index

            gf_t = torch.tensor(
                prefix.global_feature, dtype=torch.float32, device=device
            ).unsqueeze(0)
            nf_t = torch.tensor(dummy_nf, dtype=torch.float32, device=device).unsqueeze(
                0
            )
            e_t = torch.tensor(dummy_e, dtype=torch.int64, device=device)
            pid_t = torch.tensor([prefix.phase_id], dtype=torch.int64, device=device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                v_pred, ctx = agent.encode_prefix(gf_t, nf_t, e_t, pid_t)

            B_g = len(items)
            max_A = max(len(it["action_features"]) for it in items)
            if max_A == 0:
                for it in items:
                    it["loss"] = 0.0
                continue

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

            logits_f32 = logits.float()
            logits_f32 = logits_f32.masked_fill(~action_mask, -1e4)
            log_probs = F.log_softmax(logits_f32, dim=1)

            safe_log_probs = torch.where(
                action_mask, log_probs, torch.zeros_like(log_probs)
            )
            per_item_p_loss = -(padded_pis * safe_log_probs).sum(dim=1)
            p_loss = per_item_p_loss.mean()

            zs = torch.tensor(
                [it["z"] for it in items], dtype=torch.float32, device=device
            )
            v_preds = v_pred.float().view(-1).expand_as(zs)

            per_item_v_loss = F.smooth_l1_loss(v_preds, zs, reduction="none")
            v_loss = per_item_v_loss.mean()

            group_loss = p_loss + v_loss

            if torch.isnan(group_loss) or torch.isinf(group_loss):
                print(
                    f"[Learner] Warning: NaN/Inf loss encountered for prefix {pkey}, neutralizing group."
                )
                for it in items:
                    it["loss"] = 0.0
                continue

            group_loss.backward()

            per_item_loss = (per_item_p_loss + per_item_v_loss).detach().cpu().tolist()
            for it, l_val in zip(items, per_item_loss):
                if not (math.isnan(l_val) or math.isinf(l_val)):
                    it["loss"] = float(l_val)
                    buffer.update_priority(float(l_val))
                else:
                    it["loss"] = 0.0

            total_loss += float(group_loss.detach().item()) * B_g
            n_transitions += B_g

        if n_transitions > 0:
            has_nan_grad = False
            for param in agent.parameters():
                if param.grad is not None and (
                    torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
                ):
                    has_nan_grad = True
                    break

            if has_nan_grad:
                print(
                    "[Learner] Warning: NaN/Inf gradient detected, skipping optimizer step."
                )
                optimizer.zero_grad()
            else:
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


def learner_process(config: TrainConfig, replay_queue: queue.Queue):
    model_type = getattr(config, "model_type", "rnn").lower()
    algo = getattr(config, "algo", "gumbel_alphazero").lower().replace("-", "_")

    if model_type == "rnn":
        if algo in ["reinforce", "ppo", "rnn"]:
            rnn_reinforce_learner_process(config, replay_queue)
        else:
            rnn_mcts_learner_process(config, replay_queue)
    else:  # "transformer"
        if algo in ["reinforce", "ppo", "rnn"]:
            # Transformer reinforce also uses rnn_reinforce/ppo protocol if episodes are emitted
            rnn_reinforce_learner_process(config, replay_queue)
        else:
            transformer_mcts_learner_process(config, replay_queue)


def main():
    parser = argparse.ArgumentParser(
        description="Modular TensorGraph Optimization Server (AlphaZero / Gumbel / REINFORCE x RNN / Transformer)"
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="gumbel_alphazero",
        choices=["gumbel_alphazero", "alphazero", "reinforce", "gumbel", "az", "ppo"],
        help="Optimization algorithm (default: gumbel_alphazero)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="rnn",
        choices=["rnn", "transformer"],
        help="Neural model architecture (default: rnn)",
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
        "--bt-address",
        type=str,
        default="AC:F2:3C:A7:F7:EC",
        help="Bluetooth host MAC",
    )
    parser.add_argument(
        "--bt-port", type=int, default=4, help="Bluetooth RFCOMM channel"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Replay buffer or episode batch size"
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
    parser.add_argument("--depth-gamma", type=float, default=0.99)
    parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument(
        "--value-coef", type=float, default=0.5, help="Value loss coefficient"
    )
    parser.add_argument(
        "--entropy-coef", type=float, default=0.01, help="Entropy coefficient"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="C++ threads for graph operations (default: auto)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to train on (e.g. gemma-3-270m, qwen-3.6-35b-a3b, krea, vae, qwen3-vl-bf16)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model file or directory",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="LLM model sequence length (e.g. gemma-3-270m; defaults to saved config or 8)",
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

    norm_algo = args.algo.lower().replace("-", "_")
    if norm_algo in ["az", "puct"]:
        norm_algo = "alphazero"
    elif norm_algo in ["gumbel"]:
        norm_algo = "gumbel_alphazero"
    elif norm_algo in ["ppo", "rnn"]:
        norm_algo = "reinforce"

    config.algo = norm_algo
    config.model_type = args.model_type.lower()
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
    config.clip_eps = args.clip_eps
    config.value_coef = args.value_coef
    config.entropy_coef = args.entropy_coef

    if args.model is not None:
        config.model_name = args.model
        if args.model_path is None:
            config.model_path = get_default_model_path(args.model)
    if args.model_path is not None:
        config.model_path = args.model_path
    if args.seq_len is not None:
        config.seq_len = args.seq_len

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
    print(f" Algorithm: {config.algo.upper()}")
    print(f" Model Architecture: {config.model_type.upper()}")
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

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["KMP_AFFINITY"] = "none"

import argparse
import collections
import dataclasses
import json
import random
import struct
import sys
import traceback
from pathlib import Path
from queue import Empty

import psutil
import tensor_graphs
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn, optim


@dataclasses.dataclass
class TrainConfig:
    run_dir: str = ""
    model_name: str = "gemma-3-270m"
    model_path: str = "models/google/gemma-3-270m"
    workers: int = 4
    num_simulations: int = 30  # Search Method: MCTS-style simulations per episode
    replay_buffer_size: int = 50000  # Architecture: Central Replay Buffer
    batch_size: int = 1024
    save_interval: int = 20  # Save / Sync weights every N batches
    hidden_dim: int = 64
    lr: float = 1e-3
    log_cost_calls: bool = False


class GNNModel(nn.Module):
    def __init__(self, in_features, hidden_dim=64):
        super().__init__()
        self.node_emb = nn.Linear(in_features, hidden_dim)

        self.msg_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, node_features, edge_src, edge_dst):
        x = F.gelu(self.node_emb(node_features))

        if len(edge_dst) > 0:
            src_x = x[edge_src]
            dst_x = x[edge_dst]
            msg = self.msg_net(torch.cat([src_x, dst_x], dim=-1))
            aggr_msg = torch.zeros_like(x)
            aggr_msg.index_add_(0, edge_dst, msg)
            x = x + self.update_net(torch.cat([x, aggr_msg], dim=-1))

        # Global graph state
        return x.mean(dim=0)


class DecisionModel(nn.Module):
    """
    Standard AlphaZero dual-head architecture.
    Evaluates options purely based on current global graph state and option features.
    Removes the RNN requirement, allowing i.i.d. random sampling from the Replay Buffer.
    """

    def __init__(self, global_dim, feature_dim, hidden_dim=64):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(global_dim + feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.value = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_state, options_features):
        N = options_features.size(0)
        global_expanded = global_state.unsqueeze(0).expand(N, -1)

        # Policy Head (p)
        policy_in = torch.cat([global_expanded, options_features], dim=1)
        scores = self.policy(policy_in).squeeze(1)  # (N,)

        # Value Head (v)
        val = self.value(global_state.unsqueeze(0)).squeeze(0)  # (1,)

        return scores, val


class AlphaZeroAgent(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.extract_gnn = GNNModel(in_features=4, hidden_dim=hidden_dim)
        self.extract_dec = DecisionModel(
            global_dim=hidden_dim, feature_dim=6, hidden_dim=hidden_dim
        )

        self.dispatch_gnn = GNNModel(in_features=4, hidden_dim=hidden_dim)
        self.dispatch_dec = DecisionModel(
            global_dim=hidden_dim, feature_dim=6, hidden_dim=hidden_dim
        )

        self.bufferize_gnn = GNNModel(in_features=5, hidden_dim=hidden_dim)
        self.bufferize_dec = DecisionModel(
            global_dim=hidden_dim, feature_dim=4, hidden_dim=hidden_dim
        )

        self.malloc_gnn = GNNModel(in_features=3, hidden_dim=hidden_dim)
        self.malloc_dec = DecisionModel(
            global_dim=hidden_dim, feature_dim=3, hidden_dim=hidden_dim
        )


class ActorDelegate(tensor_graphs.SearchDelegate):
    def __init__(self, agent, exploration_noise=0.25):
        super().__init__()
        self.agent = agent
        self.exploration_noise = exploration_noise
        self.trajectory = []
        self.globals = {}

    def _prepare_graphs(self, node_features, edge_src, edge_dst, feat_dim):
        if not node_features:
            return None, None, None
        nf = torch.tensor(node_features, dtype=torch.float32).view(-1, feat_dim)
        nf = torch.nan_to_num(nf, posinf=1e9, neginf=-1e9)
        src = torch.tensor(edge_src, dtype=torch.int64)
        dst = torch.tensor(edge_dst, dtype=torch.int64)
        return nf, src, dst

    def init_egraph(self, node_features, edge_src, edge_dst):
        nf, src, dst = self._prepare_graphs(node_features, edge_src, edge_dst, 4)
        if nf is not None:
            self.globals["extract_dec"] = self.agent.extract_gnn(nf, src, dst)

    def init_dispatch_graph(self, node_features, edge_src, edge_dst):
        nf, src, dst = self._prepare_graphs(node_features, edge_src, edge_dst, 4)
        if nf is not None:
            self.globals["dispatch_dec"] = self.agent.dispatch_gnn(nf, src, dst)

    def init_bufferize_graph(self, node_features, edge_src, edge_dst):
        nf, src, dst = self._prepare_graphs(node_features, edge_src, edge_dst, 5)
        if nf is not None:
            self.globals["bufferize_dec"] = self.agent.bufferize_gnn(nf, src, dst)

    def init_malloc_graph(self, node_features, edge_src, edge_dst):
        nf, src, dst = self._prepare_graphs(node_features, edge_src, edge_dst, 3)
        if nf is not None:
            self.globals["malloc_dec"] = self.agent.malloc_gnn(nf, src, dst)

    def _order_items(self, items, dec_type, extract_fn):
        if len(items) <= 1:
            return list(range(len(items)))

        features = extract_fn(items)
        global_state = self.globals.get(dec_type, torch.zeros(self.agent.hidden_dim))

        # Evaluate Neural Network (No gradients needed for actors)
        with torch.no_grad():
            dec_model = getattr(self.agent, dec_type)
            scores, _ = dec_model(global_state, features)

        P = torch.softmax(scores, dim=0).cpu().numpy()

        # Add AlphaZero Dirichlet Noise for root exploration
        if self.exploration_noise > 0:
            noise = (
                torch.distributions.Dirichlet(torch.full_like(scores, 0.3))
                .sample()
                .numpy()
            )
            P = (1 - self.exploration_noise) * P + self.exploration_noise * noise

        # Get sorted order
        order = torch.argsort(torch.tensor(P), descending=True).tolist()

        # Record decision step for Replay Buffer tracking
        self.trajectory.append(
            {
                "type": dec_type,
                "global_state": global_state.cpu(),
                "features": features.cpu(),
                "P": P,
                "top_action": order[0],  # The action that C++ will attempt first
            }
        )

        return order

    def order_enodes(self, enodes):
        return self._order_items(enodes, "extract_dec", self._extract_dispatch_features)

    def order_dispatch(self, ready_nodes):
        return self._order_items(
            ready_nodes, "dispatch_dec", self._extract_dispatch_features
        )

    def order_bufferize(self, choices):
        return self._order_items(
            choices, "bufferize_dec", self._extract_bufferize_features
        )

    def order_malloc(self, avail_buffers):
        return self._order_items(
            avail_buffers, "malloc_dec", self._extract_malloc_features
        )

    def _extract_dispatch_features(self, items):
        feats = []
        for f in items:
            num_nodes = len(f.graph.nodes) if hasattr(f, "graph") and f.graph else 0
            num_edges = (
                sum(len(n.child_ids) for n in f.graph.nodes.values())
                if num_nodes
                else 0
            )
            mem_type = float(f.mem_space.type) if hasattr(f, "mem_space") else 0.0
            eng_len = float(len(f.engine_idxs)) if hasattr(f, "engine_idxs") else 0.0
            feats.append(
                [
                    float(f.cost),
                    float(f.size),
                    mem_type,
                    eng_len,
                    float(num_nodes),
                    float(num_edges),
                ]
            )
        return torch.nan_to_num(
            torch.tensor(feats, dtype=torch.float32), posinf=1e9, neginf=-1e9
        )

    def _extract_bufferize_features(self, items):
        feats = [
            [
                float(f.is_new_buffer),
                float(f.size),
                float(f.parent_size),
                float(f.parent_birth_time),
            ]
            for f in items
        ]
        return torch.nan_to_num(
            torch.tensor(feats, dtype=torch.float32), posinf=1e9, neginf=-1e9
        )

    def _extract_malloc_features(self, items):
        feats = [[float(f.size), float(f.start), float(f.end)] for f in items]
        return torch.nan_to_num(
            torch.tensor(feats, dtype=torch.float32), posinf=1e9, neginf=-1e9
        )


# ==============================================================================
# WORKER PROCESS (ACTORS)
# ==============================================================================
@torch.inference_mode()
def actor_worker(worker_id: int, config: TrainConfig, replay_queue: mp.Queue):
    """
    Search Method: Generates self-play data by running multiple Monte Carlo simulations per episode.
    """
    # Fix for intra-op multithreading
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    id_size = len(str(config.workers))
    worker_id_str = f"{worker_id:0{id_size}d}/{config.workers:0{id_size}d}"
    torch.manual_seed(42 + worker_id)
    log_path = os.path.join(config.run_dir, f"train_worker_{worker_id}.log")

    f_log = open(log_path, "w", encoding="utf-8")

    sys.stdout.flush()
    sys.stderr.flush()

    # Duplicate low-level C CRT stdout (1) and stderr (2) descriptors to log file
    os.dup2(f_log.fileno(), 1)
    os.dup2(f_log.fileno(), 2)

    if os.name == "nt":
        import ctypes
        import msvcrt

        os_handle = msvcrt.get_osfhandle(f_log.fileno())
        ctypes.windll.kernel32.SetStdHandle(-11, os_handle)  # STD_OUTPUT_HANDLE
        ctypes.windll.kernel32.SetStdHandle(-12, os_handle)  # STD_ERROR_HANDLE

    sys.stdout = f_log
    sys.stderr = f_log

    try:
        agent = AlphaZeroAgent(hidden_dim=config.hidden_dim)
        model_filepath = Path(config.run_dir) / "model.safetensors"

        episode = 0
        while True:
            # Periodically sync latest weights from Learner
            if model_filepath.exists():
                try:
                    state_dict = load_file(model_filepath)
                    agent.load_state_dict(state_dict)
                except Exception:
                    pass  # Learner might be writing to it exactly now

            agent.eval()
            best_cost = float("inf")
            state_visits = {}

            # Search Method: Run multiple simulations to build Search Distribution (pi)
            for sim in range(config.num_simulations):
                delegate = ActorDelegate(agent, exploration_noise=0.25)
                try:
                    cost = tensor_graphs.plan_graph(
                        config.model_name,
                        config.model_path,
                        delegate,
                        config.log_cost_calls,
                    )
                except Exception as e:
                    err_msg = f"Worker {worker_id_str} | Error during planning: {e}"
                    print(err_msg, flush=True)
                    traceback.print_exc()
                    replay_queue.put({"type": "log", "msg": err_msg})
                    cost = float("inf")

                if cost < float("inf"):
                    replay_queue.put({"type": "cost_metric", "cost": float(cost)})

                best_cost = min(best_cost, cost)

                # Aggregate visit counts (MCTS analogue)
                for step in delegate.trajectory:
                    h = hash(step["features"].numpy().tobytes())
                    if h not in state_visits:
                        state_visits[h] = {
                            "counts": torch.zeros_like(torch.tensor(step["P"])),
                            "data": step,
                        }
                    state_visits[h]["counts"][step["top_action"]] += 1

            # Determine final episode return (Z)
            if best_cost < float("inf"):
                Z = 1000.0 / (best_cost + 1.0)
            else:
                Z = -1.0

            # Push AlphaZero training targets to Replay Buffer
            for h, state_info in state_visits.items():
                data = state_info["data"]
                counts = state_info["counts"]

                # Policy Target (pi): Search visit count distribution
                if counts.sum() > 0:
                    pi = counts / counts.sum()
                else:
                    pi = torch.tensor(data["P"])

                replay_queue.put(
                    {
                        "type": data["type"],
                        "global_state": data["global_state"],
                        "features": data["features"],
                        "pi": pi,
                        "Z": Z,
                    }
                )

            log_msg = f"Worker {worker_id_str} | Episode {episode:03d} | Best Cost: {best_cost:8.4f} ms | Reward: {Z:.4f}"
            print(log_msg, flush=True)
            replay_queue.put({"type": "log", "msg": log_msg})
            episode += 1

    except Exception as e:
        err_msg = f"Worker {worker_id_str} ERROR: {e}"
        print(err_msg, flush=True)
        traceback.print_exc()
        replay_queue.put({"type": "log", "msg": err_msg})
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        f_log.close()


# ==============================================================================
# CENTRAL LEARNER PROCESS
# ==============================================================================
def learner_process(config: TrainConfig, replay_queue: mp.Queue):
    """
    Architecture: Centralized Learner process handling Replay Buffer and Gradient Updates.
    """
    torch.set_num_threads(4)  # Learner can use more threads for batch processing

    agent = AlphaZeroAgent(hidden_dim=config.hidden_dim)
    optimizer = optim.Adam(agent.parameters(), lr=config.lr)
    buffer = collections.deque(maxlen=config.replay_buffer_size)
    model_filepath = Path(config.run_dir) / "model.safetensors"

    # Save initial weights for actors to load
    save_file(agent.state_dict(), model_filepath)

    batches_processed = 0
    losses_bin_path = os.path.join(config.run_dir, "losses.bin")
    costs_bin_path = os.path.join(config.run_dir, "costs.bin")
    pack_fmt = "<If"
    cost_count = 0

    while True:
        # Drain incoming queue into Replay Buffer or process log/metric events
        while not replay_queue.empty():
            try:
                item = replay_queue.get_nowait()
                if isinstance(item, dict) and item.get("type") == "cost_metric":
                    cost_count += 1
                    cost_val = float(item["cost"])
                    with open(costs_bin_path, "ab") as f_bin:
                        f_bin.write(struct.pack(pack_fmt, cost_count, cost_val))
                        f_bin.flush()
                elif isinstance(item, dict) and item.get("type") == "log":
                    print(item["msg"], flush=True)
                else:
                    buffer.append(item)
            except Empty:
                break

        # Gradient Updates
        if len(buffer) >= config.batch_size:
            agent.train()
            batch = random.sample(buffer, config.batch_size)
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0)

            # Group batch by decision type to process through distinct network heads
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

                # Forward pass for each sample
                for sample in sub_batch:
                    g_state = sample["global_state"]
                    feats = sample["features"]
                    pi_target = sample["pi"]
                    z_target = torch.tensor([sample["Z"]], dtype=torch.float32)

                    scores, val = dec_model(g_state, feats)

                    # Policy Loss Target: Cross Entropy (p vs pi)
                    log_p = F.log_softmax(scores, dim=0)
                    policy_loss = -(pi_target * log_p).sum()

                    # Value Loss Target: Mean Squared Error (v vs Z)
                    value_loss = F.mse_loss(val, z_target)

                    type_loss = type_loss + policy_loss + value_loss

                total_loss = total_loss + type_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()

            batches_processed += 1
            print(
                f"Learner | Batches: {batches_processed:04d} | Replay Buffer Size: {len(buffer)} | Total Loss: {total_loss.item():.4f}",
                flush=True,
            )

            with open(losses_bin_path, "ab") as f_bin:
                f_bin.write(
                    struct.pack(pack_fmt, batches_processed, float(total_loss.detach()))
                )
                f_bin.flush()

            # Periodically sync updated weights to disk for Actors
            if batches_processed % config.save_interval == 0:
                save_file(agent.state_dict(), model_filepath)


def setup_run_dir() -> str:
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)
    existing = [int(d) for d in os.listdir("runs") if d.isdigit()]
    run_idx = max(existing) + 1 if existing else 1
    run_dir = runs_dir / str(run_idx)
    run_dir.mkdir()
    return run_dir.as_posix()


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gemma-3-270m")
    parser.add_argument("--model-path", type=str, default="models/google/gemma-3-270m")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (psutil.cpu_count(logical=False) or 4) - 1),
    )
    parser.add_argument("--log-cost-calls", action="store_true")
    args = parser.parse_args()

    config = TrainConfig(
        run_dir=setup_run_dir(),
        model_name=args.model,
        model_path=args.model_path,
        workers=args.workers,
        log_cost_calls=args.log_cost_calls,
    )

    with open(os.path.join(config.run_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(config), f, indent=4)

    # Setup cross-process queue
    replay_queue = mp.Queue()
    processes = []

    print(
        f"Starting Training in {config.run_dir} with {config.workers} Actor workers + 1 Learner...",
        flush=True,
    )

    # 1. Spawn Central Learner
    learner_p = mp.Process(target=learner_process, args=(config, replay_queue))
    learner_p.start()
    processes.append(learner_p)

    # 2. Spawn Actors
    for rank in range(config.workers):
        p = mp.Process(target=actor_worker, args=(rank, config, replay_queue))
        p.start()
        processes.append(p)

    # Wait for completion (Manual exit via Ctrl+C)
    for p in processes:
        p.join()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    train()

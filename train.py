# train.py
import psutil
import argparse
import dataclasses
import json
import os
import struct
import sys
import traceback
from pathlib import Path
from queue import Empty
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import nn, optim

import tensor_graphs


@dataclasses.dataclass
class TrainConfig:
    run_name: str = "default"
    run_dir: str = ""
    model_name: str = "gemma-3-270m"
    model_path: str = "models/google/gemma-3-270m"
    workers: int = 4
    epochs: int = 10000
    save_interval: int = 10
    hidden_dim: int = 64
    lr: float = 1e-3
    log_cost_calls: bool = False


class GNNModel(nn.Module):
    def __init__(self, in_features, hidden_dim=64):
        super().__init__()
        self.lin1 = nn.Linear(in_features, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, node_features, edge_src, edge_dst):
        x = F.relu(self.lin1(node_features))
        msg = self.lin2(x)
        out = torch.zeros_like(msg)
        if len(edge_dst) > 0:
            out.index_add_(0, edge_dst, msg[edge_src])
        x = F.relu(x + out)

        msg = self.lin3(x)
        out = torch.zeros_like(msg)
        if len(edge_dst) > 0:
            out.index_add_(0, edge_dst, msg[edge_src])
        x = F.relu(x + out)

        global_state = x.mean(dim=0)
        return global_state


class RNNModel(nn.Module):
    def __init__(self, global_dim, feature_dim, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRUCell(feature_dim, hidden_dim)
        self.policy = nn.Linear(global_dim + hidden_dim + feature_dim, 1)
        self.value = nn.Linear(global_dim + hidden_dim, 1)

    def forward(self, global_state, hidden_state, options_features):
        step_input = options_features.mean(dim=0).unsqueeze(0)  # (1, input_dim)
        new_state = self.rnn(step_input, hidden_state.unsqueeze(0)).squeeze(
            0
        )  # (hidden_dim,)

        N = options_features.size(0)
        global_expanded = global_state.unsqueeze(0).expand(N, -1)
        hidden_expanded = new_state.unsqueeze(0).expand(N, -1)

        policy_in = torch.cat(
            [global_expanded, hidden_expanded, options_features], dim=1
        )
        scores = self.policy(policy_in).squeeze(1)  # (N,)

        value_in = torch.cat([global_state, new_state], dim=0)
        val = self.value(value_in)  # (1,)

        return new_state, scores, val


class AdvancedAgent(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.extract_gnn = GNNModel(in_features=4, hidden_dim=hidden_dim)
        self.extract_rnn = RNNModel(
            global_dim=hidden_dim, feature_dim=6, hidden_dim=hidden_dim
        )
        self.dispatch_gnn = GNNModel(in_features=4, hidden_dim=hidden_dim)
        self.dispatch_rnn = RNNModel(
            global_dim=hidden_dim, feature_dim=6, hidden_dim=hidden_dim
        )
        self.bufferize_gnn = GNNModel(in_features=5, hidden_dim=hidden_dim)
        self.bufferize_rnn = RNNModel(
            global_dim=hidden_dim, feature_dim=4, hidden_dim=hidden_dim
        )
        self.malloc_gnn = GNNModel(in_features=3, hidden_dim=hidden_dim)
        self.malloc_rnn = RNNModel(
            global_dim=hidden_dim, feature_dim=3, hidden_dim=hidden_dim
        )


class AgentDelegate(tensor_graphs.SearchDelegate):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent
        self.hidden_states = []
        self.current_hidden = torch.zeros(agent.hidden_dim)
        self.log_probs = []
        self.values = []
        self.extract_global = torch.zeros(agent.hidden_dim)
        self.dispatch_global = torch.zeros(agent.hidden_dim)
        self.bufferize_global = torch.zeros(agent.hidden_dim)
        self.malloc_global = torch.zeros(agent.hidden_dim)

    def push_state(self):
        self.hidden_states.append(self.current_hidden.clone())

    def pop_state(self):
        if self.hidden_states:
            self.current_hidden = self.hidden_states.pop()
        if len(self.log_probs) > len(self.hidden_states):
            self.log_probs.pop()
            self.values.pop()

    def init_egraph(self, node_features, edge_src, edge_dst):
        if not node_features:
            return
        nf = torch.tensor(node_features, dtype=torch.float32).view(-1, 4)
        nf = torch.nan_to_num(nf, posinf=1e9, neginf=-1e9)
        src = torch.tensor(edge_src, dtype=torch.int64)
        dst = torch.tensor(edge_dst, dtype=torch.int64)
        self.extract_global = self.agent.extract_gnn(nf, src, dst)

    def init_dispatch_graph(self, node_features, edge_src, edge_dst):
        if not node_features:
            return
        nf = torch.tensor(node_features, dtype=torch.float32).view(-1, 4)
        nf = torch.nan_to_num(nf, posinf=1e9, neginf=-1e9)
        src = torch.tensor(edge_src, dtype=torch.int64)
        dst = torch.tensor(edge_dst, dtype=torch.int64)
        self.dispatch_global = self.agent.dispatch_gnn(nf, src, dst)

    def init_bufferize_graph(self, node_features, edge_src, edge_dst):
        if not node_features:
            return
        nf = torch.tensor(node_features, dtype=torch.float32).view(-1, 5)
        nf = torch.nan_to_num(nf, posinf=1e9, neginf=-1e9)
        src = torch.tensor(edge_src, dtype=torch.int64)
        dst = torch.tensor(edge_dst, dtype=torch.int64)
        self.bufferize_global = self.agent.bufferize_gnn(nf, src, dst)

    def init_malloc_graph(self, node_features, edge_src, edge_dst):
        if not node_features:
            return
        nf = torch.tensor(node_features, dtype=torch.float32).view(-1, 3)
        nf = torch.nan_to_num(nf, posinf=1e9, neginf=-1e9)
        src = torch.tensor(edge_src, dtype=torch.int64)
        dst = torch.tensor(edge_dst, dtype=torch.int64)
        self.malloc_global = self.agent.malloc_gnn(nf, src, dst)

    def _order_items(self, items, rnn_model, global_state, extract_fn):
        if len(items) <= 1:
            if len(items) == 1:
                self.log_probs.append(torch.tensor(0.0, requires_grad=True))
                self.values.append(torch.tensor([0.0], requires_grad=True))
            return list(range(len(items)))

        features = extract_fn(items)
        new_state, scores, val = rnn_model(global_state, self.current_hidden, features)
        self.current_hidden = new_state

        probs = torch.softmax(scores, dim=0)
        dist = torch.distributions.Categorical(probs)

        # Gumbel-Max trick for sampling permutations without replacement during exploration
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-10) + 1e-10)
        noisy_scores = scores + gumbel_noise

        sorted_indices = torch.argsort(noisy_scores, descending=True).tolist()

        # Log prob and predicted value of the top choice (DFS searches this first)
        top_choice = sorted_indices[0]
        self.log_probs.append(dist.log_prob(torch.tensor(top_choice)))
        self.values.append(val)

        return sorted_indices

    def order_enodes(self, enodes):
        return self._order_items(
            enodes,
            self.agent.extract_rnn,
            self.extract_global,
            self._extract_dispatch_features,
        )

    def order_dispatch(self, ready_nodes):
        return self._order_items(
            ready_nodes,
            self.agent.dispatch_rnn,
            self.dispatch_global,
            self._extract_dispatch_features,
        )

    def order_bufferize(self, choices):
        return self._order_items(
            choices,
            self.agent.bufferize_rnn,
            self.bufferize_global,
            self._extract_bufferize_features,
        )

    def order_malloc(self, avail_buffers):
        return self._order_items(
            avail_buffers,
            self.agent.malloc_rnn,
            self.malloc_global,
            self._extract_malloc_features,
        )

    def _extract_dispatch_features(self, items):
        feats = []
        for f in items:
            num_nodes = 0
            num_edges = 0
            assert hasattr(f, "graph")
            assert f.graph
            assert hasattr(f.graph, "nodes")
            num_nodes = len(f.graph.nodes)
            num_edges = sum(len(n.child_ids) for n in f.graph.nodes.values())

            mem_type = (
                float(f.mem_space.type)
                if hasattr(f, "mem_space") and hasattr(f.mem_space, "type")
                else 0.0
            )
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
        t = torch.tensor(feats, dtype=torch.float32)
        return torch.nan_to_num(t, posinf=1e9, neginf=-1e9)

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
        t = torch.tensor(feats, dtype=torch.float32)
        return torch.nan_to_num(t, posinf=1e9, neginf=-1e9)

    def _extract_malloc_features(self, items):
        feats = [[float(f.size), float(f.start), float(f.end)] for f in items]
        t = torch.tensor(feats, dtype=torch.float32)
        return torch.nan_to_num(t, posinf=1e9, neginf=-1e9)


class SharedAdam(optim.Adam):
    """
    A Shared Optimizer ensuring internal states are preserved across multi-processing environments.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(
            params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = torch.tensor(0, dtype=torch.float32).share_memory_()
                state["exp_avg"] = torch.zeros_like(p.data).share_memory_()
                state["exp_avg_sq"] = torch.zeros_like(p.data).share_memory_()


def worker_process(
    worker_id: int,
    num_workers: int,
    config: TrainConfig,
    shared_agent: nn.Module,
    optimizer: optim.Optimizer,
    queue: mp.Queue,
):
    id_size = len(str(num_workers))
    worker_id_str = f"{worker_id:0{id_size}d}/{num_workers:0{id_size}d}"
    torch.manual_seed(42 + worker_id)
    log_path = os.path.join(config.run_dir, f"train_worker_{worker_id}.log")

    f_log = open(log_path, "w", encoding="utf-8")

    # Save original Python streams and OS file descriptors
    saved_stdout = sys.stdout
    saved_stderr = sys.stderr
    saved_out_fd = os.dup(1)
    saved_err_fd = os.dup(2)

    # Redirect Python streams
    sys.stdout = f_log
    sys.stderr = f_log

    # Redirect C++ / OS-level file descriptors
    os.dup2(f_log.fileno(), 1)
    os.dup2(f_log.fileno(), 2)

    try:
        for epoch in range(config.epochs):
            delegate = AgentDelegate(shared_agent)
            try:
                cost = tensor_graphs.plan_graph(
                    config.model_name,
                    config.model_path,
                    delegate,
                    config.log_cost_calls,
                )
            except Exception as e:
                print(f"Worker {worker_id_str} | Error during planning: {e}", file=saved_stdout, flush=True)
                traceback.print_exc(file=saved_stdout)
                continue

            cost_val = float(cost)
            if cost_val == float("inf") or cost_val != cost_val:
                cost_val = 1e9

            reward = -cost_val
            reward_t = torch.tensor([reward], dtype=torch.float32)

            loss = torch.tensor(0.0)

            for lp, v in zip(delegate.log_probs, delegate.values):
                advantage = reward_t - v.detach()
                policy_loss = -lp * advantage
                value_loss = F.mse_loss(v, reward_t)
                loss = loss + policy_loss + value_loss

            optimizer.zero_grad()
            if loss.requires_grad and loss.item() != 0.0 and loss.item() == loss.item():
                loss.backward()
                torch.nn.utils.clip_grad_norm_(shared_agent.parameters(), 1.0)
                optimizer.step()

            queue.put((epoch, worker_id, float(cost_val), float(loss.item())))
    except Exception as e:
        print(f"Worker {worker_id_str} ERROR: {e}", file=saved_stdout, flush=True)
        traceback.print_exc(file=saved_stdout)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = saved_stdout
        sys.stderr = saved_stderr
        os.dup2(saved_out_fd, 1)
        os.dup2(saved_err_fd, 2)
        os.close(saved_out_fd)
        os.close(saved_err_fd)
        f_log.close()


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
    parser.add_argument("--workers", type=int, default=psutil.cpu_count(logical=False))
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument(
        "--log-cost-calls",
        action="store_true",
        help="Enable logging unbenchmarked cost calls to benchmarks/calls.bin",
    )
    args = parser.parse_args()

    config = TrainConfig(
        run_dir=setup_run_dir(),
        model_name=args.model,
        model_path=args.model_path,
        workers=args.workers,
        epochs=args.epochs,
        save_interval=args.save_interval,
        log_cost_calls=args.log_cost_calls,
    )

    with open(os.path.join(config.run_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(config), f, indent=4)

    agent = AdvancedAgent(hidden_dim=config.hidden_dim)
    agent.share_memory()
    optimizer = SharedAdam(agent.parameters(), lr=config.lr)

    queue = mp.Queue()
    processes = []

    print(f"Starting Training in {config.run_dir} with {config.workers} workers...", flush=True)
    for rank in range(config.workers):
        p = mp.Process(
            target=worker_process, args=(rank, config, agent, optimizer, queue)
        )
        p.start()
        processes.append(p)

    active_workers = config.workers
    total_epochs = 0
    losses_bin_path = os.path.join(config.run_dir, "losses.bin")

    pack_fmt = "<IIff"

    with open(losses_bin_path, "wb") as f_bin:
        while active_workers > 0:
            try:
                epoch, worker_id, cost, loss = queue.get(timeout=1.0)
                f_bin.write(struct.pack(pack_fmt, epoch, worker_id, cost, loss))
                f_bin.flush()

                print(
                    f"Worker {worker_id:02d} | Epoch {epoch:03d} | Cost: {cost:8.4f} ms | Loss: {loss:.4f}",
                    flush=True,
                )

                total_epochs += 1
                if total_epochs % config.save_interval == 0:
                    save_file(
                        agent.state_dict(),
                        os.path.join(config.run_dir, "model.safetensors"),
                    )
                    print(f"Saved model to {config.run_dir}/model.safetensors", flush=True)

            except Empty:
                active_workers = sum(1 for p in processes if p.is_alive())

    for p in processes:
        p.join()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    train()
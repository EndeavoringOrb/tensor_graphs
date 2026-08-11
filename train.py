# train.py
import argparse
import tensor_graphs
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn, optim


class GNNModel(nn.Module):
    def __init__(self, in_features, hidden_dim=64):
        super().__init__()
        self.lin1 = nn.Linear(in_features, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, node_features, edge_src, edge_dst):
        x = F.relu(self.lin1(node_features))

        # Message passing 1
        msg = self.lin2(x)
        out = torch.zeros_like(msg)
        if len(edge_dst) > 0:
            out.index_add_(0, edge_dst, msg[edge_src])
        x = F.relu(x + out)

        # Message passing 2
        msg = self.lin3(x)
        out = torch.zeros_like(msg)
        if len(edge_dst) > 0:
            out.index_add_(0, edge_dst, msg[edge_src])
        x = F.relu(x + out)

        # Global pooling
        global_state = x.mean(dim=0)
        return global_state


class RNNModel(nn.Module):
    def __init__(self, global_dim, feature_dim, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRUCell(feature_dim, hidden_dim)
        self.policy = nn.Linear(global_dim + hidden_dim + feature_dim, 1)
        # Value head to predict the expected cost (AlphaDev-esque)
        self.value = nn.Linear(global_dim + hidden_dim, 1)

    def forward(self, global_state, hidden_state, options_features):
        step_input = options_features.mean(dim=0).unsqueeze(0)  # (1, input_dim)
        new_state = self.rnn(step_input, hidden_state.unsqueeze(0)).squeeze(0)  # (hidden_dim,)

        N = options_features.size(0)
        global_expanded = global_state.unsqueeze(0).expand(N, -1)
        hidden_expanded = new_state.unsqueeze(0).expand(N, -1)

        policy_in = torch.cat([global_expanded, hidden_expanded, options_features], dim=1)
        scores = self.policy(policy_in).squeeze(1)  # (N,)

        value_in = torch.cat([global_state, new_state], dim=0)
        val = self.value(value_in)  # (1,)

        return new_state, scores, val


class AdvancedAgent(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 1. Extraction GNN
        self.extract_gnn = GNNModel(in_features=4, hidden_dim=hidden_dim)
        # Extraction RNN (cost, size, type, engines, nodes, edges)
        self.extract_rnn = RNNModel(
            global_dim=hidden_dim, feature_dim=6, hidden_dim=hidden_dim
        )

        # 2. Dispatch GNN
        self.dispatch_gnn = GNNModel(in_features=4, hidden_dim=hidden_dim)
        # Dispatch RNN
        self.dispatch_rnn = RNNModel(
            global_dim=hidden_dim, feature_dim=6, hidden_dim=hidden_dim
        )

        # 3. Bufferize GNN
        self.bufferize_gnn = GNNModel(in_features=5, hidden_dim=hidden_dim)
        # Bufferize RNN
        self.bufferize_rnn = RNNModel(
            global_dim=hidden_dim, feature_dim=4, hidden_dim=hidden_dim
        )

        # 4. Malloc GNN
        self.malloc_gnn = GNNModel(in_features=3, hidden_dim=hidden_dim)
        # Malloc RNN (size, start, end)
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
        src = torch.tensor(edge_src, dtype=torch.int64)
        dst = torch.tensor(edge_dst, dtype=torch.int64)
        self.extract_global = self.agent.extract_gnn(nf, src, dst)

    def init_dispatch_graph(self, node_features, edge_src, edge_dst):
        if not node_features:
            return
        nf = torch.tensor(node_features, dtype=torch.float32).view(-1, 4)
        src = torch.tensor(edge_src, dtype=torch.int64)
        dst = torch.tensor(edge_dst, dtype=torch.int64)
        self.dispatch_global = self.agent.dispatch_gnn(nf, src, dst)

    def init_bufferize_graph(self, node_features, edge_src, edge_dst):
        if not node_features:
            return
        nf = torch.tensor(node_features, dtype=torch.float32).view(-1, 5)
        src = torch.tensor(edge_src, dtype=torch.int64)
        dst = torch.tensor(edge_dst, dtype=torch.int64)
        self.bufferize_global = self.agent.bufferize_gnn(nf, src, dst)

    def init_malloc_graph(self, node_features, edge_src, edge_dst):
        if not node_features:
            return
        nf = torch.tensor(node_features, dtype=torch.float32).view(-1, 3)
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
        return torch.tensor(feats, dtype=torch.float32)

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
        return torch.tensor(feats, dtype=torch.float32)

    def _extract_malloc_features(self, items):
        feats = [[float(f.size), float(f.start), float(f.end)] for f in items]
        return torch.tensor(feats, dtype=torch.float32)


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
                state["step"] = torch.tensor(0, dtype=torch.float32)
                state["exp_avg"] = torch.zeros_like(p.data)
                state["exp_avg_sq"] = torch.zeros_like(p.data)
                
                # Share in memory natively for async updates across parallel workers
                state["step"].share_memory_()
                state["exp_avg"].share_memory_()
                state["exp_avg_sq"].share_memory_()


def worker_process(worker_id, shared_agent, optimizer, num_epochs):
    # Set seed differently for each worker to ensure diverse exploration
    torch.manual_seed(42 + worker_id)

    model_name = "gemma-3-270m"
    model_path = "models/google/gemma-3-270m"

    for epoch in range(num_epochs):
        delegate = AgentDelegate(shared_agent)

        try:
            cost = tensor_graphs.plan_graph(model_name, model_path, delegate)
        except Exception as e:
            print(f"Worker {worker_id} | Error during planning: {e}")
            continue

        reward = -cost if cost < float("inf") else -1e6
        reward_t = torch.tensor([reward], dtype=torch.float32)

        loss = torch.tensor(0.0)

        # Advantage Actor-Critic Loss formulation replacing standard REINFORCE
        # Advantage A = R - V
        # Policy Loss = -log_prob * A
        # Value Loss = (R - V)^2
        for lp, v in zip(delegate.log_probs, delegate.values):
            advantage = reward_t - v.detach()
            policy_loss = -lp * advantage
            value_loss = F.mse_loss(v, reward_t)
            loss = loss + policy_loss + value_loss

        optimizer.zero_grad()
        if loss.requires_grad and loss.item() != 0.0:
            loss.backward()
            optimizer.step()

        print(
            f"Worker {worker_id:02d} | Epoch {epoch:03d} | Cost: {cost:8.4f} ms | Loss: {loss.item():.4f}"
        )


def train():
    parser = argparse.ArgumentParser(description="Train GNN/RNN Agent via distributed A2C")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--epochs", type=int, default=1_000_000, help="Number of epochs per worker")
    args = parser.parse_args()

    agent = AdvancedAgent(hidden_dim=64)
    # Enable weights sharing to pass across PyTorch mp.Process calls
    agent.share_memory()

    optimizer = SharedAdam(agent.parameters(), lr=1e-3)

    processes = []
    for rank in range(args.workers):
        p = mp.Process(
            target=worker_process, args=(rank, agent, optimizer, args.epochs)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    train()
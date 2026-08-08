import torch
import torch.nn as nn
import torch.optim as optim
import tensor_graphs


class RNNAgent(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRUCell(input_dim, hidden_dim)
        self.policy = nn.Linear(hidden_dim + input_dim, 1)

    def forward(self, state, options_features):
        """
        state: Tensor of shape (hidden_dim,)
        options_features: Tensor of shape (N, input_dim)
        """
        step_input = options_features.mean(dim=0).unsqueeze(0)  # (1, input_dim)
        new_state = self.rnn(step_input, state.unsqueeze(0)).squeeze(0)  # (hidden_dim,)

        N = options_features.size(0)
        state_expanded = new_state.unsqueeze(0).expand(N, -1)  # (N, hidden_dim)
        policy_in = torch.cat([state_expanded, options_features], dim=1)
        scores = self.policy(policy_in).squeeze(1)  # (N,)

        return new_state, scores


class AgentDelegate(tensor_graphs.SearchDelegate):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent
        self.hidden_states = []
        self.current_hidden = torch.zeros(agent.hidden_dim)
        self.log_probs = []

    def push_state(self):
        self.hidden_states.append(self.current_hidden.clone())

    def pop_state(self):
        if self.hidden_states:
            self.current_hidden = self.hidden_states.pop()
        if len(self.log_probs) > len(self.hidden_states):
            self.log_probs.pop()

    def _extract_features(self, items):
        feats = [[float(f.cost), float(f.size), float(f.op_type)] for f in items]
        return torch.tensor(feats, dtype=torch.float32)

    def _order_items(self, items):
        if len(items) <= 1:
            if len(items) == 1:
                self.log_probs.append(torch.tensor(0.0, requires_grad=True))
            return list(range(len(items)))

        features = self._extract_features(items)
        new_state, scores = self.agent(self.current_hidden, features)
        self.current_hidden = new_state

        probs = torch.softmax(scores, dim=0)
        dist = torch.distributions.Categorical(probs)

        # Gumbel-Max trick for sampling permutations without replacement
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-10) + 1e-10)
        noisy_scores = scores + gumbel_noise

        sorted_indices = torch.argsort(noisy_scores, descending=True).tolist()

        # Log prob of top choice (which C++ DFS attempts first)
        top_choice = sorted_indices[0]
        self.log_probs.append(dist.log_prob(torch.tensor(top_choice)))

        return sorted_indices

    def order_enodes(self, eclass_id, enodes):
        return self._order_items(enodes)

    def order_dispatch(self, ready_nodes):
        return self._order_items(ready_nodes)

    def order_malloc(self, avail_buffers):
        return self._order_items(avail_buffers)


def train():
    agent = RNNAgent(input_dim=3, hidden_dim=64)
    optimizer = optim.Adam(agent.parameters(), lr=1e-3)

    model_name = "gemma-3-270m"
    model_path = "models/gemma-3-270m"

    for epoch in range(100):
        delegate = AgentDelegate(agent)

        cost = tensor_graphs.plan_graph(model_name, model_path, delegate)

        reward = -cost if cost < float("inf") else -1e6

        loss = torch.tensor(0.0, requires_grad=True)
        for lp in delegate.log_probs:
            loss = loss - lp * reward

        optimizer.zero_grad()
        if loss.requires_grad and loss.item() != 0.0:
            loss.backward()
            optimizer.step()

        print(
            f"Epoch {epoch:03d} | Execution Cost: {cost:.4f} ms | REINFORCE Loss: {loss.item():.4f}"
        )


if __name__ == "__main__":
    train()

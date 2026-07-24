import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import math
import malloc_rl
from tqdm import trange, tqdm


# ----------------------------------------
# 1. Variable-Size Neural Network (DeepSets)
# ----------------------------------------
class MallocNet(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        # Item-wise features
        self.enc1 = nn.Linear(5, hidden_dim)
        self.enc2 = nn.Linear(hidden_dim, hidden_dim)

        # Policy Head
        self.policy_fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.policy_out = nn.Linear(hidden_dim, 1)

        # Value Head
        self.value_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.value_out = nn.Linear(hidden_dim, 1)

    def forward(self, state_batch):
        # state_batch: [B, N, 5] where N can theoretically change
        B, N, _ = state_batch.shape

        x = F.relu(self.enc1(state_batch))
        x = F.relu(self.enc2(x))  # [B, N, hidden_dim]

        # Global max pool -> [B, 1, hidden_dim]
        global_features = torch.max(x, dim=1, keepdim=True)[0]

        # Value is determined by global state
        val = F.relu(self.value_fc1(global_features.squeeze(1)))
        val = torch.tanh(self.value_out(val))  # [-1, 1]

        # Policy is determined by item state + global context
        global_expanded = global_features.expand(-1, N, -1)
        pi_features = torch.cat([x, global_expanded], dim=2)
        pi = F.relu(self.policy_fc1(pi_features))
        logits = self.policy_out(pi).squeeze(-1)  # [B, N]

        return logits, val


class SmartMallocNet(nn.Module):
    def __init__(self, feature_dim=5, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Linear(feature_dim, hidden_dim)

        # Self-Attention allows items to "talk" to each other based on overlaps
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.policy_out = nn.Linear(hidden_dim, 1)
        self.value_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, state_batch):
        x = F.relu(self.embedding(state_batch))
        x = self.transformer(x)  # [B, N, hidden_dim]

        # Policy
        logits = self.policy_out(x).squeeze(-1)

        # Value (Global context max pool)
        global_features = torch.max(x, dim=1)[0]
        val = torch.tanh(self.value_out(global_features))

        return logits, val


# ----------------------------------------
# 2. Monte Carlo Tree Search Node
# ----------------------------------------
class MCTSNode:
    def __init__(self, env, parent=None, action_taken=None, prior=0.0):
        self.env = env
        self.parent = parent
        self.action_taken = action_taken

        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.valid_actions = env.get_valid_actions()
        self.is_expanded = False

        # New attributes to cleanly track terminal states
        self.is_terminal = False
        self.reward = 0.0

    def value(self):
        if self.is_terminal:
            return self.reward
        return 0 if self.visit_count == 0 else self.value_sum / self.visit_count

    def expand(self, action_probs):
        self.is_expanded = True
        for action, prob in enumerate(action_probs):
            if self.valid_actions[action]:
                next_env = self.env.clone()
                reward, done = next_env.step(action)

                child = MCTSNode(next_env, parent=self, action_taken=action, prior=prob)
                if done:
                    # Mark as terminal without polluting visit_count
                    child.is_terminal = True
                    child.reward = reward
                    child.is_expanded = (
                        True  # A terminal node has no children to expand
                    )
                self.children[action] = child

    def best_child(self, c_puct=1.5):
        best_score = -float("inf")
        best_act, best_child = None, None

        for act, child in self.children.items():
            u = (
                c_puct
                * child.prior
                * math.sqrt(self.visit_count)
                / (1 + child.visit_count)
            )
            score = child.value() + u
            if score > best_score:
                best_score = score
                best_act = act
                best_child = child
        return best_act, best_child


# ----------------------------------------
# 3. MCTS Search
# ----------------------------------------
@torch.inference_mode()
def mcts_search(root_env, model, num_simulations=50):
    root = MCTSNode(root_env.clone())

    for _ in range(num_simulations):
        node = root
        search_path = [node]

        # Selection
        while node.is_expanded and node.children:
            action, node = node.best_child()
            search_path.append(node)

        # Evaluation & Expansion
        if not node.is_terminal:
            if not node.is_expanded:
                state_tensor = torch.tensor(
                    node.env.get_state(), dtype=torch.float32
                ).unsqueeze(0)

                logits, value = model(state_tensor)

                # Mask invalid actions
                logits = logits[0]
                valid_mask = torch.tensor(node.valid_actions, dtype=torch.bool)
                logits[~valid_mask] = -1e9
                probs = F.softmax(logits, dim=0).cpu().numpy()

                node.expand(probs)
                v = value.item()
            else:
                v = node.value()
        else:
            # If the node is an exact dead-end/success, grab its real reward
            v = node.reward

        # Backpropagation
        for n in reversed(search_path):
            n.value_sum += v
            n.visit_count += 1

    # Return action probabilities proportional to visit counts
    visits = np.zeros(root.env.N)
    for act, child in root.children.items():
        visits[act] = child.visit_count

    visit_sum = np.sum(visits)
    if visit_sum > 0:
        probs = visits / visit_sum
    else:
        probs = np.ones(root.env.N) / root.env.N

    return probs


# ----------------------------------------
# 4. Self-Play & Training Loop
# ----------------------------------------
def generate_random_env(N, max_cap=2000):
    buffers = []
    for i in range(N):
        sz = random.randint(10, 500)
        s_time = random.uniform(0, 100)
        e_time = s_time + random.uniform(5, 50)
        buffers.append(malloc_rl.ParallelBuffer(i, sz, s_time, e_time))
    return malloc_rl.MallocEnv(max_cap, buffers)


def self_play(model, N=50, num_games=10, mcts_sims=50):
    model.eval()
    dataset = []

    for g in trange(num_games):
        env = generate_random_env(N)
        states, policies = [], []

        done = False
        reward = 0
        while not done:
            probs = mcts_search(env, model, num_simulations=mcts_sims)

            states.append(env.get_state())
            policies.append(probs)

            # Sample action proportionally or greedily
            action = np.argmax(probs)
            reward, done = env.step(action)

        for s, p in zip(states, policies):
            dataset.append((s, p, reward))

    return dataset


def train(model, dataset, optimizer, epochs=3, batch_size=32):
    model.train()
    random.shuffle(dataset)
    print(f"dataset has {len(dataset):,} items")

    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            states = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
            targets_pi = torch.tensor(
                np.array([b[1] for b in batch]), dtype=torch.float32
            )
            targets_v = torch.tensor(
                np.array([b[2] for b in batch]), dtype=torch.float32
            ).unsqueeze(1)

            optimizer.zero_grad()
            out_logits, out_v = model(states)

            # Policy loss (Cross Entropy with soft targets)
            loss_pi = -torch.sum(
                targets_pi * F.log_softmax(out_logits, dim=1), dim=1
            ).mean()
            # Value loss (MSE)
            loss_v = F.mse_loss(out_v, targets_v)

            loss = loss_pi + loss_v
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(
            f"Epoch {epoch+1} Loss: {total_loss / (len(dataset)//batch_size + 1):.4f}"
        )


def evaluate_model(model, N=20, num_games=10, max_steps=5000, compare_all=True):
    model.eval()

    def dfs(env, use_model, steps_counter):
        # --- NEW: Abort if we've searched too many nodes ---
        if steps_counter[0] >= max_steps:
            return False

        steps_counter[0] += 1

        if env.num_allocated == env.N:
            return True  # Success

        valid_mask = env.get_valid_actions()
        valid_indices = [i for i, valid in enumerate(valid_mask) if valid]

        if not valid_indices:
            return False  # Dead end, trigger backtrack

        # --- HEURISTIC ORDERING ---
        if use_model:
            state_tensor = torch.tensor(env.get_state(), dtype=torch.float32).unsqueeze(
                0
            )
            with torch.no_grad():
                logits, _ = model(state_tensor)

            probs = logits[0].cpu().numpy()
            valid_indices.sort(key=lambda idx: probs[idx], reverse=True)
        # --------------------------

        for act in valid_indices:
            env.step(act)
            if dfs(env, use_model, steps_counter):
                return True
            env.undo()

        return False

    guided_steps = []
    unguided_steps = []
    failures = 0

    print(f"\n--- Evaluating Model over {num_games} games (N={N}) ---")
    for g in trange(num_games):
        env = generate_random_env(N)

        unguided_counter = [0]
        # We only clone here once to give both strategies the exact same starting scenario
        success_un = dfs(env.clone(), use_model=False, steps_counter=unguided_counter)

        guided_counter = [0]
        success_gu = dfs(env.clone(), use_model=True, steps_counter=guided_counter)

        tqdm.write(
            f"Game {g+1}: Unguided = {unguided_counter[0]:<5} steps | Guided = {guided_counter[0]:<5} steps"
        )
        if (success_un and success_gu) or compare_all:
            unguided_steps.append(unguided_counter[0])
            guided_steps.append(guided_counter[0])
        else:
            failures += 1
            tqdm.write(f"Game {g+1}: OOM / No solution exists under capacity.")

    if unguided_steps:
        avg_un = sum(unguided_steps) / len(unguided_steps)
        avg_gu = sum(guided_steps) / len(guided_steps)
        print(
            f"\nResults over {len(unguided_steps)}{' solvable' if not compare_all else ''} games:"
        )
        print(f"Average Unguided Steps: {avg_un:.1f}")
        print(f"Average Guided Steps:   {avg_gu:.1f}")
        print(f"Improvement Factor:     {avg_un / max(avg_gu, 1):.2f}x fewer steps")


if __name__ == "__main__":
    net = SmartMallocNet()
    print(
        f"Initialized model with {sum(p.numel() for p in net.parameters()):,} parameters"
    )
    opt = optim.Adam(net.parameters(), lr=1e-3)
    N = 16

    # Simple training loop iteration
    iteration = 0
    while True:
        iteration += 1
        print(f"--- Iteration {iteration} ---")
        print("Self-Playing...")
        data = self_play(net, N=N, num_games=20, mcts_sims=1000)

        print("Training Network...")
        train(net, data, opt, 3)

        evaluate_model(net, N=N, num_games=10)

import numpy as np
import torch
from torch import nn


class CostPredictorRNN(nn.Module):
    """RNN model that predicts the final execution score directly:

    hidden, pred_cost = model(hidden, action_feat, phase_id)
    """

    ACTION_DIMS = {
        0: 6,  # Cache: [is_cached, log_size, num_users, logical_id, mem_type, log_mem_cap]
        1: 9,  # Extract (ENode): [log_cost, log_dp_cost, log_min_dp_cp, log_size, mem_type, eng_len, num_nodes, num_edges, log_mem_cap]
        2: 10,  # Dispatch: [log_cost, log_dp_cost, log_min_dp_cp, log_rev_cp, log_size, mem_type, eng_len, num_nodes, num_edges, log_mem_cap]
        3: 6,  # Bufferize: [is_new_buffer, log_size, log_parent_size, parent_birth_time, mem_type, log_mem_cap]
        4: 5,  # Malloc: [log_size, start, end, mem_type, log_mem_cap]
        5: 8,  # Frontier: [eclass_id, num_enodes, log_dp_cp, log_dp, log_size, dtype, mem_type, log_mem_cap]
    }

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Action feature encoders per decision phase
        self.action_encoders = nn.ModuleDict(
            {
                str(phase): nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for phase, dim in self.ACTION_DIMS.items()
            }
        )

        # FAST C++ RNN: Replaced GRUCell with GRU to avoid the extremely slow Python unrolling
        self.rnn = nn.GRU(hidden_dim, hidden_dim)

        # Direct score/cost prediction head
        self.cost_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def init_hidden(
        self, batch_size: int = 1, device: torch.device | None = None
    ) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(
            (batch_size, self.hidden_dim), dtype=torch.float32, device=device
        )

    def encode_action(self, action_feat: torch.Tensor, phase_id: int) -> torch.Tensor:
        encoder = self.action_encoders[str(phase_id)]
        expected_dim = self.ACTION_DIMS[phase_id]
        if action_feat.shape[-1] < expected_dim:
            pad = torch.zeros(
                (*action_feat.shape[:-1], expected_dim - action_feat.shape[-1]),
                dtype=action_feat.dtype,
                device=action_feat.device,
            )
            action_feat = torch.cat([action_feat, pad], dim=-1)
        elif action_feat.shape[-1] > expected_dim:
            action_feat = action_feat[..., :expected_dim]
        return encoder(action_feat)

    def forward(
        self, hidden: torch.Tensor, action_feat: torch.Tensor, phase_id: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Step transition: next_hidden, pred_cost = model(hidden, action_feat, phase_id)"""
        act_emb = self.encode_action(action_feat, phase_id)

        # nn.GRU expects sequence length and num_layers dims: (Seq, Batch, Hidden)
        _, next_hidden = self.rnn(act_emb.unsqueeze(0), hidden.unsqueeze(0))
        next_hidden = next_hidden.squeeze(0)

        pred_cost = self.cost_head(next_hidden).squeeze(-1)
        return next_hidden, pred_cost

    def evaluate_candidates(
        self, hidden: torch.Tensor, action_candidates: torch.Tensor, phase_id: int
    ) -> torch.Tensor:
        """Evaluates all candidate actions for a state in a single vectorized pass."""
        if hidden.dim() == 1:
            hidden = hidden.unsqueeze(0)
        num_actions = action_candidates.shape[0]
        hidden_exp = hidden.expand(num_actions, -1)
        act_emb = self.encode_action(action_candidates, phase_id)

        # Format for nn.GRU single-step rollout
        _, next_hidden = self.rnn(act_emb.unsqueeze(0), hidden_exp.unsqueeze(0))
        next_hidden = next_hidden.squeeze(0)

        pred_costs = self.cost_head(next_hidden).squeeze(-1)
        return pred_costs

    def unroll_trajectories(
        self, batch_trajectories: list[dict], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Vectorized server-side unroll of a batch of trajectories from h_0 = 0."""
        B = len(batch_trajectories)
        lengths = [int(t["length"]) for t in batch_trajectories]
        max_len = max(lengths)

        # 1. Pack padded numpy arrays on CPU
        padded_actions = np.zeros((max_len, B, 8), dtype=np.float32)
        padded_phases = np.zeros((max_len, B), dtype=np.int32)
        mask = np.zeros((max_len, B), dtype=bool)

        flat_targets_list: list[float] = []

        for b_idx, traj in enumerate(batch_trajectories):
            L = lengths[b_idx]
            acts = traj["actions"]
            if acts.ndim == 1:
                acts = acts.reshape(L, -1)
            act_dim = acts.shape[1]
            padded_actions[:L, b_idx, : min(act_dim, 8)] = acts[:L, : min(act_dim, 8)]
            padded_phases[:L, b_idx] = traj["phases"][:L]
            mask[:L, b_idx] = True

            raw_c = float(traj["cost"])
            if raw_c < 0.0:
                t_val = raw_c
            else:
                t_val = float(np.log1p(min(max(raw_c, 0.0), 1e20)))
            flat_targets_list.extend([t_val] * L)

        act_t = torch.from_numpy(padded_actions).to(device)  # (max_len, B, 8)
        phases_t = torch.from_numpy(padded_phases).to(device)  # (max_len, B)
        mask_t = torch.from_numpy(mask).to(device)  # (max_len, B)
        targets_flat = torch.tensor(
            flat_targets_list, dtype=torch.float32, device=device
        )

        # 2. Vectorized Action Encoding (6 MLP calls across the entire batch)
        flat_act = act_t.reshape(max_len * B, 8)
        flat_phases = phases_t.reshape(max_len * B)
        flat_mask = mask_t.reshape(max_len * B)

        flat_emb = torch.zeros(
            max_len * B, self.hidden_dim, dtype=torch.float32, device=device
        )

        for phase_id, dim in self.ACTION_DIMS.items():
            p_idx = torch.nonzero(
                flat_mask & (flat_phases == phase_id), as_tuple=False
            ).squeeze(-1)
            if p_idx.numel() > 0:
                p_feats = flat_act[p_idx, :dim]
                flat_emb[p_idx] = self.action_encoders[str(phase_id)](p_feats)

        seq_emb = flat_emb.reshape(max_len, B, self.hidden_dim)

        # 3. Recurrent GRU Rollout through time
        # REPLACED the slow Python `for t in range(max_len):` loop with optimized C++ engine
        hidden = torch.zeros(1, B, self.hidden_dim, dtype=torch.float32, device=device)
        all_hiddens_t, _ = self.rnn(seq_emb, hidden)  # (max_len, B, hidden_dim)

        # 4. Predict cost for all active steps in a single batched pass
        # FIXED BUG: Transpose to (B, max_len, H) before reshaping so that predictions
        # are grouped by batch. This perfectly matches the batch-grouped order of `targets_flat`.
        all_hiddens_b = all_hiddens_t.transpose(0, 1).reshape(
            B * max_len, self.hidden_dim
        )
        mask_b = mask_t.transpose(0, 1).reshape(B * max_len)

        active_hiddens = all_hiddens_b[mask_b]
        pred_costs_flat = self.cost_head(active_hiddens).squeeze(-1)

        return pred_costs_flat, targets_flat, lengths

import torch
from torch import nn


class CostPredictorRNN(nn.Module):
    """RNN model that predicts the final execution score directly:

    hidden, pred_cost = model(hidden, action_feat, phase_id)
    """

    ACTION_DIMS = {
        0: 5,  # Cache: [is_cached, log_size, mem_type, op_type, num_users]
        1: 7,  # Extract (ENode): [log_cost, log_dp_cost, log_size, mem_type, eng_len, num_nodes, num_edges]
        2: 7,  # Dispatch: [log_cost, log_dp_cost, log_size, mem_type, eng_len, num_nodes, num_edges]
        3: 4,  # Bufferize: [is_new_buffer, log_size, log_parent_size, parent_birth_time]
        4: 4,  # Malloc: [log_size, start, end, log_mem_cap]
        5: 7,  # Frontier: [eclass_id, num_enodes, log_dp_cp, log_dp, log_size, dtype, mem_type]
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

        # Recurrent state transition cell
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)

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
        """Step transition: next_hidden, pred_cost = model(hidden, action_feat,
        phase_id)"""
        act_emb = self.encode_action(action_feat, phase_id)
        next_hidden = self.rnn(act_emb, hidden)
        pred_cost = self.cost_head(next_hidden).squeeze(-1)
        return next_hidden, pred_cost

    def evaluate_candidates(
        self, hidden: torch.Tensor, action_candidates: torch.Tensor, phase_id: int
    ) -> torch.Tensor:
        """Evaluates all candidate actions for a state in a single vectorized
        pass.

        hidden: [1, H] or [H] action_candidates: [A, D] returns: pred_costs [A]
        """
        if hidden.dim() == 1:
            hidden = hidden.unsqueeze(0)
        num_actions = action_candidates.shape[0]
        hidden_exp = hidden.expand(num_actions, -1)
        act_emb = self.encode_action(action_candidates, phase_id)
        next_hidden = self.rnn(act_emb, hidden_exp)
        pred_costs = self.cost_head(next_hidden).squeeze(-1)
        return pred_costs

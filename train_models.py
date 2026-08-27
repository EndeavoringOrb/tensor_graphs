# File: train_models.py
import torch
import torch.nn.functional as F
from torch import nn


class GlobalLocalGraphAttention(nn.Module):
    """
    Multi-Head Sparse Graph Attention with Global-Local Factorization.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        assert d_model % nhead == 0, (
            f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        )

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = 1.0 / (self.head_dim**0.5)
        self.add_self_loops = add_self_loops
        self.dropout_p = dropout

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def _prepare_edges(
        self, edge_index: torch.Tensor, num_nodes: int, device: torch.device
    ) -> torch.Tensor:
        if not self.add_self_loops or num_nodes == 0:
            return edge_index
        loop_index = torch.arange(0, num_nodes, dtype=torch.int64, device=device)
        loop_edges = torch.stack([loop_index, loop_index], dim=0)
        if edge_index.numel() == 0:
            return loop_edges
        return torch.cat([edge_index, loop_edges], dim=1)

    def forward(
        self,
        x_global: torch.Tensor,
        x_nodes: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, G, D = x_global.shape
        _, N, _ = x_nodes.shape
        device = x_nodes.device

        if N == 0:
            q_g = (
                self.q_proj(x_global)
                .view(B, G, self.nhead, self.head_dim)
                .transpose(1, 2)
            )
            k_g = (
                self.k_proj(x_global)
                .view(B, G, self.nhead, self.head_dim)
                .transpose(1, 2)
            )
            v_g = (
                self.v_proj(x_global)
                .view(B, G, self.nhead, self.head_dim)
                .transpose(1, 2)
            )
            scores_g = torch.matmul(q_g, k_g.transpose(-2, -1)) * self.scale
            attn_g = F.softmax(scores_g, dim=-1)
            out_g = torch.matmul(attn_g, v_g).transpose(1, 2).contiguous().view(B, G, D)
            return self.out_proj(out_g), x_nodes

        edges = self._prepare_edges(edge_index, N, device)
        src = edges[0].long()
        dst = edges[1].long()
        E = src.size(0)

        q_g = (
            self.q_proj(x_global).view(B, G, self.nhead, self.head_dim).transpose(1, 2)
        )
        k_g = (
            self.k_proj(x_global).view(B, G, self.nhead, self.head_dim).transpose(1, 2)
        )
        v_g = (
            self.v_proj(x_global).view(B, G, self.nhead, self.head_dim).transpose(1, 2)
        )

        q_n = self.q_proj(x_nodes).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        k_n = self.k_proj(x_nodes).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        v_n = self.v_proj(x_nodes).view(B, N, self.nhead, self.head_dim).transpose(1, 2)

        k_all = torch.cat([k_g, k_n], dim=2)
        v_all = torch.cat([v_g, v_n], dim=2)
        scores_g = torch.matmul(q_g, k_all.transpose(-2, -1)) * self.scale
        attn_g = self.dropout(F.softmax(scores_g, dim=-1))
        out_g = torch.matmul(attn_g, v_all).transpose(1, 2).contiguous().view(B, G, D)
        out_g = self.out_proj(out_g)

        scores_n_to_g = torch.matmul(q_n, k_g.transpose(-2, -1)) * self.scale
        q_dst = q_n[:, :, dst, :]
        k_src = k_n[:, :, src, :]
        scores_local = (q_dst * k_src).sum(dim=-1) * self.scale

        global_max = scores_n_to_g.amax(dim=-1)
        dst_expanded = dst.view(1, 1, E).expand(B, self.nhead, E)
        local_max = torch.full(
            (B, self.nhead, N),
            -torch.inf,
            dtype=scores_local.dtype,
            device=scores_local.device,
        )
        local_max.scatter_reduce_(
            dim=2,
            index=dst_expanded,
            src=scores_local,
            reduce="amax",
            include_self=False,
        )
        joint_max = torch.maximum(global_max, local_max)

        exp_g = torch.exp(scores_n_to_g - joint_max.unsqueeze(-1))
        joint_max_dst = joint_max.gather(dim=2, index=dst_expanded)
        exp_local = torch.exp(scores_local - joint_max_dst)

        sum_exp_g = exp_g.sum(dim=-1)
        sum_exp_local = torch.zeros(
            (B, self.nhead, N), dtype=exp_local.dtype, device=exp_local.device
        )
        sum_exp_local.scatter_reduce_(
            dim=2, index=dst_expanded, src=exp_local, reduce="sum", include_self=False
        )
        denom = (sum_exp_g + sum_exp_local).clamp(min=1e-8)

        attn_n_to_g = self.dropout(exp_g / denom.unsqueeze(-1))
        denom_dst = denom.gather(dim=2, index=dst_expanded)
        attn_local = self.dropout(exp_local / denom_dst)

        out_n_from_g = torch.matmul(attn_n_to_g, v_g)

        v_src = v_n[:, :, src, :]
        msg = attn_local.unsqueeze(-1) * v_src
        dst_msg_expanded = dst.view(1, 1, E, 1).expand(B, self.nhead, E, self.head_dim)
        out_n_from_local = torch.zeros(
            (B, self.nhead, N, self.head_dim), dtype=msg.dtype, device=msg.device
        )
        out_n_from_local.scatter_reduce_(
            dim=2, index=dst_msg_expanded, src=msg, reduce="sum", include_self=False
        )

        out_n = (
            (out_n_from_g + out_n_from_local).transpose(1, 2).contiguous().view(B, N, D)
        )
        out_n = self.out_proj(out_n)

        return out_g, out_n


class GraphTransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int):
        super().__init__()
        self.norm1_g = nn.LayerNorm(d_model)
        self.norm1_n = nn.LayerNorm(d_model)
        self.attn = GlobalLocalGraphAttention(d_model, nhead)

        self.norm2_g = nn.LayerNorm(d_model)
        self.norm2_n = nn.LayerNorm(d_model)
        self.mlp_g = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)
        )
        self.mlp_n = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)
        )

    def forward(
        self,
        x_global: torch.Tensor,
        x_nodes: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        norm_g = self.norm1_g(x_global)
        norm_n = self.norm1_n(x_nodes) if x_nodes.size(1) > 0 else x_nodes

        attn_g, attn_n = self.attn(norm_g, norm_n, edge_index)
        x_global = x_global + attn_g
        x_nodes = x_nodes + attn_n if x_nodes.size(1) > 0 else x_nodes

        x_global = x_global + self.mlp_g(self.norm2_g(x_global))
        if x_nodes.size(1) > 0:
            x_nodes = x_nodes + self.mlp_n(self.norm2_n(x_nodes))

        return x_global, x_nodes


class ActionCrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = 1.0 / (self.head_dim**0.5)

        self.norm_act = nn.LayerNorm(d_model)
        self.norm_ctx = nn.LayerNorm(d_model)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x_act: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, A, D = x_act.shape
        _, C, _ = context.shape

        q_in = self.norm_act(x_act)
        kv_in = torch.cat([self.norm_ctx(context), q_in], dim=1)

        q = self.q_proj(q_in).view(B, A, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_in).view(B, C + A, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_in).view(B, C + A, self.nhead, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, A, D)
        x_act = x_act + self.out_proj(out)
        x_act = x_act + self.mlp(self.norm_mlp(x_act))
        return x_act


class AlphaZeroTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        max_feat_dim: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_feat_dim = max_feat_dim

        self.feat_proj = nn.Linear(max_feat_dim, d_model)
        self.type_emb = nn.Embedding(4, d_model)
        self.phase_emb = nn.Embedding(6, d_model)

        self.graph_layers = nn.ModuleList(
            [GraphTransformerBlock(d_model, nhead) for _ in range(num_layers)]
        )
        self.action_layers = nn.ModuleList(
            [ActionCrossAttentionBlock(d_model, nhead) for _ in range(num_layers)]
        )

        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )

    def encode_prefix(
        self,
        global_features: torch.Tensor,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        phase_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = node_features.shape
        device = node_features.device

        pid_g = phase_ids.view(B, 1)
        x_g = (
            self.feat_proj(global_features)
            + self.type_emb(torch.zeros((B, 1), dtype=torch.int64, device=device))
            + self.phase_emb(pid_g)
        )

        if N > 0:
            pid_n = phase_ids.view(B, 1).expand(B, N)
            x_n = (
                self.feat_proj(node_features)
                + self.type_emb(torch.ones((B, N), dtype=torch.int64, device=device))
                + self.phase_emb(pid_n)
            )
        else:
            x_n = torch.zeros((B, 0, self.d_model), dtype=x_g.dtype, device=device)

        for layer in self.graph_layers:
            x_g, x_n = layer(x_g, x_n, edge_index)

        v_pred = self.value_head(x_g[:, 0]).squeeze(-1)
        return v_pred, x_g

    def evaluate_actions(
        self,
        action_features: torch.Tensor,
        phase_ids: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        B, A, _ = action_features.shape
        device = action_features.device

        token_types = torch.full((B, A), 3, dtype=torch.int64, device=device)
        x_act = (
            self.feat_proj(action_features)
            + self.type_emb(token_types)
            + self.phase_emb(phase_ids)
        )

        for layer in self.action_layers:
            x_act = layer(x_act, context)

        logits = self.policy_head(x_act).squeeze(-1)
        return logits


# =============================================================================
# REINFORCE Policy-Value RNN Architecture
# =============================================================================
class PolicyValueRNN(nn.Module):
    """
    Recurrent Policy-Value Network for REINFORCE search tree optimization.

    Signatures:
      value, hidden = model(hidden, global_context, chosen_action, phase_id)
      logits, value = model.evaluate(hidden, global_context, candidate_actions, phase_id)

    Features:
      - Dedicated action embedding networks per action type (Cache, Extract, Dispatch, Bufferize, Malloc, Frontier).
      - Global context encoder incorporating memory limits, search depth, and phase IDs.
      - GRU-based state transition cell for tracking tree search paths.
      - Value head V(s) and action scoring Policy head pi(a | s).
    """

    ACTION_DIMS = {
        0: 5,  # Cache: is_cached, log_size, mem_type, op_type, num_users
        1: 7,  # Extract (ENode): log_cost, log_dp_cost, log_size, mem_type, eng_len, num_nodes, num_edges
        2: 7,  # Dispatch: log_cost, log_dp_cost, log_size, mem_type, eng_len, num_nodes, num_edges
        3: 4,  # Bufferize: is_new_buf, log_size, log_parent_size, parent_birth_time
        4: 4,  # Malloc: log_size, start, end, log_mem_cap
        5: 7,  # Frontier: eclass_id, num_enodes, log_dp_cp, log_dp, log_size, dtype, mem_type
    }

    def __init__(self, hidden_dim: int = 64, global_dim: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.global_dim = global_dim

        # 1. Dedicated Action Encoders for each decision phase
        self.action_encoders = nn.ModuleDict(
            {
                "0": nn.Sequential(
                    nn.Linear(self.ACTION_DIMS[0], hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                ),
                "1": nn.Sequential(
                    nn.Linear(self.ACTION_DIMS[1], hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                ),
                "2": nn.Sequential(
                    nn.Linear(self.ACTION_DIMS[2], hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                ),
                "3": nn.Sequential(
                    nn.Linear(self.ACTION_DIMS[3], hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                ),
                "4": nn.Sequential(
                    nn.Linear(self.ACTION_DIMS[4], hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                ),
                "5": nn.Sequential(
                    nn.Linear(self.ACTION_DIMS[5], hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                ),
            }
        )

        # 2. Global Context Encoder (includes memory limits, normalized depth, phase)
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 3. Recurrent Cell: updates hidden state given (Action Emb + Global Emb)
        self.rnn_cell = nn.GRUCell(hidden_dim * 2, hidden_dim)

        # 4. State Value Head: V(s) from (Hidden + Global)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # 5. Policy Scorer: Computes matching logits between state context and action candidate embeddings
        self.policy_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
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

    def encode_global(self, global_feat: torch.Tensor) -> torch.Tensor:
        """
        global_feat: [B, global_dim] -> [B, hidden_dim]
        """
        return self.global_encoder(global_feat)

    def encode_action(self, action_feat: torch.Tensor, phase_id: int) -> torch.Tensor:
        """
        action_feat: [B, A, feat_dim] or [B, feat_dim] -> [B, A, hidden_dim] or [B, hidden_dim]
        """
        encoder = self.action_encoders[str(phase_id)]
        expected_dim = self.ACTION_DIMS[phase_id]
        if action_feat.shape[-1] < expected_dim:
            pad_shape = list(action_feat.shape)
            pad_shape[-1] = expected_dim - action_feat.shape[-1]
            pad = torch.zeros(
                pad_shape, dtype=action_feat.dtype, device=action_feat.device
            )
            action_feat = torch.cat([action_feat, pad], dim=-1)
        elif action_feat.shape[-1] > expected_dim:
            action_feat = action_feat[..., :expected_dim]
        return encoder(action_feat)

    def step(
        self,
        hidden: torch.Tensor,
        global_feat: torch.Tensor,
        chosen_action_feat: torch.Tensor,
        phase_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the recurrent state when descending a chosen branch:
          value, next_hidden = model(hidden, global_context, chosen_action, phase_id)

        Args:
            hidden: [B, hidden_dim]
            global_feat: [B, global_dim]
            chosen_action_feat: [B, feat_dim]
            phase_id: int (0..5)

        Returns:
            value: [B, 1] Expected return from the current state
            next_hidden: [B, hidden_dim] Updated tree search hidden state
        """
        g_emb = self.encode_global(global_feat)
        a_emb = self.encode_action(chosen_action_feat, phase_id)

        # Compute State Value from (hidden, g_emb)
        state_ctx = torch.cat([hidden, g_emb], dim=-1)
        value = self.value_head(state_ctx)

        # Recurrent state transition
        rnn_in = torch.cat([a_emb, g_emb], dim=-1)
        next_hidden = self.rnn_cell(rnn_in, hidden)

        return value, next_hidden

    def forward(
        self,
        hidden: torch.Tensor,
        global_feat: torch.Tensor,
        chosen_action_feat: torch.Tensor,
        phase_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.step(hidden, global_feat, chosen_action_feat, phase_id)

    def evaluate_candidates(
        self,
        hidden: torch.Tensor,
        global_feat: torch.Tensor,
        action_candidates: torch.Tensor,
        phase_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates candidate actions at a decision node:
          logits, value = model.evaluate_candidates(hidden, global_context, candidate_actions, phase_id)

        Args:
            hidden: [B, hidden_dim]
            global_feat: [B, global_dim]
            action_candidates: [B, A, feat_dim]
            phase_id: int

        Returns:
            logits: [B, A] Action selection logits
            value: [B, 1] Baseline state value estimate
        """
        B, A, _ = action_candidates.shape
        g_emb = self.encode_global(global_feat)  # [B, hidden_dim]
        a_embs = self.encode_action(action_candidates, phase_id)  # [B, A, hidden_dim]

        state_ctx = torch.cat([hidden, g_emb], dim=-1)  # [B, 2*hidden_dim]
        value = self.value_head(state_ctx)  # [B, 1]

        # Expand state context across all candidate actions [B, A, 2*hidden_dim]
        expanded_state = state_ctx.unsqueeze(1).expand(-1, A, -1)
        score_in = torch.cat([expanded_state, a_embs], dim=-1)  # [B, A, 3*hidden_dim]
        logits = self.policy_scorer(score_in).squeeze(-1)  # [B, A]

        return logits, value

import torch
import torch.nn.functional as F
from torch import nn


class GlobalLocalGraphAttention(nn.Module):
    """
    Multi-Head Sparse Graph Attention with Global-Local Factorization.

    Complexity:
      - Global tokens (G): Dense attention to [Global + Nodes] -> O(G * (G + N))
      - Node tokens (N): Joint sparse attention to [Global] + [Incoming Graph Neighbors] -> O(N*G + E)
      - Total Time & Memory: O(N + E + G*N) instead of O((G + N + E)^2)

    Zero full attention matrices are materialized, making it fast on both CPU and GPU.
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

        # Unified Q, K, V projections
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
        """
        Args:
            x_global: [B, G, D] Global/phase tokens (e.g., G=1)
            x_nodes:  [B, N, D] Graph node tokens
            edge_index: [2, E] Directed edges (src -> dst), where node indices are in [0, N-1]
        """
        B, G, D = x_global.shape
        _, N, _ = x_nodes.shape
        device = x_nodes.device

        # Handle empty node graphs
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

        # Prepare edges with self-loops, ensuring torch.int64 index dtype
        edges = self._prepare_edges(edge_index, N, device)
        src = edges[0].long()
        dst = edges[1].long()
        E = src.size(0)

        # Q, K, V Projections
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

        # ---------------------------------------------------------------------
        # Global Attention: Global tokens attend to [Global + Nodes]
        # ---------------------------------------------------------------------
        k_all = torch.cat([k_g, k_n], dim=2)
        v_all = torch.cat([v_g, v_n], dim=2)
        scores_g = torch.matmul(q_g, k_all.transpose(-2, -1)) * self.scale
        attn_g = self.dropout(F.softmax(scores_g, dim=-1))
        out_g = torch.matmul(attn_g, v_all).transpose(1, 2).contiguous().view(B, G, D)
        out_g = self.out_proj(out_g)

        # ---------------------------------------------------------------------
        # Local Node Attention: Nodes jointly attend to Global and Neighbors
        # ---------------------------------------------------------------------
        scores_n_to_g = (
            torch.matmul(q_n, k_g.transpose(-2, -1)) * self.scale
        )  # [B, H, N, G]

        q_dst = q_n[:, :, dst, :]  # [B, H, E, D_h]
        k_src = k_n[:, :, src, :]  # [B, H, E, D_h]
        scores_local = (q_dst * k_src).sum(dim=-1) * self.scale  # [B, H, E]

        # Stable Joint Softmax (matching exact dtype with src to avoid AMP mismatch)
        global_max = scores_n_to_g.amax(dim=-1)  # [B, H, N]
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

        # Message Aggregation
        out_n_from_g = torch.matmul(attn_n_to_g, v_g)  # [B, H, N, D_h]

        v_src = v_n[:, :, src, :]
        msg = attn_local.unsqueeze(-1) * v_src  # [B, H, E, D_h]
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
    """
    Evaluates candidate action choices by cross-attending to the graph's global summary token
    and applying self-attention across candidate choices.
    """

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
        """
        x_act:   [B, A, D] (Candidate action tokens)
        context: [B, C, D] (Encoded prefix context, e.g., global token [B, 1, D])
        """
        B, A, D = x_act.shape
        _, C, _ = context.shape

        q_in = self.norm_act(x_act)
        kv_in = torch.cat([self.norm_ctx(context), q_in], dim=1)  # [B, C + A, D]

        q = self.q_proj(q_in).view(B, A, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_in).view(B, C + A, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_in).view(B, C + A, self.nhead, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, A, D)
        x_act = x_act + self.out_proj(out)
        x_act = x_act + self.mlp(self.norm_mlp(x_act))
        return x_act


class AlphaZeroTransformer(nn.Module):
    """
    AlphaZero Graph-Transformer with Sparse Factorized Attention.
    """

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
        self.phase_emb = nn.Embedding(5, d_model)

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
        """
        Encodes the graph prefix in O(N + E).

        Args:
            global_features: [B, 1, 8]
            node_features:   [B, N, 8]
            edge_index:      [2, E]
            phase_ids:       [B]

        Returns:
            v_pred: [B] Value prediction
            h_global: [B, 1, D] Latent context for action evaluation
        """
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
        """
        Evaluates candidate actions against the graph prefix context.

        Args:
            action_features: [B, A, 8]
            phase_ids:       [B, A]
            context:         [B, 1, D] (Encoded global prefix token)

        Returns:
            logits: [B, A] Action logits
        """
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

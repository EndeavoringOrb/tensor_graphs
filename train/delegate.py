import math
import random

import numpy as np
import tensor_graphs
import torch

from .buffer import rank_score_indices
from .model import CostPredictorRNN


def safe_float(v: float, default: float = float("nan")) -> float:
    try:
        val = float(v)
        return val if math.isfinite(val) else default
    except Exception:
        return default


def safe_log1p(v: float, default: float = 30.0) -> float:
    try:
        val = float(v)
        if not math.isfinite(val) or val < 0.0:
            return default if (math.isinf(val) and val > 0) else 0.0
        val = min(val, 1e20)
        res = math.log1p(val)
        return min(res, 30.0) if math.isfinite(res) else default
    except Exception:
        return default


class CostPredictorDelegate(tensor_graphs.SearchDelegate):
    """In-process search delegate for TensorGraph C++ planner integration.

    Evaluates candidates using the local RNN and collects completed trajectories
    consisting only of chosen action vectors and phase IDs.
    """

    PHASE_MAP = {
        "cache": 0,
        "extract": 1,
        "dispatch": 2,
        "bufferize": 3,
        "malloc": 4,
        "frontier": 5,
    }

    def __init__(
        self,
        model: CostPredictorRNN,
        epsilon: float = 0.1,
        is_training: bool = True,
        device: torch.device | None = None,
        max_trajectories_per_episode: int = 1500,
    ):
        super().__init__()
        self.model = model
        self.device = device if device is not None else next(model.parameters()).device
        self.epsilon = epsilon
        self.is_training = is_training
        self.max_trajectories_per_episode = max_trajectories_per_episode

        self.current_hidden = self.model.init_hidden(batch_size=1, device=self.device)
        self.hidden_stack: list[torch.Tensor] = []
        self.path_len_stack: list[int] = []
        self.active_path: list[tuple[int, np.ndarray]] = []
        self.completed_trajectories: list[dict] = []

    def export_and_reset(self) -> tuple[list[dict], list[float]]:
        """Exports all completed trajectories collected during the episode and resets state."""
        trajectories = self.completed_trajectories
        leaf_costs = [float(t["cost"]) for t in trajectories]

        # Always clear internal state to avoid memory leaks regardless of whether trajectories were found
        self.completed_trajectories = []
        self.active_path.clear()
        self.hidden_stack.clear()
        self.path_len_stack.clear()

        return trajectories, leaf_costs

    def reset(self):
        self.current_hidden = self.model.init_hidden(batch_size=1, device=self.device)
        self.hidden_stack.clear()
        self.path_len_stack.clear()
        self.active_path.clear()
        self.completed_trajectories.clear()

    def fast_fail(self) -> bool:
        return False

    def push_state(self):
        self.hidden_stack.append(self.current_hidden.clone())
        self.path_len_stack.append(len(self.active_path))

    def pop_state(self):
        if self.hidden_stack:
            self.current_hidden = self.hidden_stack.pop()
        if self.path_len_stack:
            target_len = self.path_len_stack.pop()
            while len(self.active_path) > target_len:
                self.active_path.pop()

    def on_leaf_evaluated(self, cost: float):
        cost_val = safe_float(cost)
        if math.isfinite(cost_val) and self.is_training:
            if len(self.completed_trajectories) >= self.max_trajectories_per_episode:
                return
            if not self.active_path:
                return

            phases = np.array([p for p, _ in self.active_path], dtype=np.int32)
            actions = np.array([a for _, a in self.active_path], dtype=np.float32)
            self.completed_trajectories.append(
                {
                    "phases": phases,
                    "actions": actions,
                    "cost": cost_val,
                    "length": len(phases),
                }
            )

    @torch.inference_mode()
    def _order_items(self, items, phase_name: str, extract_fn) -> list[int]:
        num_actions = len(items)
        if num_actions <= 1:
            return list(range(num_actions))

        phase_id = self.PHASE_MAP[phase_name]
        action_feats_t = extract_fn(items).to(self.device)

        pred_costs = self.model.evaluate_candidates(
            self.current_hidden, action_feats_t, phase_id
        )

        order = rank_score_indices(pred_costs).cpu().tolist()

        if self.is_training and random.random() < self.epsilon:
            chosen_idx = random.randrange(num_actions)
            if chosen_idx in order:
                order.remove(chosen_idx)
                order.insert(0, chosen_idx)
        else:
            chosen_idx = order[0]

        # Record action vector into active trajectory path
        if self.is_training:
            chosen_feat_np = action_feats_t[chosen_idx].cpu().numpy()
            padded_act = np.zeros(8, dtype=np.float32)
            dim = min(len(chosen_feat_np), 8)
            padded_act[:dim] = chosen_feat_np[:dim]
            self.active_path.append((phase_id, padded_act))

        # Advance local hidden state along the chosen branch
        chosen_t = action_feats_t[chosen_idx : chosen_idx + 1]
        self.current_hidden, _ = self.model(self.current_hidden, chosen_t, phase_id)
        self.current_hidden = torch.nan_to_num(
            self.current_hidden, nan=0.0, posinf=10.0, neginf=-10.0
        )

        return order

    def order_cache(self, choices):
        return self._order_items(choices, "cache", self._extract_cache_features)

    def order_enodes(self, enodes):
        return self._order_items(enodes, "extract", self._extract_dispatch_features)

    def order_dispatch(self, ready_nodes):
        return self._order_items(
            ready_nodes, "dispatch", self._extract_dispatch_features
        )

    def order_bufferize(self, choices):
        return self._order_items(choices, "bufferize", self._extract_bufferize_features)

    def order_malloc(self, avail_buffers):
        return self._order_items(avail_buffers, "malloc", self._extract_malloc_features)

    def order_frontier(self, frontier):
        return self._order_items(frontier, "frontier", self._extract_frontier_features)

    def _extract_cache_features(self, items):
        feats = [
            [
                safe_float(f.is_cached, default=0.0),
                safe_log1p(f.size),
                safe_float(f.num_users, default=0.0),
                safe_float(f.logical_id, default=0.0),
                safe_float(
                    f.mem_space.type if hasattr(f, "mem_space") else 0,
                    default=0.0,
                ),
                safe_log1p(f.mem_cap if hasattr(f, "mem_cap") else 0),
            ]
            for f in items
        ]
        t = torch.tensor(feats, dtype=torch.float32)
        return torch.nan_to_num(t, nan=0.0, posinf=30.0, neginf=0.0)

    def _extract_dispatch_features(self, items):
        feats = [
            [
                safe_log1p(f.cost),
                safe_log1p(f.dp_cost),
                safe_log1p(f.min_dp_cp_cost, 0.0),
                safe_log1p(f.rev_cp_cost, 0.0),
                safe_log1p(f.size),
                safe_float(
                    f.mem_space.type if hasattr(f, "mem_space") else 0,
                    default=0.0,
                ),
                safe_float(
                    len(f.engine_idxs) if hasattr(f, "engine_idxs") else 0, default=0.0
                ),
                safe_float(f.num_nodes, default=0.0),
                safe_float(f.num_edges, default=0.0),
                safe_log1p(f.mem_cap if hasattr(f, "mem_cap") else 0),
            ]
            for f in items
        ]
        t = torch.tensor(feats, dtype=torch.float32)
        return torch.nan_to_num(t, nan=0.0, posinf=30.0, neginf=0.0)

    def _extract_bufferize_features(self, items):
        feats = [
            [
                safe_float(f.is_new_buffer, default=0.0),
                safe_log1p(f.size),
                safe_log1p(f.parent_size),
                safe_float(f.parent_birth_time, default=0.0),
                safe_float(
                    f.mem_space.type if hasattr(f, "mem_space") else 0,
                    default=0.0,
                ),
                safe_log1p(f.mem_cap if hasattr(f, "mem_cap") else 0),
            ]
            for f in items
        ]
        t = torch.tensor(feats, dtype=torch.float32)
        return torch.nan_to_num(t, nan=0.0, posinf=30.0, neginf=0.0)

    def _extract_malloc_features(self, items):
        feats = [
            [
                safe_log1p(f.size),
                safe_float(f.start, default=0.0),
                safe_float(f.end, default=0.0),
                safe_float(
                    f.mem_space.type if hasattr(f, "mem_space") else 0,
                    default=0.0,
                ),
                safe_log1p(f.mem_cap if hasattr(f, "mem_cap") else 0),
            ]
            for f in items
        ]
        t = torch.tensor(feats, dtype=torch.float32)
        return torch.nan_to_num(t, nan=0.0, posinf=30.0, neginf=0.0)

    def _extract_frontier_features(self, items):
        feats = [
            [
                safe_float(f.eclass_id, default=0.0),
                safe_float(f.num_enodes, default=0.0),
                safe_log1p(f.min_dp_cp_cost),
                safe_log1p(f.min_dp_cost),
                safe_log1p(f.size),
                safe_float(f.dtype, default=0.0),
                safe_float(
                    f.mem_space.type if hasattr(f, "mem_space") else 0,
                    default=0.0,
                ),
                safe_log1p(f.mem_cap if hasattr(f, "mem_cap") else 0),
            ]
            for f in items
        ]
        t = torch.tensor(feats, dtype=torch.float32)
        return torch.nan_to_num(t, nan=0.0, posinf=30.0, neginf=0.0)

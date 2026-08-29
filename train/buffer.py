from abc import ABC, abstractmethod

import numpy as np
import torch

ACTION_DIMS = {0: 6, 1: 8, 2: 8, 3: 6, 4: 5, 5: 8}


# ==============================================================================
# DUAL-OBJECTIVE SCORE RANKING
# ==============================================================================
def rank_score_indices(scores):
    """Ranks 1D scores/costs from best to worst according to dual-objective logic:
    1. Non-negative costs (>= 0) sorted ascending (lowest latency first).
    2. Negative rewards (< 0) sorted descending (highest progress closer to 0 first).
    3. Non-finite / NaNs at the end.
    """
    if isinstance(scores, torch.Tensor):
        valid = torch.isfinite(scores)
        pos_mask = valid & (scores >= 0)
        neg_mask = valid & (scores < 0)
        invalid_mask = ~valid

        pos_idx = torch.nonzero(pos_mask, as_tuple=False).squeeze(-1)
        neg_idx = torch.nonzero(neg_mask, as_tuple=False).squeeze(-1)
        invalid_idx = torch.nonzero(invalid_mask, as_tuple=False).squeeze(-1)

        pos_sorted = (
            pos_idx[torch.argsort(scores[pos_idx], descending=False)]
            if pos_idx.numel() > 0
            else torch.empty(0, dtype=torch.long, device=scores.device)
        )
        neg_sorted = (
            neg_idx[torch.argsort(scores[neg_idx], descending=True)]
            if neg_idx.numel() > 0
            else torch.empty(0, dtype=torch.long, device=scores.device)
        )

        return torch.cat([pos_sorted, neg_sorted, invalid_idx])

    scores_arr = np.asarray(scores)
    valid = np.isfinite(scores_arr)
    pos_mask = valid & (scores_arr >= 0)
    neg_mask = valid & (scores_arr < 0)
    invalid_mask = ~valid

    pos_idx = np.flatnonzero(pos_mask)
    neg_idx = np.flatnonzero(neg_mask)
    invalid_idx = np.flatnonzero(invalid_mask)

    pos_sorted = (
        pos_idx[np.argsort(scores_arr[pos_idx])]
        if len(pos_idx) > 0
        else np.empty(0, dtype=np.int64)
    )
    neg_sorted = (
        neg_idx[np.argsort(-scores_arr[neg_idx])]
        if len(neg_idx) > 0
        else np.empty(0, dtype=np.int64)
    )

    return np.concatenate([pos_sorted, neg_sorted, invalid_idx])


# ==============================================================================
# EVICTION STRATEGIES
# ==============================================================================
class EvictionStrategy(ABC):
    @abstractmethod
    def select_indices_to_keep(
        self,
        losses: np.ndarray,
        costs: np.ndarray,
        timestamps: np.ndarray,
        max_size: int,
    ) -> np.ndarray:
        """Returns array of indices of items to keep (len <= max_size)."""


class FIFOStrategy(EvictionStrategy):
    """First-In, First-Out: Keeps the most recent items."""

    def select_indices_to_keep(
        self,
        losses: np.ndarray,
        costs: np.ndarray,
        timestamps: np.ndarray,
        max_size: int,
    ) -> np.ndarray:
        total = len(timestamps)
        if total <= max_size:
            return np.arange(total)
        return np.arange(total - max_size, total)


class EjectLowestLossStrategy(EvictionStrategy):
    """Evicts items with the lowest loss (preserves high-loss and untrained items)."""

    def select_indices_to_keep(
        self,
        losses: np.ndarray,
        costs: np.ndarray,
        timestamps: np.ndarray,
        max_size: int,
    ) -> np.ndarray:
        total = len(losses)
        if total <= max_size:
            return np.arange(total)
        priorities = np.where(np.isnan(losses) | ~np.isfinite(losses), 1e9, losses)
        return np.argpartition(-priorities, max_size)[:max_size]


class EjectHighestCostStrategy(EvictionStrategy):
    """Evicts items with worst performance: preserves best plans (lowest latency / highest progress)."""

    def select_indices_to_keep(
        self,
        losses: np.ndarray,
        costs: np.ndarray,
        timestamps: np.ndarray,
        max_size: int,
    ) -> np.ndarray:
        total = len(costs)
        if total <= max_size:
            return np.arange(total)
        ranked_indices = rank_score_indices(costs)
        return ranked_indices[:max_size]


def get_eviction_strategy(name: str) -> EvictionStrategy:
    norm = name.lower().replace("-", "_")
    if norm in ["fifo", "queue"]:
        return FIFOStrategy()
    elif norm in ["lowest_loss", "loss", "min_loss"]:
        return EjectLowestLossStrategy()
    elif norm in ["highest_cost", "cost", "max_cost"]:
        return EjectHighestCostStrategy()
    raise ValueError(
        f"Unknown eviction strategy: {name}. Choose from 'fifo', 'lowest_loss', 'highest_cost'."
    )


# ==============================================================================
# TRAJECTORY REPLAY BUFFER
# ==============================================================================
class TrajectoryReplayBuffer:
    """Stores full search trajectories and manages eviction according to an EvictionStrategy."""

    def __init__(
        self,
        maxlen: int = 50_000,
        strategy: EvictionStrategy | None = None,
    ):
        self.maxlen = maxlen
        self.strategy = strategy if strategy is not None else FIFOStrategy()
        self.trajectories: list[dict] = []
        self.costs: np.ndarray = np.empty((0,), dtype=np.float32)
        self.losses: np.ndarray = np.empty((0,), dtype=np.float32)
        self.timestamps: np.ndarray = np.empty((0,), dtype=np.float64)
        self._time_counter = 0.0

    def add_trajectories(self, new_trajectories: list[dict]) -> None:
        """Adds a list of trajectories to the buffer and applies eviction if capacity is exceeded."""
        if not new_trajectories:
            return

        n = len(new_trajectories)
        new_costs = np.array(
            [float(t["cost"]) for t in new_trajectories], dtype=np.float32
        )
        new_losses = np.full((n,), np.nan, dtype=np.float32)
        new_timestamps = np.arange(
            self._time_counter, self._time_counter + n, dtype=np.float64
        )
        self._time_counter += n

        if len(self.trajectories) + n <= self.maxlen:
            self.trajectories.extend(new_trajectories)
            self.costs = np.concatenate([self.costs, new_costs], axis=0)
            self.losses = np.concatenate([self.losses, new_losses], axis=0)
            self.timestamps = np.concatenate([self.timestamps, new_timestamps], axis=0)
        else:
            all_trajectories = self.trajectories + new_trajectories
            all_costs = np.concatenate([self.costs, new_costs], axis=0)
            all_losses = np.concatenate([self.losses, new_losses], axis=0)
            all_timestamps = np.concatenate([self.timestamps, new_timestamps], axis=0)

            keep_idx = self.strategy.select_indices_to_keep(
                losses=all_losses,
                costs=all_costs,
                timestamps=all_timestamps,
                max_size=self.maxlen,
            )

            self.trajectories = [all_trajectories[i] for i in keep_idx]
            self.costs = all_costs[keep_idx]
            self.losses = all_losses[keep_idx]
            self.timestamps = all_timestamps[keep_idx]

    def sample_batch(self, batch_size: int) -> tuple[list[dict], np.ndarray] | None:
        """Samples a random batch of trajectories from the buffer."""
        total = len(self.trajectories)
        if total == 0:
            return None

        k = min(batch_size, total)
        indices = np.random.choice(total, size=k, replace=False)
        batch = [self.trajectories[i] for i in indices]
        return batch, indices

    def update_losses(
        self, indices: np.ndarray | list[int], losses: np.ndarray | list[float]
    ) -> None:
        """Updates stored loss priorities for specified trajectory indices."""
        idx_arr = np.asarray(indices, dtype=np.int64)
        loss_arr = np.asarray(losses, dtype=np.float32)
        self.losses[idx_arr] = np.nan_to_num(loss_arr, nan=0.0, posinf=1e6, neginf=0.0)

    def __len__(self) -> int:
        return len(self.trajectories)


# Compatibility alias
PhaseReplayBuffer = TrajectoryReplayBuffer

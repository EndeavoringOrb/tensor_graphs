import math
import random
from abc import ABC, abstractmethod
import numpy as np

ACTION_DIMS = {0: 5, 1: 7, 2: 7, 3: 4, 4: 4, 5: 7}


# ==============================================================================
# EVICTION STRATEGIES (Vectorized NumPy)
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
        pass


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
        # Priority: untrained (NaN or inf) gets highest priority 1e9
        priorities = np.where(np.isnan(losses) | ~np.isfinite(losses), 1e9, losses)
        # Keep top max_size priorities
        return np.argpartition(-priorities, max_size)[:max_size]


class EjectHighestCostStrategy(EvictionStrategy):
    """Evicts items with highest cost (preserves fastest/best execution plans)."""

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
        priorities = np.where(np.isnan(costs) | ~np.isfinite(costs), 1e9, costs)
        # Keep lowest max_size costs
        return np.argpartition(priorities, max_size)[:max_size]


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
# COLUMNAR PHASE-PARTITIONED REPLAY BUFFER
# ==============================================================================
class PhaseReplayBuffer:
    """Stores transitions in flat columnar NumPy arrays separated by phase."""

    def __init__(
        self,
        maxlen: int = 50_000,
        hidden_dim: int = 64,
        strategy: EvictionStrategy | None = None,
    ):
        self.maxlen = maxlen
        self.hidden_dim = hidden_dim
        self.strategy = strategy if strategy is not None else FIFOStrategy()

        self.hiddens: dict[int, np.ndarray] = {}
        self.actions: dict[int, np.ndarray] = {}
        self.costs: dict[int, np.ndarray] = {}
        self.losses: dict[int, np.ndarray] = {}
        self.timestamps: dict[int, np.ndarray] = {}
        self.sizes: dict[int, int] = {p: 0 for p in range(6)}
        self._time_counter = 0

        for phase, act_dim in ACTION_DIMS.items():
            self._init_phase(phase, act_dim)

    def _init_phase(self, phase: int, act_dim: int):
        self.hiddens[phase] = np.zeros((self.maxlen, self.hidden_dim), dtype=np.float32)
        self.actions[phase] = np.zeros((self.maxlen, act_dim), dtype=np.float32)
        self.costs[phase] = np.zeros((self.maxlen,), dtype=np.float32)
        self.losses[phase] = np.full((self.maxlen,), np.nan, dtype=np.float32)
        self.timestamps[phase] = np.zeros((self.maxlen,), dtype=np.float64)
        self.sizes[phase] = 0

    def add_batch(self, by_phase: dict[int, dict[str, np.ndarray]]):
        """Inserts pre-vectorized batches of transitions directly per phase."""
        for phase_id_key, arrays in by_phase.items():
            phase_id = int(phase_id_key)
            if phase_id not in self.hiddens:
                continue

            h_in = arrays["hiddens"]
            a_in = arrays["actions"]
            c_in = arrays["costs"]
            n = len(h_in)
            if n == 0:
                continue

            t_in = np.arange(
                self._time_counter, self._time_counter + n, dtype=np.float64
            )
            self._time_counter += n
            l_in = np.full((n,), np.nan, dtype=np.float32)

            curr_sz = self.sizes[phase_id]

            if curr_sz + n <= self.maxlen:
                self.hiddens[phase_id][curr_sz : curr_sz + n] = h_in
                self.actions[phase_id][curr_sz : curr_sz + n] = a_in
                self.costs[phase_id][curr_sz : curr_sz + n] = c_in
                self.losses[phase_id][curr_sz : curr_sz + n] = l_in
                self.timestamps[phase_id][curr_sz : curr_sz + n] = t_in
                self.sizes[phase_id] += n
            else:
                all_h = np.concatenate([self.hiddens[phase_id][:curr_sz], h_in], axis=0)
                all_a = np.concatenate([self.actions[phase_id][:curr_sz], a_in], axis=0)
                all_c = np.concatenate([self.costs[phase_id][:curr_sz], c_in], axis=0)
                all_l = np.concatenate([self.losses[phase_id][:curr_sz], l_in], axis=0)
                all_t = np.concatenate(
                    [self.timestamps[phase_id][:curr_sz], t_in], axis=0
                )

                keep_idx = self.strategy.select_indices_to_keep(
                    losses=all_l,
                    costs=all_c,
                    timestamps=all_t,
                    max_size=self.maxlen,
                )

                k_len = len(keep_idx)
                self.hiddens[phase_id][:k_len] = all_h[keep_idx]
                self.actions[phase_id][:k_len] = all_a[keep_idx]
                self.costs[phase_id][:k_len] = all_c[keep_idx]
                self.losses[phase_id][:k_len] = all_l[keep_idx]
                self.timestamps[phase_id][:k_len] = all_t[keep_idx]
                self.sizes[phase_id] = k_len

    def sample_batch(self, batch_size: int, phase_id: int | None = None) -> dict | None:
        if phase_id is None:
            eligible = [p for p, sz in self.sizes.items() if sz > 0]
            if not eligible:
                return None
            phase_id = random.choice(eligible)

        sz = self.sizes.get(phase_id, 0)
        if sz == 0:
            return None

        k = min(batch_size, sz)
        indices = np.random.choice(sz, size=k, replace=False)
        return {
            "phase_id": phase_id,
            "indices": indices,
            "hiddens": self.hiddens[phase_id][indices],
            "actions": self.actions[phase_id][indices],
            "costs": self.costs[phase_id][indices],
        }

    def update_losses(
        self, phase_id: int, indices: np.ndarray, losses: np.ndarray | list[float]
    ):
        self.losses[phase_id][indices] = np.nan_to_num(
            losses, nan=0.0, posinf=1e6, neginf=0.0
        )

    def __len__(self) -> int:
        return sum(self.sizes.values())

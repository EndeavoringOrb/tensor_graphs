from abc import ABC, abstractmethod
from typing import Any, List, Tuple


class WeightSource(ABC):
    """Abstract base class for weight loading sources."""

    @abstractmethod
    def keys(self) -> List[str]:
        """Returns list of available tensor names."""

    @abstractmethod
    def get_tensor_metadata(self, name: str) -> Tuple[Tuple[int, ...], str]:
        """Returns (shape, dtype_str) without loading full data."""

    @abstractmethod
    def get_tensor(self, name: str) -> Any:
        """Returns the tensor data (np.ndarray or Torch tensor)."""

    def close(self):
        """Release any held resources."""

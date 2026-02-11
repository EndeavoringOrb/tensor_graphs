from abc import ABC, abstractmethod
from typing import Any, Tuple, List


class WeightSource(ABC):
    """Abstract base class for weight loading sources."""

    @abstractmethod
    def keys(self) -> List[str]:
        """Returns list of available tensor names."""
        pass

    @abstractmethod
    def get_tensor_metadata(self, name: str) -> Tuple[Tuple[int, ...], str]:
        """Returns (shape, dtype_str) without loading full data."""
        pass

    @abstractmethod
    def get_tensor(self, name: str) -> Any:
        """Returns the tensor data (np.ndarray or Torch tensor)."""
        pass

    def close(self):
        """Release any held resources."""
        pass

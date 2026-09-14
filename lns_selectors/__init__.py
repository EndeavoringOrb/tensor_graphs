"""Neighborhood selectors for exact CP-SAT large neighborhood search."""

from .protocol import NeighborhoodSelector
from .random import RandomNeighborhoodSelector, RandomSubgraphSelector
from .structural import StructuralNeighborhoodSelector

__all__ = [
    "GnnNeighborhoodSelector",
    "NeighborhoodSelector",
    "NeuralNeighborhoodSelector",
    "RandomNeighborhoodSelector",
    "RandomSubgraphSelector",
    "StructuralNeighborhoodSelector",
]


def __getattr__(name):
    """Load the optional torch-backed selector only when it is requested."""
    if name in {"GnnNeighborhoodSelector", "NeuralNeighborhoodSelector"}:
        from .gnn import GnnNeighborhoodSelector, NeuralNeighborhoodSelector

        return {
            "GnnNeighborhoodSelector": GnnNeighborhoodSelector,
            "NeuralNeighborhoodSelector": NeuralNeighborhoodSelector,
        }[name]
    raise AttributeError(name)

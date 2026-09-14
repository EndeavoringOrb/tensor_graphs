"""Neighborhood selectors for exact CP-SAT large neighborhood search."""

from .random import RandomNeighborhoodSelector, RandomSubgraphSelector


class NeighborhoodSelector:
    """Small selector protocol kept dependency-free for custom selectors."""

    def selectNeighborhood(self, context):
        raise NotImplementedError


__all__ = [
    "GnnNeighborhoodSelector",
    "NeighborhoodSelector",
    "NeuralNeighborhoodSelector",
    "RandomNeighborhoodSelector",
    "RandomSubgraphSelector",
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

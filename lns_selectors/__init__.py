"""Neighborhood selectors for exact CP-SAT large neighborhood search."""

from .gnn import GnnNeighborhoodSelector, NeuralNeighborhoodSelector
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

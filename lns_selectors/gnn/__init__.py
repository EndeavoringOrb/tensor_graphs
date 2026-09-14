"""GNN model, selector, sample-generation, and training helpers."""

__all__ = [
    "GnnNeighborhoodSelector",
    "GraphMessageLayer",
    "NeighborhoodGnn",
    "NeuralNeighborhoodSelector",
]


def __getattr__(name):
    if name in __all__:
        from .gnn import (
            GnnNeighborhoodSelector,
            GraphMessageLayer,
            NeighborhoodGnn,
            NeuralNeighborhoodSelector,
        )

        return {
            "GnnNeighborhoodSelector": GnnNeighborhoodSelector,
            "GraphMessageLayer": GraphMessageLayer,
            "NeighborhoodGnn": NeighborhoodGnn,
            "NeuralNeighborhoodSelector": NeuralNeighborhoodSelector,
        }[name]
    raise AttributeError(name)

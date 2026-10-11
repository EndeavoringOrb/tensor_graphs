"""Dependency-free neighborhood selector protocol."""


class NeighborhoodSelector:
    """Small selector protocol kept dependency-free for custom selectors."""

    def selectNeighborhood(self, context):
        raise NotImplementedError


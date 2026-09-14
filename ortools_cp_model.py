"""Shared exact CP-SAT plan model.

The implementation lives in :mod:`ortools_full` for backwards compatibility.
This module is the stable import point for both the full solver and LNS.  The
full model remains the source of truth; LNS only adds temporary fixings to it.
"""

from ortools_full import OrtoolsSolver, memSpaceKey

__all__ = ["OrtoolsSolver", "memSpaceKey"]

"""
File: tensor_graphs/ops/registry.py
"""

from typing import Dict, Any, Type, Optional
from .interface import CompositeOp

_OP_REGISTRY: Dict[str, Any] = {}
_COMPOSITE_REGISTRY: Dict[str, Type[CompositeOp]] = {}


def register_op(name: str):
    """Decorator to register metadata about an op."""

    def decorator(cls):
        _OP_REGISTRY[name] = cls
        return cls

    return decorator


def get_op_metadata(name: str):
    return _OP_REGISTRY.get(name, None)


def register_composite(cls):
    """Decorator to register a CompositeOp class."""
    if not issubclass(cls, CompositeOp):
        raise ValueError("Must inherit from CompositeOp")

    # Instantiate or just register class? Registering class is safer for statics,
    # but we need an instance to call decompose easily if stateless.
    # Let's verify instance has op_type.
    instance = cls()
    _COMPOSITE_REGISTRY[instance.op_type] = cls
    return cls


def get_composite_op(op_type: str) -> Optional[CompositeOp]:
    """Returns a fresh instance of the composite op handler."""
    cls = _COMPOSITE_REGISTRY.get(op_type, None)
    return cls() if cls else None

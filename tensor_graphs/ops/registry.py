from typing import Dict, Any, Callable, Optional, List
from ..ir.node import TensorNode

# Maps OpType (str) -> Reference Factory Function
_REFERENCE_REGISTRY: Dict[
    str, Callable[[List[TensorNode], Optional[Dict[str, Any]]], TensorNode]
] = {}


def register_reference_factory(op_type: str, factory_func: Callable):
    """
    Registers a factory function that produces a TensorNode graph (reference implementation)
    for a specific OpType.
    """
    _REFERENCE_REGISTRY[op_type] = factory_func
    return factory_func


def get_reference_factory(op_type: str) -> Optional[Callable]:
    """Returns the reference graph factory for the given OpType."""
    return _REFERENCE_REGISTRY.get(op_type, None)

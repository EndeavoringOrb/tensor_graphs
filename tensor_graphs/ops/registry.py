from typing import Dict, Any

_OP_REGISTRY: Dict[str, Any] = {}

def register_op(name: str):
    """Decorator to register metadata about an op."""
    def decorator(cls):
        _OP_REGISTRY[name] = cls
        return cls
    return decorator

def get_op_metadata(name: str):
    return _OP_REGISTRY.get(name, None)
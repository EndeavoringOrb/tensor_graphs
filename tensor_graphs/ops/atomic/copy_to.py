from typing import List, Dict, Any, Optional
from ...ir.node import TensorNode
from ...ir.dtypes import Backend
from ..atomic_types import OpType


def copy_to_ref(
    inputs: List[TensorNode], attrs: Optional[Dict[str, Any]] = None
) -> TensorNode:
    """
    Reference graph for CopyTo.
    Logically acts as an Identity function, but the resulting node
    will have the target backend specified in attrs.
    """
    if len(inputs) != 1:
        raise ValueError("CopyTo requires exactly 1 input")

    if not attrs or "target_backend" not in attrs:
        raise ValueError("CopyTo requires 'target_backend' attribute")

    input_tensor = inputs[0]

    # Resolve the string value from JSON/attrs to the Enum
    target_val = attrs["target_backend"]
    if isinstance(target_val, str):
        target_backend = Backend(target_val)
    else:
        target_backend = target_val

    return TensorNode(
        OpType.COPY_TO,
        input_tensor.shape,
        input_tensor.dtype,
        [input_tensor],
        f"copy_{input_tensor.name}_to_{target_backend.value}",
        attrs=attrs,
        backend=target_backend,
    )

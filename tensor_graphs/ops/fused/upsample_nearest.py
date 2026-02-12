from ...ir.node import TensorNode
from ..atomic_types import OpType
from ..registry import register_reference_factory


def upsample_nearest_decomposition(inputs, attrs: dict = {}):
    """
    Upsample Nearest Neighbor 2x.
    Decomposed into Repeat operations.
    inputs[0]: [N, C, H, W]
    """
    x = inputs[0]
    scale = attrs.get("scale", 2)

    # Repeat along H and W (axes 2 and 3)
    # Repeat count is scale
    h_up = TensorNode(
        OpType.REPEAT,
        x.dtype,
        [x],
        attrs={"repeats": scale, "axis": 2},
    )
    result = TensorNode(
        OpType.REPEAT,
        h_up.dtype,
        [h_up],
        attrs={"repeats": scale, "axis": 3},
    )

    return result


register_reference_factory("Upsample2x", upsample_nearest_decomposition)

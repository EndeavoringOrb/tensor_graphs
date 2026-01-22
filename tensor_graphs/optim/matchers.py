from ..ir.node import TensorNode


def match_pattern(node: TensorNode, pattern_op: str, parent_ops: list = None) -> bool:
    """
    Simple helper to check if a node matches an OpType and specific parent OpTypes.
    """
    if parent_ops is None:
        parent_ops = []
    if node.op_type != pattern_op:
        return False

    if parent_ops:
        if len(node.parents) != len(parent_ops):
            return False
        for i, p_op in enumerate(parent_ops):
            if node.parents[i].op_type != p_op:
                return False

    return True

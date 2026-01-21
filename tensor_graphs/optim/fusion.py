from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ops.atomic import OpType

def fuse_graph(node: TensorNode, memo=None) -> TensorNode:
    """
    Recursively fuses operations.
    Target: Add(Mul(A, B), C) -> FusedMulAdd(A, B, C)
    """
    if memo is None:
        memo = {}
        
    if node in memo:
        return memo[node]

    # 1. Optimize parents first (Post-order traversal)
    new_parents = [fuse_graph(p, memo) for p in node.parents]
    
    # 2. Pattern Match: ADD
    if node.op_type == OpType.ADD:
        lhs, rhs = new_parents
        
        # Check for Mul on Left side: (A * B) + C
        if lhs.op_type == OpType.MUL:
            fused_node = TensorNode(
                op_type=OpType.FUSED_MUL_ADD,
                shape=node.shape,
                dtype=node.dtype,  # Preserve Type
                parents=[*lhs.parents, rhs],
                name=f"fused_{node.name}"
            )
            memo[node] = fused_node
            return fused_node
    
    # 3. No match? Return reconstruction if parents changed
    if new_parents == node.parents:
        memo[node] = node
        return node

    new_node = TensorNode(node.op_type, node.shape, node.dtype, new_parents, name=node.name)
    memo[node] = new_node
    return new_node
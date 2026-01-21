from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ops.atomic import OpType

def fuse_graph(node: TensorNode, memo=None) -> TensorNode:
    """
    Recursively fuses operations.
    Target: Add(Mul(A, B), C) -> FusedMulAdd(A, B, C)
    
    Uses memoization to preserve DAG structure and avoid infinite recursion.
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
                parents=[*lhs.parents, rhs], # Flatten parents: A, B, C
                name=f"fused_{node.name}"
            )
            memo[node] = fused_node
            return fused_node

        # (Optional) Check for Mul on Right side: C + (A * B)
        # Note: Add is commutative, so we might flip edges or support FusedAddMul
    
    # 3. No match? Return reconstruction with optimized parents
    # Check if parents actually changed. If not, return original to save memory.
    if new_parents == node.parents:
        memo[node] = node
        return node

    new_node = TensorNode(node.op_type, node.shape, new_parents, name=node.name)
    memo[node] = new_node
    return new_node
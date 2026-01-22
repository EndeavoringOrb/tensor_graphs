import sympy
from ..ir.node import TensorNode
from ..ops.atomic import OpType

def to_sympy(node: TensorNode):
    """
    Converts a TensorNode graph into a SymPy expression for analysis.
    """
    if sympy is None:
        raise ImportError("SymPy is required for symbolic analysis.")

    if node.op_type == OpType.INPUT:
        return sympy.Symbol(node.name)
    
    args = [to_sympy(p) for p in node.parents]
    
    if node.op_type == OpType.ADD:
        return args[0] + args[1]
    elif node.op_type == OpType.MUL:
        return args[0] * args[1]
    elif node.op_type == OpType.DIVIDE:
        return args[0] / args[1]
    elif node.op_type == OpType.DOT:
        # Sympy MatrixSymbol support would go here, using generic MUL for now
        return args[0] * args[1] 
    elif node.op_type == OpType.SQRT:
        return sympy.sqrt(args[0])
    elif node.op_type == OpType.SIN:
        return sympy.sin(args[0])
    elif node.op_type == OpType.COS:
        return sympy.cos(args[0])
    elif node.op_type == OpType.NEGATE:
        return -args[0]
        
    return sympy.Symbol(f"Unknown({node.name})")
class OpType:
    INPUT = "Input"
    ADD = "Add"
    MUL = "Mul"
    DIVIDE = "Divide"
    DOT = "Dot"
    SQRT = "Sqrt"
    SIN = "Sin"
    COS = "Cos"
    EXP = "Exp"
    NEGATE = "Negate"
    
    # Reductions
    SUM = "Sum"
    
    # Structural
    RESHAPE = "Reshape"
    PERMUTE = "Permute"
    SLICE = "Slice"
    CONCAT = "Concat"
    
    # Fused Operations (Created by Optimizer)
    FUSED_MUL_ADD = "FusedMulAdd"
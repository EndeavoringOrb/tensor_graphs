class OpType:
    INPUT = "Input"
    ADD = "Add"
    MUL = "Mul"
    DOT = "Dot"  # Matrix Multiplication
    SILU = "Silu"
    
    # Reductions
    SUM = "Sum"
    
    # Structural
    RESHAPE = "Reshape"
    
    # Fused Operations (Created by Optimizer)
    FUSED_MUL_ADD = "FusedMulAdd"
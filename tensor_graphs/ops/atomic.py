class OpType:
    # --- Input ---
    INPUT = "Input"

    # --- Math ---
    ADD = "Add"
    MUL = "Mul"
    DIVIDE = "Divide"
    DOT = "Dot"
    SQRT = "Sqrt"
    SIN = "Sin"
    COS = "Cos"
    EXP = "Exp"
    NEGATE = "Negate"
    TANH = "Tanh"

    # --- Reduction ---
    SUM = "Sum"
    MAX = "Max"

    # --- Manipulation ---
    RESHAPE = "Reshape"
    PERMUTE = "Permute"
    SLICE = "Slice"
    CONCAT = "Concat"
    CAST = "Cast"
    REPEAT = "Repeat"  # New Atomic Op for GQA
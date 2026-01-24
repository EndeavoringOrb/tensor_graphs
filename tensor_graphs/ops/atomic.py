class OpType:
    # --- Input ---
    INPUT = "Input"
    CONSTANT = "Constant"

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
    POWER = "Power"

    # --- Reduction ---
    SUM = "Sum"
    MAX = "Max"

    # --- Manipulation ---
    RESHAPE = "Reshape"
    PERMUTE = "Permute"
    SLICE = "Slice"
    CONCAT = "Concat"
    CAST = "Cast"
    REPEAT = "Repeat"
    ARANGE = "Arange"
    TRIU = "Triu"
    GATHER = "Gather"
    FILL = "Fill"
    WHERE = "Where"
    COPY_TO = "CopyTo"

Do NOT use try except.
When running python, use .venv/Scripts/python.exe
For kernels/ops, the inputs list is strictly Tensor Inputs (runtime data), and attrs is strictly Static Arguments (compile-time configuration)
OpType.INPUT is for values not known at compile time. OpType.CONSTANT is for values known at compile time.
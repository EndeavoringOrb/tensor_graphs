DEBUG_EXECUTION = True
DEBUG_DETAILED = False

# If True, non-contiguous regions (lists of boxes) are merged into a single bounding box.
# This reduces complexity/overhead at the cost of redundant computation.
USE_CONTIGUOUS_APPROXIMATION = False

RECORD_KERNEL_LAUNCHES = False
RECORD_KERNEL_LAUNCHES_FOLDER = "kernel_launches"
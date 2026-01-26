# Import kernels
from .cpu_numpy import *
from .reference import *

# Try importing torch kernels (will fail gracefully inside the module if no CUDA)
try:
    from .gpu_torch import *
except ImportError:
    pass
except Exception as e:
    print(f"Warning: Could not load GPU kernels: {e}")

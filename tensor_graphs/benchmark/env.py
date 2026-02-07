import platform
import psutil
import importlib.metadata
from typing import Dict, Any


class EnvironmentSniffer:
    @staticmethod
    def get_hardware_name() -> str:
        # Basic CPU info as fallback hardware name
        cpu_name = platform.processor() or "Unknown CPU"

        # Try to get GPU name if torch is available
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.get_device_name(0)
        except ImportError:
            pass

        return cpu_name

    @staticmethod
    def get_memory_bytes() -> int:
        return psutil.virtual_memory().total

    @staticmethod
    def get_platform_info() -> Dict[str, Any]:
        return {
            "os": platform.system(),
            "os_release": platform.release(),
            "os_version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        }

    @staticmethod
    def get_libs_info() -> Dict[str, Any]:
        libs = {}
        # List of interesting libraries
        interesting_libs = ["numpy", "torch", "tensor_graphs"]

        for lib in interesting_libs:
            try:
                libs[lib] = importlib.metadata.version(lib)
            except (ImportError, importlib.metadata.PackageNotFoundError):
                try:
                    # Fallback for older python or just-in-case
                    m = __import__(lib)
                    libs[lib] = getattr(m, "__version__", "unknown")
                except ImportError:
                    libs[lib] = "not_installed"

        return libs

    @classmethod
    def sniff(cls) -> Dict[str, Any]:
        return {
            "hardware_name": cls.get_hardware_name(),
            "memory_bytes": cls.get_memory_bytes(),
            "platform_info": cls.get_platform_info(),
            "libs_info": cls.get_libs_info(),
        }

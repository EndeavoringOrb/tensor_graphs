# File: utils/common.py
import json
import math
import re
from pathlib import Path
from typing import Any


def natural_sort_key(s: Any) -> list[int | str]:
    """Sort key for natural alphanumeric sorting (e.g., '1', '2', '10')."""
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", str(s))
    ]


def num_elements(shape: list[int] | tuple[int, ...]) -> int:
    """Calculates total number of elements in a tensor shape."""
    return math.prod(shape) if shape else 0


def format_size(bytes_val: float | None) -> str:
    """Formats a byte count into a human-readable string."""
    if bytes_val is None:
        return "0 B"
    bytes_float = float(bytes_val)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_float < 1024.0:
            return f"{bytes_float:.2f} {unit}"
        bytes_float /= 1024.0
    return f"{bytes_float:.2f} PB"


def format_params(num: float) -> str:
    """Formats parameter counts into human-readable strings (K, M, B)."""
    if num >= 1e9:
        return f"{num / 1e9:.2f} B"
    if num >= 1e6:
        return f"{num / 1e6:.2f} M"
    if num >= 1e3:
        return f"{num / 1e3:.2f} K"
    return str(num)


def format_ms(ms: float) -> str:
    """Formats millisecond runtimes."""
    return f"{ms:.4f} ms"


def format_num_or_str(val: Any) -> str:
    """Formats numeric values or fallback strings for table display."""
    if val is None or val == "?":
        return "?"
    if isinstance(val, str):
        try:
            val = float(val)
        except ValueError:
            return val
    if isinstance(val, float):
        if val == float("inf"):
            return "inf"
        if val == float("-inf"):
            return "-inf"
        formatted = f"{val:.4f}".rstrip("0").rstrip(".")
        return formatted or "0"
    return str(val)


def load_uids_from_cpp(header_path: str | None = None) -> dict[Any, dict]:
    """Loads kernel UID definitions from kernel_uids.json or kernel_uids.gen.hpp."""
    uid_map: dict[Any, dict] = {}
    base_dir = (
        Path(__file__).resolve().parent.parent / "tensor_graphs_cpp" / "generated"
    )
    json_path = base_dir / "kernel_uids.json"

    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for key, info in data.items():
                    if isinstance(info, str):
                        info = {"name": info, "path": "", "hex_uid": str(key)}
                    uid_map[key] = info
                    uid_map[str(key)] = info
                    try:
                        uid_map[int(key)] = info
                    except ValueError:
                        pass
            return uid_map
        except Exception:
            pass

    hdr = Path(header_path) if header_path else (base_dir / "kernel_uids.gen.hpp")
    if hdr.exists():
        pattern = re.compile(r"constexpr uint64_t\s+(\w+)\s+=\s+(0x[0-9a-fA-F]+)ULL;")
        with open(hdr, "r", encoding="utf-8") as f:
            for name, hex_val in pattern.findall(f.read()):
                val_int = int(hex_val, 16)
                info = {"name": name, "path": "", "hex_uid": hex_val}
                uid_map[val_int] = info
                uid_map[hex_val.lower()] = info
                uid_map[str(val_int)] = info
    return uid_map


def format_op_name(info: dict | str | None, default_name: str = "") -> str:
    """Formats an operation name with an optional source path annotation."""
    if isinstance(info, dict):
        name = info.get("name", default_name)
        path = info.get("path", "")
        return f"{name} [{path}]" if path else name
    elif isinstance(info, str):
        return info
    return default_name

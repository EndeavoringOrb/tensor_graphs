from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import uuid
from .dtypes import DType, TensorSignature, Backend
from .buffer import StorageType
from ..ops.atomic_types import OpType


class CachePolicy(Enum):
    NEVER = "never"
    ALWAYS = "always"
    AUTO = "auto"


@dataclass(eq=False)
class TensorNode:
    op_type: str
    dtype: DType
    parents: List["TensorNode"]
    shape: Optional[Tuple[Optional[int], ...]] = None
    name: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    attrs: Dict[str, Any] = field(default_factory=dict)
    backend: Backend = Backend.CPU_NUMPY
    storage_type: StorageType = StorageType.TRANSIENT

    # --- Caching & Runtime State ---
    cache_policy: CachePolicy = CachePolicy.AUTO

    # Runtime flags
    # dirty_region: None means CLEAN.
    # Tuple[slice, ...] means partial or full dirty.
    # (slice(None),) * rank means fully dirty.
    dirty_region: Optional[Tuple[slice, ...]] = None

    cached_output: Any = None

    # Statistics for Cache Eviction (AUTO policy)
    execution_count: int = 0  # Total times this node was required (hit + miss)
    dirty_count: int = 0  # Times this node was invalidated
    compute_cost: float = 0.0  # Last measured execution time in ms
    last_run_tick: int = 0  # Logical timestamp of last access

    def __post_init__(self):
        # If storage_type is the default TRANSIENT, check if we should promote it to PERSISTENT
        if self.storage_type == StorageType.TRANSIENT:
            if self.op_type == OpType.CONSTANT:
                object.__setattr__(self, "storage_type", StorageType.PERSISTENT)

        # Default View-only and cheap nodes to NEVER cache
        if self.op_type in [
            OpType.RESHAPE,
            OpType.SLICE,
            OpType.PERMUTE,
            OpType.INPUT,
            OpType.CONSTANT,
        ]:
            self.cache_policy = CachePolicy.NEVER

    def get_attr(self, key: str, default: Any = None) -> Any:
        return self.attrs.get(key, default)

    @property
    def signature(self) -> TensorSignature:
        return TensorSignature(self.dtype, self.shape, self.backend)

    @property
    def is_dirty(self) -> bool:
        return self.dirty_region is not None

    def get_details(self) -> str:
        out_sig = f"{self.dtype.name if hasattr(self.dtype, 'name') else self.dtype} | {self.shape}"
        lines = []
        header = f"Node: {self.name} [{self.op_type}]"
        lines.append(header)
        lines.append("-" * len(header))
        lines.append(f"Output Signature : {out_sig}")
        lines.append(f"Backend          : {self.backend}")
        lines.append(f"Cache Policy     : {self.cache_policy.value}")
        lines.append("Parents          :")
        if not self.parents:
            lines.append("  (None - Leaf Node)")
        else:
            for idx, parent in enumerate(self.parents):
                p_sig = f"{parent.dtype.name if hasattr(parent.dtype, 'name') else parent.dtype} | {parent.shape}"
                lines.append(f"  [{idx}] {parent.name:<10} -> {p_sig}")
        if self.attrs:
            lines.append("Attributes       :")
            for k, v in self.attrs.items():
                lines.append(f"  {k:<14} : {v}")
        return "\n".join(lines)

    def __getitem__(self, key) -> "TensorNode":
        if self.shape is None:
            raise ValueError(
                f"Cannot slice node '{self.name}' because its shape is undefined."
            )

        if not isinstance(key, tuple):
            key = (key,)

        if Ellipsis in key:
            idx = key.index(Ellipsis)
            num_missing = len(self.shape) - (len(key) - 1)
            key = key[:idx] + (slice(None),) * num_missing + key[idx + 1 :]

        if len(key) < len(self.shape):
            key = key + (slice(None),) * (len(self.shape) - len(key))

        key = key[: len(self.shape)]

        new_shape = []
        starts = []
        ends = []
        steps = []

        for i, (k, dim_size) in enumerate(zip(key, self.shape)):
            if isinstance(k, int):
                start = k if k >= 0 or dim_size is None else k + dim_size
                if start == -1 and dim_size is None:
                    stop = None
                else:
                    stop = start + 1
                starts.append(start)
                ends.append(stop)
                steps.append(1)
                new_shape.append(1)
            elif isinstance(k, slice):
                start = k.start if k.start is not None else 0
                stop = k.stop
                step = k.step if k.step is not None else 1

                if dim_size is not None:
                    if start < 0:
                        start += dim_size
                    if stop is not None and stop < 0:
                        stop += dim_size
                    e = stop if stop is not None else dim_size
                    out_dim = (e - start + step - (1 if step > 0 else -1)) // step
                    new_shape.append(max(0, out_dim))
                else:
                    new_shape.append(None)
                starts.append(start)
                ends.append(stop)
                steps.append(step)
            else:
                raise ValueError(f"Unsupported index type: {type(k)}")

        return TensorNode(
            "Slice",
            self.dtype,
            [self],
            tuple(new_shape),
            f"{self.name}_slice",
            attrs={"starts": starts, "ends": ends, "steps": steps},
            backend=self.backend,
        )

    def __repr__(self):
        attr_keys = list(self.attrs.keys()) if self.attrs else []
        attrs_summary = f" | attrs={attr_keys}" if attr_keys else ""
        return f"[{self.dtype.value}|{self.shape}{attrs_summary}] {self.op_type}({self.name})"

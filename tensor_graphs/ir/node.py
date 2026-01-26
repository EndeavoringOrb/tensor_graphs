from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import uuid
from .dtypes import DType, TensorSignature, Backend


@dataclass(eq=False)
class TensorNode:
    op_type: str
    shape: Tuple[Optional[int], ...]
    dtype: DType
    parents: List["TensorNode"]
    name: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    attrs: Dict[str, Any] = field(default_factory=dict)
    backend: Backend = Backend.CPU_NUMPY

    def get_attr(self, key: str, default: Any = None) -> Any:
        return self.attrs.get(key, default)

    @property
    def signature(self) -> TensorSignature:
        return TensorSignature(self.dtype, self.shape, self.backend)

    def get_details(self) -> str:
        """
        Prints a structured representation of the node, including output signature,
        backend, specific attributes, and the signatures of all parent nodes.
        """
        # Format the output signature string
        out_sig = f"{self.dtype.name if hasattr(self.dtype, 'name') else self.dtype} | {self.shape}"

        lines = []
        header = f"Node: {self.name} [{self.op_type}]"
        lines.append(header)
        lines.append("-" * len(header))
        lines.append(f"Output Signature : {out_sig}")
        lines.append(f"Backend          : {self.backend}")

        # Format Parents with their signatures
        lines.append("Parents          :")
        if not self.parents:
            lines.append("  (None - Leaf Node)")
        else:
            for idx, parent in enumerate(self.parents):
                # Get parent signature
                p_sig = f"{parent.dtype.name if hasattr(parent.dtype, 'name') else parent.dtype} | {parent.shape}"
                lines.append(f"  [{idx}] {parent.name:<10} -> {p_sig}")

        # Format Attributes
        if self.attrs:
            lines.append("Attributes       :")
            for k, v in self.attrs.items():
                lines.append(f"  {k:<14} : {v}")

        return "\n".join(lines)

    def __getitem__(self, key) -> "TensorNode":
        """
        Syntactic sugar for OpType.SLICE.
        Supports standard Python slicing logic.
        """
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
                # Treat int as slice(k, k+1) to preserve rank for now
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
            tuple(new_shape),
            self.dtype,
            [self],
            f"{self.name}_slice",
            attrs={"starts": starts, "ends": ends, "steps": steps},
            backend=self.backend,
        )

    def __repr__(self):
        # Only print keys of attrs to avoid traversing the graph via step/shape nodes
        attr_keys = list(self.attrs.keys()) if self.attrs else []
        attrs_summary = f" | attrs={attr_keys}" if attr_keys else ""
        return f"[{self.dtype.value}|{self.shape}{attrs_summary}] {self.op_type}({self.name})"

#!/usr/bin/env python3
# File: utils/inspect_graph.py
import argparse
import json
import os
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from binary import DTYPES, load_cache_file
from common import (
    format_ms,
    format_num_or_str,
    format_op_name,
    format_size,
    load_uids_from_cpp,
    num_elements,
)

console = Console()
console_err = Console(stderr=True)

DTYPE_BYTES = {
    "FLOAT32": 4,
    "INT32": 4,
    "INT64": 8,
    "BF16": 2,
    "BOOL": 1,
    "ANY": 4,
    "INT8": 1,
    "E2M1_PACKED_INT8": 1,
    "E2M1": 1,
    "F8_E8M0": 1,
    "F8_E4M3": 1,
}


def get_dtype_size(dtype_val: Any) -> int:
    if isinstance(dtype_val, int):
        dt_str = DTYPES[dtype_val] if dtype_val < len(DTYPES) else "FLOAT32"
    else:
        dt_str = str(dtype_val).upper()
    return DTYPE_BYTES.get(dt_str, 4)


def get_required_buffer_elements(shape: list[int], strides: list[int]) -> int:
    if not shape or not strides:
        return 1
    max_offset = 0
    for dim_size, stride in zip(shape, strides):
        if dim_size > 0:
            max_offset += (dim_size - 1) * stride
    return max_offset + 1


def get_view_extent_bytes(view: dict) -> int:
    shape = view.get("shape", [])
    strides = view.get("strides", [])
    dtype = view.get("dtype", "FLOAT32")
    elems = get_required_buffer_elements(shape, strides)
    return elems * get_dtype_size(dtype)


def format_constant_data(data_bytes: bytes | None, dtype: Any) -> str:
    if not data_bytes:
        return ""
    dt_str = DTYPES[dtype] if isinstance(dtype, int) and dtype < len(DTYPES) else str(dtype).upper()
    if dt_str == "INT32":
        count = len(data_bytes) // 4
        if count == 0:
            return "[]"
        vals = list(struct.unpack(f"<{count}i", data_bytes[: count * 4]))
        return str(vals) if len(vals) <= 6 else f"{vals[:6]}... ({count} ints)"
    elif dt_str == "FLOAT32":
        count = len(data_bytes) // 4
        if count == 0:
            return "[]"
        vals = [round(v, 4) for v in struct.unpack(f"<{count}f", data_bytes[: count * 4])]
        return str(vals) if len(vals) <= 6 else f"{vals[:6]}... ({count} floats)"
    return f"<{len(data_bytes)} bytes>"


def parse_inst_range(inst_str: str | None, max_len: int) -> list[int]:
    if not inst_str:
        return []
    s = inst_str.strip()
    if ":" in s:
        parts = s.split(":")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else max_len
        return list(range(max(0, start), min(max_len, end)))
    if ".." in s:
        parts = s.split("..")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) + 1 if parts[1] else max_len
        return list(range(max(0, start), min(max_len, end)))
    try:
        idx = int(s)
        if 0 <= idx < max_len:
            return [idx]
        console_err.print(f"[bold red]Error:[/bold red] Instruction index {idx} out of range [0..{max_len - 1}]")
        return []
    except ValueError:
        return []


def collect_constants(cache_entries: list[dict], graph: dict) -> dict[Any, bytes]:
    constants_map: dict[Any, bytes] = {}
    for entry in cache_entries:
        if entry.get("type") == "constants":
            constants_map.update(entry.get("constants", {}))
    for cid, data in graph.get("constStaging", []):
        constants_map[cid] = data
    return constants_map


def validate_graph(graph: dict, constants_map: dict, uid_map: dict) -> list[dict]:
    issues: list[dict] = []
    instructions = graph.get("instructions", [])
    node_views = graph.get("nodeViews", {})
    eclass_to_logical = graph.get("eclassToLogical", {})

    # 1. Track buffer allocations and lifetimes
    buffers: dict[int, dict] = {}
    buffers_by_ms: dict[tuple, list[dict]] = defaultdict(list)

    for idx, inst in enumerate(instructions):
        out_buf = inst.get("outBuffer", {})
        buf_id = out_buf.get("id")
        if buf_id is not None and buf_id not in buffers:
            b_info = {
                "id": buf_id,
                "memSpace": (out_buf.get("memSpaceType"), out_buf.get("memSpaceIdx")),
                "size": out_buf.get("size", 0),
                "offset": out_buf.get("offset", -1),
                "start": out_buf.get("start", 0),
                "end": out_buf.get("end", 0),
                "first_def": idx,
            }
            buffers[buf_id] = b_info
            if out_buf.get("memSpaceType") != "STORAGE":
                buffers_by_ms[b_info["memSpace"]].append(b_info)

    # 2. Check each instruction
    for idx, inst in enumerate(instructions):
        kid = inst.get("kernelId", 0)
        info = uid_map.get(kid) or uid_map.get(str(kid)) or uid_map.get(hex(kid).lower())
        op_name = format_op_name(info, f"Kernel_{hex(kid)}")

        eclass = inst.get("eclassId")
        out_buf = inst.get("outBuffer", {})
        out_view = node_views.get(eclass, {})
        children = inst.get("children", [])
        in_bufs = inst.get("inBuffers", [])

        # Unallocated output buffer check
        if out_buf.get("memSpaceType") != "STORAGE" and out_buf.get("offset", -1) < 0:
            issues.append({
                "severity": "ERROR",
                "inst": idx,
                "op": op_name,
                "msg": f"Output buffer {out_buf.get('id')} has negative/unallocated offset ({out_buf.get('offset')})",
            })

        # Out buffer capacity vs View extent check
        if out_buf.get("memSpaceType") != "STORAGE" and out_view:
            v_extent = get_view_extent_bytes(out_view)
            b_size = out_buf.get("size", 0)
            b_off = out_buf.get("offset", 0)
            v_off = out_view.get("offset", 0)

            if v_off < b_off:
                issues.append({
                    "severity": "ERROR",
                    "inst": idx,
                    "op": op_name,
                    "msg": f"Output view offset (0x{v_off:x}) starts before buffer offset (0x{b_off:x})",
                })
            if (v_off - b_off) + v_extent > b_size:
                issues.append({
                    "severity": "ERROR",
                    "inst": idx,
                    "op": op_name,
                    "msg": (
                        f"Output view extent ({v_extent} B at rel offset 0x{v_off - b_off:x}) "
                        f"exceeds buffer {out_buf.get('id')} capacity ({b_size} B)"
                    ),
                })

        # Input buffer checks
        for c_idx, child in enumerate(children):
            c_view = node_views.get(child, {})
            c_buf = in_bufs[c_idx] if c_idx < len(in_bufs) else {}
            if not c_view:
                issues.append({
                    "severity": "ERROR",
                    "inst": idx,
                    "op": op_name,
                    "msg": f"Child {c_idx} (EClass {child}) missing from nodeViews",
                })
                continue

            if c_buf.get("memSpaceType") != "STORAGE":
                if c_buf.get("offset", -1) < 0:
                    issues.append({
                        "severity": "ERROR",
                        "inst": idx,
                        "op": op_name,
                        "msg": f"Input #{c_idx} buffer {c_buf.get('id')} has unallocated offset ({c_buf.get('offset')})",
                    })

                v_extent = get_view_extent_bytes(c_view)
                b_size = c_buf.get("size", 0)
                b_off = c_buf.get("offset", 0)
                v_off = c_view.get("offset", 0)

                if v_off < b_off:
                    issues.append({
                        "severity": "ERROR",
                        "inst": idx,
                        "op": op_name,
                        "msg": f"Input #{c_idx} view offset (0x{v_off:x}) starts before buffer offset (0x{b_off:x})",
                    })
                if (v_off - b_off) + v_extent > b_size:
                    issues.append({
                        "severity": "ERROR",
                        "inst": idx,
                        "op": op_name,
                        "msg": (
                            f"Input #{c_idx} view extent ({v_extent} B at rel offset 0x{v_off - b_off:x}) "
                            f"exceeds buffer {c_buf.get('id')} capacity ({b_size} B)"
                        ),
                    })

                # Lifetime use-after-death check
                if idx > c_buf.get("end", 0):
                    issues.append({
                        "severity": "WARNING",
                        "inst": idx,
                        "op": op_name,
                        "msg": (
                            f"Input #{c_idx} (Buf {c_buf.get('id')}) read at step {idx} "
                            f"is after buffer death time ({c_buf.get('end')})"
                        ),
                    })

        # Specific validation for CONCAT
        if "CONCAT" in op_name.upper():
            if len(children) < 2:
                issues.append({
                    "severity": "ERROR",
                    "inst": idx,
                    "op": op_name,
                    "msg": f"CONCAT requires at least 2 inputs (axis + data), got {len(children)}",
                })
            else:
                axis_child = children[0]
                axis_raw = constants_map.get(axis_child) or constants_map.get(eclass_to_logical.get(axis_child))
                axis_val = None
                if axis_raw and len(axis_raw) >= 4:
                    axis_val = struct.unpack("<i", axis_raw[:4])[0]

                out_shape = out_view.get("shape", [])
                rank = len(out_shape)
                if axis_val is not None:
                    norm_axis = axis_val if axis_val >= 0 else axis_val + rank
                    if norm_axis < 0 or norm_axis >= rank:
                        issues.append({
                            "severity": "ERROR",
                            "inst": idx,
                            "op": op_name,
                            "msg": f"CONCAT axis {axis_val} out of bounds for output rank {rank}",
                        })
                    else:
                        sum_dim = 0
                        for d_idx, data_child in enumerate(children[1:]):
                            d_view = node_views.get(data_child, {})
                            d_shape = d_view.get("shape", [])
                            if len(d_shape) != rank:
                                issues.append({
                                    "severity": "ERROR",
                                    "inst": idx,
                                    "op": op_name,
                                    "msg": f"CONCAT input #{d_idx + 1} rank ({len(d_shape)}) != output rank ({rank})",
                                })
                            else:
                                for dim_i in range(rank):
                                    if dim_i == norm_axis:
                                        sum_dim += d_shape[dim_i]
                                    elif d_shape[dim_i] != out_shape[dim_i]:
                                        issues.append({
                                            "severity": "ERROR",
                                            "inst": idx,
                                            "op": op_name,
                                            "msg": (
                                                f"CONCAT input #{d_idx + 1} dim {dim_i} ({d_shape[dim_i]}) "
                                                f"mismatch with output ({out_shape[dim_i]})"
                                            ),
                                        })
                        if sum_dim != out_shape[norm_axis]:
                            issues.append({
                                "severity": "ERROR",
                                "inst": idx,
                                "op": op_name,
                                "msg": f"CONCAT sum of input axis dims ({sum_dim}) != output dim ({out_shape[norm_axis]})",
                            })

    # 3. Check for temporal memory collisions between live buffers
    for ms, b_list in buffers_by_ms.items():
        sorted_bufs = sorted(b_list, key=lambda b: b["offset"])
        for i in range(len(sorted_bufs)):
            b1 = sorted_bufs[i]
            if b1["offset"] < 0 or b1["size"] <= 0:
                continue
            for j in range(i + 1, len(sorted_bufs)):
                b2 = sorted_bufs[j]
                if b2["offset"] < 0 or b2["size"] <= 0:
                    continue
                if b2["offset"] >= b1["offset"] + b1["size"]:
                    break  # Sorted by offset; cannot overlap further

                # Check lifetime overlap
                if max(b1["start"], b2["start"]) <= min(b1["end"], b2["end"]):
                    issues.append({
                        "severity": "CRITICAL",
                        "inst": b2["first_def"],
                        "op": "MEMORY_COLLISION",
                        "msg": (
                            f"Live buffers {b1['id']} (lifetime [{b1['start']}..{b1['end']}], 0x{b1['offset']:x}..0x{b1['offset'] + b1['size']:x}) "
                            f"and {b2['id']} (lifetime [{b2['start']}..{b2['end']}], 0x{b2['offset']:x}..0x{b2['offset'] + b2['size']:x}) "
                            f"collide in MemSpace {ms[0]}({ms[1]})"
                        ),
                    })

    return issues


def inspect_instruction_detail(idx: int, inst: dict, graph: dict, constants_map: dict, uid_map: dict) -> None:
    kid = inst.get("kernelId", 0)
    info = uid_map.get(kid) or uid_map.get(str(kid)) or uid_map.get(hex(kid).lower())
    op_name = format_op_name(info, f"Kernel_{hex(kid)}")

    node_views = graph.get("nodeViews", {})
    eclass_to_logical = graph.get("eclassToLogical", {})
    node_costs = graph.get("nodeCosts", {})

    eclass = inst.get("eclassId")
    logical = eclass_to_logical.get(eclass, inst.get("logicalId"))
    cost = node_costs.get(eclass, 0.0)

    out_buf = inst.get("outBuffer", {})
    out_view = node_views.get(eclass, {})
    children = inst.get("children", [])
    in_bufs = inst.get("inBuffers", [])
    engines = inst.get("engines", [])
    debug_origin = inst.get("debugOrigin", "")

    header_text = Text()
    header_text.append(f"Instruction [{idx}] ", style="bold cyan")
    header_text.append(f"{op_name} ", style="bold yellow")
    header_text.append(f"(UID: 0x{kid:x})", style="dim")

    lines = []
    lines.append(f"[bold white]Debug Origin:[/bold white] [dim cyan]{debug_origin or 'N/A'}[/dim cyan]")
    lines.append(f"[bold white]Est. Cost:[/bold white]    [green]{format_ms(cost)}[/green]")

    eng_str = ", ".join(f"Engine(idx={e.get('idx')}, type={e.get('type')})" for e in engines) or "CPU(0)"
    lines.append(f"[bold white]Engines:[/bold white]      {eng_str}")
    lines.append("")

    # Output details
    out_ms = f"{out_buf.get('memSpaceType', '?')}({out_buf.get('memSpaceIdx', '?')})"
    out_extent = get_view_extent_bytes(out_view)
    out_shape = out_view.get("shape", [])
    out_strides = out_view.get("strides", [])
    out_dt = out_view.get("dtype", "FLOAT32")
    out_off = out_buf.get("offset", -1)
    out_v_off = out_view.get("offset", 0)

    lines.append("[bold magenta]Output Tensor:[/bold magenta]")
    lines.append(
        f"  [bold]EClass:[/bold] {eclass:<6} [bold]LogicalId:[/bold] {str(logical):<8} "
        f"[bold]Buffer ID:[/bold] {out_buf.get('id', '?')} | [bold]MemSpace:[/bold] {out_ms}"
    )
    lines.append(
        f"  [bold]Buffer Range:[/bold] 0x{out_off:x}..0x{out_off + out_buf.get('size', 0):x} "
        f"({format_size(out_buf.get('size', 0))}) | [bold]Lifetime:[/bold] [{out_buf.get('start', '?')}..{out_buf.get('end', '?')}]"
    )
    lines.append(
        f"  [bold]View:[/bold] {out_dt}{out_shape!s} | [bold]Strides:[/bold] {out_strides!s} | "
        f"[bold]ViewOffset:[/bold] 0x{out_v_off:x} | [bold]Extent:[/bold] {format_size(out_extent)}"
    )

    # Input details
    lines.append(f"\n[bold magenta]Inputs ({len(children)} children):[/bold magenta]")
    for c_idx, child in enumerate(children):
        c_view = node_views.get(child, {})
        c_buf = in_bufs[c_idx] if c_idx < len(in_bufs) else {}
        c_logical = eclass_to_logical.get(child)
        c_ms = f"{c_buf.get('memSpaceType', '?')}({c_buf.get('memSpaceIdx', '?')})"
        c_shape = c_view.get("shape", [])
        c_strides = c_view.get("strides", [])
        c_dt = c_view.get("dtype", "FLOAT32")
        c_extent = get_view_extent_bytes(c_view)
        c_off = c_buf.get("offset", -1)
        c_v_off = c_view.get("offset", 0)

        const_data = constants_map.get(child) or constants_map.get(c_logical)
        const_note = f" [bold green](Constant: {format_constant_data(const_data, c_dt)})[/bold green]" if const_data else ""

        lines.append(
            f"  [bold cyan][In {c_idx}][/bold cyan] [bold]EClass:[/bold] {child:<6} [bold]LogicalId:[/bold] {str(c_logical):<8}{const_note}"
        )
        lines.append(
            f"       [bold]Buffer:[/bold] ID {c_buf.get('id', '?')} | {c_ms} | 0x{c_off:x}..0x{c_off + c_buf.get('size', 0):x} "
            f"({format_size(c_buf.get('size', 0))}) | [bold]Lifetime:[/bold] [{c_buf.get('start', '?')}..{c_buf.get('end', '?')}]"
        )
        lines.append(
            f"       [bold]View:[/bold]   {c_dt}{c_shape!s} | [bold]Strides:[/bold] {c_strides!s} | "
            f"[bold]ViewOffset:[/bold] 0x{c_v_off:x} | [bold]Extent:[/bold] {format_size(c_extent)}"
        )

    content = "\n".join(lines)
    console.print(Panel(content, title=header_text, border_style="cyan", box=box.ROUNDED))


def trace_buffer_timeline(target_buf_id: int, graph: dict, uid_map: dict) -> None:
    instructions = graph.get("instructions", [])
    node_views = graph.get("nodeViews", {})
    producers = []
    consumers = []
    buf_info = None

    for idx, inst in enumerate(instructions):
        out_buf = inst.get("outBuffer", {})
        if out_buf.get("id") == target_buf_id:
            buf_info = out_buf
            producers.append((idx, inst))
        for c_idx, in_buf in enumerate(inst.get("inBuffers", [])):
            if in_buf.get("id") == target_buf_id:
                if buf_info is None:
                    buf_info = in_buf
                consumers.append((idx, c_idx, inst))

    if not producers and not consumers:
        console.print(f"[bold red]Buffer ID {target_buf_id} not found in instructions.[/bold red]")
        return

    b_ms = f"{buf_info.get('memSpaceType', '?')}({buf_info.get('memSpaceIdx', '?')})" if buf_info else "?"
    b_off = f"0x{buf_info.get('offset', 0):x}" if buf_info else "?"
    b_sz = format_size(buf_info.get("size", 0)) if buf_info else "?"
    b_life = f"[{buf_info.get('start', '?')}..{buf_info.get('end', '?')}]" if buf_info else "?"

    console.rule(f"[bold yellow]Trace Buffer ID {target_buf_id}[/bold yellow]")
    console.print(f"[bold]MemSpace:[/bold] {b_ms} | [bold]Offset:[/bold] {b_off} | [bold]Size:[/bold] {b_sz} | [bold]Lifetime:[/bold] {b_life}\n")

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
    table.add_column("Step", style="bold cyan", justify="right")
    table.add_column("Role", style="bold magenta")
    table.add_column("Operation", style="yellow")
    table.add_column("Shape / View", style="green")
    table.add_column("Origin", style="dim")

    for p_idx, inst in producers:
        kid = inst.get("kernelId", 0)
        info = uid_map.get(kid) or uid_map.get(str(kid)) or uid_map.get(hex(kid).lower())
        op = format_op_name(info, f"Kernel_{hex(kid)}")
        v = node_views.get(inst.get("eclassId"), {})
        v_str = f"{v.get('dtype', '?')}{v.get('shape', [])!s}"
        table.add_row(str(p_idx), "[green]Producer (Out)[/green]", op, v_str, inst.get("debugOrigin", ""))

    for c_idx, in_slot, inst in consumers:
        kid = inst.get("kernelId", 0)
        info = uid_map.get(kid) or uid_map.get(str(kid)) or uid_map.get(hex(kid).lower())
        op = format_op_name(info, f"Kernel_{hex(kid)}")
        child_id = inst.get("children", [])[in_slot] if in_slot < len(inst.get("children", [])) else None
        v = node_views.get(child_id, {})
        v_str = f"{v.get('dtype', '?')}{v.get('shape', [])!s}"
        table.add_row(str(c_idx), f"[blue]Consumer (In {in_slot})[/blue]", op, v_str, inst.get("debugOrigin", ""))

    console.print(table)


def trace_eclass_flow(target_eclass: int, graph: dict, uid_map: dict) -> None:
    instructions = graph.get("instructions", [])
    node_views = graph.get("nodeViews", {})
    eclass_to_logical = graph.get("eclassToLogical", {})

    producers = []
    consumers = []
    for idx, inst in enumerate(instructions):
        if inst.get("eclassId") == target_eclass:
            producers.append((idx, inst))
        if target_eclass in inst.get("children", []):
            slot = inst.get("children", []).index(target_eclass)
            consumers.append((idx, slot, inst))

    console.rule(f"[bold yellow]Trace EClass {target_eclass}[/bold yellow]")
    logical = eclass_to_logical.get(target_eclass, "N/A")
    view = node_views.get(target_eclass, {})
    console.print(f"[bold]Logical ID:[/bold] {logical} | [bold]View:[/bold] {view.get('dtype', '?')}{view.get('shape', [])!s} | [bold]Strides:[/bold] {view.get('strides', [])!s}\n")

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
    table.add_column("Step", style="bold cyan", justify="right")
    table.add_column("Role", style="bold magenta")
    table.add_column("Operation", style="yellow")
    table.add_column("Origin", style="dim")

    for p_idx, inst in producers:
        kid = inst.get("kernelId", 0)
        info = uid_map.get(kid) or uid_map.get(str(kid)) or uid_map.get(hex(kid).lower())
        table.add_row(str(p_idx), "[green]Defining Inst[/green]", format_op_name(info, f"Kernel_{hex(kid)}"), inst.get("debugOrigin", ""))

    for c_idx, slot, inst in consumers:
        kid = inst.get("kernelId", 0)
        info = uid_map.get(kid) or uid_map.get(str(kid)) or uid_map.get(hex(kid).lower())
        table.add_row(str(c_idx), f"[blue]Consumer (In {slot})[/blue]", format_op_name(info, f"Kernel_{hex(kid)}"), inst.get("debugOrigin", ""))

    console.print(table)


def print_memory_summary(graph: dict) -> None:
    instructions = graph.get("instructions", [])
    node_views = graph.get("nodeViews", {})
    peak_by_ms: dict[tuple, int] = defaultdict(int)
    buffers_seen = set()
    total_buf_bytes = 0

    for inst in instructions:
        out_buf = inst.get("outBuffer", {})
        if out_buf.get("memSpaceType") != "STORAGE" and out_buf.get("offset", -1) >= 0:
            ms = (out_buf.get("memSpaceType"), out_buf.get("memSpaceIdx"))
            end_off = out_buf.get("offset", 0) + out_buf.get("size", 0)
            peak_by_ms[ms] = max(peak_by_ms[ms], end_off)
            if out_buf.get("id") not in buffers_seen:
                buffers_seen.add(out_buf.get("id"))
                total_buf_bytes += out_buf.get("size", 0)

        for in_buf in inst.get("inBuffers", []):
            if in_buf.get("memSpaceType") != "STORAGE" and in_buf.get("offset", -1) >= 0:
                ms = (in_buf.get("memSpaceType"), in_buf.get("memSpaceIdx"))
                end_off = in_buf.get("offset", 0) + in_buf.get("size", 0)
                peak_by_ms[ms] = max(peak_by_ms[ms], end_off)
                if in_buf.get("id") not in buffers_seen:
                    buffers_seen.add(in_buf.get("id"))
                    total_buf_bytes += in_buf.get("size", 0)

    console.rule("[bold cyan]Memory Arena Allocation Summary[/bold cyan]")
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
    table.add_column("Memory Space", style="bold yellow")
    table.add_column("Required Peak Arena Extent", style="cyan", justify="right")
    table.add_column("Extent (Bytes)", style="dim", justify="right")

    for ms, extent in sorted(peak_by_ms.items()):
        ms_str = f"{ms[0]}({ms[1]})"
        table.add_row(ms_str, format_size(extent), f"{extent:,} B")

    console.print(table)
    console.print(f"[bold]Total Unique Buffers:[/bold] {len(buffers_seen):,}")
    console.print(f"[bold]Total Buffer Capacity (summed):[/bold] {format_size(total_buf_bytes)}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect, query, and statically validate TensorGraph compiled graph cache files (.bin)."
    )
    parser.add_argument("graph_pos", nargs="?", default=None, help="Path to compiled graph .bin file")
    parser.add_argument("--graph", "-g", dest="graph_opt", default=None, help="Path to compiled graph .bin file")
    parser.add_argument("--bucket", "-b", type=int, default=None, help="Bucket index (default: 0)")
    parser.add_argument("--inst", "-i", type=str, default=None, help="Instruction index or range (e.g. 30402, 30385:30405)")
    parser.add_argument("--validate", action="store_true", help="Run static safety & bounds validation checks")
    parser.add_argument("--trace-buffer", type=int, default=None, help="Trace producer and consumers of buffer ID")
    parser.add_argument("--trace-eclass", type=int, default=None, help="Trace producer and consumers of EClass ID")
    parser.add_argument("--trace-logical", type=int, default=None, help="Trace producer and consumers of Logical ID")
    parser.add_argument("--op", type=str, default=None, help="Filter instructions by op name or regex")
    parser.add_argument("--mem-summary", action="store_true", help="Show memory arena extents and summary")
    parser.add_argument("--summary", action="store_true", help="Display summary overview of the graph")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    args = parser.parse_args()
    graph_path_str = args.graph_opt or args.graph_pos

    if not graph_path_str:
        candidate_caches = list(Path("dirty_region_caches").glob("*.bin"))
        if candidate_caches:
            graph_path_str = str(candidate_caches[0])
            console.print(f"[dim]No graph path specified. Defaulting to: {graph_path_str}[/dim]")
        else:
            console_err.print("[bold red]Error:[/bold red] No graph path specified and no files found in dirty_region_caches/.")
            sys.exit(1)

    if not os.path.exists(graph_path_str):
        console_err.print(f"[bold red]Error:[/bold red] Cache file '{graph_path_str}' does not exist.")
        sys.exit(1)

    cache_entries = load_cache_file(graph_path_str, string_enums=True)
    buckets = [e for e in cache_entries if e.get("type") == "compiled_bucket"]

    if not buckets:
        console_err.print(f"[bold red]Error:[/bold red] No compiled buckets found in '{graph_path_str}'.")
        sys.exit(1)

    target_bucket_idx = args.bucket if args.bucket is not None else 0
    if target_bucket_idx < 0 or target_bucket_idx >= len(buckets):
        console_err.print(f"[bold red]Error:[/bold red] Bucket index {target_bucket_idx} out of range (0..{len(buckets) - 1})")
        sys.exit(1)

    compiled_graph = buckets[target_bucket_idx]["graph"]
    instructions = compiled_graph.get("instructions", [])
    constants_map = collect_constants(cache_entries, compiled_graph)
    uid_map = load_uids_from_cpp()

    # ---- Validation Mode ----
    if args.validate:
        console.rule(f"[bold cyan]Static Validation Check: Bucket {target_bucket_idx} ({len(instructions)} Instructions)[/bold cyan]")
        issues = validate_graph(compiled_graph, constants_map, uid_map)

        if not issues:
            console.print(
                Panel(
                    f"[bold green]ALL CHECKS PASSED[/bold green]\n"
                    f"Validated {len(instructions):,} instructions and all buffer bounds/lifetimes with 0 errors.",
                    border_style="green",
                    box=box.ROUNDED,
                )
            )
        else:
            tbl = Table(box=box.SIMPLE, show_header=True, header_style="bold red")
            tbl.add_column("Severity", style="bold")
            tbl.add_column("Step", justify="right", style="cyan")
            tbl.add_column("Operation", style="yellow")
            tbl.add_column("Issue Description", style="white")

            err_count = sum(1 for x in issues if x["severity"] in ("ERROR", "CRITICAL"))
            warn_count = sum(1 for x in issues if x["severity"] == "WARNING")

            for issue in issues:
                color = "red" if issue["severity"] in ("ERROR", "CRITICAL") else "yellow"
                tbl.add_row(
                    f"[{color}]{issue['severity']}[/{color}]",
                    str(issue["inst"]),
                    issue["op"],
                    issue["msg"],
                )

            console.print(tbl)
            console.print(
                f"[bold red]Found {err_count} critical error(s)[/bold red] and "
                f"[bold yellow]{warn_count} warning(s)[/bold yellow] in compiled execution plan."
            )
        print_memory_summary(compiled_graph)
        return

    # ---- Memory Summary Mode ----
    if args.mem_summary:
        print_memory_summary(compiled_graph)
        return

    # ---- Tracing Modes ----
    if args.trace_buffer is not None:
        trace_buffer_timeline(args.trace_buffer, compiled_graph, uid_map)
        return

    if args.trace_eclass is not None:
        trace_eclass_flow(args.trace_eclass, compiled_graph, uid_map)
        return

    if args.trace_logical is not None:
        eclass_to_log = compiled_graph.get("eclassToLogical", {})
        log_to_eclass = {v: k for k, v in eclass_to_log.items()}
        target_ec = log_to_eclass.get(args.trace_logical)
        if target_ec is None:
            console.print(f"[bold red]Logical ID {args.trace_logical} not found in eclassToLogical mapping.[/bold red]")
        else:
            trace_eclass_flow(target_ec, compiled_graph, uid_map)
        return

    # ---- Specific Instruction Query / Range ----
    inst_indices = parse_inst_range(args.inst, len(instructions))
    if inst_indices:
        if len(inst_indices) == 1:
            inspect_instruction_detail(inst_indices[0], instructions[inst_indices[0]], compiled_graph, constants_map, uid_map)
        else:
            tbl = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
            tbl.add_column("Inst", justify="right", style="bold cyan")
            tbl.add_column("Operation", style="bold yellow")
            tbl.add_column("Out Shape / Extent", style="green")
            tbl.add_column("Out Buf", justify="right", style="yellow")
            tbl.add_column("Offset", justify="right", style="cyan")
            tbl.add_column("In Bufs", style="dim")
            tbl.add_column("Origin", style="dim")

            node_views = compiled_graph.get("nodeViews", {})
            for idx in inst_indices:
                inst = instructions[idx]
                kid = inst.get("kernelId", 0)
                info = uid_map.get(kid) or uid_map.get(str(kid)) or uid_map.get(hex(kid).lower())
                op = format_op_name(info, f"Kernel_{hex(kid)}")
                out_v = node_views.get(inst.get("eclassId"), {})
                out_b = inst.get("outBuffer", {})
                v_str = f"{out_v.get('dtype', '?')}{out_v.get('shape', [])!s}"
                in_str = ", ".join(str(b.get("id")) for b in inst.get("inBuffers", []))
                tbl.add_row(
                    str(idx),
                    op,
                    v_str,
                    str(out_b.get("id")),
                    f"0x{out_b.get('offset', 0):x}",
                    f"[{in_str}]",
                    inst.get("debugOrigin", ""),
                )
            console.print(tbl)
        return

    # ---- Op Name Filter Mode ----
    if args.op:
        pattern = re.compile(args.op, re.IGNORECASE)
        matching_indices = []
        for idx, inst in enumerate(instructions):
            kid = inst.get("kernelId", 0)
            info = uid_map.get(kid) or uid_map.get(str(kid)) or uid_map.get(hex(kid).lower())
            op = format_op_name(info, f"Kernel_{hex(kid)}")
            if pattern.search(op) or pattern.search(inst.get("debugOrigin", "")):
                matching_indices.append(idx)

        console.rule(f"[bold yellow]Found {len(matching_indices)} instruction(s) matching '{args.op}'[/bold yellow]")
        tbl = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
        tbl.add_column("Inst", justify="right", style="bold cyan")
        tbl.add_column("Operation", style="bold yellow")
        tbl.add_column("Out Shape", style="green")
        tbl.add_column("Buf ID", justify="right", style="yellow")
        tbl.add_column("Offset", justify="right", style="cyan")
        tbl.add_column("Origin", style="dim")

        node_views = compiled_graph.get("nodeViews", {})
        for idx in matching_indices[:100]:
            inst = instructions[idx]
            kid = inst.get("kernelId", 0)
            info = uid_map.get(kid) or uid_map.get(str(kid)) or uid_map.get(hex(kid).lower())
            op = format_op_name(info, f"Kernel_{hex(kid)}")
            out_v = node_views.get(inst.get("eclassId"), {})
            out_b = inst.get("outBuffer", {})
            v_str = f"{out_v.get('dtype', '?')}{out_v.get('shape', [])!s}"
            tbl.add_row(str(idx), op, v_str, str(out_b.get("id")), f"0x{out_b.get('offset', 0):x}", inst.get("debugOrigin", ""))

        console.print(tbl)
        if len(matching_indices) > 100:
            console.print(f"[dim]... and {len(matching_indices) - 100} more matches.[/dim]")
        return

    # ---- Default: Summary Mode ----
    console.rule(f"[bold cyan]Compiled Graph Overview: {graph_path_str}[/bold cyan]")
    console.print(f"[bold]Total Compiled Buckets:[/bold] {len(buckets)}")
    for b_i, b in enumerate(buckets):
        g = b["graph"]
        inst_count = len(g.get("instructions", []))
        total_time = sum(g.get("nodeCosts", {}).values())
        console.print(f"  [cyan]Bucket {b_i}:[/cyan] {inst_count:,} instructions | Est. Cost: {format_ms(total_time)}")
    console.print(
        "\n[white]Available inspection options:[/white]\n"
        "  [cyan]--validate[/cyan]            Run static memory bounds and lifetime collision checker\n"
        "  [cyan]--inst <#|start:end>[/cyan] Detailed instruction IO disassembly and bounds view\n"
        "  [cyan]--trace-buffer <id>[/cyan]   Trace buffer producer and consumers\n"
        "  [cyan]--trace-eclass <id>[/cyan]   Trace EClass producer and consumers\n"
        "  [cyan]--op <name>[/cyan]           Search instructions by op name or regex\n"
        "  [cyan]--mem-summary[/cyan]         Show memory space extents and peak buffer requirements"
    )


if __name__ == "__main__":
    main()
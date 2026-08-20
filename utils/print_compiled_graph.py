#!/usr/bin/env python3
import argparse
import os
import re
import sys

from rich import box
from rich.console import Console
from rich.table import Table

# Add the script's directory to sys.path to resolve local module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_performance import load_uids_from_cpp
from binary import load_cache_file

console = Console()


def format_num_or_str(val):
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
        return formatted if formatted else "0"
    return str(val)


class NullValue:
    def __eq__(self, other):
        return False

    def __ne__(self, other):
        return False

    def __lt__(self, other):
        return False

    def __le__(self, other):
        return False

    def __gt__(self, other):
        return False

    def __ge__(self, other):
        return False

    def __bool__(self):
        return False

    def __repr__(self):
        return "None"


class BufferProxy:
    def __init__(self, buf=None, view=None):
        buf = buf or {}
        view = view or {}
        self.id = buf.get("id")
        self.offset = buf.get("offset")
        self.size = buf.get("size")
        self.start = buf.get("start")
        self.end = buf.get("end")
        self.memSpaceIdx = buf.get("memSpaceIdx")
        self.ms_idx = self.memSpaceIdx
        self.memSpaceType = buf.get("memSpaceType")
        self.ms_type = self.memSpaceType
        self.shape = view.get("shape", [])
        self.strides = view.get("strides", [])
        self.dtype = view.get("dtype")

    def __getattribute__(self, name):
        val = super().__getattribute__(name)
        if val is None:
            return NullValue()
        return val

    def __eq__(self, other):
        if (
            self.offset == other
            or self.size == other
            or self.id == other
            or self.start == other
            or self.end == other
        ):
            return True
        return False


class NullBufferProxy:
    def __getattr__(self, name):
        return NullValue()

    def __getitem__(self, item):
        return NullValue()


class AnyField:
    def __init__(self, field_name, items):
        self.field_name = field_name
        self.items = items

    def _check(self, op, other):
        for item in self.items:
            val = getattr(item, self.field_name, None)
            if val is not None and not isinstance(val, NullValue):
                try:
                    if op(val, other):
                        return True
                except TypeError:
                    try:
                        if isinstance(val, (int, float)) and isinstance(other, str):
                            num_other = float(other) if "." in other else int(other)
                            if op(val, num_other):
                                return True
                        elif isinstance(other, (int, float)) and isinstance(val, str):
                            num_val = float(val) if "." in val else int(val)
                            if op(num_val, other):
                                return True
                    except ValueError:
                        pass
        return False

    def __eq__(self, other):
        return self._check(lambda a, b: a == b, other)

    def __ne__(self, other):
        return self._check(lambda a, b: a != b, other)

    def __lt__(self, other):
        return self._check(lambda a, b: a < b, other)

    def __le__(self, other):
        return self._check(lambda a, b: a <= b, other)

    def __gt__(self, other):
        return self._check(lambda a, b: a > b, other)

    def __ge__(self, other):
        return self._check(lambda a, b: a >= b, other)

    def __bool__(self):
        return any(bool(getattr(item, self.field_name, None)) for item in self.items)


class InputListProxy:
    def __init__(self, inputs):
        self._inputs = inputs

    def __getitem__(self, idx):
        if isinstance(idx, int):
            if 0 <= idx < len(self._inputs):
                return self._inputs[idx]
        return NullBufferProxy()

    def __len__(self):
        return len(self._inputs)

    def __getattr__(self, name):
        return AnyField(name, self._inputs)


class FilterScope(dict):
    def __missing__(self, key):
        if key in ("in", "inputs", "in_bufs"):
            return self.get("input")
        return key


def preprocess_filter_expr(expr: str) -> str:
    if not expr:
        return ""

    # 1. Convert dot-index notation for inputs (e.g. input.0.offset -> input[0].offset)
    expr = re.sub(
        r"\b(input|inputs|in|in_bufs)\.([0-9]+)", r"\1[\2]", expr, flags=re.IGNORECASE
    )

    # 2. Replace single '=' with '==' (avoiding '<=', '>=', '!=', '==')
    expr = re.sub(r"(?<![<>=!])=(?![=])", "==", expr)

    # 3. Replace '&' / '&&' with 'and', and '|' / '||' with 'or'
    expr = re.sub(r"\b&&\b|&", " and ", expr)
    expr = re.sub(r"\b\|\|\b|\|", " or ", expr)

    # 4. Normalize 'in.' / 'in[' to 'input' to avoid Python keyword collision
    expr = re.sub(r"\bin\[", "input[", expr)
    expr = re.sub(r"\bin\.", "input.", expr)

    return expr


def matches_filter(
    filter_expr,
    inst,
    idx,
    op_name,
    out_buf,
    out_view,
    in_bufs,
    children,
    node_views,
):
    in_proxies = []
    max_inputs = max(len(children), len(in_bufs))
    for c_idx in range(max_inputs):
        c_child = children[c_idx] if c_idx < len(children) else None
        c_view = node_views.get(c_child, {}) if c_child is not None else {}
        c_buf = in_bufs[c_idx] if c_idx < len(in_bufs) else {}
        in_proxies.append(BufferProxy(c_buf, c_view))

    out_proxy = BufferProxy(out_buf, out_view)
    input_proxy = InputListProxy(in_proxies)
    all_buffers = [out_proxy] + in_proxies

    kid = inst.get("kernelId")
    eclass = inst.get("eclassId")
    logical = inst.get("logicalId")
    debug = inst.get("debugOrigin")

    scope = FilterScope(
        {
            "out": out_proxy,
            "output": out_proxy,
            "input": input_proxy,
            "inputs": input_proxy,
            "in": input_proxy,
            "buffers": all_buffers,
            "offset": AnyField("offset", all_buffers),
            "size": AnyField("size", all_buffers),
            "start": AnyField("start", all_buffers),
            "end": AnyField("end", all_buffers),
            "id": AnyField("id", all_buffers),
            "buf_id": AnyField("id", all_buffers),
            "dtype": AnyField("dtype", all_buffers),
            "memSpaceIdx": AnyField("memSpaceIdx", all_buffers),
            "ms_idx": AnyField("memSpaceIdx", all_buffers),
            "memSpaceType": AnyField("memSpaceType", all_buffers),
            "ms_type": AnyField("memSpaceType", all_buffers),
            "kid": kid,
            "kernelId": kid,
            "op": op_name,
            "op_name": op_name,
            "eclass": eclass,
            "logical": logical,
            "debug": debug,
            "idx": idx,
        }
    )

    preprocessed = preprocess_filter_expr(filter_expr)
    try:
        return bool(eval(preprocessed, {"__builtins__": {}}, scope))
    except Exception as e:
        console.print(
            f"[bold red]Error evaluating filter expression '{filter_expr}' (parsed as"
            f" '{preprocessed}'):[/bold red] {e}"
        )
        sys.exit(1)


def print_graph(
    cache_file,
    bucket_idx=None,
    start_idx=None,
    end_idx=None,
    filter_expr=None,
):
    cache_entries = load_cache_file(cache_file, string_enums=True)
    uid_map = load_uids_from_cpp()

    buckets = [e for e in cache_entries if e.get("type") == "compiled_bucket"]

    if not buckets:
        console.print(
            f"[bold red]No compiled buckets found in '{cache_file}'.[/bold red]"
        )
        return

    for i, b in enumerate(buckets):
        if bucket_idx is not None and i != bucket_idx:
            continue

        console.rule(f"[bold cyan]Bucket {i}[/bold cyan]")

        graph = b["graph"]
        instructions = graph.get("instructions", [])
        node_views = graph.get("nodeViews", {})

        start_offset = 0
        if start_idx is not None:
            start_offset = (
                start_idx if start_idx >= 0 else max(0, len(instructions) + start_idx)
            )

        # Single master table for the entire bucket to keep all columns aligned across instructions
        table = Table(
            box=box.SIMPLE,
            show_header=True,
            header_style="bold cyan",
            padding=(0, 1),
            collapse_padding=True,
        )

        table.add_column("Inst / IO", style="bold magenta", justify="left")
        table.add_column("Shape / Details", style="green", justify="left")
        table.add_column("Buf ID", style="yellow", justify="right")
        table.add_column("Offset", style="cyan", justify="right")
        table.add_column("Size", style="blue", justify="right")
        table.add_column("Start", style="magenta", justify="right")
        table.add_column("End", style="magenta", justify="right")
        table.add_column("MemSpace", style="white", justify="left")
        table.add_column("Strides", style="dim", justify="left")
        table.add_column("EClass", style="bright_blue", justify="right")

        first_instruction = True

        for idx, inst in enumerate(instructions[start_idx:end_idx], start=start_offset):
            kid = inst["kernelId"]
            info = (
                uid_map.get(kid)
                or uid_map.get(str(kid))
                or uid_map.get(hex(kid).lower())
            )

            op_name = f"Kernel_{hex(kid)}"
            if info and isinstance(info, dict):
                op_name = info.get("name", op_name)
            elif isinstance(info, str):
                op_name = info

            eclass = inst.get("eclassId", "?")
            logical = inst.get("logicalId", "?")
            children = inst.get("children", [])

            out_view = node_views.get(eclass, {})
            out_shape = out_view.get("shape", [])
            out_strides = out_view.get("strides", [])
            out_dtype = out_view.get("dtype", "?")
            out_buf = inst.get("outBuffer", {})
            in_bufs = inst.get("inBuffers", [])

            if filter_expr and not matches_filter(
                filter_expr,
                inst,
                idx,
                op_name,
                out_buf,
                out_view,
                in_bufs,
                children,
                node_views,
            ):
                continue

            if not first_instruction:
                table.add_section()
            first_instruction = False

            # Add instruction header row
            table.add_row(
                f"[bold cyan][{idx:4d}][/bold cyan] [bold yellow]{op_name}[/bold yellow]",
                f"[dim](0x{kid:x})[/dim]",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                f"[bold white]Log:[/bold white][magenta]{logical}[/magenta]",
            )

            # Add Output buffer row
            shape_str = f"{out_dtype}{out_shape!s}"
            out_ms_idx = out_buf.get("memSpaceIdx", "?")
            out_ms_type = out_buf.get("memSpaceType", "?")
            out_start = out_buf.get("start", "?")
            out_end = out_buf.get("end", "?")

            table.add_row(
                "  Out",
                shape_str,
                str(out_buf.get("id", "?")),
                str(out_buf.get("offset", "?")),
                str(out_buf.get("size", "?")),
                format_num_or_str(out_start),
                format_num_or_str(out_end),
                f"{out_ms_type}({out_ms_idx})",
                str(out_strides),
                str(eclass),
            )

            # Add Input buffer rows
            for c_idx, child in enumerate(children):
                c_view = node_views.get(child, {})
                c_shape = c_view.get("shape", [])
                c_strides = c_view.get("strides", [])
                c_dtype = c_view.get("dtype", "?")
                c_shape_str = f"{c_dtype}{c_shape!s}"

                c_buf = in_bufs[c_idx] if c_idx < len(in_bufs) else {}
                c_ms_idx = c_buf.get("memSpaceIdx", "?")
                c_ms_type = c_buf.get("memSpaceType", "?")
                c_start = c_buf.get("start", "?")
                c_end = c_buf.get("end", "?")

                table.add_row(
                    f"  In {c_idx}",
                    c_shape_str,
                    str(c_buf.get("id", "?")),
                    str(c_buf.get("offset", "?")),
                    str(c_buf.get("size", "?")),
                    format_num_or_str(c_start),
                    format_num_or_str(c_end),
                    f"{c_ms_type}({c_ms_idx})",
                    str(c_strides),
                    str(child),
                )

            # Add Origin row if present
            debug = inst.get("debugOrigin")
            if debug:
                table.add_row(
                    "  [bold dim]Origin[/bold dim]",
                    f"[dim]{debug}[/dim]",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                )

        if len(table.rows) > 0:
            console.print(table)
        else:
            console.print("[dim]No matching instructions in this bucket.[/dim]")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Print chronological operation instructions from a compiled"
            " TensorGraph cache."
        )
    )
    parser.add_argument(
        "graph",
        help="Path to the compiled graph .bin file",
    )
    parser.add_argument(
        "--bucket",
        "-b",
        type=int,
        default=None,
        help="Specific bucket index to print",
    )
    parser.add_argument(
        "--start-idx",
        "--start",
        "-s",
        type=int,
        default=None,
        help="Start instruction index to print (inclusive)",
    )
    parser.add_argument(
        "--end-idx",
        "--end",
        "-e",
        type=int,
        default=None,
        help="End instruction index to print (exclusive)",
    )
    parser.add_argument(
        "--filter",
        "-f",
        "--where",
        type=str,
        default=None,
        help=(
            "Filter expression for instructions (e.g. 'input.offset = 1000',"
            " 'input.0.start = 0.0', 'end > 10.5')"
        ),
    )
    args = parser.parse_args()

    if not os.path.exists(args.graph):
        console.print(
            f"[bold red]Error:[/bold red] File '{args.graph}' does not exist.",
        )
        sys.exit(1)

    print_graph(
        args.graph,
        bucket_idx=args.bucket,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        filter_expr=args.filter,
    )


if __name__ == "__main__":
    main()

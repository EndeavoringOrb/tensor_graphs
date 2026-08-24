#!/usr/bin/env python3
# File: utils/print_compiled_graph.py
import argparse
import os
import re
import sys

from rich import box
from rich.console import Console
from rich.table import Table

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from binary import load_cache_file
from common import format_num_or_str, format_op_name, load_uids_from_cpp

console = Console()


class NullValue:
    def __eq__(self, _):
        return False

    def __ne__(self, _):
        return False

    def __lt__(self, _):
        return False

    def __le__(self, _):
        return False

    def __gt__(self, _):
        return False

    def __ge__(self, _):
        return False

    def __bool__(self):
        return False

    def __repr__(self):
        return "None"


class BufferProxy:
    def __init__(self, buf=None, view=None):
        buf, view = buf or {}, view or {}
        self.id = buf.get("id")
        self.offset = buf.get("offset")
        self.size = buf.get("size")
        self.start = buf.get("start")
        self.end = buf.get("end")
        self.memSpaceIdx = self.ms_idx = buf.get("memSpaceIdx")
        self.memSpaceType = self.ms_type = buf.get("memSpaceType")
        self.shape = view.get("shape", [])
        self.strides = view.get("strides", [])
        self.dtype = view.get("dtype")

    def __getattribute__(self, name):
        val = super().__getattribute__(name)
        return NullValue() if val is None else val

    def __eq__(self, other):
        return other in (self.offset, self.size, self.id, self.start, self.end)


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
                        num = float(other) if "." in str(other) else int(other)
                        if op(val, num):
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
        return any(bool(getattr(i, self.field_name, None)) for i in self.items)


class InputListProxy:
    def __init__(self, inputs):
        self._inputs = inputs

    def __getitem__(self, idx):
        return (
            self._inputs[idx]
            if isinstance(idx, int) and 0 <= idx < len(self._inputs)
            else BufferProxy()
        )

    def __len__(self):
        return len(self._inputs)

    def __getattr__(self, name):
        return AnyField(name, self._inputs)


def matches_filter(
    filter_expr, inst, idx, op_name, out_buf, out_view, in_bufs, children, node_views
):
    in_proxies = [
        BufferProxy(
            in_bufs[i] if i < len(in_bufs) else {},
            node_views.get(children[i], {}) if i < len(children) else {},
        )
        for i in range(max(len(children), len(in_bufs)))
    ]
    out_proxy = BufferProxy(out_buf, out_view)
    input_proxy = InputListProxy(in_proxies)
    all_bufs = [out_proxy] + in_proxies

    scope = {
        "out": out_proxy,
        "output": out_proxy,
        "input": input_proxy,
        "inputs": input_proxy,
        "in": input_proxy,
        "buffers": all_bufs,
        "offset": AnyField("offset", all_bufs),
        "size": AnyField("size", all_bufs),
        "start": AnyField("start", all_bufs),
        "end": AnyField("end", all_bufs),
        "id": AnyField("id", all_bufs),
        "dtype": AnyField("dtype", all_bufs),
        "memSpaceIdx": AnyField("memSpaceIdx", all_bufs),
        "memSpaceType": AnyField("memSpaceType", all_bufs),
        "kid": inst.get("kernelId"),
        "op": op_name,
        "eclass": inst.get("eclassId"),
        "logical": inst.get("logicalId"),
        "debug": inst.get("debugOrigin"),
        "idx": idx,
    }

    expr = re.sub(
        r"\b(input|inputs|in|in_bufs)\.([0-9]+)",
        r"\1[\2]",
        filter_expr,
        flags=re.IGNORECASE,
    )
    expr = re.sub(r"(?<![<>=!])=(?![=])", "==", expr)
    expr = re.sub(r"\b&&\b|&", " and ", expr)
    expr = re.sub(r"\b\|\|\b|\|", " or ", expr)
    expr = re.sub(r"\bin\[", "input[", expr)
    expr = re.sub(r"\bin\.", "input.", expr)

    try:
        return bool(eval(expr, {"__builtins__": {}}, scope))
    except Exception as e:
        console.print(
            f"[bold red]Error evaluating filter '{filter_expr}':[/bold red] {e}"
        )
        sys.exit(1)


def print_graph(
    cache_file, bucket_idx=None, start_idx=None, end_idx=None, filter_expr=None
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
        start_offset = start_idx if (start_idx is not None and start_idx >= 0) else 0

        table = Table(
            box=box.SIMPLE,
            show_header=True,
            header_style="bold cyan",
            padding=(0, 1),
            collapse_padding=True,
        )
        for col, align, style in [
            ("Inst / IO", "left", "bold magenta"),
            ("Shape / Details", "left", "green"),
            ("Buf ID", "right", "yellow"),
            ("Offset", "right", "cyan"),
            ("Size", "right", "blue"),
            ("Start", "right", "magenta"),
            ("End", "right", "magenta"),
            ("MemSpace", "left", "white"),
            ("Strides", "left", "dim"),
            ("EClass", "right", "bright_blue"),
        ]:
            table.add_column(col, style=style, justify=align)

        first_instruction = True
        for idx, inst in enumerate(instructions[start_idx:end_idx], start=start_offset):
            kid = inst["kernelId"]
            info = (
                uid_map.get(kid)
                or uid_map.get(str(kid))
                or uid_map.get(hex(kid).lower())
            )
            op_name = format_op_name(info, f"Kernel_{hex(kid)}")

            eclass = inst.get("eclassId", "?")
            logical = inst.get("logicalId", "?")
            children = inst.get("children", [])
            out_view = node_views.get(eclass, {})
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
            table.add_row(
                "  Out",
                f"{out_view.get('dtype', '?')}{out_view.get('shape', [])!s}",
                str(out_buf.get("id", "?")),
                str(out_buf.get("offset", "?")),
                str(out_buf.get("size", "?")),
                format_num_or_str(out_buf.get("start")),
                format_num_or_str(out_buf.get("end")),
                f"{out_buf.get('memSpaceType', '?')}({out_buf.get('memSpaceIdx', '?')})",
                str(out_view.get("strides", [])),
                str(eclass),
            )

            for c_idx, child in enumerate(children):
                c_view = node_views.get(child, {})
                c_buf = in_bufs[c_idx] if c_idx < len(in_bufs) else {}
                table.add_row(
                    f"  In {c_idx}",
                    f"{c_view.get('dtype', '?')}{c_view.get('shape', [])!s}",
                    str(c_buf.get("id", "?")),
                    str(c_buf.get("offset", "?")),
                    str(c_buf.get("size", "?")),
                    format_num_or_str(c_buf.get("start")),
                    format_num_or_str(c_buf.get("end")),
                    f"{c_buf.get('memSpaceType', '?')}({c_buf.get('memSpaceIdx', '?')})",
                    str(c_view.get("strides", [])),
                    str(child),
                )

            if inst.get("debugOrigin"):
                table.add_row(
                    "  [bold dim]Origin[/bold dim]",
                    f"[dim]{inst['debugOrigin']}[/dim]",
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
        description="Print chronological operation instructions from a compiled TensorGraph cache."
    )
    parser.add_argument("graph", help="Path to compiled graph .bin file")
    parser.add_argument(
        "--bucket", "-b", type=int, default=None, help="Specific bucket index"
    )
    parser.add_argument(
        "--start-idx",
        "--start",
        "-s",
        type=int,
        default=None,
        help="Start instruction index",
    )
    parser.add_argument(
        "--end-idx", "--end", "-e", type=int, default=None, help="End instruction index"
    )
    parser.add_argument(
        "--filter", "-f", "--where", type=str, default=None, help="Filter expression"
    )
    args = parser.parse_args()

    if not os.path.exists(args.graph):
        console.print(
            f"[bold red]Error:[/bold red] File '{args.graph}' does not exist."
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

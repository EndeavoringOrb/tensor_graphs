#!/usr/bin/env python3
# File: utils/read_safetensors.py
import argparse
import json
import os
import struct
import sys
from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import format_params, format_size, natural_sort_key, num_elements

console = Console()
console_err = Console(stderr=True)


def find_safetensors_files(path_str: str) -> list[Path]:
    p = Path(path_str)
    if not p.exists():
        console_err.print(
            f"[bold red]Error:[/bold red] Path '{path_str}' does not exist."
        )
        sys.exit(1)
    if p.is_file():
        return [p]
    files = sorted(p.rglob("*.safetensors"), key=lambda f: natural_sort_key(str(f)))
    if not files:
        console_err.print(
            f"[bold red]Error:[/bold red] No .safetensors files found under '{path_str}'."
        )
        sys.exit(1)
    return files


def get_header(file_path: Path) -> tuple[dict, int]:
    try:
        with open(file_path, "rb") as f:
            header_size_bytes = f.read(8)
            if len(header_size_bytes) < 8:
                console_err.print(
                    f"[bold red]Error:[/bold red] File '{file_path}' is too short."
                )
                sys.exit(1)
            header_size = struct.unpack("<Q", header_size_bytes)[0]
            header_bytes = f.read(header_size)
            if len(header_bytes) < header_size:
                console_err.print(
                    f"[bold red]Error:[/bold red] Incomplete header in '{file_path}'."
                )
                sys.exit(1)
            return json.loads(header_bytes.decode("utf-8")), header_size
    except Exception as e:
        console_err.print(
            f"[bold red]Error reading safetensors file '{file_path}':[/bold red] {e}"
        )
        sys.exit(1)


class TensorNode:
    def __init__(self, name=""):
        self.name = name
        self.children = {}
        self.info = None

    def add_tensor(self, path_parts, info):
        if not path_parts:
            self.info = info
            return
        head = path_parts[0]
        if head not in self.children:
            self.children[head] = TensorNode(head)
        self.children[head].add_tensor(path_parts[1:], info)

    def get_structure_signature(self):
        if self.info is not None:
            return ("leaf", self.info.get("dtype"), tuple(self.info.get("shape", [])))
        return tuple(
            ("*" if k.isdigit() else k, child.get_structure_signature())
            for k, child in self.children.items()
        )

    def get_stats(self):
        if self.info is not None:
            shape = self.info.get("shape", [])
            offsets = self.info.get("data_offsets", [0, 0])
            return num_elements(shape), offsets[1] - offsets[0], 1
        tot_elems = tot_bytes = tot_tensors = 0
        for child in self.children.values():
            e, b, t = child.get_stats()
            tot_elems += e
            tot_bytes += b
            tot_tensors += t
        return tot_elems, tot_bytes, tot_tensors

    def collect_dtypes(self, dtypes):
        if self.info:
            dtypes.add(self.info.get("dtype", "UNKNOWN"))
        for c in self.children.values():
            c.collect_dtypes(dtypes)


def process_tree_nodes(node, current_path, no_group, rows):
    if node.info is not None:
        shape = node.info.get("shape", [])
        offsets = node.info.get("data_offsets", [0, 0])
        rows.append(
            {
                "type": "tensor",
                "name": ".".join(current_path),
                "dtype": node.info.get("dtype", "UNKNOWN"),
                "shape": str(shape),
                "elements": num_elements(shape),
                "byte_size": offsets[1] - offsets[0],
            }
        )
        return

    if no_group or len(node.children) <= 1:
        for k, c in node.children.items():
            process_tree_nodes(c, current_path + [k], no_group, rows)
        return

    numeric_children = [k for k in node.children if k.isdigit()]
    if len(numeric_children) > 1:
        sig_groups = {}
        for k in numeric_children:
            sig_groups.setdefault(
                node.children[k].get_structure_signature(), []
            ).append(k)

        handled = set()
        for k, child in node.children.items():
            if k in handled:
                continue
            if k in numeric_children:
                group_keys = sig_groups[child.get_structure_signature()]
                if len(group_keys) > 1 and group_keys[0] == k:
                    process_tree_nodes(child, current_path + [k], no_group, rows)
                    handled.update(group_keys)

                    collapsed_keys = group_keys[1:]
                    tot_elems = tot_bytes = tot_tensors = 0
                    dtypes = set()
                    for ck in collapsed_keys:
                        e, b, t = node.children[ck].get_stats()
                        tot_elems += e
                        tot_bytes += b
                        tot_tensors += t
                        node.children[ck].collect_dtypes(dtypes)

                    count = len(collapsed_keys)
                    group_str = ".".join(
                        current_path + [f"[{collapsed_keys[0]}..{collapsed_keys[-1]}]"]
                    )
                    rows.append(
                        {
                            "type": "summary",
                            "name": f"{group_str} ({count} similar structures collapsed)",
                            "dtype": list(dtypes)[0] if len(dtypes) == 1 else "MIXED",
                            "shape": f"[{count} items, {tot_tensors // count} tensors/item]",
                            "elements": tot_elems,
                            "byte_size": tot_bytes,
                        }
                    )
                    continue
            process_tree_nodes(child, current_path + [k], no_group, rows)
            handled.add(k)
    else:
        for k, c in node.children.items():
            process_tree_nodes(c, current_path + [k], no_group, rows)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect metadata and tensor layouts of a .safetensors file or directory of shards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("file", help="Path to .safetensors file or folder.")
    parser.add_argument(
        "--name-only", action="store_true", help="Only list names of tensors."
    )
    parser.add_argument(
        "--meta-only", action="store_true", help="Only display global metadata."
    )
    parser.add_argument(
        "--no-summary", action="store_true", help="Do not display summary statistics."
    )
    parser.add_argument(
        "--no-group", action="store_true", help="Do not collapse repetitive structures."
    )
    parser.add_argument(
        "--json", action="store_true", help="Output raw parsed header as JSON."
    )
    args = parser.parse_args()

    files = find_safetensors_files(args.file)
    combined_header = {}
    combined_metadata = {}
    total_header_size = 0

    for file_path in files:
        header, header_size = get_header(file_path)
        total_header_size += header_size
        meta = header.pop("__metadata__", None)
        if meta:
            combined_metadata.update(meta)
        combined_header.update(header)

    if args.json:
        payload = (
            {"__metadata__": combined_metadata, **combined_header}
            if combined_metadata
            else combined_header
        )
        console.print_json(data=payload)
        return

    if args.meta_only:
        if combined_metadata:
            console.print_json(data=combined_metadata)
        else:
            console.print("[yellow]No global metadata (__metadata__) found.[/yellow]")
        return

    sorted_keys = sorted(combined_header.keys(), key=natural_sort_key)
    if args.name_only:
        for name in sorted_keys:
            console.print(name)
        return

    target_desc = (
        str(files[0]) if len(files) == 1 else f"{args.file} ({len(files)} files)"
    )
    console.print(f"[bold cyan]Target:[/bold cyan] {target_desc}")
    console.print(
        f"[bold cyan]Header Size:[/bold cyan] {format_size(total_header_size)} [dim]({total_header_size:,} bytes)[/dim]"
    )

    if combined_metadata:
        console.print()
        meta_table = Table(
            title="Global Metadata (__metadata__)",
            box=box.SIMPLE,
            show_header=True,
            header_style="bold cyan",
        )
        meta_table.add_column("Key", style="bold yellow")
        meta_table.add_column("Value", style="white")
        for k, v in combined_metadata.items():
            meta_table.add_row(str(k), str(v))
        console.print(meta_table)

    console.print()
    if not sorted_keys:
        console.print("[yellow]No tensors found.[/yellow]")
        return

    table = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="bold cyan",
        padding=(0, 1),
        collapse_padding=True,
    )
    table.add_column("Tensor Name", style="bold yellow", justify="left")
    table.add_column("Dtype", style="green", justify="left")
    table.add_column("Shape", style="magenta", justify="left")
    table.add_column("Elements", style="cyan", justify="right")
    table.add_column("Byte Size", style="blue", justify="right")

    root = TensorNode("root")
    for name in sorted_keys:
        root.add_tensor(name.split("."), combined_header[name])

    rows = []
    process_tree_nodes(root, [], args.no_group, rows)

    for row in rows:
        shape_str = row["shape"]
        if len(shape_str) > 30 and row["type"] == "tensor":
            shape_str = shape_str[:27] + "..."

        style_wrap = (
            (lambda s: f"[dim {s.split()[0]}]{s}[/dim {s.split()[0]}]")
            if row["type"] == "summary"
            else (lambda s: s)
        )
        table.add_row(
            f"[dim yellow]{row['name']}[/dim yellow]"
            if row["type"] == "summary"
            else row["name"],
            f"[dim green]{row['dtype']}[/dim green]"
            if row["type"] == "summary"
            else row["dtype"],
            f"[dim magenta]{shape_str}[/dim magenta]"
            if row["type"] == "summary"
            else shape_str,
            f"[dim cyan]{format_params(row['elements'])}[/dim cyan]"
            if row["type"] == "summary"
            else format_params(row["elements"]),
            f"[dim blue]{format_size(row['byte_size'])}[/dim blue]"
            if row["type"] == "summary"
            else format_size(row["byte_size"]),
        )

    console.print(table)

    if not args.no_summary:
        total_elements = sum(
            num_elements(info.get("shape", [])) for info in combined_header.values()
        )
        total_bytes = sum(
            info.get("data_offsets", [0, 0])[1] - info.get("data_offsets", [0, 0])[0]
            for info in combined_header.values()
        )
        console.print("\n[bold cyan]Summary:[/bold cyan]")
        if len(files) > 1:
            console.print(
                f"  [bold white]Total Files:[/bold white]      [yellow]{len(files)}[/yellow]"
            )
        console.print(
            f"  [bold white]Total Tensors:[/bold white]    [yellow]{len(combined_header)}[/yellow]"
        )
        console.print(
            f"  [bold white]Total Parameters:[/bold white] [cyan]{format_params(total_elements)}[/cyan] [dim]({total_elements:,} elements)[/dim]"
        )
        console.print(
            f"  [bold white]Total Data Size:[/bold white]  [blue]{format_size(total_bytes)}[/blue] [dim]({total_bytes:,} bytes)[/dim]"
        )


if __name__ == "__main__":
    main()

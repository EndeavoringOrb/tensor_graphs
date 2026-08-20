#!/usrKey/env python3
import argparse
import json
import os
import re
import struct
import sys
from functools import reduce

from rich import box
from rich.console import Console
from rich.table import Table

console = Console()
console_err = Console(stderr=True)


def natural_sort_key(s):
    """
    Sort key for natural sorting (e.g., '1', '2', '10' instead of '1', '10', '2').
    """
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)
    ]


def get_header(file_path):
    """
    Reads and parses the JSON header from a .safetensors file.
    """
    if not os.path.exists(file_path):
        console_err.print(
            f"[bold red]Error:[/bold red] File '{file_path}' does not exist."
        )
        sys.exit(1)

    try:
        with open(file_path, "rb") as f:
            header_size_bytes = f.read(8)
            if len(header_size_bytes) < 8:
                console_err.print(
                    f"[bold red]Error:[/bold red] File '{file_path}' is too short to be a valid .safetensors file."
                )
                sys.exit(1)

            header_size = struct.unpack("<Q", header_size_bytes)[0]
            header_bytes = f.read(header_size)
            if len(header_bytes) < header_size:
                console_err.print(
                    f"[bold red]Error:[/bold red] Failed to read the full header of size {header_size} bytes."
                )
                sys.exit(1)

            header_str = header_bytes.decode("utf-8")
            return json.loads(header_str), header_size

    except Exception as e:
        console_err.print(f"[bold red]Error reading safetensors file:[/bold red] {e}")
        sys.exit(1)


def num_elements(shape):
    """
    Calculates total number of elements in a tensor given its shape.
    """
    if not shape:
        return 0
    return reduce(lambda x, y: x * y, shape, 1)


def format_size(bytes_size):
    """
    Utility to format bytes size to human-readable strings.
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def format_params(num):
    """
    Formats the parameter count for clean readability.
    """
    if num >= 1e9:
        return f"{num / 1e9:.2f} B"
    if num >= 1e6:
        return f"{num / 1e6:.2f} M"
    if num >= 1e3:
        return f"{num / 1e3:.2f} K"
    return str(num)


class TensorNode:
    """
    Prefix tree node representing tensor name components.
    """

    def __init__(self, name=""):
        self.name = name
        self.children = {}  # key -> TensorNode (insertion order)
        self.info = None  # Dict with metadata if this is a leaf tensor node

    def add_tensor(self, path_parts, info):
        if not path_parts:
            self.info = info
            return
        head = path_parts[0]
        if head not in self.children:
            self.children[head] = TensorNode(head)
        self.children[head].add_tensor(path_parts[1:], info)

    def get_structure_signature(self):
        """
        Generates a structural signature to determine if sibling sub-trees are identical.
        Numeric key components are wildcards (*) so 'experts.0' matches 'experts.1'.
        """
        if self.info is not None:
            return ("leaf", self.info.get("dtype"), tuple(self.info.get("shape", [])))

        child_sigs = []
        for k, child in self.children.items():
            key_pattern = "*" if k.isdigit() else k
            child_sigs.append((key_pattern, child.get_structure_signature()))
        return tuple(child_sigs)

    def get_stats(self):
        """
        Recursively calculates total elements, bytes, and total leaf tensors.
        """
        if self.info is not None:
            shape = self.info.get("shape", [])
            offsets = self.info.get("data_offsets", [0, 0])
            elems = num_elements(shape)
            bytes_sz = offsets[1] - offsets[0]
            return elems, bytes_sz, 1

        tot_elems, tot_bytes, tot_tensors = 0, 0, 0
        for child in self.children.values():
            e, b, t = child.get_stats()
            tot_elems += e
            tot_bytes += b
            tot_tensors += t
        return tot_elems, tot_bytes, tot_tensors


def process_tree_nodes(node, current_path, no_group, rows):
    """
    Traverses tree nodes to produce display rows. Expands the first instance
    of repeating structures and collapses subsequent identical siblings.
    """
    if node.info is not None:
        full_name = ".".join(current_path)
        dtype = node.info.get("dtype", "UNKNOWN")
        shape = node.info.get("shape", [])
        offsets = node.info.get("data_offsets", [0, 0])
        elements = num_elements(shape)
        byte_size = offsets[1] - offsets[0]

        rows.append(
            {
                "type": "tensor",
                "name": full_name,
                "dtype": dtype,
                "shape": str(shape),
                "elements": elements,
                "byte_size": byte_size,
            }
        )
        return

    if no_group or len(node.children) <= 1:
        for child_key, child_node in node.children.items():
            process_tree_nodes(child_node, current_path + [child_key], no_group, rows)
        return

    # Check for numeric sibling components eligible for grouping (e.g. expert IDs)
    numeric_children = [k for k in node.children.keys() if k.isdigit()]

    if len(numeric_children) > 1:
        # Group numeric children by structural signature
        sig_groups = {}
        for k in numeric_children:
            sig = node.children[k].get_structure_signature()
            sig_groups.setdefault(sig, []).append(k)

        handled_keys = set()
        for child_key, child_node in node.children.items():
            if child_key in handled_keys:
                continue

            if child_key in numeric_children:
                sig = child_node.get_structure_signature()
                group_keys = sig_groups.get(sig, [])

                if len(group_keys) > 1 and group_keys[0] == child_key:
                    # Render the first instance fully expanded
                    process_tree_nodes(
                        child_node, current_path + [child_key], no_group, rows
                    )
                    handled_keys.add(child_key)

                    # Group remaining identical siblings into a single summary row
                    collapsed_keys = group_keys[1:]
                    for ck in collapsed_keys:
                        handled_keys.add(ck)

                    tot_elems, tot_bytes, tot_tensors = 0, 0, 0
                    dtypes = set()

                    for ck in collapsed_keys:
                        e, b, t = node.children[ck].get_stats()
                        tot_elems += e
                        tot_bytes += b
                        tot_tensors += t

                        def collect_dtypes(n):
                            if n.info:
                                dtypes.add(n.info.get("dtype", "UNKNOWN"))
                            for c in n.children.values():
                                collect_dtypes(c)

                        collect_dtypes(node.children[ck])

                    first_idx = collapsed_keys[0]
                    last_idx = collapsed_keys[-1]
                    count = len(collapsed_keys)
                    group_path_str = ".".join(
                        current_path + [f"[{first_idx}..{last_idx}]"]
                    )
                    dtype_str = list(dtypes)[0] if len(dtypes) == 1 else "MIXED"
                    tensors_per_item = tot_tensors // count

                    rows.append(
                        {
                            "type": "summary",
                            "name": f"{group_path_str} ({count} similar structures collapsed)",
                            "dtype": dtype_str,
                            "shape": f"[{count} items, {tensors_per_item} tensors/item]",
                            "elements": tot_elems,
                            "byte_size": tot_bytes,
                        }
                    )
                    continue

            process_tree_nodes(child_node, current_path + [child_key], no_group, rows)
            handled_keys.add(child_key)
    else:
        for child_key, child_node in node.children.items():
            process_tree_nodes(child_node, current_path + [child_key], no_group, rows)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect and pretty-print metadata and tensor layouts of a .safetensors file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("file", help="Path to the .safetensors file to inspect.")
    parser.add_argument(
        "--name-only",
        action="store_true",
        help="Only list the names of the tensors, one per line.",
    )
    parser.add_argument(
        "--meta-only",
        action="store_true",
        help="Only display global metadata (__metadata__) if present.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Do not display the final summary statistics.",
    )
    parser.add_argument(
        "--no-group",
        action="store_true",
        help="Do not group/collapse repetitive tensor structures (e.g. MoE experts).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output the raw parsed header as JSON.",
    )
    args = parser.parse_args()

    header, header_size = get_header(args.file)

    if args.json:
        console.print_json(data=header)
        return

    metadata = header.pop("__metadata__", None)

    if args.meta_only:
        if metadata:
            console.print_json(data=metadata)
        else:
            console.print(
                "[yellow]No global metadata (__metadata__) found in this file.[/yellow]"
            )
        return

    # Sort all keys naturally
    sorted_keys = sorted(header.keys(), key=natural_sort_key)

    if args.name_only:
        for name in sorted_keys:
            console.print(name)
        return

    console.print(f"[bold cyan]File:[/bold cyan] {args.file}")
    console.print(
        f"[bold cyan]Header Size:[/bold cyan] {format_size(header_size)} [dim]({header_size:,} bytes)[/dim]"
    )

    if metadata:
        console.print()
        meta_table = Table(
            title="Global Metadata (__metadata__)",
            box=box.SIMPLE,
            show_header=True,
            header_style="bold cyan",
        )
        meta_table.add_column("Key", style="bold yellow")
        meta_table.add_column("Value", style="white")
        for k, v in metadata.items():
            meta_table.add_row(str(k), str(v))
        console.print(meta_table)

    console.print()

    if not sorted_keys:
        console.print("[yellow]No tensors found.[/yellow]")
    else:
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

        # Build prefix tree using naturally ordered tensor paths
        root = TensorNode("root")
        for name in sorted_keys:
            parts = name.split(".")
            root.add_tensor(parts, header[name])

        rows = []
        process_tree_nodes(root, [], args.no_group, rows)

        for row in rows:
            shape_str = row["shape"]
            if len(shape_str) > 30 and row["type"] == "tensor":
                shape_str = shape_str[:27] + "..."

            if row["type"] == "summary":
                # Styled subtly for collapsed summary rows
                table.add_row(
                    f"[dim yellow]{row['name']}[/dim yellow]",
                    f"[dim green]{row['dtype']}[/dim green]",
                    f"[dim magenta]{shape_str}[/dim magenta]",
                    f"[dim cyan]{format_params(row['elements'])}[/dim cyan]",
                    f"[dim blue]{format_size(row['byte_size'])}[/dim blue]",
                )
            else:
                table.add_row(
                    row["name"],
                    row["dtype"],
                    shape_str,
                    format_params(row["elements"]),
                    format_size(row["byte_size"]),
                )

        console.print(table)

        if not args.no_summary:
            total_elements = sum(
                num_elements(info.get("shape", [])) for info in header.values()
            )
            total_bytes = sum(
                info.get("data_offsets", [0, 0])[1]
                - info.get("data_offsets", [0, 0])[0]
                for info in header.values()
            )

            console.print("\n[bold cyan]Summary:[/bold cyan]")
            console.print(
                f"  [bold white]Total Tensors:[/bold white]    [yellow]{len(header)}[/yellow]"
            )
            console.print(
                f"  [bold white]Total Parameters:[/bold white] [cyan]{format_params(total_elements)}[/cyan] [dim]({total_elements:,} elements)[/dim]"
            )
            console.print(
                f"  [bold white]Total Data Size:[/bold white]  [blue]{format_size(total_bytes)}[/blue] [dim]({total_bytes:,} bytes)[/dim]"
            )


if __name__ == "__main__":
    main()

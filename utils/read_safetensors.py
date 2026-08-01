#!/usr/bin/env python3
import argparse
import json
import os
import struct
import sys
from functools import reduce

from rich import box
from rich.console import Console
from rich.table import Table

console = Console()
console_err = Console(stderr=True)


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

            # Read the 8-byte little-endian header length
            header_size = struct.unpack("<Q", header_size_bytes)[0]

            # Read the header JSON content
            header_bytes = f.read(header_size)
            if len(header_bytes) < header_size:
                console_err.print(
                    f"[bold red]Error:[/bold red] Failed to read the full header of size {header_size} bytes."
                )
                sys.exit(1)

            header_str = header_bytes.decode("utf-8")
            return json.loads(header_str), header_size

    except Exception as e:
        console_err.print(
            f"[bold red]Error reading safetensors file:[/bold red] {e}"
        )
        sys.exit(1)


def num_elements(shape):
    """
    Calculates total number of elements in a tensor given its shape.
    """
    if not shape:
        return 0
    # Multiplies all dimensions together
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
        "--json",
        action="store_true",
        help="Output the raw parsed header as JSON.",
    )
    args = parser.parse_args()

    header, header_size = get_header(args.file)

    if args.json:
        console.print_json(data=header)
        return

    # Safetensors allow an optional global metadata dictionary under '__metadata__'
    metadata = header.pop("__metadata__", None)

    if args.meta_only:
        if metadata:
            console.print_json(data=metadata)
        else:
            console.print(
                "[yellow]No global metadata (__metadata__) found in this file.[/yellow]"
            )
        return

    if args.name_only:
        for name in header.keys():
            console.print(name)
        return

    # Standard detailed visualization output
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

    names = list(header.keys())
    if not names:
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

        total_elements = 0
        total_bytes = 0

        for name, info in header.items():
            dtype = info.get("dtype", "UNKNOWN")
            shape = info.get("shape", [])
            offsets = info.get("data_offsets", [0, 0])

            elements = num_elements(shape)
            byte_size = offsets[1] - offsets[0]

            total_elements += elements
            total_bytes += byte_size

            # Format shape to prevent rendering excessively long lists
            shape_str = str(shape)
            if len(shape_str) > 30:
                shape_str = shape_str[:27] + "..."

            table.add_row(
                name,
                dtype,
                shape_str,
                format_params(elements),
                format_size(byte_size),
            )

        console.print(table)

        if not args.no_summary:
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
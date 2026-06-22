#!/usr/bin/env python3
import argparse
import json
import os
import struct
import sys
from functools import reduce


def get_header(file_path):
    """
    Reads and parses the JSON header from a .safetensors file.
    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    try:
        with open(file_path, "rb") as f:
            header_size_bytes = f.read(8)
            if len(header_size_bytes) < 8:
                print(
                    f"Error: File '{file_path}' is too short to be a valid .safetensors file.",
                    file=sys.stderr,
                )
                sys.exit(1)

            # Read the 8-byte little-endian header length
            header_size = struct.unpack("<Q", header_size_bytes)[0]

            # Read the header JSON content
            header_bytes = f.read(header_size)
            if len(header_bytes) < header_size:
                print(
                    f"Error: Failed to read the full header of size {header_size} bytes.",
                    file=sys.stderr,
                )
                sys.exit(1)

            header_str = header_bytes.decode("utf-8")
            return json.loads(header_str), header_size

    except Exception as e:
        print(f"Error reading safetensors file: {e}", file=sys.stderr)
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
    parser.add_argument(
        "file", help="Path to the .safetensors file to inspect."
    )
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
        print(json.dumps(header, indent=2))
        return

    # Safetensors allow an optional global metadata dictionary under '__metadata__'
    metadata = header.pop("__metadata__", None)

    if args.meta_only:
        if metadata:
            print(json.dumps(metadata, indent=2))
        else:
            print("No global metadata (__metadata__) found in this file.")
        return

    if args.name_only:
        for name in header.keys():
            print(name)
        return

    # Standard detailed visualization output
    print("=" * 85)
    print(f"File: {args.file}")
    print(f"Header Size: {format_size(header_size)} ({header_size} bytes)")

    if metadata:
        print("-" * 85)
        print("Global Metadata (__metadata__):")
        for k, v in metadata.items():
            print(f"  {k}: {v}")

    print("-" * 85)
    print("Tensors:")

    names = list(header.keys())
    if not names:
        print("  No tensors found.")
    else:
        # Determine comfortable column width for alignment
        max_name_len = max(len(n) for n in names)
        max_name_len = max(max_name_len, 11)  # Minimum width matching "Tensor Name"
        max_name_len = min(max_name_len, 60)  # Soft limit to prevent overflow of wide terminal views

        header_line = f"  {'Tensor Name':<{max_name_len}} | {'Dtype':<8} | {'Shape':<25} | {'Elements':<10} | {'Byte Size':<10}"
        print(header_line)
        print("  " + "-" * (len(header_line) - 2))

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
            if len(shape_str) > 25:
                shape_str = shape_str[:22] + "..."

            display_name = name
            if len(display_name) > max_name_len:
                display_name = display_name[: max_name_len - 3] + "..."

            print(
                f"  {display_name:<{max_name_len}} | {dtype:<8} | {shape_str:<25} | {format_params(elements):<10} | {format_size(byte_size):<10}"
            )

        if not args.no_summary:
            print("-" * 85)
            print("Summary:")
            print(f"  Total Tensors:    {len(header)}")
            print(f"  Total Parameters: {format_params(total_elements)} ({total_elements:,} elements)")
            print(f"  Total Data Size:  {format_size(total_bytes)} ({total_bytes:,} bytes)")
    print("=" * 85)


if __name__ == "__main__":
    main()
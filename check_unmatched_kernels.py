#!/usr/bin/env python3
import struct
import os
import argparse
import re
from pathlib import Path


class BinaryReader:
    """Helper class to read the binary benchmarks format exactly as the C++ backend outputs it."""

    def __init__(self, f):
        self.f = f

    def read_u8(self):
        buf = self.f.read(1)
        if not buf: return None
        return struct.unpack("<B", buf)[0]

    def read_u32(self):
        buf = self.f.read(4)
        if not buf: return None
        return struct.unpack("<I", buf)[0]

    def read_u64(self):
        buf = self.f.read(8)
        if not buf: return None
        return struct.unpack("<Q", buf)[0]

    def read_float(self):
        buf = self.f.read(4)
        if not buf: return None
        return struct.unpack("<f", buf)[0]

    def read_string(self):
        size = self.read_u32()
        if size is None:
            return None
        if size == 0:
            return ""
        return self.f.read(size).decode("utf-8", errors="ignore")

    def read_vector(self, read_func):
        size = self.read_u32()
        if size is None:
            return None
        return [read_func() for _ in range(size)]

    def read_dtype(self):
        return self.read_u32()

    def read_backend(self):
        return self.read_u32()

    def read_record(self):
        kernelUid = self.read_u64()
        if kernelUid is None:
            return None

        self.read_u64()  # buildContextId
        self.read_string()  # hwTag
        self.read_vector(lambda: self.read_vector(self.read_u32))  # inputShapes
        self.read_vector(lambda: self.read_vector(self.read_u32))  # outputShapes
        self.read_vector(lambda: self.read_vector(self.read_u64))  # inputStrides
        self.read_vector(lambda: self.read_vector(self.read_u64))  # outputStrides
        self.read_vector(self.read_dtype)  # inputDTypes
        self.read_vector(self.read_dtype)  # outputDTypes
        self.read_vector(lambda: self.f.read(self.read_u32()))  # inputConstants
        self.read_vector(self.read_backend)  # backends
        self.read_vector(lambda: self.read_vector(self.read_backend))  # inputBackends
        self.read_float()  # runTime

        return kernelUid


def get_registered_kernels(header_path):
    """Parses kernel_uids.gen.hpp to map UIDs to their macro names."""
    uid_to_macro = {}
    if not os.path.exists(header_path):
        print(
            f"Warning: Header file {header_path} not found. Please run the build script first."
        )
        return uid_to_macro

    pattern = re.compile(r"constexpr uint64_t\s+(\w+)\s+=\s+(0x[0-9a-fA-F]+)ULL;")
    with open(header_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                name, hex_val = match.groups()
                uid = int(hex_val, 16)
                uid_to_macro[uid] = name
    return uid_to_macro


def get_called_uids(calls_path):
    """Reads benchmarks/calls.bin and returns a set of called kernel UIDs."""
    called_uids = set()
    if not os.path.exists(calls_path):
        print(f"Warning: Calls file {calls_path} not found. (Assuming 0 calls)")
        return called_uids

    with open(calls_path, "rb") as f:
        br = BinaryReader(f)
        while True:
            uid = br.read_record()
            if uid is None:
                break
            called_uids.add(uid)
    return called_uids


def get_kernel_metadata(root_dir):
    """Scans the source code to map macro names to actual kernel names and file paths."""
    metadata = {}
    kernels_dir = Path(root_dir) / "kernels"

    if not kernels_dir.exists():
        return metadata

    for path in kernels_dir.rglob("*"):
        if path.suffix in [".hpp", ".cu", ".cpp"]:
            rel_path = path.relative_to(root_dir)
            # Replicate the logic build.py uses to formulate the macro string
            const_name = (
                str(rel_path)
                .replace("\\", "_")
                .replace("/", "_")
                .replace(".", "_")
                .upper()
            )

            with open(path, "r", encoding="utf-8", errors="ignore") as f_in:
                content = f_in.read()
                # Parse the REGISTER_... macro to get the Kernel string name or OpType enum
                match = re.search(
                    r"REGISTER_[\w_]+\s*\(\s*(?:OpType::(\w+)|\"([^\"]+)\")", content
                )
                if match:
                    op_enum = match.group(1)
                    op_str = match.group(2)
                    actual_name = op_str if op_str else f"OpType::{op_enum}"
                    metadata[const_name] = (actual_name, str(rel_path))
                else:
                    metadata[const_name] = ("Unknown", str(rel_path))
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Find kernels that are registered but missing from benchmarks/calls.bin"
    )
    parser.add_argument(
        "--calls", default="benchmarks/calls.bin", help="Path to calls.bin"
    )
    parser.add_argument(
        "--header",
        default="tensor_graphs_cpp/generated/kernel_uids.gen.hpp",
        help="Path to kernel_uids.gen.hpp",
    )
    parser.add_argument(
        "--root",
        default="tensor_graphs_cpp",
        help="Root directory for tensor_graphs_cpp",
    )
    args = parser.parse_args()

    registered_kernels = get_registered_kernels(args.header)
    called_uids = get_called_uids(args.calls)
    metadata = get_kernel_metadata(args.root)

    unmatched = []
    for uid, macro_name in registered_kernels.items():
        if uid not in called_uids:
            actual_name, file_path = metadata.get(
                macro_name, ("Unknown Name", "Unknown file")
            )
            unmatched.append(
                {
                    "uid": uid,
                    "macro": macro_name,
                    "name": actual_name,
                    "file": file_path,
                }
            )

    if not unmatched:
        print(
            "✅ All registered kernels have matching entries in benchmarks/calls.bin!"
        )
        return

    print(f"❌ Found {len(unmatched)} unmatched kernel(s):\n")

    # Sort by file path for clean grouping
    unmatched.sort(key=lambda x: x["file"])

    for k in unmatched:
        print(f"Kernel Name : {k['name']}")
        print(f"File Path   : {k['file']}")
        print(f"UID         : {hex(k['uid'])}")
        print("-" * 60)


if __name__ == "__main__":
    main()

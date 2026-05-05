import argparse
import os
import sys
import hashlib
import subprocess
import re
from pathlib import Path
import platform
from rich.console import Console
from rich.panel import Panel

console = Console()

# --- Configuration ---
ROOT_DIR = Path("tensor_graphs_cpp")
GENERATED_DIR = ROOT_DIR / "generated"
KERNELS_DIR = ROOT_DIR / "kernels"
VCVARS_PATH = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"

# Core files that affect the ABI/ID of all kernels
CORE_DEPENDENCIES = [
    ROOT_DIR / "core" / "types.hpp",
    ROOT_DIR / "core" / "kernels.hpp",
    ROOT_DIR / "core" / "graph.hpp",
]

USE_CUDA = False
DEBUG_MODE = False

# List of macros that register a kernel with a unique UID
REGISTER_MACROS = [
    "REGISTER_REF_KERNEL",
    "REGISTER_REF_KERNEL_INPLACE",
    "REGISTER_REF_KERNEL_VIEW",
    "REGISTER_KERNEL",
    "REGISTER_KERNEL_INPLACE",
    "REGISTER_KERNEL_VIEW",
]


def get_compiler_cmd(fname: str):
    out_ext = ".exe" if os.name == "nt" else ""
    out_name = f"tensor_graphs_cpp/{fname.split('.')[0]}{out_ext}"
    is_arm64 = platform.machine().lower() in ("aarch64", "arm64")

    if USE_CUDA:
        cmd = [
            "nvcc",
            "-std=c++17",
            f"-I{ROOT_DIR}",
            "-DUSE_CUDA",
            "-x",
            "cu",
        ]

        # Fix for ARM64 NEON errors when using nvcc (pass flag to host compiler)
        if is_arm64:
            cmd.extend(["-Xcompiler", "-march=armv8-a"])

        if DEBUG_MODE:
            cmd.extend(["-g", "-G", "-O0", "-DDEBUG"])
        else:
            cmd.extend(["-O3"])

        cmd.append(str(ROOT_DIR / fname))
        cmd.extend(["-o", out_name])
        return cmd
    else:
        if os.name == "nt":
            cmd = [
                "cl.exe",
                "/std:c++17",
                "/EHsc",
                f"/I{ROOT_DIR}",
            ]

            if DEBUG_MODE:
                cmd.extend(["/Zi", "/Od", "/DDEBUG"])
            else:
                cmd.extend(["/O2"])

            cmd.append(str(ROOT_DIR / fname))
            cmd.extend([f"/Fe:{out_name}"])
            return cmd
        else:
            cmd = [
                "g++",
                "-std=c++17",
                f"-I{ROOT_DIR}",
            ]

            if DEBUG_MODE:
                cmd.extend(["-g", "-O0", "-DDEBUG", "-fno-omit-frame-pointer"])
            else:
                cmd.extend(["-O3"])

            cmd.append(str(ROOT_DIR / fname))
            cmd.extend(["-o", out_name])
            return cmd


def get_file_hash(filepath):
    """Return SHA256 hash of a file's content."""
    h = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            h.update(f.read())
        return h.hexdigest()
    except FileNotFoundError:
        print(f"Warning: Dependency file not found: {filepath}")
        return "0" * 64


def generate_core_seed():
    """Hashes core files to create a stable seed for all kernel IDs."""
    content_hashes = [get_file_hash(p) for p in CORE_DEPENDENCIES]
    combined = "".join(content_hashes)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def generate_kernel_uids(core_seed):
    os.makedirs(GENERATED_DIR, exist_ok=True)
    uids_hpp = GENERATED_DIR / "kernel_uids.gen.hpp"
    kernel_map = {}
    uid_to_path = {}
    kernel_exts = [".hpp", ".cu"]

    for root, _, files in os.walk(KERNELS_DIR):
        for f in files:
            if any(f.endswith(ext) for ext in kernel_exts):
                path = Path(root) / f
                rel_path = path.relative_to(ROOT_DIR)

                # --- Validation: Ensure single registration per file ---
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f_in:
                        content = f_in.read()

                    reg_count = 0
                    for macro in REGISTER_MACROS:
                        # Use \b for word boundary to avoid matching substring macros
                        # Multiline mode allows ^ to match start of lines
                        matches = re.findall(rf"^\s*{macro}\b", content, re.MULTILINE)
                        reg_count += len(matches)

                    if reg_count > 1:
                        console.print(
                            Panel(
                                f"[bold red]FATAL ERROR:[/bold red] Found {reg_count} kernel registrations in [cyan]{rel_path}[/cyan].\n\n"
                                f"The build system generates UIDs based on file paths. To prevent ID collisions "
                                f"and ensure correct kernel selection, each kernel variation must be in its own file.",
                                title="Multiple Registrations Detected",
                                border_style="red",
                            )
                        )
                        sys.exit(1)
                except Exception as e:
                    console.print(
                        f"[yellow]Warning: Could not scan {rel_path} for macros: {e}[/yellow]"
                    )

                file_content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                combined = core_seed + file_content_hash
                full_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()

                uid_val_raw = int(full_hash[:16], 16)
                uid_val = f"0x{uid_val_raw:016x}ULL"

                if uid_val in uid_to_path:
                    if uid_to_path[uid_val] != str(rel_path):
                        raise Exception(
                            f"CRITICAL COLLISION: Kernels '{rel_path}' and "
                            f"'{uid_to_path[uid_val]}' produced the same UID: {uid_val}. "
                            f"Change the kernel content slightly or update the core seed."
                        )

                uid_to_path[uid_val] = str(rel_path)
                const_name = (
                    str(rel_path)
                    .replace("/", "_")
                    .replace("\\", "_")
                    .replace(".", "_")
                    .upper()
                )
                kernel_map[const_name] = uid_val

    with open(uids_hpp, "w") as f:
        f.write("#pragma once\n")
        f.write("#include <cstdint>\n\n")
        f.write("// Generated by build.py - DO NOT EDIT\n")
        f.write(f"// Core Seed: {core_seed[:16]}...\n\n")
        f.write("namespace KernelIDs {\n")
        for name, uid in sorted(kernel_map.items()):
            f.write(f"    constexpr uint64_t {name} = {uid};\n")
        f.write("}\n")

    console.print(f"[dim]Generated {len(kernel_map)} Kernel UIDs.[/dim]")
    return kernel_map


def generate_kernel_includes(core_seed):
    """Generates kernels_all.gen.hpp with UID injection logic."""
    includes_hpp = GENERATED_DIR / "kernels_all.gen.hpp"
    kernel_entries = []
    kernel_exts = [".hpp", ".cu"]

    for root, _, files in os.walk(KERNELS_DIR):
        for f in files:
            if any(f.endswith(ext) for ext in kernel_exts):
                path = Path(root) / f
                rel_path = path.relative_to(ROOT_DIR)

                file_content_hash = get_file_hash(path)
                combined = core_seed + file_content_hash
                full_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()
                uid_val = f"0x{full_hash[:16]}ULL"

                inc_path = str(rel_path).replace("\\", "/")
                kernel_entries.append((inc_path, uid_val))

    with open(includes_hpp, "w") as f:
        f.write("#pragma once\n")
        f.write('#include "core/kernels.hpp"\n\n')
        f.write("// Generated by build.py - Injects UIDs and includes all kernels\n\n")

        for inc_path, uid in sorted(kernel_entries):
            f.write(f"// --- {inc_path} ---\n")
            for macro in REGISTER_MACROS:
                f.write(f"#undef {macro}\n")

            f.write(
                f"#define REGISTER_REF_KERNEL(op, n, m, r, ...) REGISTER_REF_KERNEL_INTERNAL({uid}, op, n, m, r, __VA_ARGS__)\n"
            )
            f.write(
                f"#define REGISTER_REF_KERNEL_INPLACE(op, n, m, r, ...) REGISTER_REF_KERNEL_INPLACE_INTERNAL({uid}, op, n, m, r, __VA_ARGS__)\n"
            )
            f.write(
                f"#define REGISTER_REF_KERNEL_VIEW(op, n, m, inview, ...) REGISTER_REF_KERNEL_VIEW_INTERNAL({uid}, op, n, m, inview, __VA_ARGS__)\n"
            )
            f.write(
                f"#define REGISTER_KERNEL(name, n, m, r, ref, ...) REGISTER_KERNEL_INTERNAL({uid}, name, n, m, r, ref, __VA_ARGS__)\n"
            )
            f.write(
                f"#define REGISTER_KERNEL_INPLACE(name, n, m, r, ref, ...) REGISTER_KERNEL_INPLACE_INTERNAL({uid}, name, n, m, r, ref, __VA_ARGS__)\n"
            )
            f.write(
                f"#define REGISTER_KERNEL_VIEW(name, n, m, ref, inview, ...) REGISTER_KERNEL_VIEW_INTERNAL({uid}, name, n, m, ref, inview, __VA_ARGS__)\n"
            )
            f.write(f'#include "{inc_path}"\n\n')

        f.write(f"// --- Clean up macros ---\n")
        for macro in REGISTER_MACROS:
            f.write(f"#undef {macro}\n")

    console.print(
        f"[dim]Generated {len(kernel_entries)} Kernel Includes with UID injection.[/dim]"
    )


def generate_build_context():
    """Hashes compiler command arguments to detect build flag changes."""
    ctx_hpp = GENERATED_DIR / "build_context.gen.hpp"
    cmd_str = " ".join(get_compiler_cmd("")[:-2])
    ctx_hash = hashlib.sha256(cmd_str.encode("utf-8")).hexdigest()

    with open(ctx_hpp, "w") as f:
        f.write("#pragma once\n")
        f.write("#include <cstdint>\n\n")
        f.write("// Generated by build.py - Represents compile flags\n")
        f.write(f"// Mode: {'Debug' if DEBUG_MODE else 'Release'}\n")
        f.write(f"constexpr uint64_t BUILD_CONTEXT_ID = 0x{ctx_hash[:16]}ULL;\n")

    console.print(
        f"[dim]Build Context ID: 0x{ctx_hash[:16]} ({'DEBUG' if DEBUG_MODE else 'RELEASE'})[/dim]"
    )


def compile_binary(fname: str):
    console.print(f"\n[bold blue]Compiling {fname}...[/bold blue]")
    compiler_args_str = " ".join(get_compiler_cmd(fname))

    if os.name == "nt":
        arch = "amd64" if USE_CUDA else "arm64"
        full_command = f'"{VCVARS_PATH}" {arch} && {compiler_args_str}'
    else:
        full_command = compiler_args_str

    result = subprocess.run(full_command, capture_output=True, text=True, shell=True)

    if result.returncode != 0:
        console.print(
            Panel(
                f"[red]{result.stdout}[/red]\n\n[red]{result.stderr}[/red]",
                title="[bold red]COMPILER ERROR[/bold red]",
                border_style="red",
            )
        )
        sys.exit(1)
    else:
        console.print(
            Panel(
                (
                    f"[green]{result.stdout}[/green]"
                    if result.stdout.strip()
                    else "[green]No output[/green]"
                ),
                title="[bold green]BUILD SUCCESS[/bold green]",
                border_style="green",
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true", help="Enable CUDA build")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Build with debug symbols and no optimization",
    )
    args = parser.parse_args()

    global USE_CUDA, DEBUG_MODE
    USE_CUDA = args.cuda
    DEBUG_MODE = args.debug

    console.print(
        f"\n[bold cyan]Starting One-Click Build [{'DEBUG' if DEBUG_MODE else 'RELEASE'}]...[/bold cyan]\n"
    )
    core_seed = generate_core_seed()
    generate_kernel_uids(core_seed)
    generate_kernel_includes(core_seed)
    generate_build_context()

    compile_binary("main.cpp")
    compile_binary("bench.cpp")
    compile_binary("test.cpp")


if __name__ == "__main__":
    main()

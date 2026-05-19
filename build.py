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
NO_LINT = False

# List of macros that register a kernel with a unique UID
REGISTER_MACROS = [
    "REGISTER_REF_KERNEL",
    "REGISTER_REF_KERNEL_INPLACE",
    "REGISTER_REF_KERNEL_VIEW",
    "REGISTER_KERNEL",
    "REGISTER_KERNEL_INPLACE",
    "REGISTER_KERNEL_VIEW",
]


def validate_kernel_match_logic(rel_path, content):
    """
    Enforces the 'Clean Match Function' rule.
    Match functions should only contain logic that cannot be expressed
    in the registration macro.
    """
    # Find all registration macros
    reg_pattern = r"(REGISTER_[\w_]+)\s*\(\s*.*?\s*,\s*.*?\s*,\s*([\w_]+)\s*,.*?\)\s*;"
    registrations = re.findall(reg_pattern, content, re.DOTALL)

    for macro_type, match_func_name in registrations:
        # Extract function body
        func_body_pattern = rf"bool\s+{match_func_name}\s*\([^{{]+\{{(.*?)\}}"
        func_body_match = re.search(func_body_pattern, content, re.DOTALL)

        if func_body_match:
            body = func_body_match.group(1)

            # Map of regex patterns to (Error Name, Reason)
            redundancies = {
                r"inputs\.size\(\)": (
                    "Input Count Check",
                    "The engine already validates input count based on the macro arguments.",
                ),
                r"inputs\s*\[\d+\]\.backend": (
                    "Input Backend Check",
                    "Input backends are validated via the backend list in the registration macro.",
                ),
                r"output\.backend": (
                    "Output Backend Check",
                    "The output backend is validated by the registry before calling match().",
                ),
                r"isContiguous\s*\(\s*(inputs|inViews)\s*\[": (
                    "Input Contiguity Check",
                    "The Planner handles 'Contiguity Repair'. Use the boolean list in the macro instead.",
                ),
                r"inputs\s*\[\d+\]\.dtype != DType::": (
                    "Input DType Check",
                    "DTypes are already validated against the DType list in the registration macro.",
                ),
                r"inputs\s*\[0\]\.storageType\s*==\s*StorageType::PERSISTENT": (
                    "Persistent Storage Check",
                    "The engine's inplace safety logic now handles this automatically.",
                ),
            }

            for pattern, (name, reason) in redundancies.items():
                if re.search(pattern, body):
                    console.print(
                        Panel(
                            f"[bold red]REDUNDANT LOGIC DETECTED:[/bold red] in [cyan]{ROOT_DIR / rel_path}[/cyan]\n\n"
                            f"The match function [yellow]{match_func_name}[/yellow] contains a manual [bold]{name}[/bold].\n\n"
                            f"[white]Reason:[/white] {reason}\n\n"
                            f"[white]Fix:[/white] Remove the check from the C++ body. Use the registration macro "
                            f"parameters to define these constraints.",
                            title="Linter Violation",
                            border_style="red",
                        )
                    )
                    sys.exit(1)


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
            cmd.extend(["-Xcompiler", "-march=armv8.6-a+bf16+i8mm"])

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
                r'"C:\Program Files\LLVM\bin\clang++.exe"',
                "-target",
                "aarch64-windows",
                "-march=armv8.6-a+bf16+i8mm",
                "-std=c++17",
                f"-I{ROOT_DIR}",
            ]

            if DEBUG_MODE:
                cmd.extend(["-g", "-O0", "-DDEBUG"])
            else:
                cmd.extend(["-O3"])

            cmd.append(str(ROOT_DIR / fname))
            cmd.extend(["-o", out_name])
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

                    # LINT: Check for redundant match logic
                    if not NO_LINT:
                        validate_kernel_match_logic(rel_path, content)

                    reg_count = 0
                    for macro in REGISTER_MACROS:
                        # Use \b for word boundary to avoid matching substring macros
                        # Multiline mode allows ^ to match start of lines
                        matches = re.findall(rf"^\s*{macro}\b", content, re.MULTILINE)
                        reg_count += len(matches)

                    if reg_count > 1 and not NO_LINT:
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
                    # If it's the specific exit we triggered, just propagate
                    if isinstance(e, SystemExit):
                        raise e
                    console.print(
                        f"[yellow]Warning: Could not scan {rel_path} for macros: {e}[/yellow]"
                    )

                file_content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                combined = core_seed + file_content_hash
                full_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()

                uid_val_raw = int(full_hash[:16], 16) # TODO: use full hash instead of first 16 chars
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
    """Generates cpu_kernels.gen.hpp and cuda_kernels.gen.cu with UID injection logic."""
    cpu_includes_hpp = GENERATED_DIR / "cpu_kernels.gen.hpp"
    cuda_includes_cu = GENERATED_DIR / "cuda_kernels.gen.cu"
    kernels_all_hpp = GENERATED_DIR / "kernels_all.gen.hpp"

    kernel_entries_cpu = []
    kernel_entries_cuda = []

    for root, _, files in os.walk(KERNELS_DIR):
        for f in files:
            path = Path(root) / f
            rel_path = path.relative_to(ROOT_DIR)

            if f.endswith(".hpp"):
                file_content_hash = get_file_hash(path)
                combined = core_seed + file_content_hash
                full_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()
                uid_val = f"0x{full_hash[:16]}ULL"
                inc_path = str(rel_path).replace("\\", "/")
                kernel_entries_cpu.append((inc_path, uid_val))
            elif f.endswith(".cu"):
                file_content_hash = get_file_hash(path)
                combined = core_seed + file_content_hash
                full_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()
                uid_val = f"0x{full_hash[:16]}ULL"
                inc_path = str(rel_path).replace("\\", "/")
                kernel_entries_cuda.append((inc_path, uid_val))

    def write_includes(filepath, entries, is_cu=False):
        with open(filepath, "w") as f:
            if not is_cu:
                f.write("#pragma once\n")
            f.write('#include "core/kernels.hpp"\n\n')
            f.write("// Generated by build.py - Injects UIDs and includes kernels\n\n")

            for inc_path, uid in sorted(entries):
                f.write(f"// --- {inc_path} ---\n")
                for macro in REGISTER_MACROS:
                    f.write(f"#undef {macro}\n")

                f.write(
                    f"#define REGISTER_REF_KERNEL(op, n, match, run, ...) REGISTER_REF_KERNEL_INTERNAL({uid}, op, n, match, run, __VA_ARGS__)\n"
                )
                f.write(
                    f"#define REGISTER_REF_KERNEL_INPLACE(op, n, match, run, ...) REGISTER_REF_KERNEL_INPLACE_INTERNAL({uid}, op, n, match, run, __VA_ARGS__)\n"
                )
                f.write(
                    f"#define REGISTER_REF_KERNEL_VIEW(op, n, match, inferView, ...) REGISTER_REF_KERNEL_VIEW_INTERNAL({uid}, op, n, match, inferView, __VA_ARGS__)\n"
                )
                f.write(
                    f"#define REGISTER_KERNEL(name, n, match, run, ref, ...) REGISTER_KERNEL_INTERNAL({uid}, name, n, match, run, ref, __VA_ARGS__)\n"
                )
                f.write(
                    f"#define REGISTER_KERNEL_INPLACE(name, n, match, run, ref, ...) REGISTER_KERNEL_INPLACE_INTERNAL({uid}, name, n, match, run, ref, __VA_ARGS__)\n"
                )
                f.write(
                    f"#define REGISTER_KERNEL_VIEW(name, n, match, ref, inferView, ...) REGISTER_KERNEL_VIEW_INTERNAL({uid}, name, n, match, ref, inferView, __VA_ARGS__)\n"
                )
                f.write(f'#include "{inc_path}"\n\n')

            f.write(f"// --- Clean up macros ---\n")
            for macro in REGISTER_MACROS:
                f.write(f"#undef {macro}\n")

    write_includes(cpu_includes_hpp, kernel_entries_cpu, is_cu=False)
    write_includes(cuda_includes_cu, kernel_entries_cuda, is_cu=True)

    with open(kernels_all_hpp, "w") as f:
        f.write("#pragma once\n")
        f.write('#include "cpu_kernels.gen.hpp"\n')

    console.print(
        f"[dim]Generated {len(kernel_entries_cpu)} CPU and {len(kernel_entries_cuda)} CUDA Kernel Includes.[/dim]"
    )


def generate_build_context():
    """Hashes compiler command arguments to detect build flag changes."""
    ctx_hpp = GENERATED_DIR / "build_context.gen.hpp"
    mode = "DEBUG" if DEBUG_MODE else "RELEASE"
    backend = "CUDA" if USE_CUDA else "CPU"
    cmd_str = f"{mode}_{backend}_{platform.machine()}"
    ctx_hash = hashlib.sha256(cmd_str.encode("utf-8")).hexdigest()

    with open(ctx_hpp, "w") as f:
        f.write("#pragma once\n")
        f.write("#include <cstdint>\n\n")
        f.write("// Generated by build.py - Represents compile flags\n")
        f.write(f"// Mode: {mode}\n")
        f.write(f"constexpr uint64_t BUILD_CONTEXT_ID = 0x{ctx_hash[:16]}ULL;\n")

    console.print(f"[dim]Build Context ID: 0x{ctx_hash[:16]} ({mode})[/dim]")


def compile_project():
    out_ext = ".exe" if os.name == "nt" else ""
    is_arm64 = platform.machine().lower() in ("aarch64", "arm64")

    # Use clang++ instead of cl.exe/g++
    if os.name == "nt":
        cxx = r'"C:\Program Files\LLVM\bin\clang++.exe"'
        nvcc = "nvcc"
    else:
        cxx = "g++"
        nvcc = "nvcc"

    cxx_flags = [f"-I{ROOT_DIR}"]
    nvcc_flags = [f"-I{ROOT_DIR}", "-std=c++17", "-x", "cu"]

    if os.name == "nt":
        if not USE_CUDA:
            cxx_flags.extend(
                ["-target", "aarch64-windows", "-march=armv8.6-a+bf16+i8mm"]
            )
        cxx_flags.extend(["-std=c++17"])
        if DEBUG_MODE:
            cxx_flags.extend(["-g", "-O0", "-DDEBUG"])
            nvcc_flags.extend(["-g", "-G", "-O0", "-DDEBUG"])
        else:
            cxx_flags.extend(["-O3"])
            nvcc_flags.extend(["-O3"])
    else:
        cxx_flags.extend(["-std=c++17"])
        if is_arm64:
            cxx_flags.append("-march=armv8.6-a+bf16+i8mm")
        if DEBUG_MODE:
            cxx_flags.extend(["-g", "-O0", "-DDEBUG", "-fno-omit-frame-pointer"])
            nvcc_flags.extend(["-g", "-G", "-O0", "-DDEBUG"])
        else:
            cxx_flags.extend(["-O3"])
            nvcc_flags.extend(["-O3"])

    if USE_CUDA:
        # Detect CUDA Path
        cuda_path = os.environ.get("CUDA_PATH", "/usr/local/cuda")

        if os.name == "nt":
            cxx_flags.append("-DUSE_CUDA")
            cxx_flags.append(f'-I"{cuda_path}\\include"')  # Add CUDA include for clang
            nvcc_flags.append("-DUSE_CUDA")
        else:
            cxx_flags.append("-DUSE_CUDA")
            cxx_flags.append(f"-I{cuda_path}/include")  # Add CUDA include for clang
            nvcc_flags.append("-DUSE_CUDA")
            if is_arm64:
                nvcc_flags.extend(["-Xcompiler", "-march=armv8.6-a+bf16+i8mm"])

    mains = ["main.cpp", "bench.cpp", "test.cpp", "test_model.cpp"]

    obj_ext = ".obj" if os.name == "nt" else ".o"
    cuda_obj = str(GENERATED_DIR / f"cuda_kernels{obj_ext}")

    def run_cmd(cmd):
        cmd_str = " ".join(cmd)
        if os.name == "nt":
            arch = "amd64" if USE_CUDA else "arm64"
            # Setting environment headers logic continues to be applied nicely before invocation
            full_command = f'"{VCVARS_PATH}" {arch} && {cmd_str}'
        else:
            full_command = cmd_str

        print(f"Running {full_command}")
        result = subprocess.run(
            full_command, capture_output=True, text=True, shell=True
        )
        if result.returncode != 0:
            console.print(
                Panel(
                    f"[red]{result.stdout}[/red]\n\n[red]{result.stderr}[/red]",
                    title="[bold red]COMPILER ERROR[/bold red]",
                    border_style="red",
                )
            )
            sys.exit(1)
        return result

    if USE_CUDA:
        console.print(f"\n[bold blue]Compiling CUDA Kernels...[/bold blue]")
        cuda_src = str(GENERATED_DIR / "cuda_kernels.gen.cu")
        cmd = [nvcc] + nvcc_flags + ["-c", cuda_src, "-o", cuda_obj]

        result = run_cmd(cmd)
        if result.stdout.strip():
            console.print(
                Panel(
                    f"[green]{result.stdout}[/green]",
                    title="[bold green]BUILD SUCCESS[/bold green]",
                    border_style="green",
                )
            )
        else:
            console.print(
                Panel(
                    f"[green]No output[/green]",
                    title="[bold green]BUILD SUCCESS[/bold green]",
                    border_style="green",
                )
            )

    for main_file in mains:
        console.print(f"\n[bold blue]Compiling {main_file}...[/bold blue]")
        main_src = str(ROOT_DIR / main_file)
        out_name = f"tensor_graphs_cpp/{main_file.split('.')[0]}{out_ext}"

        if USE_CUDA:
            main_obj = str(GENERATED_DIR / f"{main_file.split('.')[0]}{obj_ext}")

            cmd = [cxx] + cxx_flags + ["-c", main_src, "-o", main_obj]
            run_cmd(cmd)

            cmd = [nvcc] + [main_obj, cuda_obj, "-o", out_name]
            if os.name == "nt" and DEBUG_MODE:
                cmd.append("-g")
            result = run_cmd(cmd)
        else:
            cmd = [cxx] + cxx_flags + [main_src, "-o", out_name]
            result = run_cmd(cmd)

        if result.stdout.strip():
            console.print(
                Panel(
                    f"[green]{result.stdout}[/green]",
                    title="[bold green]BUILD SUCCESS[/bold green]",
                    border_style="green",
                )
            )
        else:
            console.print(
                Panel(
                    f"[green]No output[/green]",
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
    parser.add_argument(
        "--no-lint", action="store_true", help="Skip kernel validation checks"
    )
    args = parser.parse_args()

    global USE_CUDA, DEBUG_MODE, NO_LINT
    USE_CUDA = args.cuda
    DEBUG_MODE = args.debug
    NO_LINT = args.no_lint

    console.print(
        f"\n[bold cyan]Starting One-Click Build [{'DEBUG' if DEBUG_MODE else 'RELEASE'}]...[/bold cyan]\n"
    )
    core_seed = generate_core_seed()
    generate_kernel_uids(core_seed)
    generate_kernel_includes(core_seed)
    generate_build_context()

    compile_project()


if __name__ == "__main__":
    main()

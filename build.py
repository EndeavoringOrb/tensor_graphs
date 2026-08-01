# build.py
import argparse
import os
import sys
import hashlib
import json
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
PROFILE_MODE = False
DISABLE_OPENCL = False

# List of macros that register a kernel with a unique UID
REGISTER_MACROS = [
    "REGISTER_REF_KERNEL",
    "REGISTER_REF_KERNEL_VIEW",
    "REGISTER_KERNEL",
    "REGISTER_KERNEL_INPLACE",
    "REGISTER_KERNEL_VIEW",
]


def validate_kernel_match_logic(rel_path):
    with open(rel_path, "r", encoding="utf-8") as f:
        content = f.read()

    reg_pattern = r"(REGISTER_[\w_]+)\s*\(\s*.*?\s*,\s*.*?\s*,\s*([\w_]+)\s*,.*?\)\s*;"
    registrations = re.findall(reg_pattern, content, re.DOTALL)

    for macro_type, match_func_name in registrations:
        func_body_pattern = rf"bool\s+{match_func_name}\s*\([^{{]+\{{(.*?)\}}"
        func_body_match = re.search(func_body_pattern, content, re.DOTALL)

        if func_body_match:
            body = func_body_match.group(1)

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
            "-std=c++20",
            f"-I{ROOT_DIR}",
            "-DUSE_CUDA",
            "-x",
            "cu",
        ]

        if is_arm64:
            cmd.extend(["-Xcompiler", "-march=armv8.6-a+bf16+i8mm"])

        if DEBUG_MODE:
            cmd.extend(["-g", "-G", "-O0", "-DDEBUG"])
        else:
            cmd.extend(["-O3"])

        if DISABLE_OPENCL:
            cmd.append("-DTG_DISABLE_OPENCL")

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
                "-std=c++20",
                f"-I{ROOT_DIR}",
            ]

            if DEBUG_MODE:
                cmd.extend(["-g", "-gcodeview", "-gfull", "-O0", "-DDEBUG"])
            else:
                cmd.extend(["-O3"])

            if DISABLE_OPENCL:
                cmd.append("-DTG_DISABLE_OPENCL")

            cmd.append(str(ROOT_DIR / fname))
            cmd.extend(["-o", out_name])
            return cmd
        else:
            cmd = [
                "g++",
                "-std=c++20",
                f"-I{ROOT_DIR}",
            ]

            if DEBUG_MODE:
                cmd.extend(["-g", "-O0", "-DDEBUG", "-fno-omit-frame-pointer"])
            else:
                cmd.extend(["-O3"])

            if DISABLE_OPENCL:
                cmd.append("-DTG_DISABLE_OPENCL")

            cmd.append(str(ROOT_DIR / fname))
            cmd.extend(["-o", out_name])
            return cmd


def get_file_hash(filepath):
    h = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            h.update(f.read())
        return h.hexdigest()
    except FileNotFoundError:
        print(f"Warning: Dependency file not found: {filepath}")
        return "0" * 64


def generate_core_seed():
    content_hashes = [get_file_hash(p) for p in CORE_DEPENDENCIES]
    combined = "".join(content_hashes)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def generate_opencl_strings():
    cl_files = []
    for root, _, files in os.walk(KERNELS_DIR):
        for f in files:
            if f.endswith(".cl"):
                cl_files.append(Path(root) / f)

    out_file = GENERATED_DIR / "opencl_kernels.gen.hpp"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("#pragma once\n")
        f.write("#include <unordered_map>\n")
        f.write("#include <string>\n\n")
        f.write(
            "inline const std::unordered_map<std::string, const char*> OPENCL_SOURCE_MAP = {\n"
        )
        for cl_path in cl_files:
            rel_path = cl_path.relative_to(ROOT_DIR).as_posix()
            with open(cl_path, "r", encoding="utf-8") as cl_f:
                content = cl_f.read()
            f.write(f'    {{"{rel_path}", R"TG_OPENCL(\n{content}\n)TG_OPENCL"}},\n')
        f.write("};\n")
    console.print(f"[dim]Generated {len(cl_files)} OpenCL kernel strings.[/dim]")


def generate_kernel_includes(core_seed):
    cpu_includes_hpp = GENERATED_DIR / "cpu_kernels.gen.hpp"
    cuda_includes_cu = GENERATED_DIR / "cuda_kernels.gen.cu"
    kernels_all_hpp = GENERATED_DIR / "kernels_all.gen.hpp"
    kernel_uids_json = GENERATED_DIR / "kernel_uids.json"
    kernel_uids_hpp = GENERATED_DIR / "kernel_uids.gen.hpp"

    kernel_entries_cpu = []
    kernel_entries_cuda = []
    uid_info_map = {}
    hpp_lines = ["#pragma once\n", "#include <cstdint>\n\n"]

    for root, _, files in os.walk(KERNELS_DIR):
        for f in files:
            path = Path(root) / f
            validate_kernel_match_logic(path)
            rel_path = path.relative_to(ROOT_DIR)

            if f.endswith(".hpp") or f.endswith(".cu"):
                file_content_hash = get_file_hash(path)
                combined = core_seed + file_content_hash
                full_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()
                uid_hex = f"0x{full_hash[:16]}"
                uid_val = f"{uid_hex}ULL"
                inc_path = str(rel_path).replace("\\", "/")

                if f.endswith(".hpp"):
                    kernel_entries_cpu.append((inc_path, uid_val))
                else:
                    kernel_entries_cuda.append((inc_path, uid_val))

                op_name = path.stem
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as kf:
                        kcontent = kf.read()
                    m_name = re.search(
                        r'REGISTER_KERNEL(?:_INPLACE|_VIEW)?\s*\(\s*"([^"]+)"', kcontent
                    )
                    m_ref = re.search(
                        r"REGISTER_REF_KERNEL(?:_VIEW)?\s*\(\s*OpType::(\w+)", kcontent
                    )
                    if m_name:
                        op_name = m_name.group(1)
                    elif m_ref:
                        op_name = f"REF_{m_ref.group(1)}"
                except Exception:
                    pass

                info = {
                    "name": op_name,
                    "path": inc_path,
                    "hex_uid": uid_hex,
                }

                uid_int = int(full_hash[:16], 16)
                uid_info_map[str(uid_int)] = info
                uid_info_map[uid_hex.lower()] = info

                const_name = (
                    inc_path.replace("/", "_")
                    .replace("\\", "_")
                    .replace(".", "_")
                    .upper()
                )
                hpp_lines.append(f"constexpr uint64_t {const_name} = {uid_val};\n")

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

                uid_str = "KernelId{" + uid + "}"
                f.write(
                    f"#define REGISTER_REF_KERNEL(op, n_min, n_max, match, run, ...) REGISTER_REF_KERNEL_INTERNAL({uid_str}, op, n_min, n_max, match, run, __VA_ARGS__)\n"
                )
                f.write(
                    f"#define REGISTER_REF_KERNEL_VIEW(op, n_min, n_max, match, inferView, ...) REGISTER_REF_KERNEL_VIEW_INTERNAL({uid_str}, op, n_min, n_max, match, inferView, __VA_ARGS__)\n"
                )
                f.write(
                    f"#define REGISTER_KERNEL(name, n_min, n_max, match, run, ref, ...) REGISTER_KERNEL_INTERNAL({uid_str}, name, n_min, n_max, match, run, ref, __VA_ARGS__)\n"
                )
                f.write(
                    f"#define REGISTER_KERNEL_INPLACE(name, n_min, n_max, match, run, ref, ...) REGISTER_KERNEL_INPLACE_INTERNAL({uid_str}, name, n_min, n_max, match, run, ref, __VA_ARGS__)\n"
                )
                f.write(
                    f"#define REGISTER_KERNEL_VIEW(name, n_min, n_max, match, ref, inferView, ...) REGISTER_KERNEL_VIEW_INTERNAL({uid_str}, name, n_min, n_max, match, ref, inferView, __VA_ARGS__)\n"
                )
                f.write(f'#include "{inc_path}"\n\n')

            f.write("// --- Clean up macros ---\n")
            for macro in REGISTER_MACROS:
                f.write(f"#undef {macro}\n")

    write_includes(cpu_includes_hpp, kernel_entries_cpu, is_cu=False)
    write_includes(cuda_includes_cu, kernel_entries_cuda, is_cu=True)

    with open(kernels_all_hpp, "w") as f:
        f.write("#pragma once\n")
        f.write('#include "cpu_kernels.gen.hpp"\n')

    # Save kernel UIDs JSON and header files
    with open(kernel_uids_json, "w", encoding="utf-8") as f:
        json.dump(uid_info_map, f, indent=2)

    with open(kernel_uids_hpp, "w", encoding="utf-8") as f:
        f.writelines(hpp_lines)

    console.print(
        f"[dim]Generated {len(kernel_entries_cpu)} CPU and {len(kernel_entries_cuda)} CUDA Kernel Includes.[/dim]"
    )
    console.print("[dim]Saved UID metadata mapping to kernel_uids.json.[/dim]")


def generate_build_context():
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


def compile_project(targets=None):
    out_ext = ".exe" if os.name == "nt" else ""
    is_arm64 = platform.machine().lower() in ("aarch64", "arm64")

    if os.name == "nt":
        cxx = r'"C:\Program Files\LLVM\bin\clang++.exe"'
        nvcc = "nvcc"
    else:
        cxx = "g++"
        nvcc = "nvcc"

    cxx_flags = [f"-I{ROOT_DIR}"]
    nvcc_flags = [f"-I{ROOT_DIR}", "-std=c++20", "-x", "cu"]

    if os.name == "nt":
        if not USE_CUDA:
            cxx_flags.extend(
                ["-target", "aarch64-windows", "-march=armv8.6-a+bf16+i8mm"]
            )
        cxx_flags.extend(
            [
                "-std=c++20",
                "-I./OpenCL-SDK/install/include",
                "-L./OpenCL-SDK/install/lib",
                "-lOpenCL",
                "-DCL_TARGET_OPENCL_VERSION=310",
                "-v",
            ]
        )
        if DEBUG_MODE:
            cxx_flags.extend(["-g", "-O0", "-DDEBUG"])
            nvcc_flags.extend(["-g", "-G", "-O0", "-DDEBUG"])
        else:
            cxx_flags.extend(["-O3"])
            nvcc_flags.extend(["-O3"])
            if PROFILE_MODE:
                cxx_flags.extend(["-g", "-gcodeview", "-Wl,-debug"])
    else:
        cxx_flags.extend(["-std=c++20", "-lOpenCL"])
        if is_arm64:
            cxx_flags.append("-march=armv8.6-a+bf16+i8mm")
        if DEBUG_MODE:
            cxx_flags.extend(["-g", "-O0", "-DDEBUG", "-fno-omit-frame-pointer"])
            nvcc_flags.extend(["-g", "-G", "-O0", "-DDEBUG"])
        else:
            cxx_flags.extend(["-O3"])
            nvcc_flags.extend(["-O3"])

    if USE_CUDA:
        cuda_path = os.environ.get("CUDA_PATH", "/usr/local/cuda")

        if os.name == "nt":
            cxx_flags.append("-DUSE_CUDA")
            cxx_flags.append(f'-I"{cuda_path}\\include"')
            nvcc_flags.append("-DUSE_CUDA")
        else:
            cxx_flags.append("-DUSE_CUDA")
            cxx_flags.append(f"-I{cuda_path}/include")
            nvcc_flags.append("-DUSE_CUDA")
            if is_arm64:
                nvcc_flags.extend(["-Xcompiler", "-march=armv8.6-a+bf16+i8mm"])

    if DISABLE_OPENCL:
        cxx_flags.append("-DTG_DISABLE_OPENCL")
        if USE_CUDA:
            nvcc_flags.append("-DTG_DISABLE_OPENCL")

    if targets is None:
        mains = [
            "main.cpp",
            "bench.cpp",
            "test.cpp",
            "test_model.cpp",
            "write_ref_tensors.cpp",
            "embed.cpp",
        ]
    else:
        mains = []
        for t in targets:
            if not t.endswith(".cpp"):
                t = t + ".cpp"
            mains.append(t)

    obj_ext = ".obj" if os.name == "nt" else ".o"
    cuda_obj = str(GENERATED_DIR / f"cuda_kernels{obj_ext}")

    def run_cmd(cmd):
        cmd_str = " ".join(cmd)
        if os.name == "nt":
            arch = "amd64" if USE_CUDA else "arm64"
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
                    f"{result.stdout}\n\n[red]{result.stderr}[/red]",
                    title="[bold red]COMPILER ERROR[/bold red]",
                    border_style="red",
                )
            )
            sys.exit(1)
        else:
            if result.stderr.strip():
                console.print(
                    Panel(
                        f"{result.stdout}[yellow]{result.stderr}[/yellow]",
                        title="[bold yellow]BUILD WARNINGS[/bold yellow]",
                        border_style="yellow",
                    )
                )
        return result

    if USE_CUDA:
        console.print("\n[bold blue]Compiling CUDA Kernels...[/bold blue]")
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
                    "[green]No output[/green]",
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
            if os.name == "nt":
                cmd.extend(["-L./OpenCL-SDK/install/lib", "-lOpenCL"])
            else:
                cmd.append("-lOpenCL")
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
                    "[green]No output[/green]",
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
        "--profile",
        action="store_true",
        help="Build with profiling symbols (-g, -gcodeview, -Wl,-debug) while keeping optimizations",
    )
    parser.add_argument(
        "--no-lint", action="store_true", help="Skip kernel validation checks"
    )
    parser.add_argument(
        "--disable-opencl", action="store_true", help="Disable OpenCL backend"
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        help="Specify which target C++ files to build (e.g. main, bench, test, test_model)",
    )
    args = parser.parse_args()

    global USE_CUDA, DEBUG_MODE, NO_LINT, PROFILE_MODE, DISABLE_OPENCL
    USE_CUDA = args.cuda
    DEBUG_MODE = args.debug
    NO_LINT = args.no_lint
    PROFILE_MODE = args.profile
    DISABLE_OPENCL = args.disable_opencl

    console.print(
        f"\n[bold cyan]Starting One-Click Build [{'DEBUG' if DEBUG_MODE else 'RELEASE'}]...[/bold cyan]\n"
    )
    core_seed = generate_core_seed()
    generate_opencl_strings()
    generate_kernel_includes(core_seed)
    generate_build_context()

    compile_project(targets=args.targets)


if __name__ == "__main__":
    main()

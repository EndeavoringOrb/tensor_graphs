# build.py
import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from rich.console import Console
from rich.panel import Panel
from tqdm import tqdm

console = Console()

# --- Path Constants ---
ROOT_DIR = Path("tensor_graphs_cpp")
GENERATED_DIR = ROOT_DIR / "generated"
KERNELS_DIR = ROOT_DIR / "kernels"

CORE_DEPENDENCIES = [
    ROOT_DIR / "core" / "types.hpp",
    ROOT_DIR / "core" / "kernels.hpp",
    ROOT_DIR / "core" / "graph.hpp",
]

ALL_TARGETS = [
    "bench.cpp",
    "chat.cpp",
    "embed.cpp",
    "main.cpp",
    "test.cpp",
    "write_ref_tensors.cpp",
]

REGISTER_MACROS = [
    "REGISTER_REF_KERNEL",
    "REGISTER_REF_KERNEL_VIEW",
    "REGISTER_KERNEL",
    "REGISTER_KERNEL_INPLACE",
    "REGISTER_KERNEL_VIEW",
]

LOG_LEVEL_MAP = {
    "DEBUG": 0,
    "INFO": 1,
    "WARNING": 2,
    "ERROR": 3,
    "CRITICAL": 4,
    "OFF": 5,
}


def strip_cpp_comments_and_strings(text: str) -> str:
    """Replaces C++ comments and string/char literals with spaces, preserving newlines."""

    def replacer(match):
        s = match.group(0)
        return "".join("\n" if c == "\n" else " " for c in s)

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE,
    )
    return pattern.sub(replacer, text)


def extract_macro_call_args(content: str, paren_start: int) -> List[str]:
    """Extracts top-level comma-separated arguments from a C++ macro call starting at opening parenthesis paren_start."""
    idx = paren_start + 1
    depth_paren = 1
    depth_brace = 0
    depth_bracket = 0
    in_string = False
    string_char = ""

    current_arg = []
    args = []

    while idx < len(content) and depth_paren > 0:
        ch = content[idx]

        if in_string:
            current_arg.append(ch)
            if ch == string_char and content[idx - 1] != "\\":
                in_string = False
        elif ch in ('"', "'"):
            in_string = True
            string_char = ch
            current_arg.append(ch)
        elif ch == "(":
            depth_paren += 1
            current_arg.append(ch)
        elif ch == ")":
            depth_paren -= 1
            if depth_paren > 0:
                current_arg.append(ch)
        elif ch == "{":
            depth_brace += 1
            current_arg.append(ch)
        elif ch == "}":
            depth_brace -= 1
            current_arg.append(ch)
        elif ch == "[":
            depth_bracket += 1
            current_arg.append(ch)
        elif ch == "]":
            depth_bracket -= 1
            current_arg.append(ch)
        elif ch == "," and depth_paren == 1 and depth_brace == 0 and depth_bracket == 0:
            args.append("".join(current_arg).strip())
            current_arg = []
        else:
            current_arg.append(ch)
        idx += 1

    if current_arg:
        args.append("".join(current_arg).strip())

    return args


# =============================================================================
# 1. Configuration & Platform Detection
# =============================================================================


@dataclass
class BuildConfig:
    """Holds all compile and build pipeline configuration parameters."""

    use_cuda: bool = False
    debug: bool = False
    profile: bool = False
    no_lint: bool = False
    disable_opencl: bool = False
    log_level_str: str = "INFO"
    log_level_val: int = 1
    targets: List[str] = field(default_factory=lambda: list(ALL_TARGETS))

    def __post_init__(self):
        level_upper = self.log_level_str.upper()
        self.log_level_val = LOG_LEVEL_MAP.get(level_upper, 1)
        self.log_level_str = level_upper

        if self.targets:
            formatted_targets = []
            for t in self.targets:
                if not t.endswith(".cpp"):
                    t = f"{t}.cpp"
                formatted_targets.append(t)
            self.targets = formatted_targets
        else:
            self.targets = list(ALL_TARGETS)


@dataclass
class PlatformInfo:
    """Detects host operating system, architecture, and toolchain paths."""

    os_name: str  # "nt" or "posix"
    machine: str  # e.g. "x86_64", "arm64", "aarch64"
    is_arm64: bool
    is_windows: bool
    vcvars_path: str
    cuda_path: str
    opencl_sdk_path: str
    clang_cpp_path: str

    @classmethod
    def detect(cls) -> "PlatformInfo":
        os_name = os.name
        machine = platform.machine().lower()
        is_windows = os_name == "nt"
        is_arm64 = machine in ("aarch64", "arm64")

        # Environment path resolution with sensible fallbacks
        vcvars_path = os.environ.get(
            "VCVARS_PATH",
            r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat",
        )
        cuda_path = os.environ.get(
            "CUDA_PATH",
            (
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0"
                if is_windows
                else "/usr/local/cuda"
            ),
        )
        opencl_sdk_path = os.environ.get("OPENCL_SDK_ROOT", "./OpenCL-SDK/install")
        clang_cpp_path = os.environ.get(
            "CLANG_CXX",
            r"C:\Program Files\LLVM\bin\clang++.exe" if is_windows else "clang++",
        )

        return cls(
            os_name=os_name,
            machine=machine,
            is_arm64=is_arm64,
            is_windows=is_windows,
            vcvars_path=vcvars_path,
            cuda_path=cuda_path,
            opencl_sdk_path=opencl_sdk_path,
            clang_cpp_path=clang_cpp_path,
        )


# =============================================================================
# 2. Kernel & Rewrite Linter
# =============================================================================


class KernelLinter:
    """Validates kernel match logic and rewrite rules for correctness and safety."""

    REDUNDANCY_PATTERNS = {
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

    EGRAPH_MUTATION_PATTERNS = [
        r"\baddOpToEGraph\b",
        r"\baddFusedNode\b",
        r"\bcopyTo\b",
        r"\bcreateCacheInputNode\b",
        r"\binjectPartialPath\b",
        r"\b\.addEClass\b",
        r"\b\.addENode\b",
        r"\b\.merge\b",
        r"\b\.rebuild\b",
        r"\b\.getOrAddConstant\b",
        r"\b\.addIntConst\b",
    ]

    def _validate_kernel_file(self, file_path: Path):
        if not file_path.is_file() or file_path.suffix not in [".hpp", ".cu"]:
            return
        content = file_path.read_text(encoding="utf-8")
        rel_path = file_path.relative_to(ROOT_DIR)
        clean_content = strip_cpp_comments_and_strings(content)

        macro_pattern = re.compile(r"\b(REGISTER_[\w_]+)\s*\(")
        n_matches = 0
        for match in macro_pattern.finditer(content):
            macro_name = match.group(1)
            paren_start = match.end() - 1
            args = extract_macro_call_args(content, paren_start)

            if len(args) < 4:
                continue

            match_func_name = args[3].strip()
            if not re.match(r"^[a-zA-Z_]\w*$", match_func_name):
                continue

            func_def_pattern = re.compile(
                r"\bbool\s+"
                + re.escape(match_func_name)
                + r"\s*\(([\s\S]*?)\)\s*(?:const\s*)?(?:noexcept\s*)?\{",
                re.MULTILINE,
            )
            func_def_match = func_def_pattern.search(clean_content)

            n_matches += 1
            if func_def_match:
                start_brace = func_def_match.end() - 1
                brace_count = 1
                end_idx = start_brace + 1
                while end_idx < len(clean_content) and brace_count > 0:
                    if clean_content[end_idx] == "{":
                        brace_count += 1
                    elif clean_content[end_idx] == "}":
                        brace_count -= 1
                    end_idx += 1

                body = content[start_brace + 1 : end_idx - 1]
                body_start = start_brace + 1

                for pattern, (name, reason) in self.REDUNDANCY_PATTERNS.items():
                    pat_match = re.search(pattern, body)
                    if pat_match:
                        match_pos = body_start + pat_match.start()
                        line_num = content[:match_pos].count("\n") + 1
                        console.print(
                            Panel(
                                f"[bold red]REDUNDANT LOGIC DETECTED:[/bold red] in [cyan]{ROOT_DIR / rel_path}:{line_num}[/cyan]\n\n"
                                f"The match function [yellow]{match_func_name}[/yellow] contains a manual [bold]{name}[/bold] on line {line_num}.\n\n"
                                f"[white]Reason:[/white] {reason}\n\n"
                                f"[white]Fix:[/white] Remove the check from the C++ body. Use registration macro parameters.",
                                title="Linter Violation",
                                border_style="red",
                            )
                        )
                        sys.exit(1)
        if (not file_path.name.endswith("utils.hpp")) and (n_matches == 0):
            console.print(
                Panel(
                    f"[bold red]KERNEL REGISTRATION DETECTION ERROR:[/bold red] no kernel registration found in [cyan]{ROOT_DIR / rel_path}[/cyan]",
                    title="Linter Error",
                    border_style="red",
                )
            )

    def _validate_rewrite_file(self, file_path: Path):
        if not file_path.is_file() or file_path.suffix not in [
            ".hpp",
            ".cu",
            ".cpp",
        ]:
            return
        content = file_path.read_text(encoding="utf-8")
        clean_content = strip_cpp_comments_and_strings(content)

        func_pattern = re.compile(
            r"(?:inline\s+|virtual\s+|static\s+)*(?:void|bool|auto|EClassId|ExtractionResult|CompiledGraph|uint32_t|int|size_t|uint64_t)\s+([a-zA-Z_]\w*)\s*\(([\s\S]*?)\)\s*(?:const\s*)?(?:override\s*)?(?:noexcept\s*)?\{",
            re.MULTILINE,
        )

        for match in func_pattern.finditer(clean_content):
            func_name = match.group(1)
            param_str = match.group(2)

            if (
                ";" in param_str
                or "= 0" in match.group(0)
                or "= default" in match.group(0)
            ):
                continue

            start_brace = match.end() - 1

            brace_count = 1
            end_idx = start_brace + 1
            while end_idx < len(clean_content) and brace_count > 0:
                if clean_content[end_idx] == "{":
                    brace_count += 1
                elif clean_content[end_idx] == "}":
                    brace_count -= 1
                end_idx += 1

            func_body = clean_content[start_brace + 1 : end_idx - 1]

            mutation_matches = []
            for pat in self.EGRAPH_MUTATION_PATTERNS:
                for m in re.finditer(pat, func_body):
                    mutation_matches.append(m.start())

            if not mutation_matches:
                continue

            ref_param_match = re.search(r"\b(const\s+)?(EClass|ENode)\s*&", param_str)
            if ref_param_match:
                param_pos = match.start(2) + ref_param_match.start()
                line_num = content[:param_pos].count("\n") + 1
                rel_path = file_path.relative_to(ROOT_DIR)
                console.print(
                    Panel(
                        f"[bold red]DANGLING REFERENCE HAZARD DETECTED:[/bold red] in [cyan]{ROOT_DIR / rel_path}:{line_num}[/cyan]\n\n"
                        f"In function [yellow]{func_name}[/yellow] (line {line_num}):\n"
                        f"Parameter [bold]{ref_param_match.group(0)}[/bold] is passed by reference in a function that mutates [bold]egraph[/bold].\n\n"
                        f"[white]Reason:[/white] Modifying the egraph (via addOpToEGraph, addEClass, addENode, merge, etc.) "
                        f"may cause the underlying std::vector in EGraph to reallocate, invalidating references to EClass or ENode.\n\n"
                        f"[white]Fix:[/white] Pass EClass and ENode by value (e.g., 'const EClass' instead of 'const EClass &').",
                        title="Linter Violation",
                        border_style="red",
                    )
                )
                sys.exit(1)

            depths = [0] * len(func_body)
            curr_d = 1
            for i, ch in enumerate(func_body):
                if ch == "{":
                    curr_d += 1
                elif ch == "}":
                    curr_d -= 1
                depths[i] = curr_d

            ref_patterns = [
                r"\b(const\s+)?(EClass|ENode)\s*&\s*([a-zA-Z_]\w*)",
                r"\b(const\s+)?auto\s*&\s*([a-zA-Z_]\w*)\s*=\s*.*?\b(getEClass|getENode|classes|enodes)\b",
            ]

            for ref_pat in ref_patterns:
                for local_match in re.finditer(ref_pat, func_body):
                    ref_start = local_match.start()
                    ref_end = local_match.end()
                    decl_depth = depths[ref_start]

                    end_of_block = len(func_body)
                    for i in range(ref_end, len(func_body)):
                        if depths[i] < decl_depth:
                            end_of_block = i
                            break

                    has_hazard = any(
                        ref_end <= mut_pos < end_of_block
                        for mut_pos in mutation_matches
                    )

                    if has_hazard:
                        match_pos = (start_brace + 1) + ref_start
                        line_num = content[:match_pos].count("\n") + 1
                        rel_path = file_path.relative_to(ROOT_DIR)
                        console.print(
                            Panel(
                                f"[bold red]DANGLING REFERENCE HAZARD DETECTED:[/bold red] in [cyan]{ROOT_DIR / rel_path}:{line_num}[/cyan]\n\n"
                                f"In function [yellow]{func_name}[/yellow] (line {line_num}):\n"
                                f"Local variable reference [bold]{local_match.group(0)}[/bold] is active while [bold]egraph[/bold] is mutated.\n\n"
                                f"[white]Reason:[/white] Modifying the egraph (via addOpToEGraph, addEClass, addENode, merge, etc.) "
                                f"may cause the underlying std::vector in EGraph to reallocate, invalidating references to EClass or ENode.\n\n"
                                f"[white]Fix:[/white] Store EClass and ENode by value (e.g., 'const EClass cls = ...' instead of 'const EClass &cls = ...').",
                                title="Linter Violation",
                                border_style="red",
                            )
                        )
                        sys.exit(1)

    def lint(self, config: BuildConfig):
        if config.no_lint:
            return

        validators = [
            (ROOT_DIR / "kernels", self._validate_kernel_file),
            (ROOT_DIR / "core", self._validate_rewrite_file),
        ]

        for val_idx, (dir_path, func) in enumerate(validators):
            if not dir_path.exists():
                continue
            files = [p for p in dir_path.rglob("*") if p.is_file()]
            with tqdm(
                files,
                desc=f"linting [{val_idx + 1}/{len(validators)}]",
            ) as pbar:
                for path in pbar:
                    pbar.set_postfix_str(path.as_posix())
                    func(path)


# =============================================================================
# 3. Code Generation Pipeline
# =============================================================================


class CodeGenerator:
    """Handles static code and header generation prior to compilation."""

    @staticmethod
    def get_file_hash(filepath: Path) -> str:
        h = hashlib.sha256()
        try:
            h.update(filepath.read_bytes())
            return h.hexdigest()
        except FileNotFoundError:
            console.print(
                f"[yellow]Warning: Dependency file not found: {filepath}[/yellow]"
            )
            return "0" * 64

    def generate_core_seed(self) -> str:
        hashes = [self.get_file_hash(p) for p in CORE_DEPENDENCIES]
        return hashlib.sha256("".join(hashes).encode("utf-8")).hexdigest()

    def generate_opencl_strings(self) -> None:
        GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        cl_files = sorted(list(KERNELS_DIR.rglob("*.cl")))
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
                content = cl_path.read_text(encoding="utf-8")
                f.write(
                    f'    {{"{rel_path}", R"TG_OPENCL(\n{content}\n)TG_OPENCL"}},\n'
                )
            f.write("};\n")

        console.print(f"[dim]Generated {len(cl_files)} OpenCL kernel strings.[/dim]")

    def generate_kernel_includes(self, core_seed: str) -> None:
        GENERATED_DIR.mkdir(parents=True, exist_ok=True)

        cpu_includes_hpp = GENERATED_DIR / "cpu_kernels.gen.hpp"
        cuda_includes_cu = GENERATED_DIR / "cuda_kernels.gen.cu"
        kernels_all_hpp = GENERATED_DIR / "kernels_all.gen.hpp"
        kernel_uids_json = GENERATED_DIR / "kernel_uids.json"
        kernel_uids_hpp = GENERATED_DIR / "kernel_uids.gen.hpp"

        kernel_entries_cpu: List[Tuple[str, str]] = []
        kernel_entries_cuda: List[Tuple[str, str]] = []
        uid_info_map: Dict[str, Dict[str, str]] = {}
        hpp_lines = ["#pragma once\n", "#include <cstdint>\n\n"]

        kernel_files = sorted(
            [
                p
                for p in KERNELS_DIR.rglob("*")
                if p.is_file() and p.suffix in (".hpp", ".cu")
            ]
        )

        for path in kernel_files:
            rel_path = path.relative_to(ROOT_DIR)
            file_hash = self.get_file_hash(path)
            combined_hash = hashlib.sha256(
                (core_seed + file_hash).encode("utf-8")
            ).hexdigest()
            uid_hex = f"0x{combined_hash[:16]}"
            uid_val = f"{uid_hex}ULL"
            inc_path = rel_path.as_posix()

            if path.suffix == ".hpp":
                kernel_entries_cpu.append((inc_path, uid_val))
            else:
                kernel_entries_cuda.append((inc_path, uid_val))

            op_name = path.stem
            try:
                kcontent = path.read_text(encoding="utf-8", errors="ignore")
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

            info = {"name": op_name, "path": inc_path, "hex_uid": uid_hex}
            uid_int = int(combined_hash[:16], 16)

            uid_info_map[str(uid_int)] = info
            uid_info_map[uid_hex.lower()] = info

            const_name = (
                inc_path.replace("/", "_").replace("\\", "_").replace(".", "_").upper()
            )
            hpp_lines.append(f"constexpr uint64_t {const_name} = {uid_val};\n")

        self._write_includes_file(cpu_includes_hpp, kernel_entries_cpu, is_cu=False)
        self._write_includes_file(cuda_includes_cu, kernel_entries_cuda, is_cu=True)

        with open(kernels_all_hpp, "w", encoding="utf-8") as f:
            f.write("#pragma once\n")
            f.write('#include "cpu_kernels.gen.hpp"\n')

        with open(kernel_uids_json, "w", encoding="utf-8") as f:
            json.dump(uid_info_map, f, indent=2)

        with open(kernel_uids_hpp, "w", encoding="utf-8") as f:
            f.writelines(hpp_lines)

        console.print(
            f"[dim]Generated {len(kernel_entries_cpu)} CPU and {len(kernel_entries_cuda)} CUDA Kernel Includes.[/dim]"
        )
        console.print("[dim]Saved UID metadata mapping to kernel_uids.json.[/dim]")

    def generate_build_context(self) -> None:
        GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        ctx_hpp = GENERATED_DIR / "build_context.gen.hpp"
        cmd_str = f"{platform.machine()}"
        ctx_hash = hashlib.sha256(cmd_str.encode("utf-8")).hexdigest()

        with open(ctx_hpp, "w", encoding="utf-8") as f:
            f.write("#pragma once\n")
            f.write("#include <cstdint>\n\n")
            f.write(
                "// Generated by build.py - Represents compile flags relevant to kernel benchmarks\n"
            )
            f.write(f"constexpr uint64_t BUILD_CONTEXT_ID = 0x{ctx_hash[:16]}ULL;\n")

        console.print(f"[dim]Build Context ID: 0x{ctx_hash[:16]}[/dim]")

    def _write_includes_file(
        self, filepath: Path, entries: List[Tuple[str, str]], is_cu: bool
    ) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            if not is_cu:
                f.write("#pragma once\n")
            f.write('#include "core/kernels.hpp"\n\n')
            f.write("// Generated by build.py - Injects UIDs and includes kernels\n\n")

            for inc_path, uid in sorted(entries):
                f.write(f"// --- {inc_path} ---\n")
                f.writelines(f"#undef {macro}\n" for macro in REGISTER_MACROS)

                uid_str = f"KernelId{{{uid}}}"
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
            f.writelines(f"#undef {macro}\n" for macro in REGISTER_MACROS)


# =============================================================================
# 4. Toolchain & Compiler Driver
# =============================================================================


class Toolchain:
    """Assembles flags and executes compiler/linker invocations."""

    def __init__(self, config: BuildConfig, platform_info: PlatformInfo):
        self.config = config
        self.platform = platform_info

    def get_cxx_binary(self) -> str:
        if self.platform.is_windows:
            return f'"{self.platform.clang_cpp_path}"'
        return "g++"

    def get_nvcc_binary(self) -> str:
        return "nvcc"

    def get_cxx_flags(self) -> List[str]:
        flags = [f"-I{ROOT_DIR}", f"-DTG_LOG_LEVEL={self.config.log_level_val}"]

        if self.platform.is_windows:
            if not self.config.use_cuda:
                flags.extend(
                    ["-target", "aarch64-windows", "-march=armv8.6-a+bf16+i8mm"]
                )

            opencl_inc = f"{self.platform.opencl_sdk_path}/include"
            opencl_lib = f"{self.platform.opencl_sdk_path}/lib"
            flags.extend(
                [
                    "-std=c++20",
                    f"-I{opencl_inc}",
                    f"-L{opencl_lib}",
                    "-lOpenCL",
                    "-DCL_TARGET_OPENCL_VERSION=310",
                    "-v",
                ]
            )
            if self.config.debug:
                flags.extend(["-g", "-O0", "-DDEBUG"])
            else:
                flags.append("-O3")
                if self.config.profile:
                    flags.extend(["-g", "-gcodeview", "-Wl,-debug"])
        else:
            flags.extend(["-std=c++20", "-lOpenCL"])
            if self.platform.is_arm64:
                flags.append("-march=armv8.6-a+bf16+i8mm")

            if self.config.debug:
                flags.extend(["-g", "-O0", "-DDEBUG", "-fno-omit-frame-pointer"])
            else:
                flags.append("-O3")

        if self.config.use_cuda:
            flags.append("-DUSE_CUDA")
            cuda_inc = (
                f'"{self.platform.cuda_path}\\include"'
                if self.platform.is_windows
                else f"-I{self.platform.cuda_path}/include"
            )
            flags.append(cuda_inc)

        if self.config.disable_opencl:
            flags.append("-DTG_DISABLE_OPENCL")

        return flags

    def get_nvcc_flags(self) -> List[str]:
        flags = [
            f"-I{ROOT_DIR}",
            "-std=c++20",
            "-x",
            "cu",
            f"-DTG_LOG_LEVEL={self.config.log_level_val}",
        ]

        if self.config.debug:
            flags.extend(["-g", "-G", "-O0", "-DDEBUG"])
        else:
            flags.append("-O3")

        if self.config.use_cuda:
            flags.append("-DUSE_CUDA")
            if not self.platform.is_windows and self.platform.is_arm64:
                flags.extend(["-Xcompiler", "-march=armv8.6-a+bf16+i8mm"])

        if self.config.disable_opencl:
            flags.append("-DTG_DISABLE_OPENCL")

        return flags

    def run_cmd(self, cmd: List[str]) -> subprocess.CompletedProcess:
        cmd_str = " ".join(cmd)
        if self.platform.is_windows:
            arch = "amd64" if self.config.use_cuda else "arm64"
            full_command = f'"{self.platform.vcvars_path}" {arch} && {cmd_str}'
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
        elif result.stderr.strip():
            console.print(
                Panel(
                    f"{result.stdout}[yellow]{result.stderr}[/yellow]",
                    title="[bold yellow]BUILD WARNINGS[/bold yellow]",
                    border_style="yellow",
                )
            )

        return result


# =============================================================================
# 5. Build Orchestrator
# =============================================================================


class BuildOrchestrator:
    """Coordinates Lint -> CodeGen -> Compile -> Link phases."""

    def __init__(self, config: BuildConfig):
        self.config = config
        self.platform = PlatformInfo.detect()
        self.toolchain = Toolchain(config, self.platform)
        self.linter = KernelLinter()
        self.code_gen = CodeGenerator()

    def run(self) -> None:
        console.print(
            f"\n[bold cyan]Starting Build [{'DEBUG' if self.config.debug else 'RELEASE'}] "
            f"(Log Level: {self.config.log_level_str}, CUDA: {self.config.use_cuda})...[/bold cyan]\n"
        )

        # 1. Lint Phase
        self.linter.lint(self.config)

        # 2. Code Generation Phase
        core_seed = self.code_gen.generate_core_seed()
        self.code_gen.generate_opencl_strings()
        self.code_gen.generate_kernel_includes(core_seed)
        self.code_gen.generate_build_context()

        # 3. Compilation Phase
        self._compile_project()

    def _compile_project(self) -> None:
        obj_ext = ".obj" if self.platform.is_windows else ".o"
        out_ext = ".exe" if self.platform.is_windows else ""

        cuda_obj = str(GENERATED_DIR / f"cuda_kernels{obj_ext}")

        # Step 3a: Compile CUDA Object if enabled
        if self.config.use_cuda:
            console.print("\n[bold blue]Compiling CUDA Kernels...[/bold blue]")
            cuda_src = str(GENERATED_DIR / "cuda_kernels.gen.cu")
            cmd = (
                [self.toolchain.get_nvcc_binary()]
                + self.toolchain.get_nvcc_flags()
                + ["-c", cuda_src, "-o", cuda_obj]
            )

            res = self.toolchain.run_cmd(cmd)
            self._render_success_panel(res.stdout)

        # Step 3b: Compile each target
        for main_file in self.config.targets:
            console.print(f"\n[bold blue]Compiling {main_file}...[/bold blue]")
            main_src = str(ROOT_DIR / main_file)
            target_stem = main_file.split(".")[0]
            out_name = f"tensor_graphs_cpp/{target_stem}{out_ext}"

            if self.config.use_cuda:
                main_obj = str(GENERATED_DIR / f"{target_stem}{obj_ext}")

                # Compile C++ source to object
                cmd = (
                    [self.toolchain.get_cxx_binary()]
                    + self.toolchain.get_cxx_flags()
                    + ["-c", main_src, "-o", main_obj]
                )
                self.toolchain.run_cmd(cmd)

                # Link objects via NVCC
                cmd = [
                    self.toolchain.get_nvcc_binary(),
                    main_obj,
                    cuda_obj,
                    "-o",
                    out_name,
                ]
                if self.platform.is_windows:
                    opencl_lib = f"{self.platform.opencl_sdk_path}/lib"
                    cmd.extend([f"-L{opencl_lib}", "-lOpenCL"])
                else:
                    cmd.append("-lOpenCL")

                if self.platform.is_windows and self.config.debug:
                    cmd.append("-g")

                res = self.toolchain.run_cmd(cmd)
            else:
                # Direct compile + link
                cmd = (
                    [self.toolchain.get_cxx_binary()]
                    + self.toolchain.get_cxx_flags()
                    + [main_src, "-o", out_name]
                )
                res = self.toolchain.run_cmd(cmd)

            self._render_success_panel(res.stdout)

    @staticmethod
    def _render_success_panel(stdout: str) -> None:
        content = stdout.strip() if stdout.strip() else "No output"
        console.print(
            Panel(
                f"[green]{content}[/green]",
                title="[bold green]BUILD SUCCESS[/bold green]",
                border_style="green",
            )
        )


# =============================================================================
# 6. CLI Entry Point
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="TensorGraph C++ Build System")
    parser.add_argument("--cuda", action="store_true", help="Enable CUDA build")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Build with debug symbols and no optimization",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Build with profiling symbols while keeping optimizations",
    )
    parser.add_argument(
        "--no-lint", action="store_true", help="Skip kernel validation checks"
    )
    parser.add_argument(
        "--disable-opencl", action="store_true", help="Disable OpenCL backend"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=[
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
            "OFF",
            "debug",
            "info",
            "warning",
            "error",
            "critical",
            "off",
        ],
        help="Set compile-time minimum logging level (default: INFO)",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        help="Specify which target C++ files to build (e.g. main, bench, test)",
    )
    args = parser.parse_args()

    config = BuildConfig(
        use_cuda=args.cuda,
        debug=args.debug,
        profile=args.profile,
        no_lint=args.no_lint,
        disable_opencl=args.disable_opencl,
        log_level_str=args.log_level,
        targets=args.targets,
    )

    orchestrator = BuildOrchestrator(config)
    orchestrator.run()


if __name__ == "__main__":
    main()

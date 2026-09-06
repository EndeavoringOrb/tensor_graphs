import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from tqdm import tqdm

console = Console()

ROOT_DIR = Path("tensor_graphs_cpp")
GENERATED_DIR = ROOT_DIR / "generated"
KERNELS_DIR = ROOT_DIR / "kernels"
CACHE_FILE = GENERATED_DIR / ".build_cache.json"


def writeIfChanged(filepath: Path, content: str) -> bool:
    """Writes content to filepath only if the file does not exist or its content differs."""
    if filepath.exists():
        try:
            if filepath.read_text(encoding="utf-8") == content:
                return False
        except Exception:
            pass
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(content, encoding="utf-8")
    return True


def parseDepFile(dep_path: Path) -> list[Path]:
    """Parses Makefile-style dependency .d file into list of existing local paths."""
    if not dep_path.exists():
        return []
    try:
        text = dep_path.read_text(encoding="utf-8", errors="ignore")
        parts = text.replace("\\\n", " ").replace("\n", " ").split()
        if not parts:
            return []
        deps = []
        for p in parts[1:]:
            p_clean = p.strip()
            if not p_clean or p_clean.startswith("/usr/") or p_clean.startswith("/opt/"):
                continue
            dep_file = Path(p_clean)
            if dep_file.exists():
                deps.append(dep_file)
        return deps
    except Exception:
        return []


def isObjectUpToDate(
    obj_path: Path,
    src_path: Path,
    dep_path: Path,
    cmd_key: str,
    cache: dict,
    force: bool = False,
) -> bool:
    """Checks if a compiled object file is up to date relative to its source and dependencies."""
    if force or not obj_path.exists():
        return False
    cached_entry = cache.get(str(obj_path.resolve()))
    if not cached_entry or cached_entry.get("cmd_key") != cmd_key:
        return False

    obj_mtime = obj_path.stat().st_mtime
    if src_path.exists() and src_path.stat().st_mtime > obj_mtime:
        return False

    deps = parseDepFile(dep_path)
    for d in deps:
        if d.exists() and d.stat().st_mtime > obj_mtime:
            return False

    return True


def isBinaryUpToDate(
    bin_path: Path,
    obj_paths: list[Path],
    link_key: str,
    cache: dict,
    force: bool = False,
) -> bool:
    """Checks if a linked binary is up to date relative to its constituent object files."""
    if force or not bin_path.exists():
        return False
    cached_entry = cache.get(str(bin_path.resolve()))
    if not cached_entry or cached_entry.get("link_key") != link_key:
        return False

    bin_mtime = bin_path.stat().st_mtime
    for obj in obj_paths:
        if not obj.exists() or obj.stat().st_mtime > bin_mtime:
            return False

    return True


def loadBuildCache() -> dict:
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def saveBuildCache(cache: dict) -> None:
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass


CORE_DEPENDENCIES = [
    ROOT_DIR / "core" / "types.hpp",
    ROOT_DIR / "core" / "kernels.hpp",
]

ALL_TARGETS = [
    "bench.cpp",
    "bench_model.cpp",
    "chat.cpp",
    "embed.cpp",
    "main.cpp",
    "test.cpp",
    "test_inst.cpp",
    "bindings.cpp",
]

REGISTER_MACROS = [
    "REGISTER_REF_KERNEL",
    "REGISTER_REF_KERNEL_VIEW",
    "REGISTER_KERNEL",
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
        r'\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"|/\*.*?\*/|//[^\r\n]*',
        re.DOTALL,
    )
    return pattern.sub(replacer, text)


def extract_macro_call_args(content: str, paren_start: int) -> list[str]:
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


def find_vcvarsall() -> str:
    """Locates Visual Studio vcvarsall.bat across standard installation directories."""
    if "VCVARS_PATH" in os.environ and Path(os.environ["VCVARS_PATH"]).exists():
        return os.environ["VCVARS_PATH"]

    candidates = [
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return ""


@dataclass
class CompilerInfo:
    path: str
    kind: str  # "clang", "gcc", or "msvc"

    @classmethod
    def detect(cls) -> "CompilerInfo":
        cxx_env = os.environ.get("CXX") or os.environ.get("CLANG_CXX")
        if cxx_env and (shutil.which(cxx_env) or Path(cxx_env).exists()):
            kind = "clang" if "clang" in Path(cxx_env).name.lower() else "gcc"
            return cls(path=cxx_env, kind=kind)

        which_clang = shutil.which("clang++")
        if which_clang:
            return cls(path=which_clang, kind="clang")

        default_llvm = Path(r"C:\Program Files\LLVM\bin\clang++.exe")
        if default_llvm.exists():
            return cls(path=str(default_llvm), kind="clang")

        which_gxx = shutil.which("g++")
        if which_gxx:
            return cls(path=which_gxx, kind="gcc")

        which_cl = shutil.which("cl")
        if which_cl:
            return cls(path=which_cl, kind="msvc")

        return cls(path="clang++", kind="clang")


@dataclass
class PlatformInfo:
    os_name: str
    machine: str
    is_arm64: bool
    is_python_arm64: bool
    is_windows: bool
    vcvars_path: str
    cuda_path: str
    opencl_sdk_path: str
    compiler: CompilerInfo
    has_cuda: bool
    has_opencl: bool
    cuda_inc_dir: str | None = None
    cuda_lib_dir: str | None = None
    opencl_inc_dir: str | None = None
    opencl_lib_dir: str | None = None

    @classmethod
    def detect(cls) -> "PlatformInfo":
        os_name = os.name
        machine = platform.machine().lower()
        is_windows = os_name == "nt"
        is_arm64 = machine in ("aarch64", "arm64")

        python_plat = sysconfig.get_platform().lower()
        is_python_arm64 = "arm64" in python_plat or "aarch64" in python_plat

        vcvars_path = find_vcvarsall()
        compiler = CompilerInfo.detect()

        which_nvcc = shutil.which("nvcc")
        nvcc_cuda_dir = (
            str(Path(which_nvcc).resolve().parent.parent) if which_nvcc else None
        )

        cuda_path = os.environ.get(
            "CUDA_PATH",
            os.environ.get(
                "CUDA_HOME",
                (
                    nvcc_cuda_dir
                    or (
                        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0"
                        if is_windows
                        else "/usr/local/cuda"
                    )
                ),
            ),
        )
        opencl_sdk_path = os.environ.get("OPENCL_SDK_ROOT", "./OpenCL-SDK/install")

        has_cuda = False
        cuda_inc_dir = None
        cuda_lib_dir = None

        cuda_inc_candidates = [
            Path(cuda_path) / "include",
            Path("/usr/local/cuda/include"),
            Path("/opt/cuda/include"),
            Path("/usr/include"),
        ]
        for inc_dir in cuda_inc_candidates:
            if (inc_dir / "cuda_runtime.h").exists():
                has_cuda = True
                cuda_inc_dir = str(inc_dir)
                break

        if not has_cuda and (which_nvcc is not None or Path(cuda_path).exists()):
            has_cuda = True
            cuda_inc_dir = str(Path(cuda_path) / "include")

        if has_cuda:
            cuda_lib_candidates = [
                Path(cuda_path) / ("lib/x64" if is_windows else "lib64"),
                Path(cuda_path) / "lib",
                Path("/usr/local/cuda/lib64"),
                Path("/usr/local/cuda/lib"),
                Path("/usr/lib/x86_64-linux-gnu"),
            ]
            for lib_dir in cuda_lib_candidates:
                if lib_dir.exists():
                    cuda_lib_dir = str(lib_dir)
                    break

        has_opencl = False
        opencl_inc_dir = None
        opencl_lib_dir = None

        inc_candidates = [
            Path(opencl_sdk_path) / "include",
            Path(cuda_path) / "include",
            Path("/usr/include"),
            Path("/usr/local/include"),
            Path("/usr/local/cuda/include"),
            Path("/opt/cuda/include"),
        ]

        for env_var in ["CPATH", "CPLUS_INCLUDE_PATH", "INCLUDE"]:
            if env_var in os.environ:
                for p in os.environ[env_var].split(os.pathsep):
                    if p.strip():
                        inc_candidates.append(Path(p.strip()))

        for inc_dir in inc_candidates:
            if (inc_dir / "CL" / "cl.h").exists() or (
                inc_dir / "OpenCL" / "cl.h"
            ).exists():
                has_opencl = True
                opencl_inc_dir = str(inc_dir)
                break

        lib_candidates = [
            Path(opencl_sdk_path) / "lib",
            Path(opencl_sdk_path) / "lib64",
            Path(cuda_path) / "lib64",
            Path(cuda_path) / "lib",
            Path(cuda_path) / "lib/x64",
            Path("/usr/local/cuda/lib64"),
            Path("/usr/lib/x86_64-linux-gnu"),
            Path("/usr/lib64"),
            Path("/usr/local/lib"),
        ]

        for env_var in ["LIBRARY_PATH", "LD_LIBRARY_PATH", "LIB"]:
            if env_var in os.environ:
                for p in os.environ[env_var].split(os.pathsep):
                    if p.strip():
                        lib_candidates.append(Path(p.strip()))

        for lib_dir in lib_candidates:
            if lib_dir.exists():
                if any(lib_dir.glob("*OpenCL*")) or any(lib_dir.glob("*opencl*")):
                    opencl_lib_dir = str(lib_dir)
                    break

        return cls(
            os_name=os_name,
            machine=machine,
            is_arm64=is_arm64,
            is_python_arm64=is_python_arm64,
            is_windows=is_windows,
            vcvars_path=vcvars_path,
            cuda_path=cuda_path,
            opencl_sdk_path=opencl_sdk_path,
            compiler=compiler,
            has_cuda=has_cuda,
            has_opencl=has_opencl,
            cuda_inc_dir=cuda_inc_dir,
            cuda_lib_dir=cuda_lib_dir,
            opencl_inc_dir=opencl_inc_dir,
            opencl_lib_dir=opencl_lib_dir,
        )


def ensure_toolchain(platform_info: PlatformInfo) -> None:
    if platform_info.is_windows:
        has_compiler = (
            shutil.which(platform_info.compiler.path) is not None
            or Path(platform_info.compiler.path).exists()
        )
        if not has_compiler:
            console.print(
                "[yellow]No C++ compiler found. Installing LLVM via winget...[/yellow]"
            )
            try:
                subprocess.run(
                    [
                        "winget",
                        "install",
                        "--id",
                        "LLVM.LLVM",
                        "-e",
                        "--accept-source-agreements",
                        "--accept-package-agreements",
                    ],
                    check=True,
                )
                llvm_bin = Path(r"C:\Program Files\LLVM\bin")
                if llvm_bin.exists():
                    os.environ["PATH"] = f"{llvm_bin};" + os.environ.get("PATH", "")
                    platform_info.compiler = CompilerInfo(
                        path=str(llvm_bin / "clang++.exe"), kind="clang"
                    )
                    console.print(
                        "[bold green]LLVM/clang++ installed successfully![/bold green]"
                    )
            except Exception as e:
                console.print(
                    f"[bold red]Auto-installation of LLVM via winget failed: {e}[/bold red]\n"
                    "[white]Please install LLVM or MinGW manually and add it to your PATH.[/white]"
                )

        vcvars = find_vcvarsall()
        platform_info.vcvars_path = vcvars
        platform_info.compiler = CompilerInfo.detect()
    else:
        has_compiler = (
            shutil.which(platform_info.compiler.path) is not None
            or shutil.which("clang++") is not None
            or shutil.which("g++") is not None
        )
        if not has_compiler:
            console.print(
                "[yellow]No C++ compiler found. Attempting automatic installation...[/yellow]"
            )
            if shutil.which("apt-get"):
                subprocess.run(
                    "sudo apt-get update && sudo apt-get install -y clang build-essential",
                    shell=True,
                    check=False,
                )
            elif shutil.which("dnf"):
                subprocess.run(
                    ["sudo", "dnf", "install", "-y", "clang", "gcc-c++", "make"],
                    check=False,
                )
            elif shutil.which("pacman"):
                subprocess.run(
                    ["sudo", "pacman", "-S", "--noconfirm", "clang", "base-devel"],
                    check=False,
                )
            platform_info.compiler = CompilerInfo.detect()


@dataclass
class BuildConfig:
    cuda_override: int | None = None
    opencl_override: int | None = None
    use_cuda: bool = False
    use_opencl: bool = False
    debug: bool = False
    profile: bool = False
    no_lint: bool = False
    force: bool = False
    clean: bool = False
    log_level_str: str = "INFO"
    log_level_val: int = 1
    targets: list[str] = field(default_factory=lambda: list(ALL_TARGETS))

    def resolve_overrides(self, platform_info: PlatformInfo):
        if self.cuda_override is not None:
            self.use_cuda = bool(self.cuda_override)
        else:
            self.use_cuda = platform_info.has_cuda

        if self.opencl_override is not None:
            self.use_opencl = bool(self.opencl_override)
        else:
            self.use_opencl = platform_info.has_opencl

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


class KernelLinter:
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
        for match in macro_pattern.finditer(clean_content):
            paren_start = match.end() - 1
            args = extract_macro_call_args(clean_content, paren_start)

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


class CodeGenerator:
    def __init__(self, config: BuildConfig):
        self.config = config

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

        lines = [
            "#pragma once\n",
            "#include <unordered_map>\n",
            "#include <string>\n\n",
            "inline const std::unordered_map<std::string, const char*> OPENCL_SOURCE_MAP = {\n",
        ]
        for cl_path in cl_files:
            rel_path = cl_path.relative_to(ROOT_DIR).as_posix()
            content = cl_path.read_text(encoding="utf-8")
            lines.append(f'    {{"{rel_path}", R"TG_OPENCL(\n{content}\n)TG_OPENCL"}},\n')
        lines.append("};\n")

        writeIfChanged(out_file, "".join(lines))
        console.print(f"[dim]Generated {len(cl_files)} OpenCL kernel strings.[/dim]")

    def generate_kernel_includes(self, core_seed: str) -> None:
        GENERATED_DIR.mkdir(parents=True, exist_ok=True)

        cpu_includes_hpp = GENERATED_DIR / "cpu_kernels.gen.hpp"
        cuda_includes_cu = GENERATED_DIR / "cuda_kernels.gen.cu"
        cuda_stable_cu = GENERATED_DIR / "cuda_stable.gen.cu"
        cuda_gen_cu = GENERATED_DIR / "cuda_generated.gen.cu"
        kernels_all_hpp = GENERATED_DIR / "kernels_all.gen.hpp"
        kernel_uids_json = GENERATED_DIR / "kernel_uids.json"
        kernel_uids_hpp = GENERATED_DIR / "kernel_uids.gen.hpp"

        kernel_entries_cpu: list[tuple[str, str]] = []
        kernel_entries_cuda: list[tuple[str, str]] = []
        kernel_entries_cuda_stable: list[tuple[str, str]] = []
        kernel_entries_cuda_gen: list[tuple[str, str]] = []
        uid_info_map: dict[str, dict[str, str]] = {}
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
            inc_path = rel_path.as_posix()

            if not self.config.use_opencl and ("kernels/opencl" in inc_path.lower()):
                continue
            if not self.config.use_cuda and (
                "kernels/cuda" in inc_path.lower()
                or "kernels/cublas" in inc_path.lower()
            ):
                continue

            file_hash = self.get_file_hash(path)
            combined_hash = hashlib.sha256(
                (core_seed + file_hash).encode("utf-8")
            ).hexdigest()
            uid_hex = f"0x{combined_hash[:16]}"
            uid_val = f"{uid_hex}ULL"

            if path.suffix == ".hpp":
                kernel_entries_cpu.append((inc_path, uid_val))
            else:
                kernel_entries_cuda.append((inc_path, uid_val))
                if "kernels/generated" in inc_path.lower():
                    kernel_entries_cuda_gen.append((inc_path, uid_val))
                else:
                    kernel_entries_cuda_stable.append((inc_path, uid_val))

            op_name = path.stem
            try:
                kcontent = path.read_text(encoding="utf-8", errors="ignore")
                m_name = re.search(
                    r'REGISTER_KERNEL(?:_VIEW)?\s*\(\s*"([^"]+)"', kcontent
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
        self._write_includes_file(cuda_stable_cu, kernel_entries_cuda_stable, is_cu=True)
        if kernel_entries_cuda_gen:
            self._write_includes_file(cuda_gen_cu, kernel_entries_cuda_gen, is_cu=True)
        else:
            if cuda_gen_cu.exists():
                cuda_gen_cu.unlink(missing_ok=True)

        writeIfChanged(kernels_all_hpp, '#pragma once\n#include "cpu_kernels.gen.hpp"\n')
        writeIfChanged(kernel_uids_json, json.dumps(uid_info_map, indent=2))
        writeIfChanged(kernel_uids_hpp, "".join(hpp_lines))

        console.print(
            f"[dim]Generated {len(kernel_entries_cpu)} CPU and {len(kernel_entries_cuda)} CUDA Kernel Includes ({len(kernel_entries_cuda_stable)} stable, {len(kernel_entries_cuda_gen)} generated).[/dim]"
        )
        console.print("[dim]Saved UID metadata mapping to kernel_uids.json.[/dim]")

    def generate_build_context(self) -> None:
        GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        ctx_hpp = GENERATED_DIR / "build_context.gen.hpp"
        cmd_str = f"{platform.machine()}"
        ctx_hash = hashlib.sha256(cmd_str.encode("utf-8")).hexdigest()

        content = (
            "#pragma once\n"
            "#include <cstdint>\n\n"
            "// Generated by build.py - Represents compile flags relevant to kernel benchmarks\n"
            f"constexpr uint64_t BUILD_CONTEXT_ID = 0x{ctx_hash[:16]}ULL;\n"
        )
        writeIfChanged(ctx_hpp, content)
        console.print(f"[dim]Build Context ID: 0x{ctx_hash[:16]}[/dim]")

    def _write_includes_file(
        self, filepath: Path, entries: list[tuple[str, str]], is_cu: bool
    ) -> None:
        lines = []
        if not is_cu:
            lines.append("#pragma once\n")
        lines.append('#include "core/kernels.hpp"\n\n')
        lines.append("// Generated by build.py - Injects UIDs and includes kernels\n\n")

        for inc_path, uid in sorted(entries):
            lines.append(f"// --- {inc_path} ---\n")
            for macro in REGISTER_MACROS:
                lines.append(f"#undef {macro}\n")

            uid_str = f"KernelId{{{uid}}}"
            lines.append(
                f"#define REGISTER_REF_KERNEL(op, n_min, n_max, match, run, ...) REGISTER_REF_KERNEL_INTERNAL({uid_str}, op, n_min, n_max, match, run, __VA_ARGS__)\n"
            )
            lines.append(
                f"#define REGISTER_REF_KERNEL_VIEW(op, n_min, n_max, match, inferView, ...) REGISTER_REF_KERNEL_VIEW_INTERNAL({uid_str}, op, n_min, n_max, match, inferView, __VA_ARGS__)\n"
            )
            lines.append(
                f"#define REGISTER_KERNEL(name, n_min, n_max, match, run, ref, ...) REGISTER_KERNEL_INTERNAL({uid_str}, name, n_min, n_max, match, run, ref, __VA_ARGS__)\n"
            )
            lines.append(
                f"#define REGISTER_KERNEL_VIEW(name, n_min, n_max, match, ref, inferView, ...) REGISTER_KERNEL_VIEW_INTERNAL({uid_str}, name, n_min, n_max, match, ref, inferView, __VA_ARGS__)\n"
            )
            lines.append(f'#include "{inc_path}"\n\n')

        lines.append("// --- Clean up macros ---\n")
        for macro in REGISTER_MACROS:
            lines.append(f"#undef {macro}\n")

        writeIfChanged(filepath, "".join(lines))


class Toolchain:
    def __init__(self, config: BuildConfig, platform_info: PlatformInfo):
        self.config = config
        self.platform = platform_info

    def get_cxx_binary(self) -> str:
        compiler_path = self.platform.compiler.path
        if (
            self.platform.is_windows
            and " " in compiler_path
            and not compiler_path.startswith('"')
        ):
            return f'"{compiler_path}"'
        return compiler_path

    def get_nvcc_binary(self) -> str:
        return "nvcc"

    def get_cxx_flags(self, is_python_ext: bool = False) -> list[str]:
        flags = [
            f"-I{ROOT_DIR}",
            "-std=c++17",
            f"-DTG_LOG_LEVEL={self.config.log_level_val}",
        ]
        if self.platform.is_windows:
            flags.append("-DNOMINMAX")

        if self.config.use_opencl:
            flags.append("-DTG_USE_OPENCL")
            flags.append("-DCL_TARGET_OPENCL_VERSION=300")
            if self.platform.opencl_inc_dir:
                flags.append(f"-I{self.platform.opencl_inc_dir}")

        target_arm64 = (
            self.platform.is_python_arm64 if is_python_ext else self.platform.is_arm64
        )

        if self.config.profile:
            flags.append("-DTG_PROFILE")

        if self.platform.is_windows:
            if target_arm64 and not self.config.use_cuda:
                flags.extend(
                    ["-target", "aarch64-windows", "-march=armv8.6-a+bf16+i8mm"]
                )

            if self.config.debug:
                flags.extend(["-g", "-O0", "-DTG_DEBUG"])
            else:
                flags.append("-O3")
                if self.config.profile:
                    flags.extend(["-g", "-gcodeview"])
        else:
            if target_arm64:
                flags.append("-march=armv8.6-a+bf16+i8mm")

            if self.config.debug:
                flags.extend(["-g", "-O0", "-DTG_DEBUG", "-fno-omit-frame-pointer"])
            else:
                flags.append("-O3")

        if self.config.use_cuda:
            flags.append("-DTG_USE_CUDA")
            cuda_inc = self.platform.cuda_inc_dir or str(
                Path(self.platform.cuda_path) / "include"
            )
            if Path(cuda_inc).exists():
                flags.append(f"-I{cuda_inc}")

        return flags

    def get_pybind11_flags(self) -> tuple[list[str], list[str], str]:
        py_includes = (
            subprocess.check_output(
                [sys.executable, "-m", "pybind11", "--includes"], text=True
            )
            .strip()
            .split()
        )

        ext_suffix = subprocess.check_output(
            [
                sys.executable,
                "-c",
                "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')",
            ],
            text=True,
        ).strip()

        inc_flags = list(py_includes)
        link_flags = ["-shared"]

        if self.platform.is_windows:
            py_lib_dir_base = Path(sys.base_prefix) / "libs"
            py_lib_dir_prefix = Path(sys.prefix) / "libs"
            link_flags.extend([f"-L{py_lib_dir_base}", f"-L{py_lib_dir_prefix}"])
            py_version_nodot = f"{sys.version_info.major}{sys.version_info.minor}"
            link_flags.append(f"-lpython{py_version_nodot}")
        else:
            inc_flags.append("-fPIC")

        return [f for f in inc_flags if f], [f for f in link_flags if f], ext_suffix

    def get_ld_flags(self, is_python_ext: bool = False) -> list[str]:
        flags = []
        if self.config.use_cuda:
            if self.platform.cuda_lib_dir:
                flags.append(f"-L{self.platform.cuda_lib_dir}")
                if not self.platform.is_windows:
                    flags.append(f"-Wl,-rpath,{self.platform.cuda_lib_dir}")
            flags.append("-lcudart")
            flags.append("-lcublas")

        if self.config.use_opencl:
            if self.platform.opencl_lib_dir:
                flags.append(f"-L{self.platform.opencl_lib_dir}")
                if not self.platform.is_windows:
                    flags.append(f"-Wl,-rpath,{self.platform.opencl_lib_dir}")
            flags.append("-lOpenCL")

            if self.platform.is_windows and self.config.profile:
                flags.append("-Wl,-debug")

        if self.platform.is_windows:
            flags.append("-ldbghelp")

        if not is_python_ext and not self.config.use_cuda:
            flags.extend(["-static"])
        return flags

    def get_nvcc_flags(self, is_python_ext: bool = False) -> list[str]:
        flags = [
            f"-I{ROOT_DIR}",
            "-std=c++17",
            "-x",
            "cu",
            f"-DTG_LOG_LEVEL={self.config.log_level_val}",
        ]

        if not self.platform.is_windows:
            flags.extend(["-Xcompiler", "-fPIC"])
        else:
            flags.append("-DNOMINMAX")

        if self.config.profile:
            flags.append("-DTG_PROFILE")

        if self.config.debug:
            flags.extend(["-g", "-G", "-O0", "-DTG_DEBUG"])
        else:
            flags.append("-O3")

        target_arm64 = (
            self.platform.is_python_arm64 if is_python_ext else self.platform.is_arm64
        )

        if self.config.use_cuda:
            flags.append("-DTG_USE_CUDA")
            if not self.platform.is_windows and target_arm64:
                flags.extend(["-Xcompiler", "-march=armv8.6-a+bf16+i8mm"])

        if self.config.use_opencl:
            flags.append("-DTG_USE_OPENCL")

        return flags

    def run_cmd(
        self, cmd: list[str], is_python_ext: bool = False
    ) -> subprocess.CompletedProcess:
        cmd_str = " ".join(cmd)

        if (
            self.platform.is_windows
            and self.platform.compiler.kind in ("clang", "msvc")
            and self.platform.vcvars_path
            and Path(self.platform.vcvars_path).exists()
        ):
            target_arm64 = (
                self.platform.is_python_arm64
                if is_python_ext
                else self.platform.is_arm64
            )
            arch = "arm64" if (target_arm64 and not self.config.use_cuda) else "amd64"
            full_command = f'"{self.platform.vcvars_path}" {arch} && {cmd_str}'
        else:
            full_command = cmd_str

        console.print(f"[dim]Running:[/dim] [cyan]{full_command}[/cyan]")
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


class BuildOrchestrator:
    def __init__(self, config: BuildConfig):
        self.platform = PlatformInfo.detect()
        config.resolve_overrides(self.platform)
        self.config = config
        self.toolchain = Toolchain(config, self.platform)
        self.linter = KernelLinter()
        self.code_gen = CodeGenerator(config)

    def clean(self) -> None:
        console.print("[bold yellow]Cleaning build artifacts and cache...[/bold yellow]")
        if CACHE_FILE.exists():
            CACHE_FILE.unlink(missing_ok=True)
        if GENERATED_DIR.exists():
            for p in GENERATED_DIR.glob("*"):
                if p.suffix in (".o", ".obj", ".d", ".cu", ".hpp", ".json"):
                    p.unlink(missing_ok=True)
        for target_file in ALL_TARGETS:
            target_stem = target_file.split(".")[0]
            bin_path = ROOT_DIR / target_stem
            if bin_path.exists():
                bin_path.unlink(missing_ok=True)
            bin_exe = ROOT_DIR / f"{target_stem}.exe"
            if bin_exe.exists():
                bin_exe.unlink(missing_ok=True)
        for p in Path(".").glob("tensor_graphs.*"):
            if p.is_file() and p.suffix in (".so", ".pyd", ".dylib"):
                p.unlink(missing_ok=True)
        self._render_success_panel("Clean completed successfully.")

    def run(self) -> None:
        if self.config.clean:
            self.clean()
            return

        ensure_toolchain(self.platform)

        console.print(
            f"\n[bold cyan]Starting Build [{'DEBUG' if self.config.debug else 'RELEASE'}] "
            f"(Compiler: {self.platform.compiler.path} [{self.platform.compiler.kind}], Log Level: {self.config.log_level_str}, CUDA: {self.config.use_cuda}, OpenCL: {self.config.use_opencl})...[/bold cyan]\n"
        )

        self.linter.lint(self.config)

        core_seed = self.code_gen.generate_core_seed()
        self.code_gen.generate_opencl_strings()
        self.code_gen.generate_kernel_includes(core_seed)
        self.code_gen.generate_build_context()

        self._compile_project()

    def _compile_project(self) -> None:
        obj_ext = ".obj" if self.platform.is_windows else ".o"
        out_ext = ".exe" if self.platform.is_windows else ""

        build_cache = loadBuildCache()
        if self.config.force:
            build_cache.clear()

        cuda_objs: list[str] = []
        cuda_objs_recompiled = False

        if self.config.use_cuda:
            nvcc_bin = self.toolchain.get_nvcc_binary()
            nvcc_flags = self.toolchain.get_nvcc_flags()

            cuda_stable_src = GENERATED_DIR / "cuda_stable.gen.cu"
            cuda_stable_obj = GENERATED_DIR / f"cuda_stable{obj_ext}"
            cuda_stable_dep = GENERATED_DIR / "cuda_stable.d"
            if not cuda_stable_src.exists() and (GENERATED_DIR / "cuda_kernels.gen.cu").exists():
                cuda_stable_src = GENERATED_DIR / "cuda_kernels.gen.cu"

            cmd_key_stable = f"{nvcc_bin} {' '.join(nvcc_flags)} {cuda_stable_src}"
            if isObjectUpToDate(cuda_stable_obj, cuda_stable_src, cuda_stable_dep, cmd_key_stable, build_cache, self.config.force):
                console.print(f"[dim]Using cached CUDA kernels: {cuda_stable_obj.name}[/dim]")
            else:
                console.print(f"\n[bold blue]Compiling CUDA Kernels ({cuda_stable_src.name})...[/bold blue]")
                dep_flag = ["-MMD", "-MF", str(cuda_stable_dep)] if not self.platform.is_windows else []
                cmd = (
                    [nvcc_bin]
                    + nvcc_flags
                    + dep_flag
                    + ["-c", str(cuda_stable_src), "-o", str(cuda_stable_obj)]
                )
                res = self.toolchain.run_cmd(cmd)
                self._render_success_panel(res.stdout)
                cuda_objs_recompiled = True
                build_cache[str(cuda_stable_obj.resolve())] = {
                    "cmd_key": cmd_key_stable,
                    "built_at": time.time(),
                }

            cuda_objs.append(str(cuda_stable_obj))

            cuda_gen_src = GENERATED_DIR / "cuda_generated.gen.cu"
            cuda_gen_obj = GENERATED_DIR / f"cuda_generated{obj_ext}"
            cuda_gen_dep = GENERATED_DIR / "cuda_generated.d"

            if cuda_gen_src.exists() and cuda_gen_src.stat().st_size > 0:
                cmd_key_gen = f"{nvcc_bin} {' '.join(nvcc_flags)} {cuda_gen_src}"
                if isObjectUpToDate(cuda_gen_obj, cuda_gen_src, cuda_gen_dep, cmd_key_gen, build_cache, self.config.force):
                    console.print(f"[dim]Using cached CUDA generated kernels: {cuda_gen_obj.name}[/dim]")
                else:
                    console.print(f"\n[bold blue]Compiling CUDA Generated Kernels ({cuda_gen_src.name})...[/bold blue]")
                    dep_flag = ["-MMD", "-MF", str(cuda_gen_dep)] if not self.platform.is_windows else []
                    cmd = (
                        [nvcc_bin]
                        + nvcc_flags
                        + dep_flag
                        + ["-c", str(cuda_gen_src), "-o", str(cuda_gen_obj)]
                    )
                    res = self.toolchain.run_cmd(cmd)
                    self._render_success_panel(res.stdout)
                    cuda_objs_recompiled = True
                    build_cache[str(cuda_gen_obj.resolve())] = {
                        "cmd_key": cmd_key_gen,
                        "built_at": time.time(),
                    }
                cuda_objs.append(str(cuda_gen_obj))
            else:
                if cuda_gen_obj.exists():
                    cuda_gen_obj.unlink(missing_ok=True)
                if cuda_gen_dep.exists():
                    cuda_gen_dep.unlink(missing_ok=True)

            cuda_combined_obj = GENERATED_DIR / f"cuda_kernels{obj_ext}"
            if len(cuda_objs) == 1:
                if not cuda_combined_obj.exists() or cuda_objs_recompiled:
                    shutil.copy2(cuda_objs[0], cuda_combined_obj)
            elif len(cuda_objs) > 1:
                if not self.platform.is_windows:
                    if not cuda_combined_obj.exists() or cuda_objs_recompiled:
                        combine_cmd = ["ld", "-r"] + cuda_objs + ["-o", str(cuda_combined_obj)]
                        subprocess.run(combine_cmd, check=True)
                else:
                    if not cuda_combined_obj.exists() or cuda_objs_recompiled:
                        shutil.copy2(cuda_objs[0], cuda_combined_obj)

        targets_to_compile = []
        target_info = {}

        for main_file in self.config.targets:
            main_src = ROOT_DIR / main_file
            target_stem = main_file.split(".")[0]
            is_py = (main_file == "bindings.cpp")

            main_obj = GENERATED_DIR / f"{target_stem}{obj_ext}"
            dep_file = GENERATED_DIR / f"{target_stem}.d"
            cxx_bin = self.toolchain.get_cxx_binary()
            cxx_flags = self.toolchain.get_cxx_flags(is_python_ext=is_py)
            py_inc = self.toolchain.get_pybind11_flags()[0] if is_py else []
            cmd_key = f"{cxx_bin} {' '.join(cxx_flags)} {' '.join(py_inc)} {main_src}"

            up_to_date = isObjectUpToDate(main_obj, main_src, dep_file, cmd_key, build_cache, self.config.force)
            target_info[main_file] = {
                "main_src": main_src,
                "target_stem": target_stem,
                "is_py": is_py,
                "main_obj": main_obj,
                "dep_file": dep_file,
                "cxx_bin": cxx_bin,
                "cxx_flags": cxx_flags,
                "py_inc": py_inc,
                "cmd_key": cmd_key,
                "recompiled": not up_to_date,
            }

            if not up_to_date:
                targets_to_compile.append(main_file)
            else:
                console.print(f"[dim]Using cached object: {main_obj.name}[/dim]")

        def compileTargetObj(target_file: str) -> tuple[str, bool, str]:
            info = target_info[target_file]
            console.print(f"\n[bold blue]Compiling {target_file}...[/bold blue]")
            dep_flag = ["-MMD", "-MF", str(info["dep_file"])] if not self.platform.is_windows else []
            cmd = (
                [info["cxx_bin"]]
                + info["cxx_flags"]
                + info["py_inc"]
                + dep_flag
                + ["-c", str(info["main_src"]), "-o", str(info["main_obj"])]
            )
            res = self.toolchain.run_cmd(cmd, is_python_ext=info["is_py"])
            return target_file, True, res.stdout

        if targets_to_compile:
            num_workers = min(len(targets_to_compile), os.cpu_count() or 4)
            if num_workers > 1:
                console.print(f"[cyan]Compiling {len(targets_to_compile)} targets in parallel ({num_workers} workers)...[/cyan]")
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = [executor.submit(compileTargetObj, tf) for tf in targets_to_compile]
                    for fut in concurrent.futures.as_completed(futures):
                        tf, ok, out = fut.result()
                        build_cache[str(target_info[tf]["main_obj"].resolve())] = {
                            "cmd_key": target_info[tf]["cmd_key"],
                            "built_at": time.time(),
                        }
                        self._render_success_panel(out)
            else:
                for tf in targets_to_compile:
                    tf, ok, out = compileTargetObj(tf)
                    build_cache[str(target_info[tf]["main_obj"].resolve())] = {
                        "cmd_key": target_info[tf]["cmd_key"],
                        "built_at": time.time(),
                    }
                    self._render_success_panel(out)

        for main_file in self.config.targets:
            info = target_info[main_file]
            target_stem = info["target_stem"]
            is_py = info["is_py"]
            main_obj = info["main_obj"]

            if is_py:
                _, py_link_flags, ext_suffix = self.toolchain.get_pybind11_flags()
                out_path = Path(f"tensor_graphs{ext_suffix}")
                dep_objs = [main_obj] + [Path(co) for co in cuda_objs]
                ld_flags = py_link_flags + self.toolchain.get_ld_flags(is_python_ext=True)
                link_key = f"{info['cxx_bin']} {' '.join(ld_flags)} {' '.join(str(o) for o in dep_objs)}"

                bin_up_to_date = isBinaryUpToDate(out_path, dep_objs, link_key, build_cache, self.config.force)
                if bin_up_to_date and not info["recompiled"] and not cuda_objs_recompiled:
                    console.print(f"[dim]Target up-to-date: {out_path.name}[/dim]")
                else:
                    console.print(f"\n[bold blue]Linking {out_path.name}...[/bold blue]")
                    cmd = (
                        [info["cxx_bin"]]
                        + [str(main_obj)]
                        + [str(co) for co in cuda_objs]
                        + ["-o", str(out_path)]
                        + ld_flags
                    )
                    res = self.toolchain.run_cmd(cmd, is_python_ext=True)
                    self._render_success_panel(res.stdout)
                    build_cache[str(out_path.resolve())] = {
                        "link_key": link_key,
                        "built_at": time.time(),
                    }
                continue

            out_path = Path(f"tensor_graphs_cpp/{target_stem}{out_ext}")
            dep_objs = [main_obj] + ([Path(co) for co in cuda_objs] if self.config.use_cuda else [])
            ld_flags = self.toolchain.get_cxx_flags() + self.toolchain.get_ld_flags()
            link_key = f"{info['cxx_bin']} {' '.join(ld_flags)} {' '.join(str(o) for o in dep_objs)}"

            bin_up_to_date = isBinaryUpToDate(out_path, dep_objs, link_key, build_cache, self.config.force)
            if bin_up_to_date and not info["recompiled"] and not cuda_objs_recompiled:
                console.print(f"[dim]Target up-to-date: {out_path.name}[/dim]")
            else:
                console.print(f"\n[bold blue]Linking {out_path.name}...[/bold blue]")
                link_cmd = (
                    [info["cxx_bin"]]
                    + [str(main_obj)]
                    + ([str(co) for co in cuda_objs] if self.config.use_cuda else [])
                    + ["-o", str(out_path)]
                    + ld_flags
                )
                if self.platform.is_windows and self.config.debug:
                    link_cmd.append("-g")

                res = self.toolchain.run_cmd(link_cmd)
                self._render_success_panel(res.stdout)
                build_cache[str(out_path.resolve())] = {
                    "link_key": link_key,
                    "built_at": time.time(),
                }

        saveBuildCache(build_cache)

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


def main() -> None:
    parser = argparse.ArgumentParser(description="TensorGraph C++ Build System")
    parser.add_argument(
        "--cuda",
        type=int,
        choices=[0, 1],
        default=None,
        help="Override CUDA support: 1 to enable, 0 to disable. Default: auto-detect",
    )
    parser.add_argument(
        "--opencl",
        type=int,
        choices=[0, 1],
        default=None,
        help="Override OpenCL support: 1 to enable, 0 to disable. Default: auto-detect",
    )
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
        "--force",
        action="store_true",
        help="Force rebuild of all targets ignoring cache",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean generated build artifacts and cache",
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
        help="Specify which target C++ files to build (e.g. main, bench, test, bindings)",
    )
    args = parser.parse_args()

    config = BuildConfig(
        cuda_override=args.cuda,
        opencl_override=args.opencl,
        debug=args.debug,
        profile=args.profile,
        no_lint=args.no_lint,
        force=args.force,
        clean=args.clean,
        log_level_str=args.log_level,
        targets=args.targets,
    )

    orchestrator = BuildOrchestrator(config)
    orchestrator.run()


if __name__ == "__main__":
    main()

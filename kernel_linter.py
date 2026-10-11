"""Portable kernel source and repository linter; no compiler or native build required."""
import argparse
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from rich.console import Console
from rich.panel import Panel
from tqdm import tqdm
from kernel_linter_rules import REDUNDANCY_PATTERNS, ruleLabel

ROOT_DIR = Path("tensor_graphs_cpp")
SERVER_JOB = os.environ.get("KERNEL_BENCH_SERVER_JOB") == "1"
console = Console(force_terminal=False, no_color=True, width=160) if SERVER_JOB else Console()

def stripCppCommentsAndStrings(text: str) -> str:
    """Replaces C++ comments and string/char literals with spaces, preserving newlines."""

    def replacer(match):
        s = match.group(0)
        return "".join("\n" if c == "\n" else " " for c in s)

    pattern = re.compile(
        r'\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"|/\*.*?\*/|//[^\r\n]*',
        re.DOTALL,
    )
    return pattern.sub(replacer, text)


def extractMacroCallArgs(content: str, paren_start: int) -> list[str]:
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


class KernelLinter:
    REF_FACTORY_DEFINITION_PATTERN = re.compile(
        r"\bLogicalId\s+([a-zA-Z_]\w*)\s*\(\s*"
        r"const\s+std::vector\s*<\s*LogicalId\s*>\s*&\s*[a-zA-Z_]\w*\s*,\s*"
        r"Graph\s*&\s*[a-zA-Z_]\w*\s*\)\s*\{",
        re.MULTILINE,
    )
    REF_FACTORY_PATTERN = re.compile(
        r"\bLogicalId\s+[a-zA-Z_]\w*\s*\(\s*"
        r"const\s+std::vector\s*<\s*LogicalId\s*>\s*&\s*[a-zA-Z_]\w*\s*,\s*"
        r"Graph\s*&\s*[a-zA-Z_]\w*\s*\)\s*\{",
        re.MULTILINE,
    )

    REDUNDANCY_PATTERNS = REDUNDANCY_PATTERNS

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

    def validateKernelFile(self, file_path: Path):
        if (
            not file_path.is_file()
            or file_path.suffix not in [".hpp", ".cu"]
            or file_path.name == "ref.hpp"
        ):
            return
        content = file_path.read_text(encoding="utf-8")
        rel_path = Path(os.path.relpath(file_path, ROOT_DIR))
        clean_content = stripCppCommentsAndStrings(content)

        macro_pattern = re.compile(r"\b(REGISTER_[\w_]+)\s*\(")
        n_matches = 0
        for match in macro_pattern.finditer(clean_content):
            paren_start = match.end() - 1
            args = extractMacroCallArgs(clean_content, paren_start)

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

                body = clean_content[start_brace + 1 : end_idx - 1]
                body_start = start_brace + 1

                for pattern, (name, reason) in self.REDUNDANCY_PATTERNS.items():
                    pat_match = re.search(pattern, body)
                    if pat_match:
                        match_pos = body_start + pat_match.start()
                        line_num = content[:match_pos].count("\n") + 1
                        console.print(
                            Panel(
                                f"[bold red]REDUNDANT LOGIC DETECTED:[/bold red] in [cyan]{ROOT_DIR / rel_path}:{line_num}[/cyan]\n\n"
                                f"Rule: {name.lower().replace(chr(32), chr(95))}\nThe match function [yellow]{match_func_name}[/yellow] contains a manual [bold]{name}[/bold] on line {line_num}.\n\n"
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
                    f"[bold red]{ruleLabel('kernel_registration_detection_error')}:[/bold red] no kernel registration found in [cyan]{ROOT_DIR / rel_path}[/cyan]",
                    title="Linter Error",
                    border_style="red",
                )
            )
            sys.exit(1)

    def validateFusedRefLayout(self):
        fused_dir = ROOT_DIR / "kernels" / "fused"
        if not fused_dir.is_dir():
            return

        kernel_files = [
            path
            for path in fused_dir.rglob("*")
            if path.is_file()
            and path.suffix in {".hpp", ".cu", ".cl"}
            and path.name != "ref.hpp"
        ]
        nested_ref_paths = [
            path
            for path in fused_dir.rglob("ref.hpp")
            if len(path.relative_to(fused_dir).parts) > 2
        ]
        if nested_ref_paths:
            nested_ref = sorted(nested_ref_paths)[0]
            console.print(
                Panel(
                    f"[bold red]{ruleLabel('nested_reference_factory_header')}:[/bold red] [cyan]{nested_ref}[/cyan]\n\n"
                    "Keep ref.hpp at the fused kernel folder root and have nested kernel implementations include it.",
                    title="Linter Violation",
                    border_style="red",
                )
            )
            sys.exit(1)

        factory_dirs = {
            fused_dir / path.relative_to(fused_dir).parts[0] for path in kernel_files
        }

        for folder in sorted(factory_dirs):
            ref_path = folder / "ref.hpp"
            if not ref_path.is_file():
                console.print(
                    Panel(
                        f"[bold red]{ruleLabel('reference_factory_header_missing')}:[/bold red] [cyan]{ref_path}[/cyan]\n\n"
                        "Every fused kernel folder must contain a ref.hpp.",
                        title="Linter Violation",
                        border_style="red",
                    )
                )
                sys.exit(1)
            content = ref_path.read_text(encoding="utf-8")
            clean_content = stripCppCommentsAndStrings(content)
            factories = list(
                self.REF_FACTORY_DEFINITION_PATTERN.finditer(clean_content)
            )
            if len(factories) != 1:
                console.print(
                    Panel(
                        f"[bold red]{ruleLabel('invalid_reference_factory_count')}:[/bold red] [cyan]{ref_path}[/cyan]\n\n"
                        f"Expected exactly one ref factory definition, found {len(factories)}.",
                        title="Linter Violation",
                        border_style="red",
                    )
                )
                sys.exit(1)

        for file_path in kernel_files:
            rel_path = file_path.relative_to(fused_dir)
            is_opencl_kernel = file_path.suffix == ".cl"
            clean_content = ""
            if file_path.suffix in {".hpp", ".cu"}:
                content = file_path.read_text(encoding="utf-8")
                clean_content = stripCppCommentsAndStrings(content)
                is_opencl_kernel = bool(
                    re.search(
                        r"\b(?:OpenCL::getKernel|clEnqueueNDRangeKernel)\s*\(",
                        clean_content,
                    )
                )

            if is_opencl_kernel and len(rel_path.parts) < 3:
                console.print(
                    Panel(
                        f"[bold red]{ruleLabel('opencl_kernel_must_be_nested')}:[/bold red] [cyan]{ROOT_DIR / file_path.relative_to(ROOT_DIR)}[/cyan]\n\n"
                        "Place each OpenCL source and wrapper in a subfolder beneath its fused kernel folder.",
                        title="Linter Violation",
                        border_style="red",
                    )
                )
                sys.exit(1)

            if is_opencl_kernel and file_path.suffix in {".hpp", ".cu"}:
                source_match = re.search(
                    r'\bOpenCL::getKernel\s*\(\s*"([^"]+\.cl)"', content
                )
                if source_match and (ROOT_DIR / source_match.group(1)).parent != file_path.parent:
                    console.print(
                        Panel(
                            f"[bold red]{ruleLabel('opencl_source_and_wrapper_must_be_together')}:[/bold red] [cyan]{ROOT_DIR / file_path.relative_to(ROOT_DIR)}[/cyan]\n\n"
                            f"The referenced source {source_match.group(1)} must be in the wrapper's subfolder.",
                            title="Linter Violation",
                            border_style="red",
                        )
                    )
                    sys.exit(1)

            if self.REF_FACTORY_PATTERN.search(clean_content):
                rel_path = Path(os.path.relpath(file_path, ROOT_DIR))
                console.print(
                    Panel(
                        f"[bold red]{ruleLabel('reference_factory_in_kernel_file')}:[/bold red] [cyan]{ROOT_DIR / rel_path}[/cyan]\n\n"
                        "Move the LogicalId ref factory definition into the folder's ref.hpp.",
                        title="Linter Violation",
                        border_style="red",
                    )
                )
                sys.exit(1)

    def validateRewriteFile(self, file_path: Path):
        if not file_path.is_file() or file_path.suffix not in [
            ".hpp",
            ".cu",
            ".cpp",
        ]:
            return
        content = file_path.read_text(encoding="utf-8")
        clean_content = stripCppCommentsAndStrings(content)

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
                rel_path = Path(os.path.relpath(file_path, ROOT_DIR))
                console.print(
                    Panel(
                        f"[bold red]{ruleLabel('dangling_reference_hazard_detected')}:[/bold red] in [cyan]{ROOT_DIR / rel_path}:{line_num}[/cyan]\n\n"
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
                        rel_path = Path(os.path.relpath(file_path, ROOT_DIR))
                        console.print(
                            Panel(
                                f"[bold red]{ruleLabel('dangling_reference_hazard_detected')}:[/bold red] in [cyan]{ROOT_DIR / rel_path}:{line_num}[/cyan]\n\n"
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

    def lint(self, config):
        if config.no_lint:
            return

        self.validateFusedRefLayout()

        validators = [
            (ROOT_DIR / "kernels", self.validateKernelFile),
            (ROOT_DIR / "core", self.validateRewriteFile),
        ]

        for val_idx, (dir_path, func) in enumerate(validators):
            if not dir_path.exists():
                continue
            files = [p for p in dir_path.rglob("*") if p.is_file()]
            with tqdm(
                files,
                desc=f"linting [{val_idx + 1}/{len(validators)}]",
                disable=SERVER_JOB,
            ) as pbar:
                for path in pbar:
                    pbar.set_postfix_str(path.as_posix())
                    func(path)



def lintCandidate(path):
    path = Path(path).resolve()
    if not path.is_file() or path.suffix not in (".hpp", ".cu") or path.name == "ref.hpp":
        raise ValueError("Candidate must be an existing implementation .hpp or .cu file")
    KernelLinter().validateKernelFile(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", nargs="?", type=Path)
    parser.add_argument("--repository", action="store_true")
    args = parser.parse_args()
    if args.repository == bool(args.candidate):
        parser.error("Provide a candidate path or --repository")
    if args.repository:
        KernelLinter().lint(SimpleNamespace(no_lint=False))
    else:
        lintCandidate(args.candidate)
    print("Lint passed")


if __name__ == "__main__":
    main()

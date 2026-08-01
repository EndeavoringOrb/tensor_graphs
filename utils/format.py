import shutil
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm

# File extensions to target
CPP_EXTENSIONS = {".cpp", ".hpp"}
EXCLUDE = [
    "tensor_graphs_cpp/json.hpp",
    "tensor_graphs_cpp/stb_image_write.h",
    "tensor_graphs_cpp/stb_image.h",
]
EXCLUDE = [Path(path).as_posix() for path in EXCLUDE]


def find_clang_format() -> str:
    """Locate clang-format binary from PATH or fallback to LLVM install location."""
    clang_path = shutil.which("clang-format")
    if clang_path:
        return clang_path

    # Fallback to standard Windows LLVM installation directory
    llvm_default = Path(r"C:\Program Files\LLVM\bin\clang-format.exe")
    if llvm_default.exists():
        return str(llvm_default)

    raise FileNotFoundError(
        "clang-format was not found on PATH or at C:\\Program Files\\LLVM\\bin\\clang-format.exe"
    )


def get_cpp_files(root_dir: Path) -> list[Path]:
    """Find C/C++ files, respecting .gitignore using git ls-files.

    Falls back to rglob if Git is unavailable or if not in a Git repository.
    """
    try:
        # Ask Git for all tracked (--cached) and untracked (--others) files
        # while respecting standard gitignore rules (--exclude-standard)
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=root_dir,
            capture_output=True,
            text=True,
            check=True,
        )

        cpp_files = []
        for line in result.stdout.splitlines():
            file_path = root_dir / line.strip()
            if (
                file_path.suffix.lower() in CPP_EXTENSIONS
                and file_path.is_file()
                and not any(exclude in file_path.as_posix() for exclude in EXCLUDE)
            ):
                cpp_files.append(file_path)

        return cpp_files

    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Git unavailable or not a Git repo. Falling back to directory scan...")
        return [
            p
            for p in root_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() in CPP_EXTENSIONS
            and not any(part.startswith(".") for part in p.parts)
        ]


def main():
    root_dir = Path.cwd()

    # 1. Collect C/C++ files respecting .gitignore
    print("Searching for C++ files...")
    cpp_files = get_cpp_files(root_dir)

    # 2. Format C++ files with progress bar
    if cpp_files:
        clang_path = find_clang_format()
        for file_path in tqdm(cpp_files, desc="Formatting C++ files"):
            tqdm.write(str(file_path))
            subprocess.run(
                [clang_path, "--style=Microsoft", "-i", str(file_path)], check=True
            )
    else:
        print("No C++ files found.")

    # 3. Run Ruff via the active Python environment
    print("\nRunning Ruff format...")
    subprocess.run([sys.executable, "-m", "ruff", "format"])

    print("\nRunning Ruff check --fix...")
    subprocess.run([sys.executable, "-m", "ruff", "check", "--fix"])


if __name__ == "__main__":
    main()

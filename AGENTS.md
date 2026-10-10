-use `rg` instead of `grep`. make searches targeted (e.g rg something tensor_graphs_cpp) to avoid long search times.
-use .venv/Scripts/python.exe (or .venv/bin/python for linux) when running python. .venvx64/Scripts/python is used for ortools as there is no arm64 windows package

## kernels
- kernels should not have branching logic (e.g. to thread or not to thread based on input size), it is the compiler's responsibility to decide to use the threaded version or the unthreaded version so they should be two different kernels. this also means that any capability guards like `#if defined(TG_HAS_AVX2)` should wrap the entire kernel file, there shouldn't be fallbacks inside the run functions.

## code conventions
- class: PascalCase
- function: camelCase
- variable: snake_case

fail loud. e.g. instead of
```
if (branch_type < branch_counts.size())
    branch_counts[branch_type]++;
```
do
```
if (branch_type > branch_counts.size())
    Error::throw_err(some error message)
branch_counts[branch_type]++;
```
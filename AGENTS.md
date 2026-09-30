-use `rg` instead of `grep`
-use .venv/Scripts/python.exe (or .venv/bin/python for linux) when running python. .venvx64/Scripts/python is used for ortools as there is no arm64 windows package

conventions
- class: PascalCase
- function: camelCase
- variable: snake_case

verification:
do not use `--no-lint` when building
`python -m tests.test_gemma_output`
to debug correctness issues, if you have some configuration that works (i.e. without caching or full bucket only) you can use utils/compare_reference_tensors.py along with --write-refs, --compare-refs
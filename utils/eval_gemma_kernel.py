#!/usr/bin/env python3
"""
utils/eval_gemma_kernel.py

Automates the 11-step evaluation loop for a target kernel on Gemma 3 270M:
1. Research (manual)
2. Write kernel (manual)
3. build.py --t test bench bindings --opencl 0
4. test <targetKernel> (registered shape)
5. python -m tests.test_gemma_output --config decode --saturate-only
6. test <targetKernel> (all matched shapes)
7. bench <targetKernel>
8. compare against existing records (utils/read_benchmarks.py)
9. utils/run_gemma_benchmarks.py --target tensor_graphs
10. python -m tests.test_gemma_output --config decode --cache dirty_region_caches/gemma-decode
11. Repeat / Summary
"""

import argparse
import glob
import os
import re
import subprocess
import sys
import time

PYTHON_BIN = sys.executable or ".venv/bin/python"


def log_header(step_num: int, title: str):
    print(f"\n{'='*70}\n[STEP {step_num}] {title}\n{'='*70}")


def run_command(cmd: list[str], desc: str, fail_loud: bool = True) -> subprocess.CompletedProcess:
    cmd_str = " ".join(cmd)
    print(f"--> [{desc}] {cmd_str}")
    start = time.time()
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - start
    print(f"    (finished in {elapsed:.2f}s, exit code {res.returncode})")
    if res.returncode != 0:
        print(f"\n[ERROR] Command failed with code {res.returncode}:\n{res.stdout}")
        if fail_loud:
            raise RuntimeError(f"Step '{desc}' failed with exit code {res.returncode}")
    return res


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Tensor Graphs kernel for Gemma 3 270M.")
    parser.add_argument("targetKernel", nargs="?", default="", help="Target kernel name/filter (e.g. Gemma_Attention_CUDA).")
    parser.add_argument("--skip-build", action="store_true", help="Skip Step 3 (build).")
    parser.add_argument("--skip-bench", action="store_true", help="Skip Step 7 & 8 (standalone bench and record comparison).")
    parser.add_argument("--skip-saturate", action="store_true", help="Skip Step 5 (saturation).")
    parser.add_argument("--skip-e2e", action="store_true", help="Skip Step 9 & 10 (full Gemma benchmark and output verify).")
    parser.add_argument("--is-view", action="store_true", help="Kernel is a view kernel (skips standalone benchmarking).")
    parser.add_argument("--clean-cache", action="store_true", default=True, help="Remove dirty region caches before benchmark.")

    args = parser.parse_args()
    target = args.targetKernel.strip()

    print(f"Starting eval_gemma_kernel for target='{target}'")

    # Step 3: Build
    if not args.skip_build:
        log_header(3, "Build (test, bench, bindings)")
        run_command([PYTHON_BIN, "build.py", "--t", "test", "bench", "bindings", "--opencl", "0"], "build targets")

    # Step 4: Test initial shape
    if target and not args.is_view:
        log_header(4, f"Test initial target kernel: {target}")
        res = run_command(["./tensor_graphs_cpp/test", "--no-records", target], f"test {target}")
        print(res.stdout[-1000:] if len(res.stdout) > 1000 else res.stdout)

    # Step 5: Saturate decode graph
    if not args.skip_saturate:
        log_header(5, "Saturate Gemma decode graph (--saturate-only)")
        run_command([PYTHON_BIN, "-m", "tests.test_gemma_output", "--config", "decode", "--saturate-only"], "saturate-only")

    # Step 6: Test matched shapes
    if target and not args.is_view:
        log_header(6, f"Test target kernel on all matched shapes: {target}")
        res = run_command(["./tensor_graphs_cpp/test", target], f"test matched {target}")
        print(res.stdout[-1000:] if len(res.stdout) > 1000 else res.stdout)

    # Step 7: Bench target kernel
    if target and not args.skip_bench and not args.is_view:
        log_header(7, f"Benchmark target kernel: {target}")
        res = run_command(["./tensor_graphs_cpp/bench", target], f"bench {target}")
        print(res.stdout)

    # Step 8: Compare head-to-head records
    if target and not args.skip_bench and not args.is_view:
        log_header(8, f"Read and compare records for: {target}")
        res = run_command([PYTHON_BIN, "utils/read_benchmarks.py"], "read_benchmarks", fail_loud=False)
        lines = [line for line in res.stdout.splitlines() if target.lower() in line.lower()]
        if lines:
            print(f"Found matching benchmark records for {target}:")
            for line in lines[-10:]:
                print(f"  {line}")
        else:
            print(f"Note: No explicit record printed for {target} in records.bin yet.")

    # Step 9: Full Gemma benchmark
    bench_results = {}
    if not args.skip_e2e:
        log_header(9, "Run Gemma benchmarks (Prompt Processing & Text Generation)")
        if args.clean_cache:
            for cache_file in glob.glob("dirty_region_caches/gemma-decode*"):
                try:
                    os.remove(cache_file)
                    print(f"Removed stale cache: {cache_file}")
                except Exception:
                    pass

        res = run_command([PYTHON_BIN, "utils/run_gemma_benchmarks.py", "--target", "tensor_graphs"], "gemma benchmark")
        print(res.stdout)

        # Parse PP and Text Gen from output
        # Example lines:
        # Prompt Processing: 518.20 ms
        # Text Generation: 29.17 ms / tok
        # Overall Generation: 44.50 ms / tok
        m_pp = re.search(r"Prompt Processing:\s*([\d\.]+)\s*ms", res.stdout)
        m_tg = re.search(r"Text Generation:\s*([\d\.]+)\s*ms\s*/\s*tok", res.stdout)
        m_og = re.search(r"Overall Generation:\s*([\d\.]+)\s*ms\s*/\s*tok", res.stdout)

        if m_pp:
            bench_results["pp_ms"] = float(m_pp.group(1))
        if m_tg:
            bench_results["tg_ms_tok"] = float(m_tg.group(1))
        if m_og:
            bench_results["og_ms_tok"] = float(m_og.group(1))

    # Step 10: Verify Gemma output correctness
    verified = False
    if not args.skip_e2e:
        log_header(10, "Verify Gemma output correctness from cached decode")
        res = run_command(
            [PYTHON_BIN, "-m", "tests.test_gemma_output", "--config", "decode", "--cache", "dirty_region_caches/gemma-decode"],
            "verify decode output"
        )
        print(res.stdout)
        if "is <strong>Jasmine</strong>" in res.stdout or "Output matches expected" in res.stdout:
            verified = True
            print(">>> [SUCCESS] Output matches expected: 'is <strong>Jasmine</strong>' <<<")
        else:
            print(">>> [WARNING] Verify output did not contain confirmation substring! Check output above. <<<")

    # Summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"Target Kernel: {target or 'N/A'}")
    if bench_results:
        pp = bench_results.get("pp_ms", "N/A")
        tg = bench_results.get("tg_ms_tok", "N/A")
        og = bench_results.get("og_ms_tok", "N/A")
        print(f"Prompt Processing:    {pp} ms")
        print(f"Text Generation:      {tg} ms / tok")
        print(f"Overall Generation:   {og} ms / tok")
    print(f"Output Verified:      {'YES' if verified else ('SKIPPED' if args.skip_e2e else 'NO')}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

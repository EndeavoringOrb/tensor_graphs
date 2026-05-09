import subprocess
import sys
import time


def main():
    iteration = 1
    start_total = time.perf_counter()

    while True:
        print(f"\n{'='*20} Loop Iteration {iteration} {'='*20}")
        iter_start_time = time.perf_counter()

        # 1. Run the build command
        print("Running build.py...")
        build_start = time.perf_counter()

        build_cmd = ["./wsl_env/bin/python", "build.py"]
        result = subprocess.run(build_cmd, capture_output=True, text=True)

        build_end = time.perf_counter()
        build_duration = build_end - build_start

        # Print output so you can monitor progress
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        print(f"Build finished in {build_duration:.2f} seconds.")

        # 2. Check for Linter Violation
        if "Linter Violation" in result.stdout or "Linter Violation" in result.stderr:
            print(f"\nLinter Violation detected. Calling Claude to fix...")

            claude_command = (
                'claude "./wsl_env/bin/python build.py fix that linter error. '
                "When you run 'wsl_env/bin/python build.py' again to check, "
                "if there is a new linter error, that is ok. "
                'Only work on the first linter error you see, then when that is resolved, stop." '
                '--permission-mode "acceptEdits" -p'
            )

            claude_start = time.perf_counter()
            # Run Claude and wait for it to finish fixing the file
            subprocess.run(claude_command, shell=True)
            claude_end = time.perf_counter()

            claude_duration = claude_end - claude_start
            iter_duration = claude_end - iter_start_time

            print(f"\nClaude fix completed in {claude_duration:.2f} seconds.")
            print(f"Total iteration {iteration} time: {iter_duration:.2f} seconds.")

            iteration += 1
        else:
            iter_end_time = time.perf_counter()
            print(
                f"\nNo more Linter Violations found! (Iteration took {iter_end_time - iter_start_time:.2f}s)"
            )
            break

    end_total = time.perf_counter()
    print(f"\n{'='*20} Process Complete {'='*20}")
    print(f"Total time elapsed: {end_total - start_total:.2f} seconds.")
    print(f"Total iterations: {iteration}")


if __name__ == "__main__":
    main()

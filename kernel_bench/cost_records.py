"""Import validated microbenchmark timings into the compiler's native corpus."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tensor_graphs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--measurements", type=Path, required=True)
    args = parser.parse_args()
    measurements = json.loads(args.measurements.read_text())
    # Old baseline kernels may have been removed since the discovery snapshot.
    # The candidate itself must be present in the candidate build.
    kernels = {row["name"]: str(row["kernel_id"]) for row in tensor_graphs.kernel_bench_kernels()}
    selected = []
    for row in measurements:
        if row["kernel_name"] not in kernels:
            if row["is_candidate"]:
                raise ValueError("Measured candidate is missing from the compiler registry")
            continue
        if not row["is_candidate"] and str(row["kernel_id"]) != kernels[row["kernel_name"]]:
            continue  # Never apply another implementation's timing to this UID.
        selected.append(row)
    print("Imported compiler records:", tensor_graphs.kernel_bench_write_records(args.records, selected))


if __name__ == "__main__":
    main()

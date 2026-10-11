"""Compare two tensor reference files, matching repeated operations by logical ID."""

import argparse
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np


HEADER = struct.Struct("<II")
COUNT = struct.Struct("<Q")


def index_reference(path: Path):
    """Index record offsets by (logical ID, occurrence) without loading tensor data."""
    records = {}
    occurrences = defaultdict(int)
    with path.open("rb") as stream:
        while True:
            header = stream.read(HEADER.size)
            if not header:
                break
            if len(header) != HEADER.size:
                raise ValueError(f"truncated record header in {path}")

            logical_id, name_length = HEADER.unpack(header)
            name_data = stream.read(name_length)
            count_data = stream.read(COUNT.size)
            if len(name_data) != name_length or len(count_data) != COUNT.size:
                raise ValueError(f"truncated record metadata in {path}")
            name = name_data.decode("utf-8")
            element_count = COUNT.unpack(count_data)[0]
            data_offset = stream.tell()
            data_bytes = element_count * np.dtype("<f4").itemsize
            if data_bytes > path.stat().st_size - data_offset:
                raise ValueError(f"truncated tensor data for logical ID {logical_id} in {path}")

            occurrence = occurrences[logical_id]
            occurrences[logical_id] += 1
            records[(logical_id, occurrence)] = (data_offset, element_count, name)
            stream.seek(data_bytes, 1)

    return records, occurrences


def compare(reference_path: Path, current_path: Path, atol: float, limit: int) -> int:
    reference, reference_occurrences = index_reference(reference_path)
    _, current_occurrences = index_reference(current_path)
    reports = []
    matched = 0
    shape_mismatches = []
    current_seen = defaultdict(int)

    with reference_path.open("rb") as reference_file, current_path.open("rb") as current_file:
        current_file.seek(0)
        while True:
            header = current_file.read(HEADER.size)
            if not header:
                break
            logical_id, name_length = HEADER.unpack(header)
            name = current_file.read(name_length).decode("utf-8")
            count_data = current_file.read(COUNT.size)
            element_count = COUNT.unpack(count_data)[0]
            current_data = current_file.read(element_count * 4)

            # Count occurrences as records are streamed, so duplicate logical IDs
            # pair in the order they were written.
            record_index = current_seen[logical_id]
            current_seen[logical_id] += 1
            expected = reference.get((logical_id, record_index))
            if expected is None:
                continue
            ref_offset, ref_count, ref_name = expected
            if ref_count != element_count:
                shape_mismatches.append((logical_id, record_index, name, element_count, ref_count, ref_name))
                continue

            matched += 1
            reference_file.seek(ref_offset)
            reference_data = reference_file.read(ref_count * 4)
            actual = np.frombuffer(current_data, dtype="<f4")
            expected_values = np.frombuffer(reference_data, dtype="<f4")
            differences = np.abs(actual - expected_values)
            finite = differences[np.isfinite(differences)]
            if not np.allclose(actual, expected_values, atol=atol, rtol=0.0, equal_nan=True):
                max_difference = float(np.nanmax(differences)) if not np.isnan(differences).all() else float("nan")
                mean_difference = float(finite.mean()) if finite.size else 0.0
                max_index = int(np.nanargmax(differences)) if not np.isnan(differences).all() else -1
                reports.append((logical_id, record_index, name, max_difference, mean_difference, max_index))

    current_total = sum(current_seen.values())
    reference_total = sum(reference_occurrences.values())
    print(f"matched records: {matched}; current records: {current_total}; reference records: {reference_total}")
    if shape_mismatches:
        print("record size mismatches:")
        for row in shape_mismatches[:limit]:
            print(row)
    print(f"records differing by more than {atol}:")
    for row in reports[:limit]:
        print(row)
    if not reports and not shape_mismatches and matched == current_total == reference_total:
        print("all records match")
    return int(bool(reports or shape_mismatches or matched != current_total or matched != reference_total))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="reference tensor file")
    parser.add_argument("current", type=Path, help="tensor file to compare")
    parser.add_argument("--atol", type=float, default=1e-2, help="absolute tolerance (default: 0.01)")
    parser.add_argument("--limit", type=int, default=30, help="maximum mismatches to print (default: 30)")
    args = parser.parse_args()
    return compare(args.reference, args.current, args.atol, args.limit)


if __name__ == "__main__":
    raise SystemExit(main())

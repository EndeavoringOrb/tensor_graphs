"""Verification and scoring harness for Section 3 saturation optimizations.

Usage:
  .venv/Scripts/python.exe tests/verify_saturation.py [--generate-gold] [--seq-len 7] [--build]

Workflow:
  1. With the accepted implementation built, generate the reference dump for
     the N to evaluate:
     .venv/Scripts/python.exe tests/verify_saturation.py --generate-gold --seq-len N
  2. Modify planner.hpp / rewrite.hpp and rebuild.
  3. Run verification and get score. Graph equivalence is checked after
     pruneEGraph and EGraph.rebuild(true) compaction:
     .venv/Scripts/python.exe tests/verify_saturation.py --seq-len N

Gold lives in tests/data/gold_saturation/seq<N>/, one folder per N.
Verification refuses to run when the gold for N is missing or incomplete:
gold made from the candidate build would compare it with itself.
The gold metadata records the source tree it was built from; both fingerprints
are printed with the verdict.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "tests" / "data" / "gold_saturation"
CAND_DIR = ROOT / "logs" / "candidate_saturation"
CAND_PREFIX = str(CAND_DIR / "candidate")


def goldPaths(seq_len: int):
    """Each N keeps its own gold: (directory, dump prefix, metadata file)."""
    directory = DATA_DIR / f"seq{seq_len}"
    return directory, str(directory / "gold"), directory / "baseline_score.json"


def sourceFingerprint():
    """HEAD revision plus a hash of uncommitted native source (tracked diff and untracked files)."""
    def git(*command):
        return subprocess.check_output(["git", *command], cwd=ROOT)

    digest = hashlib.sha256(git("diff", "HEAD", "--", "tensor_graphs_cpp"))
    for name in sorted(git("ls-files", "--others", "--exclude-standard", "--", "tensor_graphs_cpp").decode().splitlines()):
        digest.update(name.encode() + bytes(1) + (ROOT / name).read_bytes())
    return {"revision": git("rev-parse", "HEAD").decode().strip(), "native_diff_sha256": digest.hexdigest()}


def expectedBucketSuffixes(seq_len: int):
    if seq_len < 1:
        raise ValueError("seq_len must be at least 1")
    return ["full"] + [
        suffix
        for bucket_seq_len in range(1, seq_len + 1)
        for suffix in (f"prefill{bucket_seq_len}", f"decode{bucket_seq_len}")
    ]


def dumpSuffixes(prefix: str):
    path = Path(prefix)
    return sorted(
        f.name[len(path.name) + 1 : -4]
        for f in path.parent.glob(f"{path.name}-*.txt")
    )


def requireBucketDumps(prefix: str, seq_len: int):
    expected = set(expectedBucketSuffixes(seq_len))
    actual = set(dumpSuffixes(prefix))
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"Expected {2 * seq_len + 1} saturation buckets for {prefix}; "
            f"missing={missing}, unexpected={unexpected}"
        )


def readGraph(path, bucket, graph):
    node_idx = 0
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Saturation dump file not found: {path}")

    for line in p.read_text().splitlines():
        parts = shlex.split(line)
        if not parts:
            continue
        if parts[0] == "B":
            base = ("b", int(parts[2]))
            graph.add_node(base, label=("base",))
            graph.add_edge((bucket, "c", int(parts[1])), base, slots="base")
            continue
        if parts[0] == "C":
            graph.add_node(
                (bucket, "c", int(parts[1])), label=("class", bucket, parts[2])
            )
        elif parts[0] == "N":
            owner = int(parts[1])
            node = (bucket, "n", node_idx)
            node_idx += 1
            content = parts[2]
            scatter = re.search(r"scatter-into:b([0-9]+)", content)
            if scatter:
                content = (
                    content[: scatter.start()]
                    + "scatter-into:base"
                    + content[scatter.end() :]
                )
                base = ("b", int(scatter.group(1)))
                graph.add_node(base, label=("base",))
                graph.add_edge(node, base, slots="scatter-base")
            graph.add_node(node, label=("expression", bucket, content))
            graph.add_edge((bucket, "c", owner), node, slots="member")
            children = {}
            for index, child in enumerate(parts[3:]):
                children.setdefault(int(child), []).append(index)
            for child, indices in children.items():
                graph.add_edge(node, (bucket, "c", child), slots=str(indices))
    return graph


def orderedCertificate(graph):
    """Attempt an exact mapping using class creation order, before VF2 search.

    This is only a fast path: unequal certificates still use general isomorphism.
    Equal certificates give an explicit bijection and compare every labeled edge.
    """
    identities = {}
    counts = Counter()
    for node, data in graph.nodes(data=True):
        if data["label"][0] == "class":
            bucket = data["label"][1]
            identities[node] = ("class", bucket, counts[bucket])
            counts[bucket] += 1
    for node, data in graph.nodes(data=True):
        if data["label"][0] != "base":
            continue
        anchors = tuple(
            sorted(
                identities[source]
                for source, _, edge in graph.in_edges(node, data=True)
                if edge["slots"] == "base"
                and identities.get(source, (None, None))[1] == "full"
            )
        )
        if not anchors:
            return None
        identities[node] = ("base", anchors)
    rows = Counter()
    for node, data in graph.nodes(data=True):
        if data["label"][0] == "expression":
            owners = tuple(
                sorted(
                    identities[source]
                    for source, _, edge in graph.in_edges(node, data=True)
                    if edge["slots"] == "member"
                )
            )
            edges = tuple(
                sorted(
                    (edge["slots"], identities[target])
                    for _, target, edge in graph.out_edges(node, data=True)
                )
            )
            rows[(data["label"], owners, edges)] += 1
        else:
            edges = tuple(
                sorted(
                    (edge["slots"], identities[target])
                    for _, target, edge in graph.out_edges(node, data=True)
                    if edge["slots"] != "member"
                )
            )
            rows[(identities[node], data["label"], edges)] += 1
    return rows


def comparePrefixes(reference_prefix: str, candidate_prefix: str, seq_len: int) -> bool:
    requireBucketDumps(reference_prefix, seq_len)
    requireBucketDumps(candidate_prefix, seq_len)
    suffixes = expectedBucketSuffixes(seq_len)
    before, after = nx.DiGraph(), nx.DiGraph()

    for suffix in suffixes:
        readGraph(f"{reference_prefix}-{suffix}.txt", suffix, before)
        readGraph(f"{candidate_prefix}-{suffix}.txt", suffix, after)

    print(f"Comparing {len(suffixes)} buckets: {suffixes}")
    print(f"Family vertices: reference={len(before)}, candidate={len(after)}", flush=True)

    reference_certificate = orderedCertificate(before)
    if (
        reference_certificate is not None
        and reference_certificate == orderedCertificate(after)
    ):
        print(
            "Verdict: EQUIVALENT (explicit class-order bijection match)",
            flush=True,
        )
        return True

    # Joint color refinement gives VF2 stronger, isomorphism-invariant labels.
    graphs = [before, after]
    palette = {}
    for graph in graphs:
        for node, data in graph.nodes(data=True):
            data["color"] = palette.setdefault(data["label"], len(palette))
    for _ in range(len(before) + len(after)):
        next_palette = {}
        updates = []
        for graph in graphs:
            colors = {}
            for node in graph:
                signature = (
                    graph.nodes[node]["color"],
                    tuple(
                        sorted(
                            (edge["slots"], graph.nodes[other]["color"])
                            for _, other, edge in graph.out_edges(node, data=True)
                        )
                    ),
                    tuple(
                        sorted(
                            (edge["slots"], graph.nodes[other]["color"])
                            for other, _, edge in graph.in_edges(node, data=True)
                        )
                    ),
                )
                colors[node] = next_palette.setdefault(signature, len(next_palette))
            updates.append(colors)
        for graph, colors in zip(graphs, updates):
            nx.set_node_attributes(graph, colors, "color")
        if len(next_palette) == len(palette):
            break
        palette = next_palette
    colors = [
        Counter(data["color"] for _, data in graph.nodes(data=True)) for graph in graphs
    ]
    if colors[0] != colors[1]:
        print(
            "Verdict: NOT EQUIVALENT (different refined partitions)", flush=True
        )
        return False

    expanded = []
    edge_palette = {}
    for graph in graphs:
        colored = nx.DiGraph()
        for node, data in graph.nodes(data=True):
            colored.add_node(("vertex", node), match=("vertex", data["color"]))
        for index, (source, destination, data) in enumerate(graph.edges(data=True)):
            edge = ("edge", index)
            color = edge_palette.setdefault(data["slots"], len(edge_palette))
            colored.add_node(edge, match=("edge", color))
            colored.add_edge(("vertex", source), edge)
            colored.add_edge(edge, ("vertex", destination))
        expanded.append(colored)

    match = nx.vf2pp_is_isomorphic(*expanded, node_label="match")
    print(f"Verdict: {'EQUIVALENT' if match else 'NOT EQUIVALENT'} (VF2++ isomorphism: {match})", flush=True)
    return bool(match)


def run_benchmark(seq_len: int, dump_prefix: str):
    import tensor_graphs

    if dump_prefix:
        Path(dump_prefix).parent.mkdir(parents=True, exist_ok=True)

    res = tensor_graphs.run_saturation_benchmark(seq_len, dump_prefix)
    return res.total_ms, res.full_ms, res.partial_ms, res.full_enodes, res.full_classes


def main():
    parser = argparse.ArgumentParser(description="Section 3 Saturation Verifier and Scorer")
    parser.add_argument("--generate-gold", action="store_true", help="Generate gold baseline dump and timing")
    parser.add_argument("--seq-len", type=int, default=4, help="Sequence length dimension (default: 4)")
    parser.add_argument("--build", action="store_true", help="Rebuild bindings before running")
    args = parser.parse_args()
    if args.seq_len < 1:
        parser.error("--seq-len must be at least 1")

    if args.build:
        print("[verify_saturation] Building bindings...")
        subprocess.check_call([sys.executable, "build.py", "--targets", "bindings"], cwd=str(ROOT))

    if args.generate_gold:
        print(f"[verify_saturation] Generating baseline gold saturation dump (seq_len={args.seq_len})...")
        gold_dir, gold_prefix, gold_meta = goldPaths(args.seq_len)
        gold_dir.mkdir(parents=True, exist_ok=True)
        for f in gold_dir.glob("gold-*.txt"):
            f.unlink()

        total_ms, full_ms, partial_ms, enodes, classes = run_benchmark(args.seq_len, gold_prefix)
        meta = {
            "seq_len": args.seq_len,
            "bucket_count": 2 * args.seq_len + 1,
            "total_ms": total_ms,
            "full_ms": full_ms,
            "partial_ms": partial_ms,
            "full_enodes": enodes,
            "full_classes": classes,
            "generated_at": time.time(),
            "source": sourceFingerprint(),
        }
        gold_meta.write_text(json.dumps(meta, indent=2))
        requireBucketDumps(gold_prefix, args.seq_len)
        print(f"[verify_saturation] Gold dump generated successfully at {gold_dir}")
        print(f"  Full Saturation:    {full_ms:.2f} ms")
        print(f"  Prefill + Decode:   {partial_ms:.2f} ms")
        print(f"  SCORE (Total Time): {total_ms:.2f} ms")
        return

    # Normal verification mode: never regenerate gold here.
    regenerate = (f"Generate it from the accepted implementation's build first: "
                  f".venv/Scripts/python.exe tests/verify_saturation.py --generate-gold --seq-len {args.seq_len}")
    _, gold_prefix, gold_meta = goldPaths(args.seq_len)
    if not gold_meta.exists():
        raise SystemExit(f"[verify_saturation] No gold baseline for seq_len={args.seq_len} ({gold_meta}). {regenerate}")
    b_data = json.loads(gold_meta.read_text())
    if b_data.get("seq_len") != args.seq_len or b_data.get("bucket_count") != 2 * args.seq_len + 1:
        raise SystemExit(f"[verify_saturation] Gold baseline is for seq_len={b_data.get('seq_len')}, "
                         f"not {args.seq_len}. {regenerate}")
    try:
        requireBucketDumps(gold_prefix, args.seq_len)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"[verify_saturation] Gold dumps are incomplete: {error}. {regenerate}")
    baseline_ms = b_data.get("total_ms")
    gold_source = b_data.get("source")
    candidate_source = sourceFingerprint()

    CAND_DIR.mkdir(parents=True, exist_ok=True)
    for f in CAND_DIR.glob("candidate-*.txt"):
        f.unlink()

    print(f"\n[verify_saturation] Running candidate saturation test (seq_len={args.seq_len})...")
    cand_total, cand_full, cand_partial, cand_enodes, cand_classes = run_benchmark(args.seq_len, CAND_PREFIX)

    print("\n[verify_saturation] Verifying pruned, compacted e-graph equivalence against baseline...")
    try:
        equivalent = comparePrefixes(gold_prefix, CAND_PREFIX, args.seq_len)
    except Exception as e:
        print(f"Equivalence comparison error: {e}")
        equivalent = False

    print("\n" + "=" * 60)
    print("             SECTION 3 SATURATION TEST RESULT               ")
    print("=" * 60)
    verdict = "PASS (EGraph Equivalent)" if equivalent else "FAIL (EGraph Divergence Detected)"
    print(f"VERDICT:            {verdict}")
    print(f"CANDIDATE SCORE:    {cand_total:.2f} ms (Full: {cand_full:.2f} ms, Prefill + Decode: {cand_partial:.2f} ms)")
    if baseline_ms:
        speedup = baseline_ms / cand_total if cand_total > 0 else 1.0
        print(f"BASELINE SCORE:     {baseline_ms:.2f} ms")
        print(f"SPEEDUP:            {speedup:.2f}x")
    print(f"E-NODES / CLASSES:  {cand_enodes} enodes / {cand_classes} classes")
    print(f"GOLD SOURCE:        {gold_source or 'unrecorded (gold predates source fingerprints)'}")
    print(f"CANDIDATE SOURCE:   {candidate_source}")
    if gold_source == candidate_source:
        print("NOTE:               gold was generated from this source tree; equivalence is only "
              "meaningful if this is the accepted implementation")
    print("=" * 60 + "\n")

    if not equivalent:
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Benchmark cases for docs/agents/improve_inference_speed.md (fixed contract).

Every case is a slice of one fixed token sequence S (stored with the HF
reference). A case `pp<n>@<c>` prefills n tokens on top of c already-processed
tokens: the timed step processes S[c:c+n] and returns the logits of position
c+n-1. A case `tg<n>@<c>` decodes n tokens one at a time on top of c tokens:
step k (k = 0..n-1) processes S[c+k] and returns the logits of position c+k
(teacher forcing: the next input is always S, never our own sample).
"""

MODEL_NAME = "gemma-3-270m"
MODEL_PATH = "models/google/gemma-3-270m"
REFERENCE_PATH = "tests/inference_reference/gemma-3-270m.json"
# Positions 0..SEQUENCE_LENGTH-1 of S are used; the compiled graph must cover them.
SEQUENCE_LENGTH = 1024

CASES = [
    {"name": "pp8@0", "kind": "pp", "context": 0, "tokens": 8},
    {"name": "tg8@8", "kind": "tg", "context": 8, "tokens": 8},
    {"name": "pp64@64", "kind": "pp", "context": 64, "tokens": 64},
    {"name": "tg64@64", "kind": "tg", "context": 64, "tokens": 64},
    {"name": "pp896@0", "kind": "pp", "context": 0, "tokens": 896},
    {"name": "tg128@896", "kind": "tg", "context": 896, "tokens": 128},
]


def checked_positions(case):
    """Positions whose logits the case compares against the reference."""
    start, count = case["context"], case["tokens"]
    if case["kind"] == "pp":
        return [start + count - 1]
    if case["kind"] == "tg":
        return list(range(start, start + count))
    raise ValueError("Unknown case kind: " + str(case["kind"]))


def all_checked_positions():
    positions = set()
    for case in CASES:
        positions.update(checked_positions(case))
    return sorted(positions)


def validate_cases():
    for case in CASES:
        if case["context"] < 0 or case["tokens"] <= 0:
            raise ValueError("Invalid case: " + case["name"])
        if case["context"] + case["tokens"] > SEQUENCE_LENGTH:
            raise ValueError("Case exceeds the sequence length: " + case["name"])


validate_cases()

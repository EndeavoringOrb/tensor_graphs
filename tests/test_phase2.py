"""
Phase 2: Graph Canonicalization & Identity Tests

Tests:
1. Structural hashing for commutative operations (add/mul)
2. Internal graph hashing with GraphHasher class
3. Graph equivalence checking
4. Subgraph isomorphism search
"""

from tensor_graphs.ir.node import TensorNode
from tensor_graphs.ir.dtypes import DType
from tensor_graphs.ir.hashing import compute_structural_hash, GraphHasher
from tensor_graphs.ops.atomic_types import OpType
from tensor_graphs.optim.symbolic import are_equivalent
from tensor_graphs.ir.graph import find_subgraph, normalize_graph


def test_simple_commutative_hash():
    """Test that a+b and b+a produce the same hash."""
    a = TensorNode(OpType.INPUT, (4, 4), DType.FP32, [], name="a")
    b = TensorNode(OpType.INPUT, (4, 4), DType.FP32, [], name="b")

    # a + b
    res1 = TensorNode(OpType.ADD, (4, 4), DType.FP32, [a, b])
    # b + a
    res2 = TensorNode(OpType.ADD, (4, 4), DType.FP32, [b, a])

    hash1 = compute_structural_hash(res1)
    hash2 = compute_structural_hash(res2)

    print(f"Hash 1 (a+b): {hash1}")
    print(f"Hash 2 (b+a): {hash2}")

    assert hash1 == hash2, f"Hashes do not match: {hash1} != {hash2}"
    print("✓ Simple commutative hash test passed\n")


def test_internal_hash_commutativity():
    """Test GraphHasher with nested commutative operations."""
    a = TensorNode(OpType.INPUT, (4, 4), DType.FP32, [], name="a")
    b = TensorNode(OpType.INPUT, (4, 4), DType.FP32, [], name="b")

    # x = a + b
    x = TensorNode(OpType.ADD, (4, 4), DType.FP32, [a, b], name="x")
    # y = b + a
    y = TensorNode(OpType.ADD, (4, 4), DType.FP32, [b, a], name="y")

    # root = x + y
    root = TensorNode(OpType.ADD, (4, 4), DType.FP32, [x, y], name="root")

    hasher = GraphHasher(root)
    hasher.compute_hash(root)

    hash_x = hasher.hashes[x]
    hash_y = hasher.hashes[y]

    print(f"Hash X (a+b): {hash_x}")
    print(f"Hash Y (b+a): {hash_y}")

    assert hash_x == hash_y, f"Hashes do not match: {hash_x} != {hash_y}"
    print("✓ Internal hash commutativity test passed\n")


def test_equivalence():
    """Test if (a+b)*c is equivalent to a*c + b*c."""
    a = TensorNode(OpType.INPUT, (4, 4), DType.FP32, [], name="a")
    b = TensorNode(OpType.INPUT, (4, 4), DType.FP32, [], name="b")
    c = TensorNode(OpType.INPUT, (4, 4), DType.FP32, [], name="c")

    # (a + b) * c
    res1 = TensorNode(
        OpType.MUL,
        (4, 4),
        DType.FP32,
        [TensorNode(OpType.ADD, (4, 4), DType.FP32, [a, b]), c],
    )

    # a*c + b*c
    res2 = TensorNode(
        OpType.ADD,
        (4, 4),
        DType.FP32,
        [
            TensorNode(OpType.MUL, (4, 4), DType.FP32, [a, c]),
            TensorNode(OpType.MUL, (4, 4), DType.FP32, [b, c]),
        ],
    )

    result = are_equivalent(res1, res2)
    print(f"Are equivalent: {result}")
    assert result, "Graphs should be equivalent"
    print("✓ Equivalence test passed\n")


def test_subgraph_matching():
    """Test finding subgraph (a+b) within larger graph (a+b)*a."""
    a = TensorNode(OpType.INPUT, (4, 4), DType.FP32, [], name="a")
    b = TensorNode(OpType.INPUT, (4, 4), DType.FP32, [], name="b")

    # x = a + b
    sub = TensorNode(OpType.ADD, (4, 4), DType.FP32, [a, b])

    # root = (a + b) * a
    root = TensorNode(
        OpType.MUL,
        (4, 4),
        DType.FP32,
        [TensorNode(OpType.ADD, (4, 4), DType.FP32, [a, b]), a],
    )

    # Normalize both for structural matching
    normalize_graph(sub)
    normalize_graph(root)

    matches = find_subgraph(root, sub)
    print(f"Found {len(matches)} matches for subgraph (a+b)")
    assert len(matches) > 0, "Subgraph should be found"
    print("✓ Subgraph matching test passed\n")


if __name__ == "__main__":
    print("Running Phase 2: Graph Canonicalization & Identity Tests\n")
    print("=" * 60)

    test_simple_commutative_hash()
    test_internal_hash_commutativity()
    test_equivalence()
    test_subgraph_matching()

    print("=" * 60)
    print("All Phase 2 tests passed!")

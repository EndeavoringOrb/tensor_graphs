# +ref counting
# +ref count invalidation
# +memory counting
# +cost
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Set


@dataclass
class ENode:
    op: str
    children: List[int]  # child eclass ids
    mem_size: int
    cost: int
    inplace: bool
    inplace_idx: int


class EGraph:
    def __init__(self):
        self.eclasses: Dict[int, List[ENode]] = defaultdict(list)

    def add_enode(
        self,
        eclass_id: int,
        op: str,
        children: List[int],
        mem_size: int,
        cost: int,
        inplace: bool,
        inplace_idx: int = -1,
    ):
        if inplace and inplace_idx == -1:
            raise RuntimeError(
                f"Must provide inplace_idx if inplace=true, but got inplace_idx={inplace_idx}"
            )
        self.eclasses[eclass_id].append(
            ENode(op, children, mem_size, cost, inplace, inplace_idx)
        )


def build_ref_counts(egraph, selection_map, root):
    ref = defaultdict(int)

    for eclass, sel in selection_map.items():
        node = egraph.eclasses[eclass][sel]
        for c in node.children:
            ref[c] += 1

    # root is "used" once (output)
    ref[root] += 1

    return ref


def compute_peak_memory(egraph: EGraph, selection_map: Dict[int, int], root: int):
    ref = build_ref_counts(egraph, selection_map, root)
    live_mem = 0
    peak_mem = 0

    visited = set()

    def visit(eclass):
        nonlocal live_mem, peak_mem

        if eclass in visited:
            return
        visited.add(eclass)

        node = egraph.eclasses[eclass][selection_map[eclass]]

        # compute children first
        for c in node.children:
            visit(c)

        # allocate this node (unless inplace)
        if node.inplace:
            # reuse child buffer → no allocation
            pass
        else:
            live_mem += node.mem_size
            peak_mem = max(peak_mem, live_mem)

        # consume children
        for i, c in enumerate(node.children):
            ref[c] -= 1

            # if last use, free unless it was reused inplace
            if ref[c] == 0:
                child_node = egraph.eclasses[c][selection_map[c]]

                # IMPORTANT: if this node reused child inplace,
                # that memory is now owned by this node → don't free
                if node.inplace and i == node.inplace_idx:
                    continue

                if not child_node.inplace:
                    live_mem -= child_node.mem_size

    visit(root)

    return peak_mem


def extract_best(egraph: EGraph, root_id: int, max_mem_size: int):
    # state
    selection_map: Dict[int, int] = {}
    path: List[int] = []
    to_process: List[int] = [root_id]
    to_process_enode: List[int] = []
    ref_counts: Dict[int, int] = defaultdict(int)
    need_single_ref: Dict[int, int] = {}
    next_sel: Dict[int, int] = {}

    best = (None, {})

    while True:
        valid = True
        reason = ""
        cost: float = 0.0
        for eclass, enode in selection_map.items():
            cost += egraph.eclasses[eclass][enode].cost

        # descend
        while to_process:
            current = to_process.pop(0)

            if current in selection_map:
                continue

            path.append(current)

            # default selection
            if current in next_sel:
                sel = next_sel.pop(current)
            else:
                sel = 0

            enodes = egraph.eclasses[current]

            if sel >= len(enodes):
                raise RuntimeError(f"Invalid selection index at eclass {current}")

            node = enodes[sel]

            # record selection
            selection_map[current] = sel
            cost += node.cost

            # Update single refs constraints
            if node.inplace:
                child = node.children[node.inplace_idx]
                if child in need_single_ref:
                    valid = False
                    reason = "inplace"
                else:
                    need_single_ref[child] = current

            # update reference counts for children uniformly
            for child in node.children:
                ref_counts[child] += 1
                # ensure inplace validity (no overlapping constraints over counts > 1)
                if child in need_single_ref and ref_counts[child] > 1:
                    valid = False
                    reason = "inplace"

            # Re-verify evaluated cost
            if best[0] is not None and cost > best[0]:
                valid = False
                reason = "cost"

            # if there are more choices later, mark for backtracking
            if len(enodes) > sel + 1:
                if current not in to_process_enode:
                    to_process_enode.append(current)

            # Break prematurely if a constraint was violated (keeps exact symmetries allowing safe rollbacks)
            if not valid:
                break

            # enqueue children (DFS)
            new_to_process = [
                child for child in node.children if child not in selection_map
            ]  # we don't need to descend into children we have already processed
            to_process = new_to_process + to_process

        # reached a full tree
        peak = "N/A"
        if valid:
            peak = compute_peak_memory(egraph, selection_map, root_id)
            if peak > max_mem_size:
                reason = "OOM"
                valid = False

        if valid:
            if best[0] is None or cost < best[0]:
                best = (cost, selection_map.copy())
                print(f"NEW BEST: {cost}")
        print(
            f"{'FOUND' if valid else f'REJECT ({reason})'}, peak={peak}, cost={cost}:",
            selection_map,
        )

        if len(to_process_enode) == 0:
            break

        # backtrack
        while path:
            current = path.pop()

            if current not in selection_map:
                continue
            sel = selection_map[current]
            enodes = egraph.eclasses[current]
            node = enodes[sel]

            # rollback child ref_counts symmetrically
            for child in node.children:
                ref_counts[child] -= 1
                if ref_counts[child] == 0:
                    del ref_counts[child]

            # rollback need_single_ref conditionally using established ownership map
            if node.inplace:
                child = node.children[node.inplace_idx]
                if need_single_ref.get(child) == current:
                    del need_single_ref[child]

            # increment selection if possible
            if sel + 1 < len(enodes):
                # advance choice cleanly for the next forward pass
                next_sel[current] = sel + 1

                # reset deeper selections
                keys_to_delete = [
                    k for k in selection_map if k not in path and k != current
                ]
                for k in keys_to_delete:
                    del selection_map[k]

                # drop exact branch reference since the next iteration acts dynamically
                del selection_map[current]

                # rebuild to_process from this node
                if current in to_process_enode:
                    to_process_enode.remove(current)

                if len(enodes) > sel + 2:
                    to_process_enode.append(current)

                to_process = []
                for eclass in path:
                    p_node = egraph.eclasses[eclass][selection_map[eclass]]
                    new_to_process = [
                        child for child in p_node.children if child not in selection_map
                    ]
                    to_process = new_to_process + to_process
                to_process = [current] + to_process

                break
            else:
                # exhausted this eclass
                if current in selection_map:
                    del selection_map[current]

                if current in to_process_enode:
                    to_process_enode.remove(current)

    return best


# ----------------------------
# Build your example egraph
# ----------------------------
def test0():
    eg = EGraph()

    # inputs
    eg.add_enode(0, "input(a)", [], 4, 0, False)
    eg.add_enode(1, "input(b)", [], 4, 0, False)
    eg.add_enode(2, "input(c)", [], 4, 0, False)

    # b + c
    eg.add_enode(3, "+", [1, 2], 4, 2, False)

    # eclass 4 has TWO enodes:
    #   0: a*(b+c)
    #   1: (a*b) + (a*c)
    eg.add_enode(4, "+", [5, 6], 4, 2, False)  # index 0
    eg.add_enode(4, "*", [0, 3], 4, 3, False)  # index 1

    # a*b
    eg.add_enode(5, "*", [0, 1], 4, 1, True, 0)
    eg.add_enode(5, "*", [0, 1], 4, 3, False)

    # a*c
    eg.add_enode(6, "*", [0, 2], 4, 3, False)

    # ----------------------------
    # Run extraction
    # ----------------------------

    best = extract_best(eg, root_id=4, max_mem_size=32)
    assert best[0] == 5.0
    assert best[1] == {4: 1, 0: 0, 3: 0, 1: 0, 2: 0}

    print(f"\nBest selection map (cost={best[0]}):", best[1])


def test_memory_cost_tradeoff():
    """
    Test that the algorithm chooses a HIGHER cost path to satisfy a Memory constraint.
    """
    eg = EGraph()
    # input
    eg.add_enode(0, "input", [], mem_size=10, cost=0, inplace=False)

    # Path 1: Fast but uses huge intermediate memory
    eg.add_enode(1, "huge_intermediate", [0], mem_size=100, cost=5, inplace=False)
    eg.add_enode(
        2, "fast_out", [1], mem_size=10, cost=5, inplace=False
    )  # index 0 for root

    # Path 2: Slow but does everything inplace (low memory)
    eg.add_enode(
        3, "slow_inplace_1", [0], mem_size=10, cost=20, inplace=True, inplace_idx=0
    )
    eg.add_enode(
        2, "slow_out", [3], mem_size=10, cost=20, inplace=True, inplace_idx=0
    )  # index 1 for root

    # If we have unlimited memory, it should pick Path 1 (Cost = 10)
    best_unlimited = extract_best(eg, root_id=2, max_mem_size=1000)
    assert best_unlimited[0] == 10.0
    assert best_unlimited[1][2] == 0  # picked fast_out

    # If we restrict memory, it MUST pick Path 2 (Cost = 40) to avoid the 100 mem allocation
    best_limited = extract_best(eg, root_id=2, max_mem_size=50)
    assert best_limited[0] == 40.0
    assert best_limited[1][2] == 1  # picked slow_out
    print("\n[PASS] test_memory_cost_tradeoff")


def test_diamond_inplace_conflict():
    """
    Test that an inplace operation is rejected if the child is referenced multiple times.
    """
    eg = EGraph()
    eg.add_enode(0, "input_a", [], mem_size=10, cost=0, inplace=False)

    # Diamond split: both 1 and 2 use 0
    # Node 1 tries to use Node 0 inplace. This SHOULD BE INVALID because Node 2 needs Node 0 intact.
    eg.add_enode(
        1, "op_left", [0], mem_size=10, cost=10, inplace=True, inplace_idx=0
    )  # Invalid
    eg.add_enode(
        1, "op_left_safe", [0], mem_size=10, cost=15, inplace=False
    )  # Valid, but higher cost

    eg.add_enode(2, "op_right", [0], mem_size=10, cost=10, inplace=False)

    # Root combines them
    eg.add_enode(3, "combine", [1, 2], mem_size=10, cost=5, inplace=False)

    best = extract_best(eg, root_id=3, max_mem_size=100)

    # The cost should be 15 (op_left_safe) + 10 (op_right) + 5 (combine) = 30
    # If it incorrectly allowed inplace, cost would be 25.
    assert best[0] == 30.0
    assert best[1][1] == 1  # Forced to pick op_left_safe
    print("\n[PASS] test_diamond_inplace_conflict")


def test_deep_inplace_chain():
    """
    Test that a deep chain of inplace operations computes memory exactly once.
    """
    eg = EGraph()
    eg.add_enode(0, "input", [], mem_size=100, cost=1, inplace=False)

    # Chain of inplace ops
    eg.add_enode(1, "op1", [0], mem_size=100, cost=1, inplace=True, inplace_idx=0)
    eg.add_enode(2, "op2", [1], mem_size=100, cost=1, inplace=True, inplace_idx=0)
    eg.add_enode(3, "op3", [2], mem_size=100, cost=1, inplace=True, inplace_idx=0)

    best = extract_best(eg, root_id=3, max_mem_size=105)

    # Because it's a direct chain of inplace ops, peak memory should just be the initial 100 allocation.
    # If memory was double-counted, peak memory would be > 105 and it would fail.
    assert best[0] == 4.0
    assert best[1] != {}
    print("\n[PASS] test_deep_inplace_chain")


def test_unsatisfiable_memory():
    """
    Test when no possible execution path can fit in memory.
    """
    eg = EGraph()
    eg.add_enode(0, "input", [], mem_size=100, cost=0, inplace=False)
    eg.add_enode(1, "out1", [0], mem_size=100, cost=5, inplace=False)
    eg.add_enode(1, "out2", [0], mem_size=100, cost=5, inplace=True, inplace_idx=0)

    # Max mem is 10. The input alone takes 100.
    best = extract_best(eg, root_id=1, max_mem_size=10)

    # Should return (None, {})
    assert best[0] is None
    assert len(best[1]) == 0
    print("\n[PASS] test_unsatisfiable_memory")


if __name__ == "__main__":
    print("Running test0...")
    test0()
    print("Running edge cases...")
    test_memory_cost_tradeoff()
    test_diamond_inplace_conflict()
    test_deep_inplace_chain()
    test_unsatisfiable_memory()
    print("\nALL TESTS PASSED!")

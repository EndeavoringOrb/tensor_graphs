# +ref counting
# +ref count invalidation
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Set


@dataclass
class ENode:
    op: str
    children: List[int]  # child eclass ids
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
        inplace: bool,
        inplace_idx: int = -1,
    ):
        if inplace and inplace_idx == -1:
            raise RuntimeError(f"Must provide inplace_idx if inplace=true, but got inplace_idx={inplace_idx}")
        self.eclasses[eclass_id].append(ENode(op, children, inplace, inplace_idx))


def extract_all(egraph: EGraph, root_id: int):
    results = []

    # state
    selection_map: Dict[int, int] = {}
    path: List[int] = []
    to_process: List[int] = [root_id]
    to_process_enode: List[int] = []
    ref_counts: Dict[int, int] = defaultdict(int)
    need_single_ref: Set[int] = set()

    while True:
        valid = True

        # descend
        while to_process:
            current = to_process.pop(0)
            path.append(current)

            # default selection
            sel = selection_map.get(current, 0)
            enodes = egraph.eclasses[current]

            if sel >= len(enodes):
                raise RuntimeError(f"Invalid selection index at eclass {current}")

            node = enodes[sel]

            if node.inplace:
                if node.children[node.inplace_idx] in need_single_ref:
                    valid = False
                    break
                need_single_ref.add(node.children[node.inplace_idx])

            # record selection
            selection_map[current] = sel

            # if there are more choices later, mark for backtracking
            if len(enodes) > sel + 1:
                if current not in to_process_enode:
                    to_process_enode.append(current)

            # update reference counts for children
            for child in node.children:
                ref_counts[child] += 1
                # ensure inplace validity
                if child in need_single_ref and ref_counts[child] > 1:
                    valid = False
                    break

            # enqueue children (DFS)
            new_to_process = [
                child for child in node.children if child not in selection_map
            ]  # we don't need to descend into children we have already processed
            to_process = new_to_process + to_process

        # reached a full tree
        results.append(selection_map.copy())
        print(f"FOUND valid={valid}:", selection_map)

        if len(to_process_enode) == 0:
            break

        # backtrack
        while path:
            current = path.pop()

            sel = selection_map[current]
            enodes = egraph.eclasses[current]
            node = enodes[sel]

            # rollback child ref_counts
            for child in node.children:
                ref_counts[child] -= 1
                if ref_counts[child] == 0:
                    del ref_counts[child]
            
            # rollback need_single_ref
            if node.inplace and node.children[node.inplace_idx] in need_single_ref:
                need_single_ref.remove(node.children[node.inplace_idx])

            # increment selection if possible
            if sel + 1 < len(enodes):
                # advance choice
                selection_map[current] = sel + 1

                # reset deeper selections
                keys_to_delete = [
                    k for k in selection_map if k not in path and k != current
                ]
                for k in keys_to_delete:
                    del selection_map[k]

                # rebuild to_process from this node
                node = enodes[sel + 1]

                if current in to_process_enode:
                    to_process_enode.remove(current)

                if len(enodes) > sel + 2:
                    to_process_enode.append(current)

                to_process = []
                for eclass in path:
                    node = egraph.eclasses[eclass][selection_map[eclass]]
                    new_to_process = [
                        child for child in node.children if child not in selection_map
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

    return results


# ----------------------------
# Build your example egraph
# ----------------------------

eg = EGraph()

# inputs
eg.add_enode(0, "input(a)", [], False)
eg.add_enode(1, "input(b)", [], False)
eg.add_enode(2, "input(c)", [], False)

# b + c
eg.add_enode(3, "+", [1, 2], False)

# eclass 4 has TWO enodes:
#   0: a*(b+c)
#   1: (a*b) + (a*c)
eg.add_enode(4, "*", [0, 3], False)  # index 0
eg.add_enode(4, "+", [5, 6], False)  # index 1

# a*b
eg.add_enode(5, "*", [0, 1], True, 0)
eg.add_enode(5, "*", [0, 1], False)

# a*c
eg.add_enode(6, "*", [0, 2], False)


# ----------------------------
# Run extraction
# ----------------------------

all_trees = extract_all(eg, root_id=4)

print("\nTotal trees:", len(all_trees))

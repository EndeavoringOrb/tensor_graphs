# +ref counting
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class ENode:
    op: str
    children: List[int]  # child eclass ids
    inplace: bool


class EGraph:
    def __init__(self):
        self.eclasses: Dict[int, List[ENode]] = defaultdict(list)

    def add_enode(self, eclass_id: int, op: str, children: List[int], inplace: bool):
        self.eclasses[eclass_id].append(ENode(op, children, inplace))


def extract_all(egraph: EGraph, root_id: int):
    results = []

    # state
    selection_map: Dict[int, int] = {}
    path: List[int] = []
    to_process: List[int] = [root_id]
    to_process_enode: List[int] = []
    ref_counts: Dict[int, int] = defaultdict(int)

    while True:
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

            # record selection
            selection_map[current] = sel

            # if there are more choices later, mark for backtracking
            if len(enodes) > sel + 1:
                if current not in to_process_enode:
                    to_process_enode.append(current)

            # update reference counts for children
            for child in node.children:
                ref_counts[child] += 1

            # enqueue children (DFS)
            new_to_process = [child for child in node.children if child not in selection_map] # we don't need to descend into children we have already processed
            to_process = new_to_process + to_process

        # reached a full tree
        results.append(selection_map.copy())
        print("FOUND:", selection_map)

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

                # rebuild forward state
                for child in node.children:
                    ref_counts[child] += 1

                path.append(current)
                to_process = []
                for eclass in path:
                    node = egraph.eclasses[eclass][selection_map[eclass]]
                    new_to_process = [child for child in node.children if child not in selection_map]
                    to_process = new_to_process + to_process

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
eg.add_enode(5, "*", [0, 1], False)
eg.add_enode(5, "*", [0, 1], True)

# a*c
eg.add_enode(6, "*", [0, 2], False)


# ----------------------------
# Run extraction
# ----------------------------

all_trees = extract_all(eg, root_id=4)

print("\nTotal trees:", len(all_trees))

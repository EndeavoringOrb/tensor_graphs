from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class ENode:
    op: str
    children: List[int]  # child eclass ids


class EGraph:
    def __init__(self):
        self.eclasses: Dict[int, List[ENode]] = defaultdict(list)

    def add_enode(self, eclass_id: int, op: str, children: List[int]):
        self.eclasses[eclass_id].append(ENode(op, children))


def extract_all(egraph: EGraph, root_id: int):
    results = []

    # state
    selection_map: Dict[int, int] = {}
    path: List[int] = []
    to_process: List[int] = [root_id]
    to_process_enode: List[int] = []

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

            # enqueue children (DFS-like but using list)
            to_process = node.children + to_process

        # reached a full tree
        results.append(selection_map.copy())
        print("FOUND:", selection_map)

        if len(to_process_enode) == 0:
            break

        # backtrack
        while path:
            current = path.pop()

            # increment selection if possible
            sel = selection_map[current]
            enodes = egraph.eclasses[current]

            if sel + 1 < len(enodes):
                # advance choice
                selection_map[current] = sel + 1

                # reset deeper selections
                keys_to_delete = [k for k in selection_map if k not in path and k != current]
                for k in keys_to_delete:
                    del selection_map[k]

                # rebuild to_process from this node
                node = enodes[sel + 1]

                if current in to_process_enode:
                    to_process_enode.remove(current)

                if len(enodes) > sel + 2:
                    to_process_enode.append(current)

                to_process = node.children.copy()
                path.append(current)

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
eg.add_enode(0, "input(a)", [])
eg.add_enode(1, "input(b)", [])
eg.add_enode(2, "input(c)", [])

# b + c
eg.add_enode(3, "+", [1, 2])

# eclass 4 has TWO enodes:
#   0: a*(b+c)
#   1: (a*b) + (a*c)
eg.add_enode(4, "*", [0, 3])     # index 0
eg.add_enode(4, "+", [5, 6])     # index 1

# a*b and a*c
eg.add_enode(5, "*", [0, 1])
eg.add_enode(6, "*", [0, 2])


# ----------------------------
# Run extraction
# ----------------------------

all_trees = extract_all(eg, root_id=4)

print("\nTotal trees:", len(all_trees))
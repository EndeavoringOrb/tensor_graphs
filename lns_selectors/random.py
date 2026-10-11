"""Random dependency-aware neighborhood selection."""

import random


class RandomNeighborhoodSelector:
    """Select a random connected set of primary decision groups."""

    def __init__(self, target_size=15, max_hops=2, seed=None):
        self.target_size = max(1, int(target_size))
        self.max_hops = max(0, int(max_hops))
        self.random = random.Random(seed)

    def selectNeighborhood(self, context):
        metadata = context.get("metadata", {})
        groups = metadata.get("groups", {})
        adjacency = metadata.get("adjacency", {})
        candidates = [
            key
            for key, group in groups.items()
            if group.get("selectable", False)
        ]
        if not candidates:
            candidates = list(groups)
        if not candidates:
            return set()

        seeds = [
            key
            for key in candidates
            if groups.get(key, {}).get("kind") == "selection"
            and int(groups[key].get("choice_count", 0)) > 1
        ]
        seed = self.random.choice(seeds or candidates)
        selected = {seed}
        frontier = [seed]

        for _ in range(self.max_hops):
            next_frontier = []
            for current in frontier:
                neighbors = list(adjacency.get(current, ()))
                self.random.shuffle(neighbors)
                for neighbor in neighbors:
                    if neighbor in selected or neighbor not in groups:
                        continue
                    selected.add(neighbor)
                    next_frontier.append(neighbor)
                    if len(selected) >= self.target_size:
                        return selected
            frontier = next_frontier
            if not frontier:
                break

        return selected

    def selectUnfrozenNodes(
        self,
        bucket,
        selection_map,
        order,
        start_times,
        end_times,
        slack,
        critical_nodes,
        classes_by_id,
        iteration,
    ):
        """Compatibility adapter for the previous selector API."""
        active = set(selection_map)
        multi_choice = [
            cid
            for cid in order
            if cid in active and len(classes_by_id[cid].get("enodes", [])) > 1
        ]
        pool = multi_choice or list(active)
        if not pool:
            return set()
        seed = self.random.choice(pool)
        selected = {seed}
        frontier = [seed]
        parents = {cid: [] for cid in active}
        children = {cid: [] for cid in active}
        for cid in active:
            index = selection_map[cid]
            enode = classes_by_id[cid]["enodes"][index]
            for child in enode.get("children", []):
                if child in active:
                    children[cid].append(child)
                    parents[child].append(cid)
        for _ in range(self.max_hops):
            next_frontier = []
            for current in frontier:
                for neighbor in parents.get(current, []) + children.get(current, []):
                    if neighbor not in selected:
                        selected.add(neighbor)
                        next_frontier.append(neighbor)
                        if len(selected) >= self.target_size:
                            return selected
            frontier = next_frontier
        return selected


RandomSubgraphSelector = RandomNeighborhoodSelector


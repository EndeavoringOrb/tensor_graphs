"""Graph neural neighborhood selector and model definition.

The training entry point lives in the root-level ``train_gnn.py`` script so
this module remains focused on inference and model architecture.
"""

import math
import random

import torch
from torch import nn

from .random import RandomNeighborhoodSelector


class GraphMessageLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.message = nn.Linear(hidden_dim, hidden_dim)
        self.update = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, node_state, edge_index):
        messages = torch.zeros_like(node_state)
        if edge_index.numel() > 0:
            source = edge_index[0]
            target = edge_index[1]
            messages.index_add_(0, target, self.message(node_state[source]))
            degree = torch.zeros(
                node_state.shape[0], device=node_state.device, dtype=node_state.dtype
            )
            degree.index_add_(
                0, target, torch.ones_like(target, dtype=node_state.dtype)
            )
            messages = messages / degree.clamp_min(1.0).unsqueeze(-1)
        return self.update(messages, node_state)


class NeighborhoodGnn(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64, layers=3):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList(
            [GraphMessageLayer(hidden_dim) for _ in range(max(1, layers))]
        )
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.runtime = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def encode(self, node_features, edge_index):
        state = self.encoder(node_features)
        for layer in self.layers:
            state = layer(state, edge_index)
        return state

    def predictRuntimeFromState(self, state, neighborhood_mask):
        mask = neighborhood_mask.to(dtype=state.dtype)
        if mask.ndim == 1:
            selected = (state * mask.unsqueeze(-1)).sum(dim=0)
            selected = selected / mask.sum().clamp_min(1.0)
        elif mask.ndim == 2:
            selected = mask @ state
            selected = selected / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        else:
            raise ValueError("neighborhood_mask must be a 1D or 2D tensor")
        global_state = state.mean(dim=0)
        if selected.ndim == 2:
            global_state = global_state.expand(selected.shape[0], -1)
        return self.runtime(torch.cat((selected, global_state), dim=-1)).squeeze(-1)

    def predictRuntime(self, node_features, edge_index, neighborhood_mask):
        return self.predictRuntimeFromState(
            self.encode(node_features, edge_index), neighborhood_mask
        )

    def forward(self, node_features, edge_index):
        state = self.encode(node_features, edge_index)
        return self.score(state).squeeze(-1)


class GnnNeighborhoodSelector:
    def __init__(
        self,
        model_path=None,
        target_size=15,
        hidden_dim=64,
        layers=3,
        seed=None,
        device=None,
    ):
        self.target_size = max(1, int(target_size))
        self.hidden_dim = int(hidden_dim)
        self.layers = int(layers)
        self.random = random.Random(seed)
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = None
        self.model_path = model_path
        self.runtime_available = False
        if model_path:
            self.loadModel(model_path)

    def loadModel(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        feature_dim = int(checkpoint["feature_dim"])
        self.model = NeighborhoodGnn(
            feature_dim, checkpoint.get("hidden_dim", self.hidden_dim), checkpoint.get("layers", self.layers)
        ).to(self.device)
        # Former checkpoints do not have a runtime head.  Shared encoder
        # weights can still be loaded while a runtime checkpoint is trained.
        state_dict = checkpoint["state_dict"]
        self.runtime_available = any(key.startswith("runtime.") for key in state_dict)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        self.model_path = model_path

    def _tensorize(self, context):
        metadata = context.get("metadata", {})
        groups = metadata.get("groups", {})
        keys = list(groups)
        feature_dim = int(metadata.get("feature_dim", 0))
        if not keys or feature_dim <= 0:
            return keys, None, None
        features = []
        for key in keys:
            row = list(groups[key].get("features", []))[:feature_dim]
            row.extend([0.0] * (feature_dim - len(row)))
            features.append([0.0 if not math.isfinite(float(value)) else float(value) for value in row])
        index = {key: position for position, key in enumerate(keys)}
        edges = []
        for source, neighbors in metadata.get("adjacency", {}).items():
            for target in neighbors:
                if source in index and target in index:
                    edges.append((index[source], index[target]))
        node_features = torch.tensor(features, dtype=torch.float32, device=self.device)
        edge_index = torch.tensor(edges, dtype=torch.long, device=self.device).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long, device=self.device)
        return keys, node_features, edge_index

    def selectNeighborhood(self, context):
        keys, node_features, edge_index = self._tensorize(context)
        groups = context.get("metadata", {}).get("groups", {})
        candidates = [key for key in keys if groups[key].get("selectable", False)]
        if not candidates:
            candidates = keys
        if not candidates:
            return set()
        if self.model is None or node_features is None:
            return RandomNeighborhoodSelector(self.target_size, seed=self.random.randrange(2**31)).selectNeighborhood(context)

        with torch.inference_mode():
            if not self.runtime_available:
                # Preserve behavior for checkpoints from the former
                # node-classification model.
                scores = self.model(node_features, edge_index).detach().cpu().tolist()
                ranked = sorted(
                    candidates,
                    key=lambda key: scores[keys.index(key)],
                    reverse=True,
                )
                return set(ranked[: self.target_size])

            state = self.model.encode(node_features, edge_index)
            candidate_positions = [keys.index(key) for key in candidates]
            # Evaluate singleton neighborhoods without materializing an
            # O(num_candidates * num_nodes) mask matrix.
            selected_state = state[candidate_positions]
            global_state = state.mean(dim=0).expand(len(candidate_positions), -1)
            scores = (
                self.model.runtime(torch.cat((selected_state, global_state), dim=-1))
                .squeeze(-1)
                .detach()
                .cpu()
                .tolist()
            )
        ranked = sorted(
            zip(candidates, scores), key=lambda item: item[1]
        )
        return {key for key, _ in ranked[: self.target_size]}

    def selectUnfrozenNodes(self, *args):
        """Compatibility adapter for the former eclass-based API."""
        if len(args) == 1 and isinstance(args[0], dict):
            return self.selectNeighborhood(args[0])
        if len(args) >= 9:
            classes_by_id = args[7]
            selection_map = args[1]
            order = args[2]
            return RandomNeighborhoodSelector(self.target_size).selectUnfrozenNodes(
                args[0], selection_map, order, args[3], args[4], args[5], args[6], classes_by_id, args[8]
            )
        raise TypeError("Expected a full model context")

NeuralNeighborhoodSelector = GnnNeighborhoodSelector

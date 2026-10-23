"""
encoder.py
──────────
Improved Graph-to-Sequence encoder.

Changes over the original
─────────────────────────
1. 4 GIN layers instead of 2 — captures longer dependency chains in the DFG.
2. Residual connections between layers — stabilises gradients in deeper nets.
3. LayerNorm after each layer — keeps activations well-scaled.
4. Global graph embedding via global_add_pool — produces a single vector
   summarising the whole graph, which is prepended to the node embeddings
   before they are passed to the decoder as cross-attention memory.
   This gives the decoder a global context at every step so it cannot ignore
   the graph even if cross-attention collapses to a few nodes.
5. Linear projection to align the global vector dimension with hidden_dim
   before prepending.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, global_add_pool


def _make_mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.ReLU(),
        nn.Linear(out_dim, out_dim),
    )


class G2SEncoder(nn.Module):
    """
    Parameters
    ──────────
    node_feat_dim : int  raw node feature width
    hidden_dim    : int  hidden / output width for all layers
    num_layers    : int  number of GIN message-passing layers (default 4)
    """

    def __init__(self, node_feat_dim: int, hidden_dim: int, num_layers: int = 4):
        super().__init__()

        # Input projection: bring raw features up to hidden_dim
        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)

        # GIN layers — each owns its own MLP
        self.convs = nn.ModuleList([
            GINConv(_make_mlp(hidden_dim, hidden_dim))
            for _ in range(num_layers)
        ])

        # LayerNorm after each conv (applied over the feature dim)
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])

        # Project the global pooled vector to hidden_dim
        # (global_add_pool output is already hidden_dim, but kept explicit)
        self.global_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, batch):
        """
        Parameters
        ──────────
        x          : FloatTensor [TotalNodes, node_feat_dim]
        edge_index : LongTensor  [2, TotalEdges]
        batch      : LongTensor  [TotalNodes]   PyG graph assignment vector

        Returns
        ───────
        memory : FloatTensor [B, 1 + max_nodes, hidden_dim]
            Per-graph node embeddings padded to the same node count, with a
            prepended global graph embedding token at position 0.
            The calling code receives this ready to pass as decoder memory.
        mask   : BoolTensor [B, 1 + max_nodes]
            True where the position is padding (to be ignored by attention).
            Position 0 (global token) is always False (never masked).
        """
        # ── Node-level encoding ────────────────────────────────────────
        h = self.input_proj(x).relu()   # [TotalNodes, H]

        for conv, norm in zip(self.convs, self.norms):
            h = norm(conv(h, edge_index).relu() + h)   # residual + norm

        # ── Graph-level global embedding ───────────────────────────────
        # global_add_pool sums node embeddings per graph → [B, H]
        g = self.global_proj(global_add_pool(h, batch)).relu()  # [B, H]

        # ── Split flat node embeddings back into per-graph tensors ─────
        num_graphs = int(batch.max().item()) + 1
        per_graph  = [h[batch == i] for i in range(num_graphs)]

        max_nodes = max(t.shape[0] for t in per_graph)
        H = h.shape[1]

        # memory: slot 0 = global token, slots 1..max_nodes = node embeddings
        memory = torch.zeros(num_graphs, 1 + max_nodes, H, device=h.device)
        mask   = torch.ones(num_graphs, 1 + max_nodes, dtype=torch.bool,
                            device=h.device)

        # Global token at position 0 — never masked
        memory[:, 0, :] = g
        mask[:, 0]      = False

        # Node embeddings at positions 1..n+1
        for i, t in enumerate(per_graph):
            n = t.shape[0]
            memory[i, 1:n + 1] = t
            mask[i, 1:n + 1]   = False

        return memory, mask
"""Gated Graph Convolution Layers.

This module implements gated graph convolution layers based on:
    Bresson, X., & Laurent, T. (2017).
    "Residual Gated Graph ConvNets"
    arXiv:1711.07553
    https://arxiv.org/abs/1711.07553

The core mechanism uses edge gates computed from node features to control
message passing, combined with residual connections for deep architectures.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor
from torch.nn import Parameter, Sigmoid
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor, PairTensor


class ResGatedConv(MessagePassing):
    """Gated graph convolution for gene-to-reaction heterogeneous bipartite graphs.

    Implements the Residual Gated Graph ConvNet mechanism with:
    - Edge gates computed as: gate = act(q_i + k_j)
    - Gated messages: message = gate * v_j
    - Residual connections via identity skip (reaction features bypass message path)

    This variant uses a bottleneck architecture (project to hidden dimension,
    process, then project back) and requires even output channels for use with
    antisymmetric flux prediction models.

    Graph Structure:
        Source nodes: Genes (provide expression data)
        Target nodes: Reactions (receive aggregated gene information)
        Messages flow: Genes → Reactions

    Args:
        in_channels: Tuple of (gene_dim, reaction_dim) input dimensions
        out_channels: Output dimension (must be even for antisymmetric operations)
        hidden_channels: Hidden bottleneck dimension (default: 64)
        act: Activation function for gates (default: Sigmoid)
        edge_dim: Optional edge feature dimension
        root_weight: Whether to use residual skip connection (default: True)
        bias: Whether to add bias terms (default: True)
        aggr: Aggregation method for message passing (default: "add"; could use "mean")
        **kwargs: Additional arguments for MessagePassing

    Reference:
        Bresson & Laurent (2017). Residual Gated Graph ConvNets. arXiv:1711.07553
    """

    def __init__(
        self,
        in_channels: tuple[int, int],  # (gene_dim, reaction_dim)
        out_channels: int,  # reaction output dimension
        hidden_channels: int = 64,
        act: Callable | None = Sigmoid(),
        edge_dim: int | None = None,  # Optional edge feature dimension
        root_weight: bool = True,
        bias: bool = True,
        aggr: str = "add",
        **kwargs,
    ):

        kwargs["aggr"] = aggr
        super().__init__(**kwargs)

        assert out_channels % 2 == 0, "out_channels must be even"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act
        self.edge_dim = edge_dim
        self.root_weight = root_weight

        if self.root_weight and in_channels[1] != out_channels:
            raise ValueError(
                "root_weight requires in_channels[1] == out_channels for identity skip"
            )

        edge_dim = edge_dim if edge_dim is not None else 0
        self.gate_query = Linear(
            hidden_channels + edge_dim, hidden_channels
        )  # Target (reaction) queries
        self.gate_key = Linear(
            hidden_channels + edge_dim, hidden_channels
        )  # Source (gene) keys
        self.gate_value = Linear(
            hidden_channels + edge_dim, hidden_channels
        )  # Source (gene) values
        self.up_project = Linear(hidden_channels, out_channels)
        self.dn_project_src = Linear(in_channels[0], hidden_channels)  # Gene projection
        self.dn_project_tgt = Linear(
            in_channels[1], hidden_channels
        )  # Reaction projection

        if bias:
            self.bias = Parameter(Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.gate_query.reset_parameters()
        self.gate_key.reset_parameters()
        self.gate_value.reset_parameters()
        self.up_project.reset_parameters()
        self.dn_project_src.reset_parameters()
        self.dn_project_tgt.reset_parameters()
        if self.bias is not None:
            zeros(self.bias)

    def forward(
        self,
        x: PairTensor,
        edge_index: Adj,
        edge_attr: OptTensor = None,
    ) -> Tensor:
        """Forward pass with gated message passing and residual connection.

        Args:
            x: PairTensor of node features (x_src, x_tgt) where:
               - x_src: Gene node features (source)
               - x_tgt: Reaction node features (target)
            edge_index: Graph connectivity in COO format (genes → reactions)
            edge_attr: Optional edge attributes

        Returns:
            Updated reaction node features
        """
        x_src, x_tgt = x  # genes (source), reactions (target)

        # Project to hidden dimension
        x_src_hidden = self.dn_project_src(x_src)
        x_tgt_hidden = self.dn_project_tgt(x_tgt)

        # If no edge features, compute key, query, value in node space
        if self.edge_dim is None:
            q = self.gate_query(x_tgt_hidden)  # Query from reactions (target)
            k = self.gate_key(x_src_hidden)  # Key from genes (source)
            v = self.gate_value(x_src_hidden)  # Value from genes (source)
        else:
            # With edge features, we compute k, q, v in message function
            q, k, v = x_tgt_hidden, x_src_hidden, x_src_hidden

        # Propagate messages from genes to reactions
        out = self.propagate(edge_index, q=q, k=k, v=v, edge_attr=edge_attr)

        # Project back to output dimension
        out = self.up_project(out)

        # Add residual connection from original reaction features
        if self.root_weight:
            out = out + x_tgt

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(
        self, q_i: Tensor, k_j: Tensor, v_j: Tensor, edge_attr: OptTensor
    ) -> Tensor:
        """Compute gated messages for each edge.

        Implements: gate = act(q_i + k_j), message = gate * v_j

        Args:
            q_i: Query vectors from target nodes (reactions)
            k_j: Key vectors from source nodes (genes)
            v_j: Value vectors from source nodes (genes)
            edge_attr: Optional edge attributes

        Returns:
            Gated messages for aggregation
        """
        assert (edge_attr is not None) == (self.edge_dim is not None)

        if edge_attr is not None:
            q_i = self.gate_query(torch.cat([q_i, edge_attr], dim=-1))
            k_j = self.gate_key(torch.cat([k_j, edge_attr], dim=-1))
            v_j = self.gate_value(torch.cat([v_j, edge_attr], dim=-1))

        return self.act(q_i + k_j) * v_j

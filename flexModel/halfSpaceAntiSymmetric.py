"""Antisymmetric pairwise scoring modules.

This module contains two antisymmetric scorers used for reaction-pair modeling:
- ``BiLinAntisymmetricFunc``: compact bilinear antisymmetric head.
- ``AntisymmetricFunc``: deeper residual odd architecture with optional
  dedicated self-term pathway.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_pair_inputs(x1: Tensor, x2: Tensor, expected_d: int, where: str) -> None:
    """Validate paired tensor inputs used by antisymmetric modules."""
    if x1.ndim != 3 or x2.ndim != 3:
        raise ValueError(
            f"{where} expects 3D tensors of shape (batch, n_reactions, d), "
            f"got x1.ndim={x1.ndim}, x2.ndim={x2.ndim}."
        )
    if x1.shape != x2.shape:
        raise ValueError(
            f"{where} requires x1 and x2 to have identical shape, "
            f"got {x1.shape} vs {x2.shape}."
        )
    if x1.shape[-1] != expected_d:
        raise ValueError(
            f"{where} expected last dimension d={expected_d}, "
            f"got d={x1.shape[-1]}."
        )


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Linear -> activation -> dropout, with residual connection.

    Args:
        k: Feature width.
        activation: One of ``"relu"``, ``"softsign"``, ``"tanh"``.
        dropout: Dropout probability applied inside the residual branch.
        bias: Whether the linear layer includes a bias term.  Must be
            ``False`` for layers on the antisymmetric (cross) pathway.
    """

    _ACTIVATIONS = {
        "relu": nn.ReLU,
        "softsign": nn.Softsign,
        "tanh": nn.Tanh,
    }

    def __init__(
        self,
        k: int,
        activation: str = "relu",
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if activation not in self._ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation '{activation}'. "
                f"Choose from {set(self._ACTIVATIONS)}."
            )
        self.linear = nn.Linear(k, k, bias=bias)
        self.act = self._ACTIVATIONS[activation]()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.xavier_uniform_(self.linear.weight, gain=0.1)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, h: Tensor) -> Tensor:
        return h + self.dropout(self.act(self.linear(h)))


class AntisymmetricLayer(nn.Module):
    """Maps ``(x1, x2) -> K`` antisymmetric features via a ``(z, s)``
    bilinear form where ``z = x1 - x2`` and ``s = x1 + x2``.

    Each output channel *k* computes::

        h_k = w_k^T z  +  (z^T P_k)(s^T Q_k)

    The linear term captures first-order antisymmetry; the bilinear term
    captures both self-quadratic (x1^T S x1 - x2^T S x2) and cross-bilinear
    interactions in a single low-rank form.

    Args:
        d: Input feature dimension.
        k: Number of output channels.
        rank: Low-rank factor size for the bilinear term.
    """

    def __init__(self, d: int, k: int, rank: int = 16) -> None:
        super().__init__()
        self.d = d
        self.k = k
        self.rank = rank
        self.W_lin = nn.Linear(d, k, bias=False)
        self.P = nn.Parameter(torch.randn(k, d, rank) * (rank * d) ** -0.5)
        self.Q = nn.Parameter(torch.randn(k, d, rank) * (rank * d) ** -0.5)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        _validate_pair_inputs(x1, x2, self.d, "AntisymmetricLayer.forward")
        z = x1 - x2
        s = x1 + x2
        lin = self.W_lin(z)                            # B x nR x K
        zP = torch.einsum("bnd,kdr->bnkr", z, self.P)
        sQ = torch.einsum("bnd,kdr->bnkr", s, self.Q)
        bili = (zP * sQ).sum(-1)                       # B x nR x K
        return lin + bili


# ---------------------------------------------------------------------------
# Legacy bilinear scorer
# ---------------------------------------------------------------------------

class BiLinAntisymmetricFunc(nn.Module):
    """Antisymmetric bilinear function with a learnable scalar gate.

    For each channel *k* this models an antisymmetric bilinear form
    ``x1^T M_k x2`` where ``M_k = U_k V_k^T - V_k U_k^T``.
    The forward output is::

        g(x1) - g(x2) + sum_k alpha_k * x1^T M_k x2

    Args:
        d: Feature dimension of each reaction embedding.
        k: Number of antisymmetric channels.
        rank: Low-rank factor size for each channel.
    """

    def __init__(self, d: int, k: int = 8, rank: int = 64) -> None:
        super().__init__()
        self.d = d
        self.k = k
        self.rank = rank
        self.g = nn.Sequential(
            nn.Linear(d, k),
            nn.ReLU(),
            nn.Linear(k, k),
            nn.ReLU(),
            nn.Linear(k, 1),
        )
        self.U = nn.Parameter(torch.randn(k, d, rank))
        self.V = nn.Parameter(torch.randn(k, d, rank))
        self.alpha = nn.Parameter(torch.ones(k))

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """Forward pass computing antisymmetric bilinear form.

        Args:
            x1: Tensor of shape ``(batch, n_reactions, d)``.
            x2: Tensor of shape ``(batch, n_reactions, d)``.

        Returns:
            Tensor of shape ``(batch, n_reactions)``.
        """
        _validate_pair_inputs(x1, x2, self.d, "BiLinAntisymmetricFunc.forward")

        g_out1 = self.g(x1).squeeze(-1)
        g_out2 = self.g(x2).squeeze(-1)

        x1_u = torch.einsum("bnd,kdr->bnkr", x1, self.U)
        x2_v = torch.einsum("bnd,kdr->bnkr", x2, self.V)
        x1_v = torch.einsum("bnd,kdr->bnkr", x1, self.V)
        x2_u = torch.einsum("bnd,kdr->bnkr", x2, self.U)
        bili_per_k = (x1_u * x2_v - x1_v * x2_u).sum(dim=-1)
        bili = torch.einsum("bnk,k->bn", bili_per_k, self.alpha)
        return g_out1 - g_out2 + bili


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

class AntisymmetricFunc(nn.Module):
    """Stacked antisymmetric function: ``(x1, x2) -> scalar``.

    Architecture::

        Cross pathway (odd):
            AntisymmetricLayer -> [ResidualBlock x n_layers_cross] -> head

        Self pathway (unconstrained, antisymmetrised by subtraction):
            g_proj -> [ResidualBlock x n_layers_self] -> g_head
            output: g(x1) - g(x2)

        Final: g(x1) - g(x2) + cross(x1, x2)

    Antisymmetry is guaranteed by construction:

    - The cross pathway is odd because every component (``AntisymmetricLayer``,
      bias-free linear maps, odd activations, residual sums) preserves
      oddness under ``(x1, x2) -> (x2, x1)``.
    - The self pathway is antisymmetrised by the ``g(x1) - g(x2)``
      subtraction, so *g* itself is unconstrained.

    Set ``n_layers_self=0`` to disable the self pathway and recover the
    pure stacked-odd architecture.

    Args:
        d: Input feature dimension.
        k: Hidden width for both pathways.
        rank: Low-rank factor size in :class:`AntisymmetricLayer`.
        n_layers_cross: Number of residual blocks in the cross pathway.
        n_layers_self: Number of residual blocks in the self pathway.
            Set to ``0`` to disable the self pathway entirely.
        activation_cross: Odd activation for the cross pathway
            (``"softsign"`` or ``"tanh"``).
        activation_self: Activation for the self pathway
            (``"relu"``, ``"softsign"``, or ``"tanh"``).
        dropout: Dropout probability used in both pathways.
    """

    def __init__(
        self,
        d: int,
        k: int = 32,
        rank: int = 16,
        n_layers_cross: int = 4,
        n_layers_self: int = 3,
        activation_cross: Literal["softsign", "tanh"] = "softsign",
        activation_self: str = "relu",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d = d
        self.k = k
        self.rank = rank
        self.n_layers_cross = n_layers_cross
        self.n_layers_self = n_layers_self

        # -- Cross pathway (odd — bias=False everywhere) --
        self.input_layer = AntisymmetricLayer(d, k, rank)
        self.cross_blocks = nn.ModuleList([
            ResidualBlock(k, activation_cross, dropout, bias=False)
            for _ in range(n_layers_cross)
        ])
        self.head = nn.Linear(k, 1, bias=False)

        # -- Self pathway (unconstrained — biases allowed) --
        if n_layers_self > 0:
            self.g_proj = nn.Linear(d, k)
            self.g_blocks = nn.ModuleList([
                ResidualBlock(k, activation_self, dropout, bias=True)
                for _ in range(n_layers_self)
            ])
            self.g_head = nn.Linear(k, 1)
        else:
            self.g_proj = None

    def _g(self, x: Tensor) -> Tensor:
        """Self-term scalar function applied independently to each input."""
        h = self.g_proj(x)
        for block in self.g_blocks:
            h = block(h)
        return self.g_head(h).squeeze(-1)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """
        Args:
            x1, x2: Tensors of shape ``(batch, n_reactions, d)``.

        Returns:
            Tensor of shape ``(batch, n_reactions)``.
        """
        _validate_pair_inputs(x1, x2, self.d, "AntisymmetricFunc.forward")

        # Cross pathway
        h = self.input_layer(x1, x2)
        for block in self.cross_blocks:
            h = block(h)
        out = self.head(h).squeeze(-1)

        # Self pathway
        if self.g_proj is not None:
            out = out + self._g(x1) - self._g(x2)

        return out
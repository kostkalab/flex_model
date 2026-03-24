"""Antisymmetric pairwise scoring modules.

This module contains two antisymmetric scorers used for reaction-pair modeling:
- `BiLinAntisymmetricFunc`: compact bilinear antisymmetric head.
- `AntisymmetricFunc`: deeper residual odd architecture.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor
import torch.nn as nn


def _validate_pair_inputs(x1: Tensor, x2: Tensor, expected_d: int, where: str) -> None:
    """Validate paired tensor inputs used by antisymmetric modules."""
    if x1.ndim != 3 or x2.ndim != 3:
        raise ValueError(
            f"{where} expects 3D tensors of shape (batch, n_reactions, d), "
            f"got x1.ndim={x1.ndim}, x2.ndim={x2.ndim}."
        )
    if x1.shape != x2.shape:
        raise ValueError(
            f"{where} requires x1 and x2 to have identical shape, got {x1.shape} vs {x2.shape}."
        )
    if x1.shape[-1] != expected_d:
        raise ValueError(f"{where} expected last dimension d={expected_d}, got d={x1.shape[-1]}.")


class BiLinAntisymmetricFunc(torch.nn.Module):
    r"""Antisymmetric bilinear function with a learnable scalar gate.

    For each channel $k$, this models an antisymmetric bilinear form
    $x_1^\top M_k x_2$ where $M_k = U_k V_k^\top - V_k U_k^\top$.
    The forward output is:
    $g(x_1) - g(x_2) + \sum_k \alpha_k x_1^\top M_k x_2$.

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
        self.g = torch.nn.Sequential(
            torch.nn.Linear(d, k),
            torch.nn.ReLU(),
            torch.nn.Linear(k, k),
            torch.nn.ReLU(),
            torch.nn.Linear(k, 1),
        )
        self.U = torch.nn.Parameter(torch.randn(k, d, rank))
        self.V = torch.nn.Parameter(torch.randn(k, d, rank))
        self.alpha = torch.nn.Parameter(torch.ones(k))

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """Forward pass computing antisymmetric bilinear form.

        Args:
            x1: Tensor of shape (batch, n_reactions, d).
            x2: Tensor of shape (batch, n_reactions, d).

        Returns:
            Tensor of shape (batch, n_reactions).
        """
        _validate_pair_inputs(x1, x2, self.d, "BiLinAntisymmetricFunc.forward")

        g_out1 = self.g(x1).squeeze(-1)
        g_out2 = self.g(x2).squeeze(-1)

        # Compute x1^T(UV^T - VU^T)x2 without explicitly forming (k, d, d).
        x1_u = torch.einsum("bnd,kdr->bnkr", x1, self.U)
        x2_v = torch.einsum("bnd,kdr->bnkr", x2, self.V)
        x1_v = torch.einsum("bnd,kdr->bnkr", x1, self.V)
        x2_u = torch.einsum("bnd,kdr->bnkr", x2, self.U)
        bili_per_k = (x1_u * x2_v - x1_v * x2_u).sum(dim=-1)
        bili = torch.einsum("bnk,k->bn", bili_per_k, self.alpha)
        return g_out1 - g_out2 + bili


class AntisymmetricLayer(nn.Module):
    """Maps (x1, x2) -> K antisymmetric features via (z, s) bilinear form."""

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
        lin = self.W_lin(z)
        zP = torch.einsum("bnd,kdr->bnkr", z, self.P)
        sQ = torch.einsum("bnd,kdr->bnkr", s, self.Q)
        bili = (zP * sQ).sum(-1)
        return lin + bili


class ResidualOddBlock(nn.Module):
    """Linear -> odd activation -> dropout, with residual connection."""

    def __init__(
        self,
        k: int,
        activation: Literal["softsign", "tanh"] = "softsign",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if activation not in {"softsign", "tanh"}:
            raise ValueError(f"Unsupported activation '{activation}'. Use 'softsign' or 'tanh'.")
        self.linear = nn.Linear(k, k, bias=False)
        self.act = nn.Tanh() if activation == "tanh" else nn.Softsign()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Init near-identity so early residuals are gentle
        nn.init.xavier_uniform_(self.linear.weight, gain=0.1)

    def forward(self, h: Tensor) -> Tensor:
        return h + self.dropout(self.act(self.linear(h)))


class AntisymmetricFunc(nn.Module):
    """
    Stacked antisymmetric function: (x1, x2) -> scalar.

    Architecture:
        AntisymLayer -> [ResidualOddBlock x n_layers] -> Linear -> scalar

    Antisymmetry is guaranteed by construction:
        - AntisymLayer is odd in (x1, x2) swap
        - Odd activations preserve oddness
        - Linear maps preserve oddness
        - Residual connections preserve oddness (sum of odd = odd)
    """

    def __init__(
        self,
        d: int,
        k: int = 32,
        rank: int = 16,
        n_layers: int = 4,
        activation: Literal["softsign", "tanh"] = "softsign",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d = d
        self.k = k
        self.rank = rank
        self.n_layers = n_layers
        self.input_layer = AntisymmetricLayer(d, k, rank)
        self.blocks = nn.ModuleList(
            [ResidualOddBlock(k, activation, dropout) for _ in range(n_layers)]
        )
        self.head = nn.Linear(k, 1, bias=False)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """
        Args:
            x1, x2: Tensors of shape (batch, n_reactions, d).
        Returns:
            Tensor of shape (batch, n_reactions).
        """
        _validate_pair_inputs(x1, x2, self.d, "AntisymmetricFunc.forward")
        h = self.input_layer(x1, x2)
        for block in self.blocks:
            h = block(h)
        return self.head(h).squeeze(-1)

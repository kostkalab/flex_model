"""FlexModel: Metabolic flux prediction from gene expression using graph neural networks.

This package provides tools for predicting metabolic reaction fluxes from gene expression
data using heterogeneous graph neural networks with biological constraints.

Main components:
    - FlexModule: PyTorch Lightning module for training flux prediction models
    - GNN architectures: Pre-built graph neural network models
    - Convolution layers: Gated and attention-based graph convolutions
    - Utilities: Correlation metrics, batch normalization, nullspace projection
"""

from __future__ import annotations

__version__ = "0.5.0"

from .conv_gadconv import GADConv
from .conv_gatedGraphConv import ResGatedConv
from .flex_gnn import (
    FlexGNN,
    FlexGNN_GCNConv_GGConv,
    FlexGNN_GCNConv_GGConv_LW,
    FluxHead,
    build_flex_gnn,
)
from .flex_module import FlexModule
from .pairwise_concordance import pairwise_concordance
from .utils import (
    MeanBatchNorm1d,
    get_S_NSprojectorSR,
    kendall_tau,
    sim_cor,
)

__all__ = [
    "FlexModule",
    "FlexGNN",
    "FluxHead",
    "build_flex_gnn",
    "FlexGNN_GCNConv_GGConv",
    "FlexGNN_GCNConv_GGConv_LW",
    "ResGatedConv",
    "GADConv",
    "pairwise_concordance",
    "kendall_tau",
    "sim_cor",
    "MeanBatchNorm1d",
    "get_S_NSprojectorSR",
]

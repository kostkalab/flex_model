"""FlexModel: Metabolic flux prediction from gene expression using graph neural networks.

This package provides tools for predicting metabolic reaction fluxes from gene expression
data using heterogeneous graph neural networks with biological constraints.

Main components:
    - FlexModule: PyTorch Lightning module for training flux prediction models
    - GNN architectures: Pre-built graph neural network models
    - Convolution layers: Gated and attention-based graph convolutions
    - Utilities: Correlation metrics, batch normalization, nullspace projection
"""

__version__ = "0.4.2"

from .flex_module import FlexModule
from .flex_gnn import FlexGNN, FluxHead, FlexGNN_GCNConv_GGConv, FlexGNN_GCNConv_GGConv_LW
from .conv_gatedGraphConv import ResGatedConv
from .conv_gadconv import GADConv
from .utils import (
    diff_spearman,
    wcor,
    sim_cor,
    MeanBatchNorm1d,
    get_S_NSprojectorSR,
)

__all__ = [
    "FlexModule",
    "FlexGNN",
    "FluxHead",
    "FlexGNN_GCNConv_GGConv",
    "FlexGNN_GCNConv_GGConv_LW",
    "ResGatedConv",
    "GADConv",
    "diff_spearman",
    "wcor",
    "sim_cor",
    "MeanBatchNorm1d",
    "get_S_NSprojectorSR",
]

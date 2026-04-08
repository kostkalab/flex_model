"""Quick test to verify package installation and imports."""

from __future__ import annotations


def test_imports() -> None:
    """Test that all main components can be imported."""
    from flexModel import FlexModule
    from flexModel import FlexGNN_GCNConv_GGConv, FlexGNN_GCNConv_GGConv_LW
    from flexModel import GADConv, ResGatedConv
    from flexModel import (
        MeanBatchNorm1d,
        get_S_NSprojectorSR,
        kendall_tau,
        pairwise_concordance,
        sim_cor,
    )

    import flexModel

    assert hasattr(flexModel, "__version__")

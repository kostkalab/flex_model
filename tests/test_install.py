"""Quick test to verify package installation and imports."""

from __future__ import annotations


def test_imports():
    """Test that all main components can be imported."""
    # Test main imports
    from flexModel import FlexModule

    print("✓ FlexModule imported successfully")

    from flexModel import FlexGNN_GCNConv_GGConv, FlexGNN_GCNConv_GGConv_LW

    print("✓ GNN architectures imported successfully")

    from flexModel import GADConv, ResGatedConv

    print("✓ Convolution layers imported successfully")

    from flexModel import (
        MeanBatchNorm1d,
        get_S_NSprojectorSR,
        kendall_tau,
        pairwise_concordance,
        sim_cor,
    )

    print("✓ Utility functions imported successfully")

    # Test version
    import flexModel

    print(f"✓ Package version: {flexModel.__version__}")

    print("\n✅ All imports successful! Package is properly installed.")

    # Test passes if all imports succeeded without exception


if __name__ == "__main__":
    test_imports()

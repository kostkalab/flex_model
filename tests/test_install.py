"""Quick test to verify package installation and imports."""

def test_imports():
    """Test that all main components can be imported."""
    # Test main imports
    from flexModel import FlexModule
    print("✓ FlexModule imported successfully")
    
    from flexModel import FlexGNN_GCNConv_GGConv, FlexGNN_GCNConv_GGConv_LW
    print("✓ GNN architectures imported successfully")
    
    from flexModel import ResGatedConv, GADConv
    print("✓ Convolution layers imported successfully")
    
    from flexModel import pairwise_concordance, kendall_tau, sim_cor, MeanBatchNorm1d, get_S_NSprojectorSR
    print("✓ Utility functions imported successfully")
    
    # Test version
    import flexModel
    print(f"✓ Package version: {flexModel.__version__}")
    
    print("\n✅ All imports successful! Package is properly installed.")
    
    # Test passes if all imports succeeded without exception

if __name__ == "__main__":
    test_imports()

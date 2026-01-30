#!/bin/bash
# Post-install script for Python packages
# Run after: mamba env create -f environment.yml

set -e

echo "Installing Python packages..."

echo "Installing Lightning..."
pip install lightning

echo "Installing PyTorch Geometric..."
pip install torch_geometric

echo "Installing PyG extensions (pyg_lib, torch_scatter, etc.)..."
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu129.html

echo "Installing torchsort (compiling from source with correct PyTorch)..."
pip uninstall -y torchsort 2>/dev/null || true
pip install torchsort --no-binary torchsort --no-cache-dir --no-build-isolation

echo "Installing flexModel package with dev dependencies..."
pip install -e .[dev]

echo ""
echo "Installation complete!"
echo "Test with: pytest tests/"

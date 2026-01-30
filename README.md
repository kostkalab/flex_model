# FlexModel

Metabolic flux prediction from gene expression using graph neural networks.

## Overview

FlexModel uses heterogeneous graph neural networks to predict metabolic reaction fluxes from expression data. The model uses biological constraints (stoichiometric balance, flux positivity) to learn patterns linking expression data with fluxes.

## Installation

### Requirements (Reference Configuration)

- **Python**: 3.12
- **PyTorch**: 2.8.0 with CUDA 12.9
- **PyTorch Geometric**: 2.7.0
- **Lightning**: 2.x

> **Note**: This installation works for us, others may alsowork. The installation requires a 2-stage process due to different pip index URLs for PyTorch and PyG packages. `torchsort` is compiled from source.

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/yourusername/flex_model.git
cd flex_model

# Step 1: Create conda environment with PyTorch
mamba env create -f environment.yml

# Step 2: Activate environment
mamba activate flex_model_env

# Step 3: Install remaining packages (PyG, Lightning, torchsort, flexModel)
bash install_pips.sh
```

### Why Two Steps?

PyTorch and PyTorch Geometric require different pip index URLs:
- PyTorch 2.8.0+cu129: `https://download.pytorch.org/whl/cu129`
- PyG extensions: `https://data.pyg.org/whl/torch-2.8.0+cu129.html`

Since conda's environment.yml only supports a single global pip index URL, we split the installation:
1. **environment.yml**: Python, CUDA toolkit, and PyTorch from the PyTorch index
2. **install_pips.sh**: PyG, Lightning, torchsort (compiled from source), and flexModel

> **Important**: `torchsort` is compiled from source using `--no-build-isolation` to ensure it links against the correct PyTorch version, avoiding ABI compatibility issues.

## Quick Start

```python
from flexModel import FlexModule
from flexModel.flex_gnn import FlexGNN_GCNConv_GGConv
import torch

# Define your GNN architecture
gnn = FlexGNN_GCNConv_GGConv(
    nr=100,           # Number of reactions
    re_edim=64,       # Reaction embedding dimension
    ge_edim=32,       # Gene embedding dimension
    nlayers=3         # Number of GNN layers
)

# Create the Lightning module
model = FlexModule(
    gnn=gnn,
    eid_g2r=edge_index_g2r,          # Gene→Reaction edges
    eid_r2r=edge_index_r2r,          # Reaction→Reaction edges
    Mcr=stoichiometry_matrix,        # S matrix
    Mmg=gene_module_matrix,          # Gene-to-module mapping
    Mmr=module_reaction_matrix,      # Module-to-reaction mapping
    cor_wts=module_weights,          # Module importance weights
    l_fb=1.0,                        # Flux balance weight
    l_pos=0.5,                       # Positivity weight
    l_cor=1.0,                       # Correlation weight
    l_sco=0.1,                       # Similarity weight
    l_ent=0.0,                       # Entropy weight (0=disabled)
    lopt_lr=1e-3,                    # Learning rate
)

# Train with PyTorch Lightning
import lightning as L

trainer = L.Trainer(max_epochs=100, accelerator="gpu")
trainer.fit(model, train_dataloader, val_dataloader)
```

## Package Structure

```
flexModel/
├── flex_module.py           # Main Lightning module (FlexModule)
├── flex_gnn.py              # GNN architectures
├── conv_gatedGraphConv.py   # Gated graph convolution layers
├── conv_gadconv.py          # Attention-based convolution 
├── utils.py                 # Correlation metrics and utilities
└── halfSpaceAntiSymmetric.py # Antisymmetric flux prediction heads
```

## Loss Components

1. **L_fb (Flux Balance)**: Enforces stoichiometric constraints S·v = 0
2. **L_pos (Positivity)**: Penalizes negative fluxes
3. **L_cor (Correlation)**: Module-level Spearman correlation between expression and flux
4. **L_sco (Similarity)**: Preserves pairwise similarity structure between samples
5. **L_ent (Entropy)**: Controls flux distribution uniformity (positive weight → uniform, negative → sparse)

## Usage with other hetero-GNNs

### Custom GNN Architectures

```python
# Define your own GNN
class CustomGNN(torch.nn.Module):
    def __init__(self, nr, re_edim, ge_edim):
        super().__init__()
        self.nr = nr
        self.re_edim = re_edim
        self.ge_edim = ge_edim
        # ... your layers
    
    def forward(self, x_dict, ei_dict):
        # x_dict: {"G": gene_embeddings, "R": reaction_embeddings}
        # ei_dict: {("G", "to", "R"): edge_index, ("R", "to", "R"): edge_index}
        # Return: flux predictions (batch_size, nr)
        pass

model = FlexModule(gnn=CustomGNN(...), ...)
```

### Nullspace Projection

```python
from flexModel.utils import get_S_NSprojectorSR

# Compute nullspace projector from stoichiometry matrix
NSP = get_S_NSprojectorSR(S_matrix, thrsh=0.999, device="cuda")

model = FlexModule(
    ...,
    flx_project=True,  # Enable projection
    NSP=NSP,           # Provide projector
)
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{flexmodel2026,
  author = {Your Name},
  title = {FlexModel: Metabolic Flux Prediction with Graph Neural Networks},
  year = {2026},
  url = {https://github.com/kostkalab/flex_model}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Development

```bash
# Install with development dependencies
bash install_pips.sh  # Already installs [dev] dependencies
```

### Running Tests

The test suite verifies core functionality using randomly generated data:

```bash
# Run all tests
pytest tests/

# Run with verbose output showing loss trajectories
pytest tests/ -v -s

# Run specific test file
pytest tests/test_training.py -v -s
```

#### Test Coverage

**`tests/test_install.py`**
- Verifies all package imports work correctly
- Checks that FlexModule, GNN architectures, and utilities are accessible

**`tests/test_flex_model.py`**
- Tests FlexModule forward pass with random graph structures
- Validates output shapes and loss computation
- Checks training_step integration
- Tests nullspace projection functionality

**`tests/test_training.py`**
- **`test_model_overfits_random_data`**: Verifies that the standard FlexGNN can reduce loss on random data (500 genes, 1000 reactions), confirming training mechanics work correctly
- **`test_model_with_layer_weights_overfits`**: Same test for FlexGNN_LW variant with learnable layer weights

> **Note**: Tests use random data. The goal is to verify that gradients flow correctly and the model can optimize, not to validate biological accuracy.

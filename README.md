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

> **Note**: This installation works for us; other configurations may also work. Installation requires two stages because PyTorch and PyG packages use different wheel indexes.

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/yourusername/flex_model.git
cd flex_model

# Step 1: Create conda environment with PyTorch
mamba env create -f environment.yml

# Step 2: Activate environment
mamba activate flex_model_env

# Step 3: Install remaining packages (PyG, Lightning, flexModel)
bash install_pips.sh
```

### Why Two Steps?

PyTorch and PyTorch Geometric require different pip index URLs:
- PyTorch 2.8.0+cu129: `https://download.pytorch.org/whl/cu129`
- PyG extensions: `https://data.pyg.org/whl/torch-2.8.0+cu129.html`

Since conda's environment.yml only supports a single global pip index URL, we split the installation:
1. **environment.yml**: Python, CUDA toolkit, and PyTorch from the PyTorch index
2. **install_pips.sh**: PyG, Lightning, and flexModel

## Quick Start

```python
from flexModel import FlexModule

# Supply tensors with consistent gene, reaction, and module ordering.
graph_inputs = dict(
    eid_g2r=edge_index_g2r,          # Gene→Reaction edges
    eid_r2r=edge_index_r2r,          # Reaction→Reaction edges
    Mcr=stoichiometry_matrix,        # S matrix
    Mmg=gene_module_matrix,          # Gene-to-module mapping
    Mmr=module_reaction_matrix,      # Module-to-reaction mapping
)

# FlexModule constructs its GNN and learns embeddings when none are supplied.
model = FlexModule(
    **graph_inputs,
    re_edim=64,                      # Reaction embedding width (even)
    ge_edim=32,                      # Gene embedding width
    nlayers=3,
    l_fb=1.0,                        # Flux balance weight
    l_pos=0.5,                       # Positivity weight
    l_cor=1.0,                       # Correlation weight
    cor_scale=1.0,                   # Module concordance margin scale
    l_sco=0.1,                       # Similarity weight
    l_ent=0.0,                       # Entropy weight (0=disabled)
    lopt_lr=1e-3,                    # Learning rate
)

# Train with PyTorch Lightning
import lightning as L

trainer = L.Trainer(max_epochs=100, accelerator="gpu", devices=[1])
trainer.fit(model, train_dataloader, val_dataloader)
```

The example uses physical GPU 1. Use `accelerator="cpu"` for CPU training.
`cor_wts` may be omitted; only uniform module weights are currently supported.

## Package Structure

```
flexModel/
├── flex_module.py           # Main Lightning module (FlexModule)
├── aligned_flex_module.py   # FlexModule with learned cross-modal alignment
├── alignment_bottleneck.py  # Denoising encoders and Pearson alignment loss
├── flex_gnn.py              # GNN architectures
├── conv_gatedGraphConv.py   # Gated graph convolution layers
├── conv_gadconv.py          # Attention-based convolution 
├── utils.py                 # Correlation metrics and utilities
└── halfSpaceAntiSymmetric.py # Antisymmetric flux prediction heads
```

## Loss Components

1. **L_fb (Flux Balance)**: Enforces stoichiometric constraints S·v = 0
2. **L_pos (Positivity)**: Penalizes negative fluxes
3. **L_cor (Module concordance)**: Pairwise ranking hinge loss between module expression and flux summaries, using relative differences and `cor_scale=1.0` by default. The scale must be finite and positive.
4. **L_sco (Sample alignment)**: One minus Pearson correlation between upper-triangle sample cosine-similarity vectors. `AlignedFlexModule` also includes reconstruction loss, as described below.
5. **L_ent (Entropy)**: Controls flux distribution uniformity (positive weight → uniform, negative → sparse)

## Choosing a model

`FlexModule` compares similarities in expression and flux spaces after centering
each feature across the batch. `AlignedFlexModule` centers each sample across
features and learns separate expression and flux encoders. It compares sample
cosine similarities in the latent spaces and combines Pearson alignment with
denoising reconstruction:

`L_sco = alpha_align * (reconstruction_flux + reconstruction_expression) + (1 - alpha_align) * (1 - Pearson_r)`

Both use the same Pearson helper with O(batch_size²) alignment storage. Constant
similarity vectors count as correlation zero; batches with fewer than three
samples contribute zero Pearson alignment loss. Reconstruction still applies.

The models also differ in `L_cor`: `FlexModule` standardizes expression and
absolute fluxes before module projection, while `AlignedFlexModule` centers
expression and signed fluxes without standard deviation normalization.

```python
from flexModel import AlignedFlexModule

model = AlignedFlexModule(
    **graph_inputs,
    re_edim=64,
    ge_edim=32,
    nlayers=3,
    z_dim=32,
    noise_std=0.1,
    alpha_align=0.5,
)
embeddings = model.get_metabolic_embeddings(expression_batch)
# embeddings["z_ge"] and embeddings["z_fl"] have shape (batch_size, z_dim).
```

### Concordant/discordant reaction messages

Both classes support `use_disc=True` together with `f_disc_orig`, a 1D tensor
containing one value in [0, 1] per non-self-loop R→R edge, in edge order.
Prepare these values from the metabolic graph: 0 means concordant and 1 means
discordant. The model updates edge strengths per sample from intermediate
predicted fluxes; the first layer assumes positive fluxes. Optional
`use_rea_rea_gate=True` enables message gating, and `use_layer_weights=True`
learns a combination of layer representations.

### Checkpoint loading

New alignment checkpoints save `z_dim`, `noise_std`, and `alpha_align` along with
the base model settings. Supply the same graph tensors at construction when
loading; also supply fixed embeddings or `f_disc_orig` if the original model
used them:

```python
restored = AlignedFlexModule.load_from_checkpoint(
    "model.ckpt", weights_only=True, **graph_inputs
)
```

Older checkpoints that lack alignment settings require those settings explicitly
when they differ from the defaults.

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
- **`test_model_overfits_random_data`**: Verifies that the standard FlexGNN can reduce loss on random data, confirming training mechanics work correctly
- **`test_model_with_layer_weights_overfits`**: Same test for FlexGNN_LW variant with learnable layer weights

**`tests/test_aligned_flex_module.py`**
- Checks Pearson agreement and finite gradients, including constant similarities and small batches
- Trains gated alignment models with `f_disc` and layer weighting independently enabled or disabled
- Checks checkpoint loading with non-default bottleneck settings and matching predictions

> **Note**: Tests use random data. The goal is to verify that gradients flow correctly and the model can optimize, not to validate biological accuracy.

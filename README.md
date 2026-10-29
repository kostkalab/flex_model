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

If you do not have access to `flex_species`, skip to
[tensor-based construction](#tensor-based-construction) and supply your own inputs.

Install [`flex_species`](https://github.com/kostkalab/flex_species#usage) first
and configure its repository credentials for the initial bundle download. This
complete CPU example loads mouse bundle **0.3.0**, creates **113 synthetic cells**
of gene expression, and predicts raw and projected reaction fluxes. Cached
bundles are reused; use `offline=True` in `load_species` to require cached data.

```python
import flex_species as fs
import numpy as np
import torch

from flexModel import AlignedFlexModule

mouse = fs.load_species("mouse", version="0.3.0")
torch.manual_seed(31)
model = AlignedFlexModule.from_bundle(
    mouse,
    nlayers=2,
    use_disc=True,
    use_rea_rea_gate=True,
    flx_project=True,
    z_dim=32,
).eval()

# Synthetic expression: (cells, genes), with explicit column IDs.
rng = np.random.default_rng(31)
expression_values = rng.lognormal(size=(113, len(mouse.genes))).astype(np.float32)
expression_gene_ids = mouse.genes
# Align gene IDs and use zero for any missing genes in nonnegative input.
expression = model.align_expression(
    expression_values, expression_gene_ids
)

# Run in small batches to bound graph-message memory.
raw_batches, projected_batches = [], []
with torch.inference_mode():
    for batch in expression.split(4):
        raw, projected = model(batch)
        raw_batches.append(raw)
        projected_batches.append(projected)
fluxes = torch.cat(raw_batches)
projected_fluxes = torch.cat(projected_batches)

print("Expression:", tuple(expression.shape))       # (113, 1571)
print("Raw fluxes:", tuple(fluxes.shape))            # (113, 2280)
print("Projected fluxes:", tuple(projected_fluxes.shape))  # (113, 2280)
print("All finite:", bool(torch.isfinite(fluxes).all()
                          & torch.isfinite(projected_fluxes).all()))
# Both flux outputs use model.bundle_identity.reactions as their column order.
```

These are predictions from an untrained model on synthetic inputs. Replace the
synthetic values and gene IDs with your expression data for actual use.
`FlexModule.from_bundle(mouse, ...)` uses the same adapter without the learned
alignment bottleneck; omit `z_dim` for that class.

### Expression alignment and missing genes

`align_expression` returns an independent float32 tensor on the model device,
with columns in `mouse.genes` order, extra genes discarded, and missing genes
filled according to the input scale:

```python
# Nonnegative expression: fill missing genes with zero (default).
expression = model.align_expression(values, gene_ids)

# Centered expression: fill with the observed minimum.
expression = model.align_expression(values, gene_ids, scale="centered")

# Or use the 1st percentile instead of the minimum.
expression = model.align_expression(
    values, gene_ids, scale="centered", percentile=1
)
```

The minimum/percentile is computed from observed modeled genes in the supplied
matrix, before padding. No normalization or fill state is stored. Duplicate IDs,
invalid values, and inputs with no modeled genes raise `ExpressionAlignmentError`.
Filling retains gene nodes and edges; it does not prune the graph. Direct calls
to `model(...)` require expression columns already in canonical gene order.

### Bundle compatibility and checkpoints

The adapter supports bundle schema 3 and the current reaction-graph,
currency-selection, projection, and module-diffusion contracts. It checks
labelled axes and reaction embedding input/output halves, then copies numerical
inputs to float32 and edge indices to int64. Undirected reaction edges expand
as `[original, reversed]` with matching `[f_disc, f_disc]`; self-loops remain the
convolution's responsibility. The released projection basis is used as-is, with
no transpose or recomputation. It is an approximate projection and does not
guarantee exact stoichiometric balance.

Layers, losses, discordance, halfspace initialization, and projection remain
constructor options with their existing defaults. `use_disc`, `halfspace_init`,
and `flx_project` are disabled unless requested. Embedding widths are inferred;
bundle-owned tensors cannot be overridden through `from_bundle`. Thermodynamic
values and compound embeddings are not consumed. Unsupported bundle contracts
or axes raise `BundleCompatibilityError`. Importing `flexModel` and using its
tensor constructor do not require installing `flex_species`.

Source species, bundle version, content ID, and ordered gene/reaction/module/
compound IDs are retained in `model.bundle_identity` and checkpoint metadata.
After saving a normal Lightning checkpoint, restore with the same bundle:

```python
# After training:
trainer.save_checkpoint("mouse-model.ckpt")
restored = AlignedFlexModule.load_from_bundle_checkpoint("mouse-model.ckpt", mouse)
```

This restores saved model settings and weights using weights-only loading.
Different bundle identities or canonical axes, and checkpoints missing bundle
metadata, raise `BundleIdentityError`. Use this loader for bundle-bound models;
the tensor-based checkpoint interface below remains available for manually
constructed models.

### Tensor-based construction

For manually prepared data, supply tensors with consistent gene, reaction, and
module ordering. The model constructs its GNN and learns embeddings when fixed
embeddings are not supplied:

```python
from flexModel import FlexModule

graph_inputs = dict(
    eid_g2r=edge_index_g2r,          # Gene→Reaction edges
    eid_r2r=edge_index_r2r,          # Reaction→Reaction edges
    Mcr=stoichiometry_matrix,        # Compound-by-reaction matrix
    Mmg=gene_module_matrix,          # Module-by-gene map
    Mmr=module_reaction_matrix,      # Module-by-reaction map
)
model = FlexModule(
    **graph_inputs,
    re_edim=64,                      # Reaction embedding width (even)
    ge_edim=32,                      # Gene embedding width
    nlayers=3,
    l_fb=1.0,
    l_pos=0.5,
    l_cor=1.0,
    cor_scale=1.0,
    l_sco=0.1,
    l_ent=0.0,
    lopt_lr=1e-3,
)
```

`cor_wts` may be omitted; only uniform module weights are currently supported.

## Package Structure

```
flexModel/
├── flex_module.py           # Main Lightning module (FlexModule)
├── aligned_flex_module.py   # FlexModule with learned cross-modal alignment
├── alignment_bottleneck.py  # Denoising encoders and Pearson alignment loss
├── bundle_adapter.py        # Species compatibility, conversion, expression order
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

**`tests/test_bundle_adapter.py`**
- Runs both models on synthetic expression with discordance, gating, and projection enabled
- Checks expression permutations, extra/missing genes, independent copies, and invalid inputs
- Checks optional zero filling against explicit zero columns for both models' predictions and losses
- Checks direct minimum/percentile filling and independent per-call behavior
- Verifies distinct edge attributes stay aligned with both reaction directions
- Checks the released projection orientation and existing flux normalization
- Trains both model classes with discordance and gating, and verifies checkpoint identity restoration/rejection

**`tests/test_mouse_bundle_smoke.py`**
- Loads the released mouse 0.3.0 bundle offline and runs both models on synthetic expression
- Checks finite raw and projected fluxes with two layers, discordance, and gating
- Skips if `flex_species` or the cached bundle is unavailable; never downloads data

```bash
python -m pytest tests/test_mouse_bundle_smoke.py -q -s
```

> **Note**: Tests use random data. The goal is to verify that gradients flow correctly and the model can optimize, not to validate biological accuracy.

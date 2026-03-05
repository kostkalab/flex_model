"""Test that the model trains correctly using the Lightning Trainer."""

import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import lightning as L
from flexModel import FlexModule
from flexModel.flex_gnn import FlexGNN_GCNConv_GGConv, FlexGNN_GCNConv_GGConv_LW


def create_flex_module(n_genes=50, n_reactions=30, gene_edim=16, reaction_edim=32, use_layer_weights=False):
    """Create a FlexModule with random graph structure and embeddings."""
    n_compounds = n_reactions // 2
    n_modules = max(5, n_genes // 10)

    eid_g2r_list = []
    for gene_idx in range(n_genes):
        n_edges = torch.randint(1, 4, (1,)).item()
        for rxn_idx in torch.randint(0, n_reactions, (n_edges,)):
            eid_g2r_list.append([gene_idx, rxn_idx.item()])
    eid_g2r = torch.tensor(eid_g2r_list, dtype=torch.long).t()

    eid_r2r_list = []
    for rxn_idx in range(n_reactions):
        for tgt in torch.randint(0, n_reactions, (3,)):
            if tgt.item() != rxn_idx:
                eid_r2r_list.append([rxn_idx, tgt.item()])
    eid_r2r = torch.tensor(eid_r2r_list, dtype=torch.long).t()

    Mcr = torch.randn(n_compounds, n_reactions) * 0.3
    Mcr[torch.rand(n_compounds, n_reactions) > 0.3] = 0

    Mmg = torch.zeros(n_modules, n_genes)
    for m in range(n_modules):
        Mmg[m, torch.randperm(n_genes)[:torch.randint(3, 9, (1,)).item()]] = 1.0

    Mmr = torch.zeros(n_modules, n_reactions)
    for m in range(n_modules):
        Mmr[m, torch.randperm(n_reactions)[:torch.randint(2, 6, (1,)).item()]] = 1.0

    cor_wts = torch.rand(n_modules) + 0.5
    cor_wts = cor_wts / cor_wts.sum()

    gen_emb = torch.randn(n_genes, gene_edim)
    rea_emb = torch.randn(n_reactions, reaction_edim)

    GNNCls = FlexGNN_GCNConv_GGConv_LW if use_layer_weights else FlexGNN_GCNConv_GGConv
    gnn = GNNCls(nr=n_reactions, re_edim=reaction_edim, ge_edim=gene_edim, nlayers=2)

    model = FlexModule(
        gnn=gnn,
        eid_g2r=eid_g2r,
        eid_r2r=eid_r2r,
        Mcr=Mcr,
        Mmg=Mmg,
        Mmr=Mmr,
        cor_wts=cor_wts,
        gen_emb=gen_emb,
        rea_emb=rea_emb,
        flx_project=False,
        l_fb=1.0,
        l_pos=1.0,
        l_cor=1.0,
        l_sco=1.0,
        l_ent=0.0,
        lopt_lr=1e-3,
    )
    return model, n_genes, n_reactions


def make_dataloaders(n_genes, n_samples=64, batch_size=16, val_frac=0.25, seed=42):
    """Create train/val DataLoaders from random gene expression data."""
    torch.manual_seed(seed)
    ge = torch.randn(n_samples, n_genes).abs()
    dataset = TensorDataset(ge)
    n_val = max(1, int(n_samples * val_frac))
    n_trn = n_samples - n_val
    trn_ds, val_ds = random_split(dataset, [n_trn, n_val],
                                  generator=torch.Generator().manual_seed(seed))
    trn_dl = DataLoader(trn_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return trn_dl, val_dl


def test_trainer_runs():
    """Smoke test: Lightning Trainer completes fit() without errors."""
    torch.manual_seed(42)
    model, n_genes, _ = create_flex_module(n_genes=50, n_reactions=30,
                                           gene_edim=16, reaction_edim=32)
    trn_dl, val_dl = make_dataloaders(n_genes, n_samples=64, batch_size=16)

    trainer = L.Trainer(
        max_epochs=3,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    trainer.fit(model, train_dataloaders=trn_dl, val_dataloaders=val_dl)
    print("Trainer completed successfully.")


def test_trainer_loss_decreases():
    """Verify that training loss decreases over multiple epochs."""
    torch.manual_seed(42)
    model, n_genes, _ = create_flex_module(n_genes=50, n_reactions=30,
                                           gene_edim=16, reaction_edim=32)
    trn_dl, val_dl = make_dataloaders(n_genes, n_samples=64, batch_size=16)

    losses = []

    class LossCallback(L.Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            losses.append(trainer.callback_metrics.get("trn_loss-all", torch.tensor(float("nan"))).item())

    trainer = L.Trainer(
        max_epochs=10,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        callbacks=[LossCallback()],
    )
    trainer.fit(model, train_dataloaders=trn_dl, val_dataloaders=val_dl)

    print(f"Loss trajectory: {[f'{l:.3f}' for l in losses]}")
    assert losses[-1] < losses[0], (
        f"Training loss did not decrease! First: {losses[0]:.4f}, Last: {losses[-1]:.4f}"
    )
    print("Training loss decreased as expected.")


def test_trainer_with_layer_weights():
    """Smoke test: Trainer runs with the layer-weighted GNN variant."""
    torch.manual_seed(42)
    model, n_genes, _ = create_flex_module(n_genes=50, n_reactions=30,
                                           gene_edim=16, reaction_edim=32,
                                           use_layer_weights=True)
    trn_dl, val_dl = make_dataloaders(n_genes, n_samples=64, batch_size=16)

    trainer = L.Trainer(
        max_epochs=3,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    trainer.fit(model, train_dataloaders=trn_dl, val_dataloaders=val_dl)
    print("Trainer (layer weights) completed successfully.")

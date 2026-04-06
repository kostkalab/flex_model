"""Test that the model trains correctly using the Lightning Trainer."""

import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import lightning as L
from flexModel import FlexModule

class DetailedLoggingFlexModule(FlexModule):
    def training_step(self, batch, batch_idx):
        x, *_ = batch
        flxs, flxs_p = self(x)
        lses = self.losses(x, flxs, flxs_p)  
        lses = lses.mean(dim=0)
        
        l_fb, l_pos, l_cor, l_sco, l_ent = lses[0], lses[1], lses[2], lses[3], lses[4]

        # Gradient calculation for each type
        opt = self.optimizers()
        opt.zero_grad()
        
        def compute_grad_norm(loss_tensor):
            if loss_tensor.requires_grad:
                grads = torch.autograd.grad(loss_tensor, [p for p in self.parameters() if p.requires_grad], retain_graph=True, allow_unused=True)
                norm = 0.0
                for g in grads:
                    if g is not None:
                        norm += g.detach().norm(2).item() ** 2
                return norm ** 0.5
            return 0.0

        if self.loss_lms[0] > 0: self.log("grad_fb", compute_grad_norm(l_fb * self.loss_lms[0]), sync_dist=True)
        if self.loss_lms[1] > 0: self.log("grad_pos", compute_grad_norm(l_pos * self.loss_lms[1]), sync_dist=True)
        if self.loss_lms[2] > 0: self.log("grad_cor", compute_grad_norm(l_cor * self.loss_lms[2]), sync_dist=True)
        if self.loss_lms[3] > 0: self.log("grad_sco", compute_grad_norm(l_sco * self.loss_lms[3]), sync_dist=True)
        if self.loss_lms[4] > 0: self.log("grad_ent", compute_grad_norm(l_ent * self.loss_lms[4]), sync_dist=True)

        loss = lses @ self.loss_lms
        
        self.log("trn_loss-all", loss.detach(), on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("trn_loss-fb", (l_fb * self.loss_lms[0]).detach(), on_epoch=True, sync_dist=True)
        self.log("trn_loss-pos", (l_pos * self.loss_lms[1]).detach(), on_epoch=True, sync_dist=True)
        self.log("trn_loss-cor", (l_cor * self.loss_lms[2]).detach(), on_epoch=True, sync_dist=True)
        self.log("trn_loss-sco", (l_sco * self.loss_lms[3]).detach(), on_epoch=True, sync_dist=True)
        self.log("trn_loss-ent", (l_ent * self.loss_lms[4]).detach(), on_epoch=True, sync_dist=True)

        return loss

from flexModel.flex_gnn_v2 import FlexGNN_GCNConv_GGConv, FlexGNN_GCNConv_GGConv_LW, FlexGNN_Disc_GGConv, FlexGNN_Disc_GGConv_LW


def create_flex_module(n_genes=50, n_reactions=30, gene_edim=16, reaction_edim=32, use_layer_weights=False, use_disc=False):
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

    if use_disc:
        f_disc = torch.rand(eid_r2r.shape[1])
        GNNCls = FlexGNN_Disc_GGConv_LW if use_layer_weights else FlexGNN_Disc_GGConv
        gnn = GNNCls(nr=n_reactions, f_disc_orig=f_disc, re_edim=reaction_edim, ge_edim=gene_edim, nlayers=2)
    else:
        GNNCls = FlexGNN_GCNConv_GGConv_LW if use_layer_weights else FlexGNN_GCNConv_GGConv
        gnn = GNNCls(nr=n_reactions, re_edim=reaction_edim, ge_edim=gene_edim, nlayers=2)

    model = DetailedLoggingFlexModule(
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


@pytest.mark.parametrize('use_disc', [False, True])
def test_trainer_runs(use_disc):
    """Smoke test: Lightning Trainer completes fit() without errors."""
    torch.manual_seed(42)
    model, n_genes, _ = create_flex_module(n_genes=50, n_reactions=30,
                                           gene_edim=16, reaction_edim=32, use_disc=use_disc)
    trn_dl, val_dl = make_dataloaders(n_genes, n_samples=64, batch_size=16)

    trainer = L.Trainer(
        max_epochs=3,
        accelerator="auto",
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    trainer.fit(model, train_dataloaders=trn_dl, val_dataloaders=val_dl)
    print("Trainer completed successfully.")


@pytest.mark.parametrize('use_disc', [False, True])
def test_trainer_loss_decreases(use_disc):
    """Verify that training loss decreases over multiple epochs."""
    torch.manual_seed(42)
    model, n_genes, _ = create_flex_module(n_genes=50, n_reactions=30,
                                           gene_edim=16, reaction_edim=32, use_disc=use_disc)
    trn_dl, val_dl = make_dataloaders(n_genes, n_samples=64, batch_size=16)

    losses = []



    class DetailedLoggingCallback(L.Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            metrics = trainer.callback_metrics
            l_all = metrics.get("trn_loss-all", torch.tensor(float("nan"))).item()
            l_fb = metrics.get("trn_loss-fb", torch.tensor(float("nan"))).item()
            l_pos = metrics.get("trn_loss-pos", torch.tensor(float("nan"))).item()
            l_cor = metrics.get("trn_loss-cor", torch.tensor(float("nan"))).item()
            l_sco = metrics.get("trn_loss-sco", torch.tensor(float("nan"))).item()
            l_ent = metrics.get("trn_loss-ent", torch.tensor(float("nan"))).item()
            
            g_fb = metrics.get("grad_fb", torch.tensor(float("nan"))).item()
            g_pos = metrics.get("grad_pos", torch.tensor(float("nan"))).item()
            g_cor = metrics.get("grad_cor", torch.tensor(float("nan"))).item()
            g_sco = metrics.get("grad_sco", torch.tensor(float("nan"))).item()
            g_ent = metrics.get("grad_ent", torch.tensor(float("nan"))).item()

            losses.append({
                "all": l_all,
            })
            
            print(f"\nEpoch {trainer.current_epoch}:")
            print(f"  Losses -> Total:{l_all:.4f}  | fb:{l_fb:.4f} pos:{l_pos:.4f} cor:{l_cor:.4f} sco:{l_sco:.4f} ent:{l_ent:.4f}")
            print(f"  Grads  -> fb:{g_fb:.4f} pos:{g_pos:.4f} cor:{g_cor:.4f} sco:{g_sco:.4f} ent:{g_ent:.4f}\n")

    trainer = L.Trainer(
        max_epochs=10,
        accelerator="auto",
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        callbacks=[DetailedLoggingCallback()],
    )
    trainer.fit(model, train_dataloaders=trn_dl, val_dataloaders=val_dl)


    assert losses[-1]["all"] < losses[0]["all"], "Total loss did not decrease"



@pytest.mark.parametrize('use_disc', [False, True])
def test_trainer_with_layer_weights(use_disc):
    """Smoke test: Trainer runs with the layer-weighted GNN variant."""
    torch.manual_seed(42)
    model, n_genes, _ = create_flex_module(n_genes=50, n_reactions=30,
                                           gene_edim=16, reaction_edim=32,
                                           use_layer_weights=True, use_disc=use_disc)
    trn_dl, val_dl = make_dataloaders(n_genes, n_samples=64, batch_size=16)

    trainer = L.Trainer(
        max_epochs=3,
        accelerator="auto",
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    trainer.fit(model, train_dataloaders=trn_dl, val_dataloaders=val_dl)
    print("Trainer (layer weights) completed successfully.")

"""Test that the model trains correctly using the Lightning Trainer."""

from __future__ import annotations

import lightning as L
import pytest
import torch

from flexModel import FlexModule

from tests.factories import create_flex_module, make_dataloaders


class DetailedLoggingFlexModule(FlexModule):
    """FlexModule that logs per-term gradient norms during training."""

    def training_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        """Compute loss and log per-component gradient norms."""
        x, *_ = batch
        flxs, flxs_p = self(x)
        lses = self.losses(x, flxs, flxs_p)
        lses = lses.mean(dim=0)

        l_fb, l_pos, l_cor, l_sco, l_ent = lses[0], lses[1], lses[2], lses[3], lses[4]

        opt = self.optimizers()
        opt.zero_grad()

        def compute_grad_norm(loss_tensor: torch.Tensor) -> float:
            if loss_tensor.requires_grad:
                grads = torch.autograd.grad(
                    loss_tensor,
                    [p for p in self.parameters() if p.requires_grad],
                    retain_graph=True,
                    allow_unused=True,
                )
                norm = 0.0
                for g in grads:
                    if g is not None:
                        norm += g.detach().norm(2).item() ** 2
                return norm**0.5
            return 0.0

        if self.loss_lms[0] > 0:
            self.log(
                "grad_fb", compute_grad_norm(l_fb * self.loss_lms[0]), sync_dist=True
            )
        if self.loss_lms[1] > 0:
            self.log(
                "grad_pos", compute_grad_norm(l_pos * self.loss_lms[1]), sync_dist=True
            )
        if self.loss_lms[2] > 0:
            self.log(
                "grad_cor", compute_grad_norm(l_cor * self.loss_lms[2]), sync_dist=True
            )
        if self.loss_lms[3] > 0:
            self.log(
                "grad_sco", compute_grad_norm(l_sco * self.loss_lms[3]), sync_dist=True
            )
        if self.loss_lms[4] > 0:
            self.log(
                "grad_ent", compute_grad_norm(l_ent * self.loss_lms[4]), sync_dist=True
            )

        loss = lses @ self.loss_lms

        self.log(
            "trn_loss-all", loss.detach(), on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log(
            "trn_loss-fb",
            (l_fb * self.loss_lms[0]).detach(),
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "trn_loss-pos",
            (l_pos * self.loss_lms[1]).detach(),
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "trn_loss-cor",
            (l_cor * self.loss_lms[2]).detach(),
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "trn_loss-sco",
            (l_sco * self.loss_lms[3]).detach(),
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "trn_loss-ent",
            (l_ent * self.loss_lms[4]).detach(),
            on_epoch=True,
            sync_dist=True,
        )

        return loss


@pytest.mark.parametrize("use_disc", [False, True])
def test_trainer_runs(use_disc: bool) -> None:
    """Smoke test: Lightning Trainer completes fit() without errors."""
    torch.manual_seed(42)
    model, n_genes, _ = create_flex_module(
        n_genes=50,
        n_reactions=30,
        gene_edim=16,
        reaction_edim=32,
        use_disc=use_disc,
        module_cls=DetailedLoggingFlexModule,
    )
    trn_dl, val_dl = make_dataloaders(n_genes, n_samples=64, batch_size=16)

    trainer = L.Trainer(
        max_epochs=3,
        accelerator="auto",
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(model, train_dataloaders=trn_dl, val_dataloaders=val_dl)


@pytest.mark.parametrize("use_disc", [False, True])
def test_trainer_loss_decreases(use_disc: bool) -> None:
    """Verify that training loss decreases over multiple epochs."""
    torch.manual_seed(42)
    model, n_genes, _ = create_flex_module(
        n_genes=50,
        n_reactions=30,
        gene_edim=16,
        reaction_edim=32,
        use_disc=use_disc,
        module_cls=DetailedLoggingFlexModule,
    )
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

            losses.append(
                {
                    "all": l_all,
                }
            )

    trainer = L.Trainer(
        max_epochs=10,
        accelerator="auto",
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
        callbacks=[DetailedLoggingCallback()],
    )
    trainer.fit(model, train_dataloaders=trn_dl, val_dataloaders=val_dl)

    assert losses[-1]["all"] < losses[0]["all"], "Total loss did not decrease"


@pytest.mark.parametrize("use_disc", [False, True])
def test_trainer_with_layer_weights(use_disc: bool) -> None:
    """Smoke test: Trainer runs with the layer-weighted GNN variant."""
    torch.manual_seed(42)
    model, n_genes, _ = create_flex_module(
        n_genes=50,
        n_reactions=30,
        gene_edim=16,
        reaction_edim=32,
        use_layer_weights=True,
        use_disc=use_disc,
        module_cls=DetailedLoggingFlexModule,
    )
    trn_dl, val_dl = make_dataloaders(n_genes, n_samples=64, batch_size=16)

    trainer = L.Trainer(
        max_epochs=3,
        accelerator="auto",
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(model, train_dataloaders=trn_dl, val_dataloaders=val_dl)

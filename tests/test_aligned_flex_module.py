from __future__ import annotations

from pathlib import Path

import lightning as L
import pytest
import torch

from flexModel.aligned_flex_module import AlignedFlexModule
from flexModel.alignment_bottleneck import alignment_loss
from tests.factories import create_flex_module


def test_default_alignment_matches_pearson_reference() -> None:
    """Default alignment agrees with an independent Pearson reference."""
    rng = torch.Generator().manual_seed(17)
    z_a = torch.randn(6, 4, generator=rng, dtype=torch.float64, requires_grad=True)
    z_b = torch.randn(6, 3, generator=rng, dtype=torch.float64, requires_grad=True)
    x = torch.zeros(6, 2, dtype=torch.float64)
    out = {"z_a": z_a, "z_b": z_b, "x_a_hat": x, "x_b_hat": x}
    loss = alignment_loss(out, x, x, alpha=0.0)

    idx = torch.triu_indices(6, 6, offset=1)
    similarities = []
    for z in (z_a, z_b):
        unit = z / (z.norm(dim=1, keepdim=True) + 1e-7)
        similarities.append((unit @ unit.t())[idx[0], idx[1]])
    expected = 1.0 - torch.corrcoef(torch.stack(similarities))[0, 1]
    torch.testing.assert_close(loss, expected, rtol=1e-10, atol=1e-10)
    loss.backward()
    for z in (z_a, z_b):
        assert z.grad is not None
        assert torch.isfinite(z.grad).all()
        assert z.grad.abs().sum() > 0


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_default_alignment_handles_degenerate_similarities(batch_size: int) -> None:
    """Small batches and constant similarities have finite losses and gradients."""
    z = torch.ones(batch_size, 3, dtype=torch.float64, requires_grad=True)
    x = torch.zeros(batch_size, 2, dtype=torch.float64)
    out = {"z_a": z, "z_b": z, "x_a_hat": x, "x_b_hat": x}
    loss = alignment_loss(out, x, x, alpha=0.0)
    expected = torch.tensor(0.0 if batch_size < 3 else 1.0, dtype=loss.dtype)
    torch.testing.assert_close(loss, expected, rtol=0.0, atol=1e-12)
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


@pytest.mark.parametrize("use_disc", [False, True])
@pytest.mark.parametrize("use_layer_weights", [False, True])
def test_aligned_module_overfits_small_batch_with_sco_only(
    use_disc: bool, use_layer_weights: bool
) -> None:
    """Aligned module should overfit a fixed batch using only loss_sco."""
    torch.manual_seed(42)
    model, n_genes, _ = create_flex_module(
        n_genes=30,
        n_reactions=20,
        use_layer_weights=use_layer_weights,
        use_disc=use_disc,
        module_cls=AlignedFlexModule,
        use_rea_rea_gate=True,
    )
    model.loss_lms[0] = 0.0
    model.loss_lms[1] = 0.0
    model.loss_lms[2] = 0.0
    model.loss_lms[3] = 1.0
    model.loss_lms[4] = 0.0

    ge = torch.randn(10, n_genes).abs()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()

    with torch.no_grad():
        flxs, flxs_p = model(ge)
        initial_terms = model.losses(ge, flxs, flxs_p).mean(dim=0)
        initial_loss = float(initial_terms[3].item())

    for _ in range(20):
        optimizer.zero_grad()
        flxs, flxs_p = model(ge)
        loss_terms = model.losses(ge, flxs, flxs_p).mean(dim=0)
        loss = loss_terms @ model.loss_lms
        loss.backward()
        if use_disc:
            assert model.gnn.needs_fluxes
            for conv in model.gnn.convs:
                grad = conv.convs[("R", "to", "R")].lin_disc.weight.grad
                assert grad is not None
                assert torch.isfinite(grad).all()
                assert grad.abs().sum() > 0
        optimizer.step()

    with torch.no_grad():
        flxs, flxs_p = model(ge)
        final_terms = model.losses(ge, flxs, flxs_p).mean(dim=0)
        final_loss = float(final_terms[3].item())

    print(
        f"Alignment use_disc={use_disc}, use_layer_weights={use_layer_weights}: "
        f"{initial_loss:.6f} -> {final_loss:.6f}"
    )
    assert final_loss < initial_loss, (
        "AlignedFlexModule failed to reduce loss_sco on a fixed batch. "
        f"Initial: {initial_loss:.4f}, Final: {final_loss:.4f}"
    )
    assert final_loss / initial_loss < 0.5, (
        "AlignedFlexModule reduced loss_sco too little on a fixed batch. "
        f"Initial: {initial_loss:.4f}, Final: {final_loss:.4f}"
    )


def test_alignment_checkpoint_restores_nondefault_settings(tmp_path: Path) -> None:
    """Checkpoint loading restores bottleneck settings and predictions."""
    torch.manual_seed(23)
    model, n_genes, _ = create_flex_module(
        n_genes=30,
        n_reactions=20,
        use_disc=True,
        module_cls=AlignedFlexModule,
        z_dim=7,
        noise_std=0.23,
        alpha_align=0.7,
        use_rea_rea_gate=True,
    )
    path = tmp_path / "alignment.ckpt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": L.__version__,
        },
        path,
    )
    restored = AlignedFlexModule.load_from_checkpoint(
        path,
        weights_only=True,
        eid_g2r=model.eid_g2r,
        eid_r2r=model.eid_r2r,
        Mcr=model.Mcr,
        Mmg=model.Mmg,
        Mmr=model.Mmr,
        cor_wts=model.cor_wts,
        gen_emb=model.gen_emb_tt,
        rea_emb=model.rea_emb_tt,
        f_disc_orig=model.gnn.convs[0].convs[("R", "to", "R")].f_disc_orig,
    )
    assert restored.bottleneck.k == 7
    assert restored.bottleneck.noise_std == 0.23
    assert restored.alpha_align == 0.7
    assert restored.cor_scale == 1.0
    model.eval()
    restored.eval()
    ge = torch.randn(6, n_genes).abs()
    with torch.no_grad():
        expected, _ = model(ge)
        actual, _ = restored(ge)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(
            restored.loss_sco(ge, actual),
            model.loss_sco(ge, expected),
            rtol=1e-6,
            atol=1e-7,
        )


def test_get_metabolic_embeddings_preserves_training_mode() -> None:
    """Embedding helper should not leave the module stuck in eval mode."""
    torch.manual_seed(42)
    model, n_genes, _ = create_flex_module(
        n_genes=30,
        n_reactions=20,
        use_layer_weights=False,
        use_disc=False,
        module_cls=AlignedFlexModule,
    )
    ge = torch.randn(10, n_genes).abs()

    model.train()
    assert model.training

    embeddings = model.get_metabolic_embeddings(ge)

    assert model.training
    assert embeddings["z_ge"].shape[0] == ge.shape[0]
    assert embeddings["z_fl"].shape[0] == ge.shape[0]

from __future__ import annotations

import torch

from flexModel.aligned_flex_module import AlignedFlexModule
from tests.factories import create_flex_module


def test_aligned_module_overfits_small_batch_with_sco_only() -> None:
    """Aligned module should overfit a fixed batch using only loss_sco."""
    torch.manual_seed(42)
    model, n_genes, _ = create_flex_module(
        n_genes=30,
        n_reactions=20,
        use_layer_weights=False,
        use_disc=False,
        module_cls=AlignedFlexModule,
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
        optimizer.step()

    with torch.no_grad():
        flxs, flxs_p = model(ge)
        final_terms = model.losses(ge, flxs, flxs_p).mean(dim=0)
        final_loss = float(final_terms[3].item())

    assert final_loss < initial_loss, (
        "AlignedFlexModule failed to reduce loss_sco on a fixed batch. "
        f"Initial: {initial_loss:.4f}, Final: {final_loss:.4f}"
    )
    assert final_loss / initial_loss < 0.5, (
        "AlignedFlexModule reduced loss_sco too little on a fixed batch. "
        f"Initial: {initial_loss:.4f}, Final: {final_loss:.4f}"
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
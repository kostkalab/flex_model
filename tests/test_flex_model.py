"""Basic test for FlexModule with random data to verify mechanics work."""

from __future__ import annotations

import pytest
import torch

from flexModel.flex_gnn import (
    FlexGNN_Disc_GGConv,
    FlexGNN_Disc_GGConv_LW,
    FlexGNN_GCNConv_GGConv,
    FlexGNN_GCNConv_GGConv_LW,
)
from flexModel.flex_module import FlexModule
from tests.factories import create_flex_module, create_random_problem


def test_sample_alignment_matches_pearson_reference() -> None:
    """Base sample alignment uses Pearson on centered sample similarities."""
    torch.manual_seed(19)
    model, n_genes, n_reactions = create_flex_module()
    ge = torch.randn(7, n_genes, dtype=torch.float64)
    flxs = torch.randn(7, n_reactions, dtype=torch.float64, requires_grad=True)
    similarities = []
    idx = torch.triu_indices(7, 7, offset=1)
    for values in (ge, flxs):
        centered = values - values.mean(dim=0, keepdim=True)
        unit = centered / (centered.norm(dim=1, keepdim=True) + 1e-7)
        similarities.append((unit @ unit.t())[idx[0], idx[1]])
    expected = 1.0 - torch.corrcoef(torch.stack(similarities))[0, 1]
    actual = model.loss_sco(ge, flxs)
    torch.testing.assert_close(actual, expected.expand(7), rtol=1e-10, atol=1e-10)
    actual.mean().backward()
    assert flxs.grad is not None
    assert torch.isfinite(flxs.grad).all()
    assert flxs.grad.abs().sum() > 0


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_module_rejects_invalid_concordance_scale(scale: float) -> None:
    with pytest.raises(ValueError, match="cor_scale must be finite and positive"):
        create_flex_module(cor_scale=scale)


@pytest.mark.parametrize("use_disc", [False, True])
def test_flex_module_forward_pass(use_disc: bool) -> None:
    """Test FlexModule forward pass with random graph structure and embeddings."""

    # Set random seed for reproducibility
    torch.manual_seed(42)

    batch_size = 8
    gene_edim = 16
    reaction_edim = 32
    problem = create_random_problem(
        n_genes=50,
        n_reactions=30,
        gene_edim=gene_edim,
        reaction_edim=reaction_edim,
    )

    f_disc_orig = None
    if use_disc:
        f_disc_orig = torch.rand(int((problem.eid_r2r[0] != problem.eid_r2r[1]).sum().item()))

    # Initialize FlexModule
    model = FlexModule(
        eid_g2r=problem.eid_g2r,
        eid_r2r=problem.eid_r2r,
        Mcr=problem.Mcr,
        Mmg=problem.Mmg,
        Mmr=problem.Mmr,
        cor_wts=problem.cor_wts,
        gen_emb=problem.gen_emb,
        rea_emb=problem.rea_emb,
        nlayers=2,
        use_disc=use_disc,
        f_disc_orig=f_disc_orig,
        flx_project=False,  # Test without nullspace projection first
        l_fb=1.0,
        l_pos=1.0,
        l_cor=1.0,
        l_sco=1.0,
        l_ent=0.0,
        lopt_lr=1e-3,
    )

    # Create random gene expression data
    ge = torch.randn(batch_size, problem.n_genes).abs()  # (batch, n_genes)

    flxs, flxs_p = model.forward(ge)

    # Verify output shapes
    assert flxs.shape == (
        batch_size,
        problem.n_reactions,
    ), f"Expected shape {(batch_size, problem.n_reactions)}, got {flxs.shape}"
    assert flxs_p is None, "Expected flxs_p to be None when flx_project=False"

    losses = model.losses(ge, flxs, flxs_p)

    # Verify loss shapes
    assert losses.shape == (
        batch_size,
        5,
    ), f"Expected loss shape {(batch_size, 5)}, got {losses.shape}"

    batch = (ge,)
    loss = model.training_step(batch, batch_idx=0)

    # Verify loss is a scalar
    assert loss.dim() == 0, f"Expected scalar loss, got shape {loss.shape}"

    # Nullspace projection path
    n_null = problem.n_reactions - 5
    NSP = torch.randn(n_null, problem.n_reactions)
    NSP = torch.nn.functional.normalize(NSP, p=2, dim=1)  # Orthonormalize rows

    # Reinitialize model with projection
    model_proj = FlexModule(
        eid_g2r=problem.eid_g2r,
        eid_r2r=problem.eid_r2r,
        Mcr=problem.Mcr,
        Mmg=problem.Mmg,
        Mmr=problem.Mmr,
        cor_wts=problem.cor_wts,
        gen_emb=problem.gen_emb,
        rea_emb=problem.rea_emb,
        nlayers=2,
        use_disc=use_disc,
        f_disc_orig=f_disc_orig,
        flx_project=True,
        l_fb=1.0,
        l_pos=1.0,
        l_cor=1.0,
        l_sco=1.0,
        l_ent=0.1,
        lopt_lr=1e-3,
        NSP=NSP,
    )

    # Test forward pass with projection
    flxs_proj, flxs_p_proj = model_proj.forward(ge)

    # Verify output shapes
    assert flxs_proj.shape == (
        batch_size,
        problem.n_reactions,
    ), f"Expected shape {(batch_size, problem.n_reactions)}, got {flxs_proj.shape}"
    assert flxs_p_proj.shape == (
        batch_size,
        problem.n_reactions,
    ), (
        f"Expected projected flux shape {(batch_size, problem.n_reactions)}, "
        f"got {flxs_p_proj.shape}"
    )

    losses_proj = model_proj.losses(ge, flxs_proj, flxs_p_proj)
    assert losses_proj.shape == (batch_size, 5)


@pytest.mark.parametrize(
    ("wrapper_cls", "kwargs"),
    [
        (FlexGNN_GCNConv_GGConv, {}),
        (FlexGNN_GCNConv_GGConv_LW, {}),
        (FlexGNN_Disc_GGConv, {"f_disc_orig": torch.rand(3)}),
        (FlexGNN_Disc_GGConv_LW, {"f_disc_orig": torch.rand(3)}),
    ],
)
def test_legacy_gnn_wrappers_warn(
    wrapper_cls: type[torch.nn.Module], kwargs: dict[str, torch.Tensor]
) -> None:
    """Legacy preset wrappers emit deprecation warnings but remain usable."""
    with pytest.warns(DeprecationWarning, match="deprecated"):
        model = wrapper_cls(nr=4, re_edim=2, ge_edim=2, nlayers=1, **kwargs)

    assert isinstance(model, torch.nn.Module)


def test_flex_module_requires_dims_without_fixed_embeddings() -> None:
    """Learned-embedding mode still requires explicit architecture widths."""
    problem = create_random_problem(n_genes=8, n_reactions=6, gene_edim=4, reaction_edim=4)

    with pytest.raises(ValueError, match="ge_edim must be provided"):
        FlexModule(
            eid_g2r=problem.eid_g2r,
            eid_r2r=problem.eid_r2r,
            Mcr=problem.Mcr,
            Mmg=problem.Mmg,
            Mmr=problem.Mmr,
            gen_emb=None,
            rea_emb=problem.rea_emb,
        )


def test_loss_cor_respects_configured_scale() -> None:
    """Higher cor_scale should relax the hinge margin for concordant pairs."""
    model_lo = FlexModule(
        eid_g2r=torch.zeros((2, 0), dtype=torch.long),
        eid_r2r=torch.zeros((2, 0), dtype=torch.long),
        Mcr=torch.zeros((1, 3)),
        Mmg=torch.eye(3),
        Mmr=torch.eye(3),
        cor_wts=torch.ones(3),
        gen_emb=torch.ones((3, 2)),
        rea_emb=torch.ones((3, 2)),
        cor_scale=1.0,
    )
    model_hi = FlexModule(
        eid_g2r=torch.zeros((2, 0), dtype=torch.long),
        eid_r2r=torch.zeros((2, 0), dtype=torch.long),
        Mcr=torch.zeros((1, 3)),
        Mmg=torch.eye(3),
        Mmr=torch.eye(3),
        cor_wts=torch.ones(3),
        gen_emb=torch.ones((3, 2)),
        rea_emb=torch.ones((3, 2)),
        cor_scale=4.0,
    )

    ge = torch.tensor([[1.0, 2.0, 4.0]])
    flxs = torch.tensor([[1.0, 1.2, 4.0]])

    loss_lo = model_lo.loss_cor(ge, flxs)
    loss_hi = model_hi.loss_cor(ge, flxs)

    assert torch.all(loss_hi < loss_lo), (loss_lo, loss_hi)

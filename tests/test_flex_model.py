"""Basic test for FlexModule with random data to verify mechanics work."""

from __future__ import annotations

import pytest
import torch

from flexModel.flex_gnn import FlexGNN_Disc_GGConv, FlexGNN_GCNConv_GGConv
from flexModel.flex_module import FlexModule
from tests.factories import create_random_problem


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

    # Initialize GNN model
    if use_disc:
        f_disc_orig = torch.rand(problem.eid_r2r.shape[1])
        gnn = FlexGNN_Disc_GGConv(
            nr=problem.n_reactions,
            f_disc_orig=f_disc_orig,
            re_edim=reaction_edim,
            ge_edim=gene_edim,
            nlayers=2,
        )
    else:
        gnn = FlexGNN_GCNConv_GGConv(
            nr=problem.n_reactions,
            re_edim=reaction_edim,
            ge_edim=gene_edim,
            nlayers=2,
        )

    # Initialize FlexModule
    model = FlexModule(
        gnn=gnn,
        eid_g2r=problem.eid_g2r,
        eid_r2r=problem.eid_r2r,
        Mcr=problem.Mcr,
        Mmg=problem.Mmg,
        Mmr=problem.Mmr,
        cor_wts=problem.cor_wts,
        gen_emb=problem.gen_emb,
        rea_emb=problem.rea_emb,
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
        gnn=gnn,
        eid_g2r=problem.eid_g2r,
        eid_r2r=problem.eid_r2r,
        Mcr=problem.Mcr,
        Mmg=problem.Mmg,
        Mmr=problem.Mmr,
        cor_wts=problem.cor_wts,
        gen_emb=problem.gen_emb,
        rea_emb=problem.rea_emb,
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

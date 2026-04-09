from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from flexModel import FlexModule


@dataclass(frozen=True)
class RandomProblem:
    """Synthetic graph and feature tensors for FlexModel tests."""

    n_genes: int
    n_reactions: int
    eid_g2r: torch.Tensor
    eid_r2r: torch.Tensor
    Mcr: torch.Tensor
    Mmg: torch.Tensor
    Mmr: torch.Tensor
    cor_wts: torch.Tensor
    gen_emb: torch.Tensor
    rea_emb: torch.Tensor


def _make_gene_reaction_edges(n_genes: int, n_reactions: int) -> torch.Tensor:
    pairs: list[list[int]] = []
    for gene_idx in range(n_genes):
        n_edges = torch.randint(1, 4, (1,)).item()
        reaction_indices = torch.randint(0, n_reactions, (n_edges,))
        for reaction_idx in reaction_indices:
            pairs.append([gene_idx, int(reaction_idx.item())])
    return torch.tensor(pairs, dtype=torch.long).t()


def _make_reaction_edges(n_reactions: int) -> torch.Tensor:
    pairs: list[list[int]] = []
    for reaction_idx in range(n_reactions):
        n_edges = torch.randint(2, 5, (1,)).item()
        target_indices = torch.randint(0, n_reactions, (n_edges,))
        for target_idx in target_indices:
            target = int(target_idx.item())
            if target != reaction_idx:
                pairs.append([reaction_idx, target])
    return torch.tensor(pairs, dtype=torch.long).t()


def _make_binary_membership(
    n_rows: int,
    n_cols: int,
    min_members: int,
    max_members: int,
) -> torch.Tensor:
    membership = torch.zeros(n_rows, n_cols)
    for row_idx in range(n_rows):
        n_members = torch.randint(min_members, max_members + 1, (1,)).item()
        indices = torch.randperm(n_cols)[:n_members]
        membership[row_idx, indices] = 1.0
    return membership


def create_random_problem(
    n_genes: int = 50,
    n_reactions: int = 30,
    gene_edim: int = 16,
    reaction_edim: int = 32,
) -> RandomProblem:
    """Create a synthetic FlexModel problem instance for tests."""
    n_compounds = n_reactions // 2
    n_modules = max(5, n_genes // 10)

    Mcr = torch.randn(n_compounds, n_reactions) * 0.3
    Mcr[torch.rand(n_compounds, n_reactions) > 0.3] = 0

    return RandomProblem(
        n_genes=n_genes,
        n_reactions=n_reactions,
        eid_g2r=_make_gene_reaction_edges(n_genes, n_reactions),
        eid_r2r=_make_reaction_edges(n_reactions),
        Mcr=Mcr,
        Mmg=_make_binary_membership(
            n_rows=n_modules,
            n_cols=n_genes,
            min_members=3,
            max_members=8,
        ),
        Mmr=_make_binary_membership(
            n_rows=n_modules,
            n_cols=n_reactions,
            min_members=2,
            max_members=5,
        ),
        cor_wts=torch.ones(n_modules),
        gen_emb=torch.randn(n_genes, gene_edim),
        rea_emb=torch.randn(n_reactions, reaction_edim),
    )


def create_flex_module(
    n_genes: int = 50,
    n_reactions: int = 30,
    gene_edim: int = 16,
    reaction_edim: int = 32,
    use_layer_weights: bool = False,
    use_disc: bool = False,
    module_cls: type[FlexModule] = FlexModule,
) -> tuple[FlexModule, int, int]:
    """Create a FlexModule-compatible test model with random synthetic data."""
    problem = create_random_problem(
        n_genes=n_genes,
        n_reactions=n_reactions,
        gene_edim=gene_edim,
        reaction_edim=reaction_edim,
    )

    f_disc_orig = None
    if use_disc:
        f_disc_orig = torch.rand(int((problem.eid_r2r[0] != problem.eid_r2r[1]).sum().item()))

    model = module_cls(
        eid_g2r=problem.eid_g2r,
        eid_r2r=problem.eid_r2r,
        Mcr=problem.Mcr,
        Mmg=problem.Mmg,
        Mmr=problem.Mmr,
        cor_wts=problem.cor_wts,
        gen_emb=problem.gen_emb,
        rea_emb=problem.rea_emb,
        re_edim=reaction_edim,
        ge_edim=gene_edim,
        nlayers=2,
        use_disc=use_disc,
        f_disc_orig=f_disc_orig,
        use_layer_weights=use_layer_weights,
        flx_project=False,
        l_fb=1.0,
        l_pos=1.0,
        l_cor=1.0,
        l_sco=1.0,
        l_ent=0.0,
        lopt_lr=1e-3,
    )
    return model, problem.n_genes, problem.n_reactions


def make_dataloaders(
    n_genes: int,
    n_samples: int = 64,
    batch_size: int = 16,
    val_frac: float = 0.25,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Create train/validation loaders from random gene expression data."""
    torch.manual_seed(seed)
    ge = torch.randn(n_samples, n_genes).abs()
    dataset = TensorDataset(ge)
    n_val = max(1, int(n_samples * val_frac))
    n_train = n_samples - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
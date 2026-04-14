"""Example: FlexModule subclass with alignment bottleneck loss_sco.

Usage:
    model = AlignedFlexModule(
        ...,               # all standard FlexModule args
        z_dim=32,          # bottleneck dimension
        noise_std=0.1,     # z-space noise
        alpha_align=0.5,   # reconstruction vs alignment balance
    )

The bottleneck parameters (W_a for fluxes, W_b for expression) are trained
jointly with the GNN. loss_sco measures cross-modal alignment in a learned
low-dim space rather than raw high-dim cosine similarity.

Convention: modality a = fluxes, modality b = expression.
"""

from __future__ import annotations

import torch

from .alignment_bottleneck import AlignmentBottleneck, alignment_loss
from .flex_module import FlexModule
from .pairwise_concordance import pairwise_concordance
from .utils import kendall_tau


class AlignedFlexModule(FlexModule):
    """FlexModule with cross-modal alignment bottleneck for loss_sco.

    Adds a denoising bottleneck that projects fluxes (modality a) and
    metabolic enzyme expression (modality b) into a shared k-dimensional
    space. Cell-cell similarity is compared in this learned space rather
    than in the raw high-dimensional modalities.

    Additional Args:
        z_dim: Bottleneck latent dimension. Default: 32.
        noise_std: Gaussian noise std in z-space during training. Default: 0.1.
        alpha_align: Balance between reconstruction (alpha) and similarity
            alignment (1-alpha) in loss_sco. Default: 0.5.
    """

    def __init__(
        self,
        *args: object,
        z_dim: int = 32,
        noise_std: float = 0.1,
        alpha_align: float = 0.5,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.alpha_align = alpha_align

        n_reactions = int(self.Mmr.shape[1])
        n_genes = int(self.Mmg.shape[1])

        # Modality a = fluxes, modality b = expression
        self.bottleneck = AlignmentBottleneck(
            d_a=n_reactions,
            d_b=n_genes,
            k=z_dim,
            noise_std=noise_std,
        )

    def loss_cor(self, ge: torch.Tensor, flxs: torch.Tensor) -> torch.Tensor:
        """Module-level concordance: centered, no std normalization."""
        ge_sdt = ge - ge.mean(dim=1, keepdim=True)
        flxs_sdt = flxs - flxs.mean(dim=1, keepdim=True)
        a = (self.Mmg @ ge_sdt.t()).t()
        b = (self.Mmr @ flxs_sdt.t()).t()
        return pairwise_concordance(a, b, diff="relative")

    def loss_sco(self, ge: torch.Tensor, flxs: torch.Tensor) -> torch.Tensor:
        """Cross-modal alignment in learned bottleneck space.

        Projects fluxes (modality a) and expression (modality b) into a
        shared low-dim z-space via denoising tied-weight autoencoders.
        Returns combined reconstruction + similarity alignment loss.
        """
        flxs_c = flxs - flxs.mean(dim=1, keepdim=True)
        ge_c = ge - ge.mean(dim=1, keepdim=True)

        # a = fluxes, b = expression
        out = self.bottleneck(flxs_c, ge_c)

        loss = alignment_loss(
            out,
            flxs_c,
            ge_c,
            alpha=self.alpha_align,
            concordance_fn=lambda a, b: pairwise_concordance(a, b, diff="relative"),
        )

        return loss.expand(ge.shape[0])

    def _compute_tau(
        self, ge: torch.Tensor, flxs: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Kendall tau diagnostics including bottleneck z-space similarity."""
        with torch.no_grad():
            # tau_cor: module-level concordance
            ge_sdt = ge - ge.mean(dim=1, keepdim=True)
            flxs_sdt = flxs - flxs.mean(dim=1, keepdim=True)
            a_cor = (self.Mmg @ ge_sdt.t()).t()
            b_cor = (self.Mmr @ flxs_sdt.t()).t()
            tau_cor = kendall_tau(a_cor, b_cor).mean()

            # tau_sco: cell-cell similarity in bottleneck z-space
            flxs_c = flxs - flxs.mean(dim=1, keepdim=True)
            ge_c = ge - ge.mean(dim=1, keepdim=True)
            z_fl, z_ge = self.bottleneck.encode(flxs_c, ge_c)

            z_fl_n = z_fl / (z_fl.norm(dim=1, keepdim=True) + 1e-7)
            z_ge_n = z_ge / (z_ge.norm(dim=1, keepdim=True) + 1e-7)

            sim_fl = z_fl_n @ z_fl_n.t()
            sim_ge = z_ge_n @ z_ge_n.t()

            n = sim_fl.shape[0]
            idx = torch.triu_indices(n, n, offset=1, device=ge.device)
            tau_sco = kendall_tau(
                sim_fl[idx[0], idx[1]],
                sim_ge[idx[0], idx[1]],
            )

        return {"tau_cor": tau_cor, "tau_sco": tau_sco}

    def get_metabolic_embeddings(
        self, ge: torch.Tensor, flxs: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Get bottleneck embeddings for downstream analysis (UMAP, etc.).

        Args:
            ge: Gene expression, shape (n_cells, n_genes).
            flxs: Optional precomputed fluxes. If None, runs forward pass.

        Returns:
            z_ge: Expression embeddings, shape (n_cells, k).
            z_fl: Flux embeddings, shape (n_cells, k).
            flxs: Predicted fluxes, shape (n_cells, n_reactions).
            flxs_p: Projected fluxes (if flx_project), else None.
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            if flxs is None:
                flxs_raw, flxs_p = self(ge)
                flxs_use = flxs_p if flxs_p is not None else flxs_raw
            else:
                flxs_use = flxs
                flxs_raw = flxs
                flxs_p = None

            flxs_c = flxs_use - flxs_use.mean(dim=1, keepdim=True)
            ge_c = ge - ge.mean(dim=1, keepdim=True)
            z_fl, z_ge = self.bottleneck.encode(flxs_c, ge_c)

        if was_training:
            self.train()

        return {
            "z_ge": z_ge,
            "z_fl": z_fl,
            "flxs": flxs_raw,
            "flxs_p": flxs_p,
        }

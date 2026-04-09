from __future__ import annotations

import math

import lightning as L
import torch

from .flex_gnn import build_flex_gnn
from .pairwise_concordance import pairwise_concordance
from .utils import kendall_tau, sim_cor


class FlexModule(L.LightningModule):
    """Lightning module for metabolic flux prediction from gene expression.

    The module owns all graph structure, stoichiometric buffers, and optional
    learnable embeddings. It delegates flux prediction to ``gnn`` and combines
    the outputs with biologically motivated loss terms during training.

    Args:
        eid_g2r: Gene-to-reaction edge index with shape ``(2, E_gr)``.
        eid_r2r: Reaction-to-reaction edge index with shape ``(2, E_rr)``.
        Mcr: Stoichiometry matrix with shape ``(n_compounds, n_reactions)``.
        Mmg: Gene-to-module map with shape ``(n_modules, n_genes)``.
        Mmr: Module-to-reaction map with shape ``(n_modules, n_reactions)``.
        cor_wts: Optional module weights for the correlation term.
        gen_emb: Optional fixed gene embeddings with shape ``(n_genes, gene_edim)``.
        rea_emb: Optional fixed reaction embeddings with shape
            ``(n_reactions, re_edim)``.
        re_edim: Reaction embedding dimension.
        ge_edim: Gene embedding dimension.
        nlayers: Number of graph-conv layers.
        use_disc: Whether to enable concordant/discordant R→R message blending.
        f_disc_orig: Static R→R edge attribute required when ``use_disc=True``.
        use_layer_weights: Whether to combine layer outputs with learned weights.
        flx_project: Whether to project fluxes into the stoichiometric nullspace.
        l_fb: Flux-balance loss weight.
        l_pos: Positivity loss weight.
        l_cor: Module concordance loss weight.
        l_sco: Sample similarity concordance loss weight.
        l_ent: Entropy loss weight.
        lopt_lr: Adam learning rate.
        NSP: Square root of the nullspace projector. Required when
            ``flx_project=True``.
    """

    # ------------------------------------------------------------------
    # Construction and state
    # ------------------------------------------------------------------

    def __init__(
        self,
        eid_g2r: torch.Tensor,
        eid_r2r: torch.Tensor,
        Mcr: torch.Tensor,
        Mmg: torch.Tensor,
        Mmr: torch.Tensor,
        cor_wts: torch.Tensor | None = None,
        gen_emb: torch.Tensor | None = None,
        rea_emb: torch.Tensor | None = None,
        re_edim: int = 1,
        ge_edim: int = 1,
        nlayers: int = 1,
        use_disc: bool = False,
        f_disc_orig: torch.Tensor | None = None,
        use_layer_weights: bool = False,
        flx_project: bool = False,
        l_fb: float = 1,
        l_pos: float = 1,
        l_cor: float = 1,
        l_sco: float = 1,
        l_ent: float = 0,
        lopt_lr: float = 1e-3,
        NSP: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        n_genes = int(Mmg.shape[1])
        n_reactions = int(Mmr.shape[1])
        self._validate_graph_inputs(
            eid_g2r=eid_g2r,
            eid_r2r=eid_r2r,
            Mcr=Mcr,
            Mmg=Mmg,
            Mmr=Mmr,
            gen_emb=gen_emb,
            rea_emb=rea_emb,
            ge_edim=ge_edim,
            re_edim=re_edim,
            use_disc=use_disc,
            f_disc_orig=f_disc_orig,
        )
        self.gnn = build_flex_gnn(
            nr=n_reactions,
            re_edim=re_edim,
            ge_edim=ge_edim,
            nlayers=nlayers,
            use_disc=use_disc,
            f_disc_orig=f_disc_orig,
            use_layer_weights=use_layer_weights,
        )
        self.flx_project = flx_project
        self.save_hyperparameters(
            ignore=[
                "eid_g2r",
                "eid_r2r",
                "Mcr",
                "Mmg",
                "Mmr",
                "cor_wts",
                "gen_emb",
                "rea_emb",
                "NSP",
                "f_disc_orig",
            ]
        )

        self.register_buffer(
            "loss_lms",
            torch.tensor([l_fb, l_pos, l_cor, l_sco, l_ent], dtype=torch.float32),
        )

        self.register_buffer("cor_wts", cor_wts)
        if cor_wts is not None and not torch.allclose(
            cor_wts, torch.ones_like(cor_wts)
        ):
            raise ValueError(
                "Non-uniform cor_wts are not supported with pairwise concordance. "
                "Pass None or all-ones weights."
            )

        # Edge indices are buffers so they follow device moves with the module.
        self.register_buffer("eid_g2r", eid_g2r)
        self.register_buffer("eid_r2r", eid_r2r)

        self.register_buffer("Mmg", Mmg)
        self.register_buffer("Mmr", Mmr)
        self.register_buffer("Mcr", Mcr)

        if self.flx_project:
            if NSP is not None:
                self.register_buffer("Pi", NSP)
            else:
                raise ValueError("NSP must be provided if flx_project is True")
        else:
            self.register_buffer("Pi", None)

        self.r_embed = None
        self.g_embed = None

        if gen_emb is not None:
            assert (
                gen_emb.shape[0] == n_genes
            ), f"{gen_emb.shape[0]} != {n_genes}"
            assert (
                gen_emb.shape[1] == self.gnn.ge_edim
            ), f"{gen_emb.shape[1]} != {self.gnn.ge_edim}"
            self.register_buffer("gen_emb_tt", gen_emb)
        else:
            self.register_buffer("gen_emb_tt", None)
            if self.gnn.ge_edim > 1:
                self.g_embed = torch.nn.Embedding(self.Mmg.shape[1], self.gnn.ge_edim)
        if rea_emb is not None:
            assert rea_emb.shape[0] == n_reactions
            assert (
                rea_emb.shape[1] == self.gnn.re_edim
            ), f"{rea_emb.shape[1]} != {self.gnn.re_edim}"
            self.register_buffer("rea_emb_tt", rea_emb)
        else:
            self.register_buffer("rea_emb_tt", None)
            self.r_embed = torch.nn.Embedding(1, self.gnn.re_edim)

    @staticmethod
    def _validate_graph_inputs(
        eid_g2r: torch.Tensor,
        eid_r2r: torch.Tensor,
        Mcr: torch.Tensor,
        Mmg: torch.Tensor,
        Mmr: torch.Tensor,
        gen_emb: torch.Tensor | None,
        rea_emb: torch.Tensor | None,
        ge_edim: int,
        re_edim: int,
        use_disc: bool,
        f_disc_orig: torch.Tensor | None,
    ) -> None:
        """Validate graph tensors and edge attributes before building the GNN."""
        n_genes = int(Mmg.shape[1])
        n_reactions = int(Mmr.shape[1])

        if Mcr.shape[1] != n_reactions:
            raise ValueError(
                f"Mcr has {Mcr.shape[1]} reactions but Mmr has {n_reactions}."
            )
        if eid_g2r.shape[0] != 2:
            raise ValueError(f"eid_g2r must have shape (2, E), got {tuple(eid_g2r.shape)}.")
        if eid_r2r.shape[0] != 2:
            raise ValueError(f"eid_r2r must have shape (2, E), got {tuple(eid_r2r.shape)}.")
        if eid_g2r.numel() > 0:
            if int(eid_g2r[0].max().item()) >= n_genes:
                raise ValueError("eid_g2r references gene indices outside Mmg.")
            if int(eid_g2r[1].max().item()) >= n_reactions:
                raise ValueError("eid_g2r references reaction indices outside Mmr.")
        if eid_r2r.numel() > 0 and int(eid_r2r.max().item()) >= n_reactions:
            raise ValueError("eid_r2r references reaction indices outside Mmr.")
        if gen_emb is not None and gen_emb.shape != (n_genes, ge_edim):
            raise ValueError(
                "gen_emb must have shape "
                f"({n_genes}, {ge_edim}), got {tuple(gen_emb.shape)}."
            )
        if rea_emb is not None and rea_emb.shape != (n_reactions, re_edim):
            raise ValueError(
                "rea_emb must have shape "
                f"({n_reactions}, {re_edim}), got {tuple(rea_emb.shape)}."
            )
        if use_disc:
            if f_disc_orig is None:
                raise ValueError("f_disc_orig must be provided when use_disc=True.")
            n_non_self = int((eid_r2r[0] != eid_r2r[1]).sum().item())
            if f_disc_orig.ndim != 1:
                raise ValueError("f_disc_orig must be a 1D tensor of R->R edge attributes.")
            if f_disc_orig.shape[0] != n_non_self:
                raise ValueError(
                    "f_disc_orig must match the non-self-loop R->R edge count: "
                    f"expected {n_non_self}, got {f_disc_orig.shape[0]}."
                )

    @property
    def eid(self) -> dict[tuple[str, str, str], torch.Tensor]:
        """Return the hetero-edge mapping expected by the GNN."""
        return {("G", "to", "R"): self.eid_g2r, ("R", "to", "R"): self.eid_r2r}

    # ------------------------------------------------------------------
    # Forward path and embedding preparation
    # ------------------------------------------------------------------

    def forward(self, ge: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Predict metabolic fluxes from gene expression.

        Args:
            ge: Gene expression values with shape ``(batch_size, n_genes)``.

        Returns:
            Tuple of raw normalized fluxes and projected fluxes. The projected
            term is ``None`` when nullspace projection is disabled.
        """
        x_dict = self.get_gen_rea_emb(ge)
        flxs = self.gnn(x_dict, self.eid)

        if len(flxs.shape) == 1:
            flxs = flxs.unsqueeze(0)

        scle = math.sqrt(flxs.shape[-1])
        flxs = flxs / (flxs.abs().sum(dim=-1, keepdim=True) + 1e-7) * scle

        if self.Pi is not None:
            flxs_p = flxs @ self.Pi.t() @ self.Pi
            flxs_p = flxs_p / (flxs_p.abs().sum(dim=1, keepdim=True) + 1e-7) * scle
        else:
            flxs_p = None
        return flxs, flxs_p

    def get_gen_rea_emb(self, ge: torch.Tensor) -> dict[str, torch.Tensor]:
        """Build gene and reaction embeddings for one mini-batch.

        Args:
            ge: Gene expression values with shape ``(batch_size, n_genes)``.

        Returns:
            Mapping with ``"G"`` and ``"R"`` entries ready for the hetero GNN.
        """
        if self.rea_emb_tt is not None:
            rea_emb = self.rea_emb_tt.expand(ge.shape[0], self.gnn.nr, self.gnn.re_edim)
        else:
            rea_emb = self.r_embed(
                torch.zeros(self.gnn.nr, dtype=torch.int64, device=self.device)
            )
            rea_emb = rea_emb.expand(ge.shape[0], self.gnn.nr, self.gnn.re_edim)
            rea_emb = torch.nn.functional.softmax(rea_emb, dim=2)

        if self.gen_emb_tt is not None:
            gen_emb = self.gen_emb_tt.expand(ge.shape[0], ge.shape[1], self.gnn.ge_edim)
            gen_emb = gen_emb * ge.unsqueeze(2)
        elif self.gnn.ge_edim == 1:
            gen_emb = ge.unsqueeze(2)
        else:
            gen_emb = self.g_embed(torch.arange(ge.shape[1], device=self.device))
            gen_emb = gen_emb.expand(ge.shape[0], ge.shape[1], self.gnn.ge_edim)
            gen_emb = torch.nn.functional.softmax(gen_emb, dim=2)
            gen_emb = gen_emb * ge.unsqueeze(2)

        return {"G": gen_emb, "R": rea_emb}

    # ------------------------------------------------------------------
    # Loss terms and diagnostics
    # ------------------------------------------------------------------

    def loss_fb(
        self, flxs: torch.Tensor, flxs_p: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return per-sample flux-balance penalties."""
        if flxs_p is None:
            return (self.Mcr @ flxs.t()).square().sum(dim=0)
        return (flxs - flxs_p).square().sum(dim=1)

    def loss_pos(self, flxs: torch.Tensor) -> torch.Tensor:
        """Return per-sample penalties for negative flux values."""
        return torch.nn.functional.relu(-flxs).square().sum(dim=1)

    def loss_cor(self, ge: torch.Tensor, flxs: torch.Tensor) -> torch.Tensor:
        """Return module-level concordance loss for each sample.

        Both gene expression and flux magnitudes are projected into module
        space before comparing relative pairwise rankings.
        """
        ge_sdt = ge - ge.mean(dim=1, keepdim=True)
        ge_sdt = ge_sdt / (ge_sdt.std(dim=1, keepdim=True) + 1e-7)
        flxs_sdt = flxs.abs() - flxs.abs().mean(dim=1, keepdim=True)
        flxs_sdt = flxs_sdt / (flxs_sdt.std(dim=1, keepdim=True) + 1e-7)
        a = (self.Mmg @ ge_sdt.t()).t()
        b = (self.Mmr @ flxs_sdt.t()).t()
        return pairwise_concordance(a, b, diff="relative")

    def loss_sco(self, ge: torch.Tensor, flxs: torch.Tensor) -> torch.Tensor:
        """Return sample-similarity concordance loss for each sample.

        Similarity is a batch-level quantity, so the same scalar is expanded to
        all samples in the batch.
        """
        flxs_n = flxs - flxs.mean(dim=0, keepdim=True)
        ge_n = ge - ge.mean(dim=0, keepdim=True)
        return sim_cor(flxs_n, ge_n).expand(ge.shape[0])

    def loss_ent(self, flxs: torch.Tensor) -> torch.Tensor:
        """Return per-sample negative entropy of absolute flux values."""
        flux_prob = flxs.abs()
        flux_prob = flux_prob / (flux_prob.sum(dim=1, keepdim=True) + 1e-7)
        return (flux_prob * torch.log(flux_prob + 1e-7)).sum(dim=1)

    def losses(
        self, ge: torch.Tensor, flxs: torch.Tensor, flxs_p: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Assemble all per-sample loss components into one tensor.

        The returned tensor has shape ``(batch_size, 5)`` ordered as
        ``[L_fb, L_pos, L_cor, L_sco, L_ent]``. When projected fluxes are
        available, downstream terms use the projected values.
        """
        nsam = ge.shape[0]
        lses = torch.empty((nsam, 5), device=self.device)
        lses[:, 0] = self.loss_fb(flxs, flxs_p)
        flxs = flxs_p if flxs_p is not None else flxs
        lses[:, 1] = self.loss_pos(flxs)
        lses[:, 2] = self.loss_cor(ge, flxs)
        lses[:, 3] = self.loss_sco(ge, flxs)
        lses[:, 4] = self.loss_ent(flxs)
        return lses

    def _compute_tau(
        self, ge: torch.Tensor, flxs: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute cheap, no-grad Kendall tau diagnostics for monitoring."""
        with torch.no_grad():
            ge_sdt = ge - ge.mean(dim=1, keepdim=True)
            ge_sdt = ge_sdt / (ge_sdt.std(dim=1, keepdim=True) + 1e-7)
            flxs_sdt = flxs.abs() - flxs.abs().mean(dim=1, keepdim=True)
            flxs_sdt = flxs_sdt / (flxs_sdt.std(dim=1, keepdim=True) + 1e-7)
            a_cor = (self.Mmg @ ge_sdt.t()).t()
            b_cor = (self.Mmr @ flxs_sdt.t()).t()
            tau_cor = kendall_tau(a_cor, b_cor).mean()

            flxs_n = flxs - flxs.mean(dim=0, keepdim=True)
            ge_n = ge - ge.mean(dim=0, keepdim=True)
            flx_u = flxs_n / flxs_n.norm(dim=1, keepdim=True)
            ge_u = ge_n / ge_n.norm(dim=1, keepdim=True)
            sim_f = flx_u @ flx_u.t()
            sim_g = ge_u @ ge_u.t()
            n = sim_f.shape[0]
            idx = torch.triu_indices(n, n, offset=1, device=ge.device)
            tau_sco = kendall_tau(sim_f[idx[0], idx[1]], sim_g[idx[0], idx[1]])

        return {"tau_cor": tau_cor, "tau_sco": tau_sco}

    # ------------------------------------------------------------------
    # Training and validation
    # ------------------------------------------------------------------

    def _log_stage_losses(self, stage: str, lses: torch.Tensor, loss: torch.Tensor) -> None:
        """Log weighted loss terms for one optimization stage."""
        names = ["fb", "pos", "cor", "sco", "ent"]
        for idx, name in enumerate(names):
            self.log(
                f"{stage}_loss-{name}",
                (lses[idx] * self.loss_lms[idx]).detach(),
                prog_bar=True,
            )
        self.log(f"{stage}_loss-all", loss.detach(), prog_bar=True)

    def _shared_step(
        self,
        stage: str,
        batch: tuple[torch.Tensor, ...],
        batch_idx: int,
    ) -> torch.Tensor:
        """Run the shared train/validation step logic for one batch."""
        del batch_idx
        x, *_ = batch
        flxs, flxs_p = self(x)
        lses = self.losses(x, flxs, flxs_p).mean(dim=0)
        loss = lses @ self.loss_lms

        self._log_stage_losses(stage, lses, loss)

        taus = self._compute_tau(x, flxs_p if flxs_p is not None else flxs)
        self.log(f"{stage}_tau-cor", taus["tau_cor"], prog_bar=False)
        self.log(f"{stage}_tau-sco", taus["tau_sco"], prog_bar=False)
        return loss

    def training_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        """Compute and log the training objective for one batch.

        Args:
            batch: Batch tuple whose first element is the gene-expression tensor.
            batch_idx: Mini-batch index.

        Returns:
            Weighted total loss.
        """
        return self._shared_step("trn", batch, batch_idx)

    def validation_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        """Compute and log the validation objective for one batch.

        Args:
            batch: Batch tuple whose first element is the gene-expression tensor.
            batch_idx: Mini-batch index.

        Returns:
            Weighted total loss.
        """
        return self._shared_step("val", batch, batch_idx)

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the Adam optimizer used by Lightning."""
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lopt_lr)

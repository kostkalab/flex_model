import math
import torch
import lightning as L
from torch_geometric.data import HeteroData

from .utils import sim_cor, kendall_tau
from .pairwise_concordance import pairwise_concordance

class FlexModule(L.LightningModule):
    """Lightning module for metabolic flux prediction from gene expression.
    
    Wraps a GNN model for training with multiple loss components that enforce
    biological constraints (flux balance, positivity) and consistency between
    predicted fluxes and gene expression patterns (correlation, similarity).
    
    Loss Components:
        L_fb (Flux Balance): Enforces stoichiometric constraints S·v = 0 or projection consistency
        L_pos (Positivity): Penalizes negative fluxes to encourage forward reactions
        L_cor (Correlation): Module-level Spearman correlation between expression and flux magnitude
        L_sco (Similarity): Sample-wise similarity preservation between expression and flux spaces
        L_ent (Entropy): With l_ent > 0, encourages uniform distributions; for sparsity use l_ent < 0
    
    Args:
        gnn: The GNN model that predicts fluxes from gene/reaction embeddings
        eid_g2r: Edge index tensor for gene→reaction edges, shape (2, E_gr)
        eid_r2r: Edge index tensor for reaction→reaction edges, shape (2, E_rr)
        Mcr: Stoichiometry matrix S, shape (n_compounds, n_reactions)
        Mmg: Gene-to-module mapping matrix, shape (n_modules, n_genes)
        Mmr: Module-to-reaction mapping matrix, shape (n_modules, n_reactions)
        cor_wts: Module importance weights for correlation loss, shape (n_modules,)
        gen_emb: Optional precomputed gene embeddings, shape (n_genes, gene_edim)
        rea_emb: Optional precomputed reaction embeddings, shape (n_reactions, re_edim)
        flx_project: If True, project fluxes into stoichiometric nullspace using NSP projector
        l_fb: Weight for flux balance loss component. Default: 1.0
        l_pos: Weight for positivity loss component. Default: 1.0
        l_cor: Weight for correlation loss component. Default: 1.0
        l_sco: Weight for similarity correlation loss component. Default: 1.0
        l_ent: Weight for entropy loss component. Default: 0.0
        lopt_lr: Learning rate for Adam optimizer. Default: 1e-3
        NSP: Nullspace projector (square root), shape (n_null, n_reactions). Required if flx_project=True
    
    Attributes:
        loss_lms: Tensor of loss component weights [l_fb, l_pos, l_cor, l_sco, l_ent]
        eid: Dictionary mapping edge types to edge indices
        Pi: Nullspace projector matrix (if flx_project=True)
        g_embed: Learnable gene embeddings (if gen_emb not provided)
        r_embed: Learnable reaction embeddings (if rea_emb not provided)
        Similarity centering is done inline in the loss with tensor ops.
    """

    def __init__(
        self,
        gnn: torch.nn.Module,
        eid_g2r: torch.Tensor,
        eid_r2r: torch.Tensor,
        Mcr: torch.Tensor,
        Mmg: torch.Tensor,
        Mmr: torch.Tensor,
        cor_wts: torch.Tensor | None = None,
        gen_emb: torch.Tensor | None = None,
        rea_emb: torch.Tensor | None = None,
        flx_project: bool = False,
        l_fb: float = 1,  # - flux balance loss weight
        l_pos: float = 1,  # - positivity loss weight
        l_cor: float = 1,  # - correlation loss weight
        l_sco: float = 1,  # - similarity correlation loss weight
        l_ent: float = 0,  # - entropy loss weight
        lopt_lr: float = 1e-3,
        NSP: torch.Tensor | None = None,
    ):
        super().__init__()
        self.gnn = gnn
        self.flx_project = flx_project
        self.save_hyperparameters(ignore=["gnn", "Mcr", "Mmg", "Mmr", "cor_wts"])
        
        self.register_buffer(
            "loss_lms", torch.tensor([l_fb, l_pos, l_cor, l_sco, l_ent], dtype=torch.float32)
        )
        
        self.register_buffer("cor_wts", cor_wts)
        if cor_wts is not None and not torch.allclose(cor_wts, torch.ones_like(cor_wts)):
            raise ValueError(
                "Non-uniform cor_wts are not supported with pairwise concordance. "
                "Pass None or all-ones weights."
            )

        # - make eids parameters of the model
        #   so that they are put on the proper device
        #   cannot use ParameterDict b/c it does not allow for keys being tuples
        #   parameters apparently need to be *direct* attributes of the module
        self.register_buffer("eid_g2r", eid_g2r)
        self.register_buffer("eid_r2r", eid_r2r)

        # - matrix connecting genes to modules (for loss)
        self.register_buffer("Mmg", Mmg)
        # - matrix connecting modules to reactions (for loss)
        self.register_buffer("Mmr", Mmr)
        # - stoichiometry matrix
        self.register_buffer("Mcr", Mcr)

        # - fixme: get rid of flx_project and just use Pi
        if self.flx_project:
            # - add square root of projection operator to model
            if NSP is not None:
                self.register_buffer("Pi", NSP)
            else:
                raise ValueError("NSP must be provided if flx_project is True")
        else:
            self.register_buffer("Pi", None)

        # - assert that are both None, or the same size as the number of genes/reactions
        # - and make embedings accessible if they are provided.
        self.r_embed = None
        self.g_embed = None

        if gen_emb is not None:
            assert (
                gen_emb.shape[0] == Mmg.shape[1]
            ), f"{gen_emb.shape[0]} != {Mmg.shape[1]}"
            assert (
                gen_emb.shape[1] == self.gnn.ge_edim
            ), f"{gen_emb.shape[1]} != {self.gnn.ge_edim}"
            self.register_buffer("gen_emb_tt", gen_emb)
        else:
            self.register_buffer("gen_emb_tt", None)
            if self.gnn.ge_edim > 1:
                # - learn embedding for each gene
                self.g_embed = torch.nn.Embedding(self.Mmg.shape[1], self.gnn.ge_edim)
            # - otherwise selg.gen_embed stays None
        if rea_emb is not None:
            assert rea_emb.shape[0] == Mmr.shape[1]
            assert (
                rea_emb.shape[1] == self.gnn.re_edim
            ), f"{rea_emb.shape[1]} != {self.gnn.re_edim}"
            self.register_buffer("rea_emb_tt", rea_emb)
        else:
            self.register_buffer("rea_emb_tt", None)
            # - all reactions get the same embedding
            self.r_embed = torch.nn.Embedding(1, self.gnn.re_edim)

    @property
    def eid(self):
        return {("G", "to", "R"): self.eid_g2r, ("R", "to", "R"): self.eid_r2r}

    def forward(self, ge: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Predict metabolic fluxes from gene expression.
        
        Args:
            ge: Gene expression values, shape (batch_size, n_genes)
        
        Returns:
            flxs: Predicted fluxes (L1-normalized and scaled), shape (batch_size, n_reactions)
            flxs_p: Projected fluxes (if flx_project=True), else None
        """
        x_dict = self.get_gen_rea_emb(ge)
        flxs = self.gnn(x_dict, self.eid)

        # Handle single-sample case (add batch dimension)
        if len(flxs.shape) == 1:
            flxs = flxs.unsqueeze(0)

        # L1-normalize fluxes and scale by sqrt(n_reactions) to preserve magnitude
        # Add epsilon to avoid NaN when flux vectors are all-zero (common at initialization)
        scle = math.sqrt(flxs.shape[-1])
        flxs = flxs / (flxs.abs().sum(dim=-1, keepdim=True) + 1e-7) * scle

        # Project fluxes into stoichiometric nullspace if enabled
        if self.Pi is not None:
            # Project: v_proj = v @ P^T @ P where P is nullspace projector square root
            flxs_p = flxs @ self.Pi.t() @ self.Pi
            # Re-normalize projected fluxes
            flxs_p = flxs_p / (flxs_p.abs().sum(dim=1, keepdim=True) + 1e-7) * scle
        else:
            flxs_p = None
        return flxs, flxs_p

    def loss_fb(
        self, flxs: torch.Tensor, flxs_p: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Flux balance loss: ||S·v||² or ||v - v_proj||² (shape: batch_size)."""
        if flxs_p is None:
            return (self.Mcr @ flxs.t()).square().sum(dim=0)
        else:
            return (flxs - flxs_p).square().sum(dim=1)

    def loss_pos(self, flxs: torch.Tensor) -> torch.Tensor:
        """Positivity loss: Σ relu(-v)² (shape: batch_size)."""
        return torch.nn.functional.relu(-flxs).square().sum(dim=1)

    def loss_cor(self, ge: torch.Tensor, flxs: torch.Tensor) -> torch.Tensor:
        """Module-level pairwise concordance loss (shape: batch_size).

        Projects gene expression and flux magnitudes into module space,
        then measures ranking agreement via pairwise concordance with
        relative differences.
        """
        ge_sdt = ge - ge.mean(dim=1, keepdim=True)
        ge_sdt = ge_sdt / (ge_sdt.std(dim=1, keepdim=True) + 1e-7)
        flxs_sdt = flxs.abs() - flxs.abs().mean(dim=1, keepdim=True)
        flxs_sdt = flxs_sdt / (flxs_sdt.std(dim=1, keepdim=True) + 1e-7)
        a = (self.Mmg @ ge_sdt.t()).t()      # (batch, n_modules)
        b = (self.Mmr @ flxs_sdt.t()).t()    # (batch, n_modules)
        return pairwise_concordance(a, b, diff="relative")

    def loss_sco(self, ge: torch.Tensor, flxs: torch.Tensor) -> torch.Tensor:
        """Sample similarity concordance loss (shape: batch_size).

        Measures ranking agreement between pairwise sample similarities in
        flux vs expression space.  Returns the same scalar for every sample
        in the batch (similarity is a batch-level property).
        """
        flxs_n = flxs - flxs.mean(dim=0, keepdim=True)
        ge_n = ge - ge.mean(dim=0, keepdim=True)
        return sim_cor(flxs_n, ge_n).expand(ge.shape[0])

    def loss_ent(self, flxs: torch.Tensor) -> torch.Tensor:
        """Entropy loss: Σ p log(p) where p = |v|/Σ|v| (shape: batch_size).
        Negative entropy — minimizing with l_ent > 0 encourages uniformity; l_ent < 0 encourages sparsity."""
        flux_prob = flxs.abs()
        flux_prob = flux_prob / (flux_prob.sum(dim=1, keepdim=True) + 1e-7)
        return (flux_prob * torch.log(flux_prob + 1e-7)).sum(dim=1)

    def losses(
        self, ge: torch.Tensor, flxs: torch.Tensor, flxs_p: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Assemble all loss components into a (batch_size, 5) tensor: [L_fb, L_pos, L_cor, L_sco, L_ent].

        Resolves flux projection once here so individual loss methods receive the
        effective fluxes (flxs_p when available, otherwise flxs).
        Override individual loss_* methods to customise specific components.
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

    def get_gen_rea_emb(self, ge) -> dict[str, torch.Tensor]:
        """Prepare gene and reaction embeddings for GNN input.
        
        Embedding Strategy:
            - If precomputed embeddings provided: Use them (weighted by expression for genes)
            - If not provided: Learn embeddings during training
                * Genes: Learned per-gene embeddings (or raw expression if ge_edim=1)
                * Reactions: Single shared embedding for all reactions
        
        Args:
            ge: Gene expression values, shape (batch_size, n_genes)
        
        Returns:
            Dictionary with keys "G" (gene embeddings) and "R" (reaction embeddings),
            both with shape (batch_size, n_nodes, embedding_dim)
        """
        # Reaction embeddings
        if self.rea_emb_tt is not None:
            # Use precomputed reaction embeddings (expand to batch size)
            rea_emb = self.rea_emb_tt.expand(
                ge.shape[0], self.gnn.nr, self.gnn.re_edim
            )
        else:
            # Learn single shared embedding for all reactions
            rea_emb = self.r_embed(
                torch.zeros(self.gnn.nr, dtype=torch.int64, device=self.device)
            )
            rea_emb = rea_emb.expand(ge.shape[0], self.gnn.nr, self.gnn.re_edim)
            # Softmax normalization for learned embeddings
            rea_emb = torch.nn.functional.softmax(rea_emb, dim=2)
        # Gene embeddings (weighted by expression values)
        if self.gen_emb_tt is not None:
            # Use precomputed gene embeddings, weighted by expression
            gen_emb = self.gen_emb_tt.expand(
                ge.shape[0], ge.shape[1], self.gnn.ge_edim
            )
            gen_emb = gen_emb * ge.unsqueeze(2)  # Element-wise multiply by expression
        else:
            # Learn gene embeddings during training
            if self.gnn.ge_edim == 1:
                # 1D case: use raw expression values
                gen_emb = ge.unsqueeze(2)
            else:
                # Multi-dimensional: learn per-gene embeddings
                gen_emb = self.g_embed(
                    torch.arange(ge.shape[1], device=self.device)
                )
                gen_emb = gen_emb.expand(
                    ge.shape[0], ge.shape[1], self.gnn.ge_edim
                )
                # Softmax normalization for learned embeddings
                gen_emb = torch.nn.functional.softmax(gen_emb, dim=2)
                # Weight embeddings by expression: (batch, genes, edim) * (batch, genes, 1)
                gen_emb = gen_emb * ge.unsqueeze(2)

        return {"G": gen_emb, "R": rea_emb}

    def _compute_tau(self, ge: torch.Tensor, flxs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute Kendall tau diagnostics for cor and sco (no grad, cheap)."""
        with torch.no_grad():
            # Module-level tau (matches loss_cor projection)
            ge_sdt = ge - ge.mean(dim=1, keepdim=True)
            ge_sdt = ge_sdt / (ge_sdt.std(dim=1, keepdim=True) + 1e-7)
            flxs_sdt = flxs.abs() - flxs.abs().mean(dim=1, keepdim=True)
            flxs_sdt = flxs_sdt / (flxs_sdt.std(dim=1, keepdim=True) + 1e-7)
            a_cor = (self.Mmg @ ge_sdt.t()).t()
            b_cor = (self.Mmr @ flxs_sdt.t()).t()
            tau_cor = kendall_tau(a_cor, b_cor).mean()

            # Sample similarity tau (matches sim_cor projection)
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

    def training_step(self, batch, batch_idx):
        """Compute training loss for one batch.
        
        Args:
            batch: Tuple where first element is gene expression tensor
            batch_idx: Batch index (unused)
        
        Returns:
            Weighted sum of all loss components
        """
        x, *_ = batch
        flxs, flxs_p = self(x)
        lses = self.losses(x, flxs, flxs_p)  # shape: (batch_size, 5)
        lses = lses.mean(dim=0)  # Average over batch, shape: (5,)
        loss = lses @ self.loss_lms  # Weighted sum with loss weights, shape: ()

        # Explicitly detach logging values so metric aggregation never retains graphs.
        self.log("trn_loss-fb", (lses[0] * self.loss_lms[0]).detach(), prog_bar=True)
        self.log("trn_loss-pos", (lses[1] * self.loss_lms[1]).detach(), prog_bar=True)
        self.log("trn_loss-cor", (lses[2] * self.loss_lms[2]).detach(), prog_bar=True)
        self.log("trn_loss-sco", (lses[3] * self.loss_lms[3]).detach(), prog_bar=True)
        self.log("trn_loss-ent", (lses[4] * self.loss_lms[4]).detach(), prog_bar=True)
        self.log("trn_loss-all", loss.detach(), prog_bar=True)

        taus = self._compute_tau(x, flxs_p if flxs_p is not None else flxs)
        self.log("trn_tau-cor", taus["tau_cor"], prog_bar=False)
        self.log("trn_tau-sco", taus["tau_sco"], prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        """Compute validation loss for one batch.
        
        Args:
            batch: Tuple where first element is gene expression tensor
            batch_idx: Batch index (unused)
        
        Returns:
            Weighted sum of all loss components
        """
        x, *_ = batch
        flxs, flxs_p = self(x)
        lses = self.losses(x, flxs, flxs_p)  # shape: (batch_size, 5)
        lses = lses.mean(dim=0)  # Average over batch, shape: (5,)
        loss = lses @ self.loss_lms  # Weighted sum with loss weights, shape: ()
        self.log("val_loss-fb", (lses[0] * self.loss_lms[0]).detach(), prog_bar=True)
        self.log("val_loss-pos", (lses[1] * self.loss_lms[1]).detach(), prog_bar=True)
        self.log("val_loss-cor", (lses[2] * self.loss_lms[2]).detach(), prog_bar=True)
        self.log("val_loss-sco", (lses[3] * self.loss_lms[3]).detach(), prog_bar=True)
        self.log("val_loss-ent", (lses[4] * self.loss_lms[4]).detach(), prog_bar=True)
        self.log("val_loss-all", loss.detach(), prog_bar=True)

        taus = self._compute_tau(x, flxs_p if flxs_p is not None else flxs)
        self.log("val_tau-cor", taus["tau_cor"], prog_bar=False)
        self.log("val_tau-sco", taus["tau_sco"], prog_bar=False)

        return loss

    def configure_optimizers(self):
        """Configure Adam optimizer with learning rate from hyperparameters.
        
        Returns:
            Configured optimizer instance
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lopt_lr)
        return optimizer

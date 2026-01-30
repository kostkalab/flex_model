import math
import torch
import lightning as L
from torch_geometric.data import HeteroData

from .utils import sim_cor, diff_spearman, MeanBatchNorm1d

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
        fl_bn, ge_bn: Batch normalization layers for similarity correlation
    """

    def __init__(
        self,
        gnn: torch.nn.Module,
        eid_g2r: torch.Tensor,
        eid_r2r: torch.Tensor,
        Mcr: torch.Tensor,
        Mmg: torch.Tensor,
        Mmr: torch.Tensor,
        cor_wts: torch.Tensor,
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
        self.loss_lms = torch.nn.Parameter(
            torch.Tensor([l_fb, l_pos, l_cor, l_sco, l_ent]), requires_grad=False
        )

        self.register_buffer("cor_wts", cor_wts)

        # - make eids parameters of the model
        #   so that they are put on the proper device
        #   cannot use ParameterDict b/c it does not allow for keys being tuples
        #   parameters apparently need to be *direct* attributes of the module
        self.register_buffer("eid_g2r", eid_g2r)
        self.register_buffer("eid_r2r", eid_r2r)
        self.eid = {("G", "to", "R"): self.eid_g2r, ("R", "to", "R"): self.eid_r2r}

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

        # - batch normalization for similarity correlation
        self.fl_bn = MeanBatchNorm1d(self.gnn.nr)
        self.ge_bn = MeanBatchNorm1d(self.Mmg.shape[1])

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

    def losses(
        self, ge: torch.Tensor, flxs: torch.Tensor, flxs_p: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute all loss components for metabolic flux prediction.
        
        Loss Formulas:
            L_fb = ||S·v||² (flux balance) or ||v - v_proj||² (projection consistency)
            L_pos = Σ max(0, -v)² (penalty for negative fluxes)
            L_cor = 1 - ρ_spearman(M_g·g_std, M_r·|v|_std) (module correlation)
            L_sco = 1 - sim_cor(v, g) (sample similarity preservation)
            L_ent = Σ p log(p) where p = |v|/Σ|v| (negative entropy; l_ent > 0 → uniformity, l_ent < 0 → sparsity)
        
        Args:
            ge: Gene expression, shape (batch_size, n_genes)
            flxs: Predicted fluxes, shape (batch_size, n_reactions)
            flxs_p: Projected fluxes (optional), shape (batch_size, n_reactions)
        
        Returns:
            Loss components tensor, shape (batch_size, 5) where columns are [L_fb, L_pos, L_cor, L_sco, L_ent]
        """
        nsam = ge.shape[0]
        losses = torch.empty((nsam, 5)).to(self.device)

        # L_fb: Flux Balance Loss
        if flxs_p is None:
            losses[:, 0] = (self.Mcr @ flxs.t()).square().sum(dim=0)
        else:
            losses[:, 0] = (flxs - flxs_p).square().sum(dim=1)
            flxs = flxs_p

        # L_pos: Positivity Loss (penalize negative fluxes)
        # Formula: Σ (max(0, -v))² = Σ (relu(-v))²
        losses[:, 1] = (
            torch.nn.functional.relu(-flxs).square()
        ).sum(dim=1)
        
        # L_cor: Module-level Correlation Loss
        # Standardize gene expression and flux magnitudes per sample (z-score)
        ge_sdt = ge - ge.mean(dim=1, keepdim=True)
        ge_sdt = ge_sdt / (ge_sdt.std(dim=1, keepdim=True) + 1e-7)  # epsilon for numerical stability
        flxs_sdt = flxs.abs() - flxs.abs().mean(dim=1, keepdim=True)
        flxs_sdt = flxs_sdt / (flxs_sdt.std(dim=1, keepdim=True) + 1e-7)
        
        # Aggregate to module level and compute weighted Spearman correlation
        losses[:, 2] = torch.ones(nsam, device=ge.device) - diff_spearman(
            (self.Mmg @ ge_sdt.t()).t(),  # Module-level expression
            (self.Mmr @ flxs_sdt.t()).t(),  # Module-level flux magnitude
            wts=self.cor_wts,  # Module importance weights
        )
        
        # L_sco: Similarity Correlation Loss (batch-level consistency)
        # Mean-center features before computing pairwise similarities
        flxs_n = self.fl_bn(flxs)
        ge_n = self.ge_bn(ge)
        lsco = sim_cor(flxs_n, ge_n)
        losses[:, 3] = torch.ones(nsam, device=ge.device) - lsco

        # L_ent: Entropy Loss (with positive l_ent: encourages uniform distributions)
        # Computes negative entropy: Σ p log(p) where p = |v|/Σ|v|
        # Note: This is -H (negative of traditional entropy H = -Σ p log(p))
        # Minimizing with l_ent > 0 → maximizes H → encourages uniformity (NOT sparsity)
        # For sparsity, use l_ent < 0 or flip sign below
        flux_prob = flxs.abs()
        flux_prob = flux_prob / (flux_prob.sum(dim=1, keepdim=True) + 1e-7)  # epsilon for numerical stability, normalizeflux_prob.sum(dim=1, keepdim=True)
        losses[:, 4] = (flux_prob * torch.log(flux_prob + 1e-7)).sum(dim=1)  # epsilon prevents log(0)

        return losses

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

        self.log("trn_loss-fb", lses[0] * self.loss_lms[0], prog_bar=True)
        self.log("trn_loss-pos", lses[1] * self.loss_lms[1], prog_bar=True)
        self.log("trn_loss-cor", lses[2] * self.loss_lms[2], prog_bar=True)
        self.log("trn_loss-sco", lses[3] * self.loss_lms[3], prog_bar=True)
        self.log("trn_loss-ent", lses[4] * self.loss_lms[4], prog_bar=True)
        self.log("trn_loss-all", loss, prog_bar=True)

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
        self.log("val_loss-fb", lses[0] * self.loss_lms[0], prog_bar=True)
        self.log("val_loss-pos", lses[1] * self.loss_lms[1], prog_bar=True)
        self.log("val_loss-cor", lses[2] * self.loss_lms[2], prog_bar=True)
        self.log("val_loss-sco", lses[3] * self.loss_lms[3], prog_bar=True)
        self.log("val_loss-ent", lses[4] * self.loss_lms[4], prog_bar=True)
        self.log("val_loss-all", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        """Configure Adam optimizer with learning rate from hyperparameters.
        
        Returns:
            Configured optimizer instance
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lopt_lr)
        return optimizer

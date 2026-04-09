"""Utility functions for metabolic flux prediction models.

Provides correlation metrics, normalization utilities, and similarity measures
for comparing predicted fluxes with gene expression data.

Key components:
- kendall_tau: Non-differentiable Kendall's tau for diagnostics
- sim_cor: Similarity-based concordance between flux and expression patterns
- MeanBatchNorm1d: Custom batch normalization with only mean centering
"""

from __future__ import annotations

import torch


@torch.no_grad()
def kendall_tau(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Non-differentiable Kendall's tau-a for diagnostics.

    O(D²) sign-product computation — no sorting needed, cheaper than Spearman.

    Args:
        a: shape (batch, D) or (D,)
        b: shape (batch, D) or (D,)

    Returns:
        Tau value(s). Scalar if 1D inputs, shape (batch,) if 2D.
    """
    squeeze = a.dim() == 1
    if squeeze:
        a = a.unsqueeze(0)
        b = b.unsqueeze(0)
    # (batch, D, 1) - (batch, 1, D) → (batch, D, D)
    da = a.unsqueeze(2) - a.unsqueeze(1)
    db = b.unsqueeze(2) - b.unsqueeze(1)
    # Upper triangle only
    D = a.shape[1]
    idx = torch.triu_indices(D, D, offset=1, device=a.device)
    da = da[:, idx[0], idx[1]]
    db = db[:, idx[0], idx[1]]
    tau = (da.sign() * db.sign()).mean(dim=1)
    return tau.squeeze(0) if squeeze else tau


def sim_cor(
    flx: torch.Tensor, ge: torch.Tensor, scale: bool = True, nsamp: int | None = None
) -> torch.Tensor:
    """Similarity-based pairwise concordance between flux and expression patterns.

    Measures whether pairwise sample similarities are rank-consistent between
    flux and expression spaces using pairwise concordance with relative diffs.

    Args:
        flx: Predicted fluxes, shape (n_samples, n_reactions)
        ge: Gene expression values, shape (n_samples, n_genes)
        scale: If True, L2-normalize to unit vectors (enables cosine similarity)
        nsamp: If provided, randomly subsample this many samples for efficiency.
               Must be < n_samples/2

    Returns:
        Scalar pairwise concordance loss between flux and expression similarities

    Raises:
        AssertionError: If sample counts don't match or nsamp is invalid
    """
    from .pairwise_concordance import pairwise_concordance

    # - assert that flx and ge have the same number of rows
    assert flx.shape[0] == ge.shape[0], "flx and ge must have the same number of rows"

    # - assert that if nsamp is not None, it is a positive integer less than the 1/2 the number of rows of flx
    if nsamp is not None:
        assert isinstance(nsamp, int), "nsamp must be an integer"
        assert nsamp > 0, "nsamp must be positive"
        assert (
            nsamp < flx.shape[0] / 2
        ), "nsamp must be less than half the number of rows of flx"
        # - subsample the rows of flx and ge
        idx = torch.randperm(flx.shape[0], device=flx.device)[:nsamp]
        flx_s = flx[idx]
        ge_s = ge[idx]
    else:
        flx_s = flx
        ge_s = ge

    if scale:
        flx_s = flx_s / (flx_s.norm(dim=1, keepdim=True) + 1e-7)
        ge_s = ge_s / (ge_s.norm(dim=1, keepdim=True) + 1e-7)

    sim_f = flx_s @ flx_s.t()
    sim_g = ge_s @ ge_s.t()

    # Extract upper triangle without allocating a full ones+triu mask
    n = sim_f.shape[0]
    idx = torch.triu_indices(n, n, offset=1, device=sim_f.device)
    sim_fv = sim_f[idx[0], idx[1]]
    sim_gv = sim_g[idx[0], idx[1]]

    # pairwise_concordance expects (batch, D); treat as single sample
    return pairwise_concordance(
        sim_gv.unsqueeze(0), sim_fv.unsqueeze(0), diff="relative"
    ).squeeze(0)


class MeanBatchNorm1d(torch.nn.Module):
    """Batch normalization with mean centering only (no variance scaling).

    Unlike standard BatchNorm, this only subtracts the running mean without
    dividing by standard deviation. Useful when you want to center features
    without changing their relative scales, particularly for correlation-based
    losses where variance normalization is handled separately.

    Maintains exponential moving average of feature means across batches.
    During training, updates running statistics. During eval, uses fixed
    running mean.

    Args:
        num_features: Number of features (C from input shape N×C)
        momentum: Momentum for running mean update. If None, uses cumulative
                  moving average. Default: 0.1

    Shape:
        - Input: (N, C) where N is batch size, C is num_features
        - Output: (N, C), same shape as input

    Attributes:
        running_mean: Exponential moving average of feature means, shape (C,)
        num_batches_tracked: Counter for batches seen during training
    """

    def __init__(
        self,
        num_features,
        momentum=0.1,
    ):
        super(MeanBatchNorm1d, self).__init__()

        self.num_features = num_features
        self.momentum = momentum
        self.running_mean = torch.nn.parameter.Buffer(torch.zeros(num_features))
        self.num_batches_tracked = torch.nn.parameter.Buffer(
            torch.tensor(0, dtype=torch.int64)
        )

    def forward(self, input):
        """Apply mean centering to input.

        Args:
            input: Input tensor, shape (N, C)

        Returns:
            Mean-centered tensor, same shape as input

        Raises:
            AssertionError: If input is not 2D
        """
        assert input.dim() == 2, "input must be 2D tensor"

        exponential_average_factor = 0.0

        if self.training:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / float(self.num_batches_tracked)
            else:  # use exponential moving average
                exponential_average_factor = self.momentum
            mean = input.mean(dim=0)
            # Update running statistics for eval mode, outside autograd to avoid
            # carrying graph references between batches.
            with torch.no_grad():
                if self.num_batches_tracked == 1:
                    self.running_mean.copy_(mean.detach())
                else:
                    self.running_mean.mul_(1 - exponential_average_factor)
                    self.running_mean.add_(exponential_average_factor * mean.detach())
            # Always use current batch mean during training so gradients flow.
        else:
            mean = self.running_mean

        input = input - mean
        return input


def get_S_NSprojectorSR(
    S: torch.Tensor, thrsh: float = 0.999, device: str = "cuda"
) -> torch.Tensor:
    """Compute square root of nullspace projector for stoichiometry matrix.

    Uses SVD to find the nullspace of the stoichiometry matrix S, which represents
    the steady-state flux space where S·v = 0. Returns the square root of the
    orthogonal projector onto this nullspace for memory-efficient projection.

    Mathematical background:
    - Full projector: P = V_null^T @ V_null, where V_null are right singular vectors
      corresponding to near-zero singular values
    - This function returns V_null^T (the square root), so projection is done as:
      v_projected = v @ V_null^T @ V_null

    The threshold determines which singular values are considered "near-zero" by
    looking at cumulative energy: singular values after thrsh fraction of cumulative
    sum are treated as nullspace components.

    Args:
        S: Stoichiometry matrix, shape (n_compounds, n_reactions). Can be sparse or dense
        thrsh: Threshold for cumulative singular value fraction. Values after this
               fraction are considered nullspace. Default: 0.999 (keep 99.9% of energy
               in the non-null space, discard remaining as nullspace)
        device: Device to perform computation on. Default: "cuda"

    Returns:
        Square root of nullspace projector, shape (n_null, n_reactions), where n_null
        is the dimension of the nullspace. To project: v_proj = v @ P.T @ P
        Returned as float32 for memory efficiency.

    Note:
        The full projector matrix P would be P.T @ P, but we return only P (square root)
        to save memory. Apply projection as: fluxes @ P.T @ P
    """

    if S.is_sparse:
        U, S, Vh = torch.linalg.svd(S.to_dense().to(device), driver="gesvd")
    else:
        U, S, Vh = torch.linalg.svd(S.to(device), driver="gesvd")

    # Compute cumulative fraction of singular values (energy distribution)
    # Values before 'cut' index represent the range space (non-null)
    # Values after 'cut' index represent the nullspace (near-zero singular values)
    frac = S.cumsum(dim=0) / S.sum()
    cut = (
        (frac < thrsh).count_nonzero().item()
    )  # Index where cumulative fraction exceeds threshold

    # Return right singular vectors corresponding to nullspace (after cutoff)
    # Full projector would be: Vh[cut:, :].T @ Vh[cut:, :]
    # We return only the square root for memory efficiency
    return Vh[cut:, :].to(torch.float32)

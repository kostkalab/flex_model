"""Utility functions for metabolic flux prediction models.

Provides correlation metrics, normalization utilities, and similarity measures
for comparing predicted fluxes with gene expression data.

Key components:
- diff_spearman: Differentiable Spearman correlation using soft ranking
- wcor: Weighted Pearson correlation coefficient
- sim_cor: Similarity-based correlation between flux and expression patterns
- MeanBatchNorm1d: Custom batch normalization with only mean centering
"""

import torch
from typing import Optional
import torchsort


def diff_spearman(pred, target, wts=None, regularization_strength=1e-2):
    """Differentiable Spearman correlation coefficient using soft ranking.

    Computes a differentiable approximation to Spearman's rank correlation
    using torchsort's soft ranking. Allows backpropagation through ranking
    operations, enabling use in loss functions.

    Args:
        pred: Predicted values, shape (n,) or (batch, n)
        target: Target values, same shape as pred
        wts: Optional feature weights, shape (n,). If None, uniform weights used
        regularization_strength: Soft ranking temperature. Lower = closer to hard ranks

    Returns:
        Correlation coefficient(s). Scalar if 1D inputs, shape (batch,) if 2D inputs

    Raises:
        AssertionError: If pred/target shapes don't match or dimensions > 2
    """
    assert pred.shape == target.shape, "pred and target must have the same shape"
    out_dtype = pred.dtype

    if pred.dim() == 1:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    if pred.dim() > 2:
        raise ValueError("pred and target must be 1D or 2D tensors")

    if wts is not None:
        assert wts.dim() == 1, "wts must be a 1D tensor"
        assert (
            wts.shape[0] == pred.shape[1]
        ), "wts must have the same length as the number of columns in pred and target"
    else:
        wts = torch.ones(pred.shape[1], dtype=pred.dtype).to(pred.device)

    # torchsort kernels are most stable in FP32 under AMP/bfloat16 training.
    # Only upcast when running in mixed precision or low-precision input dtypes.
    use_fp32_torchsort = (
        torch.is_autocast_enabled()
        or pred.dtype in (torch.float16, torch.bfloat16)
        or target.dtype in (torch.float16, torch.bfloat16)
    )
    if use_fp32_torchsort:
        pred = pred.float()
        target = target.float()
        wts = wts.float().to(pred.device)
    else:
        wts = wts.to(pred.device, dtype=pred.dtype)

    pred_rank = torchsort.soft_rank(pred, regularization_strength=regularization_strength)
    target_rank = torchsort.soft_rank(target, regularization_strength=regularization_strength)

    correlation = wcor(pred_rank, target_rank, wts)
    return correlation.to(out_dtype)


def wcor(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Weighted Pearson correlation coefficient between rows of x and y.

    Computes weighted correlation: cor(x,y) = cov_w(x,y) / sqrt(var_w(x) * var_w(y))
    where weights are normalized to sum to 1. Useful for emphasizing specific
    features when computing correlations (e.g., module-level importance).

    Args:
        x: Feature matrix, shape (n,) or (batch, n)
        y: Feature matrix, same shape as x
        w: Feature weights, shape (n,). Normalized internally

    Returns:
        Correlation coefficient(s). Scalar if 1D, shape (batch,) if 2D

    Raises:
        AssertionError: If shapes are incompatible or dimensions invalid
    """

    # - assert that x, y, are the same shape
    assert x.shape == y.shape, "x and y must be the same shape"
    # - assert x and y are either both vectors or both 2D tensors
    assert x.dim() in [1, 2], "x and y must be 1D or 2D tensors"
    if x.dim() == 1:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
    # - deal with the weights
    assert w.dim() == 1, "w must be a 1D tensor"
    # - assert that w is the same length as the rows of x
    assert w.shape[0] == x.shape[1], "w must have the same length as the rows of x and y"

    ws = w.sum()
    wn = w / ws

    mx = (x * wn).sum(dim=1)
    my = (y * wn).sum(dim=1)

    # Centre once, then broadcast-multiply (avoids repeated transposes)
    xc = x - mx.unsqueeze(1)
    yc = y - my.unsqueeze(1)
    cxy = (xc * yc * wn).sum(dim=1)
    cxx = (xc.square() * wn).sum(dim=1)
    cyy = (yc.square() * wn).sum(dim=1)

    return cxy / (cxx * cyy).sqrt()


def sim_cor(
    flx: torch.Tensor, ge: torch.Tensor, scale: bool = True, nsamp: Optional[int] = None
) -> torch.Tensor:
    """Similarity-based correlation between flux and gene expression patterns.

    Measures how well pairwise sample similarities are preserved between flux
    and expression spaces. Algorithm:
    1. Normalize samples to unit vectors (if scale=True)
    2. Compute pairwise cosine similarities for both flux and expression
    3. Extract upper triangle similarity vectors
    4. Compute Spearman correlation between similarity vectors

    This metric captures whether samples that are similar in expression space
    are also similar in predicted flux space, providing a batch-level consistency
    measure beyond pointwise correlations.

    Args:
        flx: Predicted fluxes, shape (n_samples, n_reactions)
        ge: Gene expression values, shape (n_samples, n_genes)
        scale: If True, L2-normalize to unit vectors (enables cosine similarity)
        nsamp: If provided, randomly subsample this many samples for efficiency.
               Must be < n_samples/2

    Returns:
        Scalar correlation coefficient between flux and expression similarities

    Raises:
        AssertionError: If sample counts don't match or nsamp is invalid
    """

    # - assert that flx and ge have the same number of rows
    assert flx.shape[0] == ge.shape[0], "flx and ge must have the same number of rows"

    # - assert that if nsamp is not None, it is a positive integer less than the 1/2 the number of rows of flx
    if nsamp is not None:
        assert isinstance(nsamp, int), "nsamp must be an integer"
        assert nsamp > 0, "nsamp must be positive"
        assert nsamp < flx.shape[0] / 2, "nsamp must be less than half the number of rows of flx"
        # - subsample the rows of flx and ge
        idx = torch.randperm(flx.shape[0], device=flx.device)[:nsamp]
        flx_s = flx[idx]
        ge_s = ge[idx]
    else:
        flx_s = flx
        ge_s = ge

    # Normalize to unit vectors and compute cosine similarity matrices
    flx_n = flx_s / flx_s.norm(dim=1, keepdim=True)
    sim_f = flx_n @ flx_n.t()
    ge_n = ge_s / ge_s.norm(dim=1, keepdim=True)
    sim_g = ge_n @ ge_n.t()

    # Extract upper triangle without allocating a full ones+triu mask
    n = sim_f.shape[0]
    idx = torch.triu_indices(n, n, offset=1, device=sim_f.device)
    sim_fv = sim_f[idx[0], idx[1]]
    sim_gv = sim_g[idx[0], idx[1]]

    # - calculate correlation between the two vectors
    sim_cor = diff_spearman(sim_fv, sim_gv)

    return sim_cor


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
        self.num_batches_tracked = torch.nn.parameter.Buffer(torch.tensor(0, dtype=torch.int64))

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
    cut = (frac < thrsh).count_nonzero().item()  # Index where cumulative fraction exceeds threshold

    # Return right singular vectors corresponding to nullspace (after cutoff)
    # Full projector would be: Vh[cut:, :].T @ Vh[cut:, :]
    # We return only the square root for memory efficiency
    return Vh[cut:, :].to(torch.float32)


def diff_spearman_ai(x, y, tau0=1.0, eps=1e-8):
    """Differentiable, affine-invariant Spearman correlation for use in losses.

    Each input is normalized to zero mean and unit MADM scale (1.48 * MAD,
    a consistent estimator of σ) using detached statistics before soft-ranking.
    The soft-ranking temperature is then set to tau0 / N, where N is the vector
    length. After MADM normalization the typical gap between adjacent sorted
    values is ~1.35/N (IQR/N for Gaussian data), so tau = tau0/N sits right in
    the soft zone: tau0=1 gives one gap-width of smoothing, regardless of N or
    the original input scale. This avoids both the near-hard-ranking collapse
    (tau << gap) and the over-smoothing bias (tau >> gap).

    Handles 1-D and 2-D (batched) inputs and is safe under AMP / bfloat16
    training (upcasts to fp32 for torchsort, returns in the caller's dtype).

    Args:
        x: First input, shape (n,) or (batch, n)
        y: Second input, same shape as x
        tau0: Temperature multiplier. Effective tau = tau0 / N. Default 1.0
        eps: Numerical stability term added to MADM denominator. Default 1e-8

    Returns:
        Correlation coefficient(s). Scalar if 1-D inputs, shape (batch,) if 2-D
    """
    assert x.shape == y.shape, "x and y must have the same shape"
    if x.dim() > 2:
        raise ValueError("x and y must be 1D or 2D tensors")

    out_dtype = x.dtype
    squeeze = x.dim() == 1
    if squeeze:
        x, y = x.unsqueeze(0), y.unsqueeze(0)

    # torchsort is most stable in fp32 under AMP / low-precision training
    if torch.is_autocast_enabled() or x.dtype in (torch.float16, torch.bfloat16):
        x, y = x.float(), y.float()

    # Normalize to unit-MADM scale using detached statistics.
    # Detaching mean and MAD is critical: gradient flows only through the
    # linear rescaling (x - mu) / mad, whose Jacobian is a constant diagonal
    # diag(1/mad). Without detach, the correction term -x/mad^2 * d(mad)/dx_j
    # shrinks and destabilizes the gradient as p evolves during optimization.
    def _madm_normalize(t):
        mu  = t.mean(dim=-1, keepdim=True).detach()
        mad = 1.48 * (t - mu).abs().mean(dim=-1, keepdim=True).detach().clamp(min=eps)
        return (t - mu) / mad

    x = _madm_normalize(x)
    y = _madm_normalize(y)
    # tau = tau0 / N targets one inter-element gap of smoothing after normalization
    tau = tau0 / x.shape[-1]
    r_x = torchsort.soft_rank(x, regularization_strength=tau)
    r_y = torchsort.soft_rank(y, regularization_strength=tau)

    # Pearson on ranks = Spearman
    r_x = r_x - r_x.mean(dim=-1, keepdim=True).detach()  # - detaching may improve  stability.
    r_y = r_y - r_y.mean(dim=-1, keepdim=True).detach()
    corr = (r_x * r_y).sum(dim=-1) / (r_x.norm(dim=-1) * r_y.norm(dim=-1) + eps)

    corr = corr.squeeze(0) if squeeze else corr
    return corr.to(out_dtype)


class _StraightThrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx, corr_val, corr_grad):
        return corr_val          # forward sees the accurate value

    @staticmethod
    def backward(ctx, g):
        return None, g           # gradient flows only through corr_grad


def diff_spearman_ai_st(x, y, tau_val=0.5, tau_grad=3.0, eps=1e-8):
    """Straight-through differentiable affine-invariant Spearman correlation.

    Decouples accuracy from gradient quality by using two tau values:
    - tau_val (small): controls the forward value — stays close to true Spearman
    - tau_grad (large): controls the backward gradient — keeps gradient signal
      alive throughout optimization even when ranks have largely separated

    Implemented via torch.autograd.Function: forward returns corr_val directly;
    backward ignores corr_val and routes the upstream gradient into corr_grad only.

    Both calls share the same MADM normalization convention as diff_spearman_ai,
    with effective tau = tau_val/N and tau_grad/N.

    Args:
        x: First input, shape (n,) or (batch, n)
        y: Second input, same shape as x
        tau_val:  Temperature multiplier for the forward value. Default 0.5
        tau_grad: Temperature multiplier for the gradient path. Default 3.0
        eps: Numerical stability term. Default 1e-8

    Returns:
        Correlation coefficient(s). Scalar if 1-D inputs, shape (batch,) if 2-D
    """
    corr_val  = diff_spearman_ai(x, y, tau0=tau_val,  eps=eps)
    corr_grad = diff_spearman_ai(x, y, tau0=tau_grad, eps=eps)
    return _StraightThrough.apply(corr_val, corr_grad)


import torch


def monotone_invariant_spearman(
    pred,
    target,
    K=64,
    sigma=0.1,
    eps=1e-8,
):
    """
    Differentiable, approximately monotone-invariant Spearman correlation.

    Key ideas:
    - Use detached quantile anchors (from sorting) → invariant reference frame
    - Softly interpolate values onto anchors → approximate CDF
    - Correlate centered CDF values

    Args:
        pred, target: (n,) or (batch, n)
        K: number of quantile anchors
        sigma: softness of interpolation (relative to normalized scale)
        eps: numerical stability

    Returns:
        correlation: scalar or (batch,)
    """

    assert pred.shape == target.shape

    if pred.dim() == 1:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    B, n = pred.shape
    device = pred.device
    dtype = pred.dtype

    # ---- 1. scale normalization (affine invariance) ----
    def normalize(x):
        x = x - x.mean(dim=-1, keepdim=True)
        scale = x.abs().mean(dim=-1, keepdim=True)
        return x / (scale + eps)

    pred_n = normalize(pred)
    target_n = normalize(target)

    # ---- 2. detached quantile anchors ----
    q = torch.linspace(0, 1, K, device=device, dtype=dtype)

    pred_sorted = torch.sort(pred_n, dim=-1).values.detach()
    target_sorted = torch.sort(target_n, dim=-1).values.detach()

    pred_bins = torch.quantile(pred_sorted, q, dim=-1).T  # (B, K)
    target_bins = torch.quantile(target_sorted, q, dim=-1).T

    # ---- 3. soft assignment → approximate CDF ----
    def soft_cdf(x, bins):
        diff = x.unsqueeze(-1) - bins.unsqueeze(-2)  # (B, n, K)
        weights = torch.softmax(-(diff**2) / (2 * sigma**2 + eps), dim=-1)
        return (weights * q).sum(dim=-1)  # (B, n)

    u_pred = soft_cdf(pred_n, pred_bins)
    u_target = soft_cdf(target_n, target_bins)

    # ---- 4. center (true CDF mean = 0.5) ----
    u_pred = u_pred - 0.5
    u_target = u_target - 0.5

    # ---- 5. correlation ----
    num = (u_pred * u_target).sum(dim=-1)
    denom = u_pred.norm(dim=-1) * u_target.norm(dim=-1) + eps

    corr = num / denom

    return corr.squeeze(0) if corr.shape[0] == 1 else corr


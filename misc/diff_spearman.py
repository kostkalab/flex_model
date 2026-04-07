"""Differentiable Spearman correlation variants (archived).

These functions were previously part of flexModel.utils. They are kept here
for reference and potential future use but are NOT imported by the main package.

Requires: torchsort (pip install torchsort --no-binary torchsort --no-build-isolation)
"""

import torch
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

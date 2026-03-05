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
        assert wts.shape[0] == pred.shape[1], "wts must have the same length as the number of columns in pred and target"
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
    assert (
        w.shape[0] == x.shape[1]
    ), "w must have the same length as the rows of x and y"

    ws = w.sum()
    wn = w / ws

    mx = (x * wn).sum(dim=1)
    my = (y * wn).sum(dim=1)

    cxy = ((x.t() - mx).t() * (y.t() - my).t() * wn).sum(dim=1)
    cxx = ((x.t() - mx).t().square() * wn).sum(dim=1)
    cyy = ((y.t() - my).t().square() * wn).sum(dim=1)

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

    # - nomralize to unit circle and calculate dot product
    flx_n = (flx_s.t() / flx_s.norm(dim=1)).t()
    sim_f = flx_n @ flx_n.t()
    ge_n = (ge_s.t() / ge_s.norm(dim=1)).t()
    sim_g = ge_n @ ge_n.t()

    # - vectorize upper triangle of the matrices
    sim_fv = sim_f[torch.triu(torch.ones(sim_f.shape, device=sim_f.device), diagonal=1) == 1]
    sim_gv = sim_g[torch.triu(torch.ones(sim_g.shape, device=sim_g.device), diagonal=1) == 1]

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
        self.num_batches_tracked = torch.nn.parameter.Buffer(torch.tensor(0, dtype=torch.float32))

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
            # Keep running statistics outside autograd to avoid carrying graph
            # references between batches.
            with torch.no_grad():
                if self.num_batches_tracked == 1:
                    self.running_mean.copy_(mean.detach())
                else:
                    self.running_mean.mul_(1 - exponential_average_factor)
                    self.running_mean.add_(exponential_average_factor * mean.detach())
            if self.num_batches_tracked != 1:
                mean = self.running_mean
        else:
            mean = self.running_mean

        input = input - mean
        return input


def get_S_NSprojectorSR(S: torch.Tensor, thrsh: float = 0.999, device: str = "cuda") -> torch.Tensor:
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
    cut = (frac < thrsh).count_nonzero().item() # Index where cumulative fraction exceeds threshold

    # Return right singular vectors corresponding to nullspace (after cutoff)
    # Full projector would be: Vh[cut:, :].T @ Vh[cut:, :]
    # We return only the square root for memory efficiency
    return Vh[cut:, :].to(torch.float32)

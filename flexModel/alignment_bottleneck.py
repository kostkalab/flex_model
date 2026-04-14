"""Cross-modal alignment bottleneck.

Projects two modalities into a shared low-dimensional space via tied-weight
linear autoencoders with noise injection. The shared z-space enables
meaningful cell-cell distance comparisons across modalities.

Architecture per modality:

    encode:  z = W @ x                    (d_input → k)
    noise:   z_noisy = z + ε,  ε ~ N(0, σ²I)
    decode:  x_hat = W^T @ z_noisy        (k → d_input)

Loss:
    L = α * (||x_a - x_a_hat||² + ||x_b - x_b_hat||²)
      + (1 - α) * concordance(dist(Z_a), dist(Z_b))

The reconstruction prevents collapse (projecting everything to the same
point can't reconstruct). The noise prevents memorization. The concordance
aligns the two z-spaces so cell-cell distances are comparable.

The tied weights (W^T for decode) mean the optimal W recovers PCA for
each modality, but the similarity term pulls them toward cross-modal
alignment — similar to CCA but trainable end-to-end.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor


class AlignmentBottleneck(nn.Module):
    """Denoising cross-modal alignment bottleneck.

    Two tied-weight linear autoencoders (one per modality) with shared
    latent dimension k. Noise is injected in z-space during training.
    The module provides both reconstruction loss and aligned z embeddings
    for downstream similarity comparison.

    Args:
        d_a: Input dimension for modality a.
        d_b: Input dimension for modality b.
        k: Latent bottleneck dimension.
        noise_std: Standard deviation of Gaussian noise in z-space during
            training. Set to 0 to disable noise (not recommended).
    """

    def __init__(
        self,
        d_a: int,
        d_b: int,
        k: int = 32,
        noise_std: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_a = d_a
        self.d_b = d_b
        self.k = k
        self.noise_std = noise_std

        # Tied-weight encoders: encode with W, decode with W^T
        # No bias — centering is handled externally
        self.W_a = nn.Linear(d_a, k, bias=False)
        self.W_b = nn.Linear(d_b, k, bias=False)

    def encode(
        self, x_a: Tensor, x_b: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Project both modalities into the shared latent space.

        Args:
            x_a: Modality a values, shape (batch, d_a).
            x_b: Modality b values, shape (batch, d_b).

        Returns:
            z_a: Modality a embeddings, shape (batch, k).
            z_b: Modality b embeddings, shape (batch, k).
        """
        return self.W_a(x_a), self.W_b(x_b)

    def forward(
        self, x_a: Tensor, x_b: Tensor
    ) -> dict[str, Tensor]:
        """Full forward: encode → noise → decode.

        Args:
            x_a: Modality a values, shape (batch, d_a). Should be centered.
            x_b: Modality b values, shape (batch, d_b). Should be centered.

        Returns:
            Dictionary with:
                z_a: Clean modality a embeddings, shape (batch, k).
                z_b: Clean modality b embeddings, shape (batch, k).
                x_a_hat: Reconstructed modality a, shape (batch, d_a).
                x_b_hat: Reconstructed modality b, shape (batch, d_b).
        """
        z_a, z_b = self.encode(x_a, x_b)

        # Inject noise during training only
        if self.training and self.noise_std > 0:
            z_a_noisy = z_a + self.noise_std * torch.randn_like(z_a)
            z_b_noisy = z_b + self.noise_std * torch.randn_like(z_b)
        else:
            z_a_noisy = z_a
            z_b_noisy = z_b

        # Decode with tied weights: x_hat = W^T @ z_noisy
        x_a_hat = z_a_noisy @ self.W_a.weight  # (batch, d_a)
        x_b_hat = z_b_noisy @ self.W_b.weight  # (batch, d_b)

        return {
            "z_a": z_a,
            "z_b": z_b,
            "x_a_hat": x_a_hat,
            "x_b_hat": x_b_hat,
        }


def alignment_loss(
    out: dict[str, Tensor],
    x_a: Tensor,
    x_b: Tensor,
    alpha: float = 0.5,
    concordance_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
) -> Tensor:
    """Compute combined reconstruction + alignment loss.

    Args:
        out: Output dict from AlignmentBottleneck.forward().
        x_a: Original (centered) modality a, shape (batch, d_a).
        x_b: Original (centered) modality b, shape (batch, d_b).
        alpha: Balance between reconstruction (alpha) and alignment (1-alpha).
        concordance_fn: Callable(sim_a, sim_b) → scalar loss. Both inputs are
            shape (1, n_pairs) upper-triangle similarity vectors. If None, uses
            MSE between similarity matrices.

    Returns:
        Scalar loss.
    """
    # Reconstruction: MSE normalized by dimension
    recon_a = (x_a - out["x_a_hat"]).pow(2).mean()
    recon_b = (x_b - out["x_b_hat"]).pow(2).mean()
    recon = recon_a + recon_b

    # Alignment: cell-cell similarity conservation in z-space
    z_a = out["z_a"]
    z_b = out["z_b"]

    # L2-normalize for cosine similarity
    z_a_n = z_a / (z_a.norm(dim=1, keepdim=True) + 1e-7)
    z_b_n = z_b / (z_b.norm(dim=1, keepdim=True) + 1e-7)

    sim_a = z_a_n @ z_a_n.t()  # (batch, batch)
    sim_b = z_b_n @ z_b_n.t()  # (batch, batch)

    # Extract upper triangle
    n = sim_a.shape[0]
    idx = torch.triu_indices(n, n, offset=1, device=x_a.device)
    sim_a_v = sim_a[idx[0], idx[1]]
    sim_b_v = sim_b[idx[0], idx[1]]

    if concordance_fn is not None:
        align = concordance_fn(sim_b_v.unsqueeze(0), sim_a_v.unsqueeze(0)).squeeze(0)
    else:
        align = (sim_a_v - sim_b_v).pow(2).mean()

    return alpha * recon + (1.0 - alpha) * align

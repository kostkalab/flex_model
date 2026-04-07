"""
Pairwise concordance loss for ranking consistency.

General-purpose: given two vectors per sample, penalise pairs whose ordering
disagrees.

Three computation strategies:

  - Exact (delta=None):           O(D²) pairwise.  Fine for D ≲ 1000.
  - Binned (delta>0):             O(K²) where K = number of occupied cells
    in the joint (a, b) contingency table.
  - Binned + sampled (delta>0, n_pairs>0):  O(n_pairs).  Draw cell pairs
    with probability ∝ n_r·n_s (count product).  Unbiased estimator of
    the full binned loss.  Cost independent of D and K.

Two loss modes controlled by a single `scale` parameter:
  - "hinge": relu(1 - scale * da * db).  Reaches zero when the scaled
    concordance product exceeds 1.  Larger scale → easier to satisfy.
  - "logistic": softplus(-scale * da * db).  Smooth, nonzero floor.
    Larger scale → steeper sigmoid, stronger discordance penalty.

Two difference modes: "absolute" (raw) and "relative" (per-pair normalised,
bounded O(1) products, detached denominator for clean gradients).

See pairwise_concordance.md for mathematical details.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# ===========================================================================
# Public API
# ===========================================================================


def pairwise_concordance(
    a: torch.Tensor,
    b: torch.Tensor,
    mode: str = "hinge",
    scale: float = 1.0,
    diff: str = "absolute",
    delta: float | None = None,
    n_pairs: int | None = None,
    c: float = 0.1,
) -> torch.Tensor:
    """Pairwise concordance between two vectors.

    For each sample and each pair of indices (i, j), scores whether a and b
    agree on the ordering of i vs j.

    Args:
        a:       Reference vector, shape (batch, D).
        b:       Predicted vector, shape (batch, D).
        mode:    "hinge" or "logistic". Default: "hinge".
        scale:   Multiplies the concordance product before the activation.
                 Hinge:    relu(1 - scale * da * db).  Larger → easier to
                           satisfy, loss reaches zero sooner.
                 Logistic: softplus(-scale * da * db).  Larger → steeper
                           sigmoid, stronger discordance penalty.
                 Default: 1.0.
        diff:    "absolute" or "relative". Default: "absolute".
        delta:   Bin resolution for approximate computation.
                 None → exact O(D²).  > 0 → binned O(K²).
                 Default: None.
        n_pairs: Sampled cell pairs (requires delta > 0).
                 None → exhaustive.  > 0 → stochastic O(n_pairs).
                 Default: None.
        c:       Gradient scale floor for diff="relative".  Gradient
                 capped at 2/c.  Ignored for "absolute".  Default: 0.1.

    Returns:
        Per-sample loss, shape (batch,).

    Raises:
        ValueError: If delta <= 0, n_pairs <= 0, n_pairs set without delta,
                    or unknown mode/diff.
    """
    if delta is not None and delta <= 0:
        raise ValueError(f"delta must be > 0, got {delta}")
    if n_pairs is not None:
        if n_pairs <= 0:
            raise ValueError(f"n_pairs must be > 0, got {n_pairs}")
        if delta is None:
            raise ValueError("n_pairs requires delta > 0 (binned mode)")

    if delta is not None:
        return _pairwise_binned(a, b, mode, scale, diff, delta, n_pairs, c)
    else:
        return _pairwise_exact(a, b, mode, scale, diff, c)


def _graph_connected_zero(a, b):
    """Return a per-sample zero connected to both a and b's autograd graph."""
    return a.sum(dim=-1) * 0 + b.sum(dim=-1) * 0


# ===========================================================================
# Exact O(D²) computation
# ===========================================================================


def _pairwise_exact(a, b, mode, scale, diff, c):
    D = a.shape[1]
    if D < 2:
        return _graph_connected_zero(a, b)

    da, db = _differences(a, b, diff, c)
    product = scale * da * db  # (batch, D, D)
    pair_loss = _apply_mode(product, mode)  # (batch, D, D)

    mask = torch.triu(torch.ones(D, D, device=a.device, dtype=torch.bool), diagonal=1)
    return pair_loss[:, mask].mean(dim=1)


# ===========================================================================
# Binned O(K²) computation
# ===========================================================================


def _pairwise_binned(a, b, mode, scale, diff, delta, n_pairs, c):
    """Binned approximation.  Batch-serial (Python loop over samples)."""
    batch, D = a.shape

    if D < 2:
        return _graph_connected_zero(a, b)

    a_bin = torch.round(a / delta).long()
    b_bin = torch.round(b / delta).long()

    losses = []
    for n in range(batch):
        loss_n = _binned_single(
            a[n],
            b[n],
            a_bin[n],
            b_bin[n],
            mode,
            scale,
            diff,
            delta,
            n_pairs,
            c,
        )
        losses.append(loss_n)

    return torch.stack(losses)


def _binned_single(a_n, b_n, a_bin_n, b_bin_n, mode, scale, diff, delta, n_pairs, c):
    """Binned concordance for a single sample."""
    a_off = a_bin_n - a_bin_n.min()
    b_off = b_bin_n - b_bin_n.min()
    b_range = b_off.max() + 1
    cell_key = a_off * b_range + b_off

    unique_keys, inverse, counts = cell_key.unique(
        return_inverse=True,
        return_counts=True,
    )
    K = unique_keys.shape[0]

    if K < 2:
        return a_n.sum() * 0 + b_n.sum() * 0

    cell_a = torch.zeros(K, device=a_n.device, dtype=a_n.dtype)
    cell_b = torch.zeros(K, device=a_n.device, dtype=b_n.dtype)
    cell_a.scatter_add_(0, inverse, a_n)
    cell_b.scatter_add_(0, inverse, b_n)
    cell_a = cell_a / counts.float()
    cell_b = cell_b / counts.float()

    if n_pairs is not None:
        return _binned_sampled(cell_a, cell_b, counts, K, mode, scale, diff, n_pairs, c)
    else:
        return _binned_exhaustive(cell_a, cell_b, counts, K, mode, scale, diff, c)


def _binned_exhaustive(cell_a, cell_b, counts, K, mode, scale, diff, c):
    ca = cell_a.unsqueeze(0)
    cb = cell_b.unsqueeze(0)

    da, db = _differences(ca, cb, diff, c)
    product = scale * da * db
    pair_loss = _apply_mode(product, mode).squeeze(0)

    weights = counts.float().unsqueeze(1) * counts.float().unsqueeze(0)
    mask = torch.triu(
        torch.ones(K, K, device=cell_a.device, dtype=torch.bool), diagonal=1
    )

    total_pairs = weights[mask].sum()
    if total_pairs < 1:
        return cell_a.sum() * 0 + cell_b.sum() * 0

    return (pair_loss[mask] * weights[mask]).sum() / total_pairs


def _binned_sampled(cell_a, cell_b, counts, K, mode, scale, diff, n_pairs, c):
    probs = counts.float()
    probs = probs / probs.sum()

    r = torch.multinomial(probs, n_pairs * 3, replacement=True)
    s = torch.multinomial(probs, n_pairs * 3, replacement=True)

    valid = r < s
    r, s = r[valid], s[valid]

    if r.shape[0] == 0:
        return cell_a.sum() * 0 + cell_b.sum() * 0

    n_use = min(n_pairs, r.shape[0])
    r, s = r[:n_use], s[:n_use]

    da_rs = cell_a[r] - cell_a[s]
    db_rs = cell_b[r] - cell_b[s]

    if diff == "relative":
        da_denom = (cell_a[r].abs() + cell_a[s].abs() + c).detach()
        db_denom = (cell_b[r].abs() + cell_b[s].abs() + c).detach()
        da_rs = 2.0 * da_rs / da_denom
        db_rs = 2.0 * db_rs / db_denom

    product = scale * da_rs * db_rs
    return _apply_mode_1d(product, mode)


# ===========================================================================
# Shared helpers
# ===========================================================================


def _differences(a, b, diff, c):
    if diff == "absolute":
        da = a.unsqueeze(2) - a.unsqueeze(1)
        db = b.unsqueeze(2) - b.unsqueeze(1)
    elif diff == "relative":
        da = _relative_diff(a, c)
        db = _relative_diff(b, c)
    else:
        raise ValueError(f"Unknown diff: {diff!r}. Use 'absolute' or 'relative'.")
    return da, db


def _relative_diff(x, c):
    """Per-pair relative differences with detached denominator.

        δ_ij = 2(x_i - x_j) / (|x_i| + |x_j| + c)  [denom detached]

    Forward: bounded in [-2, 2].
    Backward: gradient = 2 / d_ij^stop, capped at 2/c.
              Detach prevents quotient-rule suppression of large-difference
              pairs and removes cross-element coupling.
    """
    xi = x.unsqueeze(2)
    xj = x.unsqueeze(1)
    numer = xi - xj
    denom = (xi.abs() + xj.abs() + c).detach()
    return 2.0 * numer / denom


def _apply_mode(product, mode):
    """Apply loss activation to pairwise products (any shape)."""
    if mode == "logistic":
        return F.softplus(-product)
    elif mode == "hinge":
        return F.relu(1.0 - product)
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'logistic' or 'hinge'.")


def _apply_mode_1d(product, mode):
    """Apply loss activation and reduce to scalar mean."""
    if mode == "logistic":
        return F.softplus(-product).mean()
    elif mode == "hinge":
        return F.relu(1.0 - product).mean()
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'logistic' or 'hinge'.")

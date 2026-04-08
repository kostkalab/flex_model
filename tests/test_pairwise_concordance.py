"""Test suite for pairwise_concordance.

Covers correctness, gradient health, mode/path variants, binned consistency,
sampling bias, large-D sparsity, and Hessian off-diagonal structure.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pytest
import torch
from scipy.stats import kendalltau, spearmanr

from flexModel.pairwise_concordance import pairwise_concordance


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------


def _make_random(
    batch: int, d: int, sparse_frac: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    a = torch.randn(batch, d)
    b = torch.randn(batch, d)
    if sparse_frac > 0:
        a[torch.rand(batch, d) < sparse_frac] = 0.0
        b[torch.rand(batch, d) < sparse_frac] = 0.0
    return a, b


def _make_concordant(
    batch: int, d: int, sparse_frac: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a, b with same ordering (perfect concordance)."""
    a = torch.randn(batch, d).abs()
    b = a * torch.rand(batch, d).clamp(min=0.5) + 0.01
    if sparse_frac > 0:
        mask = torch.rand(batch, d) < sparse_frac
        a[mask] = 0.0
        b[mask] = 0.0
    return a, b


def _make_discordant(batch: int, d: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a, b with opposite ordering."""
    a = torch.randn(batch, d)
    b = -a * torch.rand(batch, d).clamp(min=0.5)
    return a, b


def _scipy_spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean Spearman rho across batch (scipy reference)."""
    rhos = []
    for i in range(a.shape[0]):
        rho, _ = spearmanr(a[i].detach().numpy(), b[i].detach().numpy())
        rhos.append(rho)
    return float(np.mean(rhos))


def _scipy_tau(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean Kendall tau across batch (scipy reference)."""
    taus = []
    for i in range(a.shape[0]):
        tau, _ = kendalltau(a[i].detach().numpy(), b[i].detach().numpy())
        taus.append(tau)
    return float(np.mean(taus))


# ---------------------------------------------------------------------------
# Helpers for optimisation / Hessian tests
# ---------------------------------------------------------------------------


def _hessian_offdiag(
    loss_fn: Callable, a: torch.Tensor, b_param: torch.Tensor, seed: int = 12345
) -> float:
    """Mean absolute off-diagonal Hessian element via finite differences.

    For hinge mode, pairs near the threshold contribute a kink where finite
    differences pick up nonzero H_od.  This is the relu non-differentiability,
    not coupling.
    """
    eps_h = 1e-3
    d = b_param.shape[1]
    b0 = b_param.detach().clone()

    b_param.data.copy_(b0)
    b_param.requires_grad_(True)
    if b_param.grad is not None:
        b_param.grad = None
    torch.manual_seed(seed)
    loss = loss_fn(a, b_param)
    loss[0].backward(retain_graph=False)
    g0 = b_param.grad[0].clone()

    n_probe = min(d, 32)
    offdiag_sum = 0.0
    count = 0
    for j in range(n_probe):
        b_param.data.copy_(b0)
        b_param.data[0, j] += eps_h
        b_param.requires_grad_(True)
        if b_param.grad is not None:
            b_param.grad = None
        torch.manual_seed(seed)
        loss = loss_fn(a, b_param)
        loss[0].backward(retain_graph=False)
        gj = b_param.grad[0].clone()

        dg = (gj - g0) / eps_h
        mask = torch.ones(d, dtype=torch.bool)
        mask[j] = False
        offdiag_sum += dg[mask].abs().mean().item()
        count += 1

    b_param.data.copy_(b0)
    b_param.requires_grad_(True)
    return offdiag_sum / count if count > 0 else 0.0


def _run_optim(
    a: torch.Tensor,
    b_init: torch.Tensor,
    steps: int,
    lr: float,
    mode: str,
    diff: str,
    delta: float | None,
    n_pairs: int | None,
    c: float,
) -> tuple[torch.Tensor, list[dict]]:
    """Run optimisation, return (b_param, trajectory)."""
    b_param = b_init.clone().requires_grad_(True)
    opt = torch.optim.Adam([b_param], lr=lr)
    log_steps = sorted(set(list(range(0, steps, max(1, steps // 10))) + [steps - 1]))
    trajectory: list[dict] = []

    for step in range(steps):
        opt.zero_grad()
        loss = pairwise_concordance(
            a, b_param, mode=mode, diff=diff, delta=delta, n_pairs=n_pairs, c=c
        )
        loss.sum().backward()
        if step in log_steps:
            trajectory.append(
                dict(
                    step=step,
                    loss=loss.mean().item(),
                    rho=_scipy_spearman(a, b_param),
                    tau=_scipy_tau(a, b_param),
                    grad_norm=b_param.grad.abs().mean().item(),
                )
            )
        opt.step()

    return b_param, trajectory


# ===========================================================================
# 1. Basic correctness
# ===========================================================================


@pytest.mark.parametrize("mode", ["hinge", "logistic"])
@pytest.mark.parametrize("diff", ["absolute", "relative"])
def test_concordant_lower_loss_than_discordant(mode: str, diff: str) -> None:
    """Concordant pairs should produce lower loss than discordant pairs."""
    batch, d = 8, 64
    a, b = _make_concordant(batch, d)
    loss_conc = pairwise_concordance(a, b, mode=mode, diff=diff).mean().item()

    a, b = _make_discordant(batch, d)
    loss_disc = pairwise_concordance(a, b, mode=mode, diff=diff).mean().item()

    assert loss_conc < loss_disc, (
        f"mode={mode}, diff={diff}: conc={loss_conc:.4f} >= disc={loss_disc:.4f}"
    )


def test_hinge_perfect_concordance_is_zero() -> None:
    """Hinge loss should be ~0 for perfectly concordant data with large separation."""
    a = torch.arange(64, dtype=torch.float32).unsqueeze(0).expand(4, -1)
    b = a * 2.0
    loss = (
        pairwise_concordance(a, b, mode="hinge", diff="absolute", scale=2.0)
        .mean()
        .item()
    )
    assert loss < 1e-6, f"loss={loss:.8f}"


def test_logistic_perfect_concordance_has_floor() -> None:
    """Logistic mode has a nonzero floor even for perfectly concordant data."""
    a = torch.arange(64, dtype=torch.float32).unsqueeze(0).expand(4, -1)
    b = a * 2.0
    loss = pairwise_concordance(a, b, mode="logistic", diff="absolute").mean().item()
    assert loss > 0.001, f"loss={loss:.4f}"


# ===========================================================================
# 2. Output shape and dtype
# ===========================================================================


@pytest.mark.parametrize("d", [64, 512])
def test_output_shape_and_dtype(d: int) -> None:
    batch = 4
    a, b = _make_random(batch, d)
    loss = pairwise_concordance(a, b)
    assert loss.shape == (batch,)
    assert loss.dtype == torch.float32


# ===========================================================================
# 3. Gradient existence and health
# ===========================================================================


@pytest.mark.parametrize(
    "d, diff",
    [(64, "absolute"), (64, "relative"), (512, "absolute"), (512, "relative")],
)
def test_gradient_health(d: int, diff: str) -> None:
    batch = 4
    a, b = _make_random(batch, d)
    b = b.clone().requires_grad_(True)
    loss = pairwise_concordance(a, b, mode="hinge", diff=diff)
    loss.sum().backward()

    g = b.grad
    assert g is not None
    assert not torch.isnan(g).any()
    assert not torch.isinf(g).any()
    assert g.abs().max().item() > 1e-10


@pytest.mark.parametrize("c_val", [0.01, 0.1, 1.0])
def test_relative_gradient_bounded(c_val: float) -> None:
    """Relative-diff gradient magnitude should be bounded."""
    a, b = _make_random(4, 64)
    b = b.clone().requires_grad_(True)
    loss = pairwise_concordance(a, b, mode="hinge", diff="relative", c=c_val)
    loss.sum().backward()
    g_max = b.grad.abs().max().item()
    bound = 4.0 / c_val  # loose per-element bound
    assert g_max < bound * 10, f"|grad|_max={g_max:.4f}, rough_bound={bound:.1f}"


# ===========================================================================
# 4. Gradient with zeros (gene expression pattern)
# ===========================================================================


@pytest.mark.parametrize(
    "sparse_frac, d",
    [(0.5, 512), (0.9, 512), (0.5, 16384), (0.9, 16384)],
)
def test_gradient_with_sparse_data(sparse_frac: float, d: int) -> None:
    a, b = _make_random(2, d, sparse_frac=sparse_frac)
    b = b.clone().requires_grad_(True)

    use_delta = 0.01 if d > 1000 else None
    loss = pairwise_concordance(a, b, mode="hinge", diff="absolute", delta=use_delta)
    loss.sum().backward()

    g = b.grad
    assert not torch.isnan(g).any()
    assert not torch.isinf(g).any()
    assert g.abs().max().item() > 1e-10


def test_gradient_sparse_relative_diff() -> None:
    """90% sparse data with relative diff should have clean gradients."""
    a, b = _make_random(2, 512, sparse_frac=0.9)
    b = b.clone().requires_grad_(True)
    loss = pairwise_concordance(a, b, mode="hinge", diff="relative", c=0.1)
    loss.sum().backward()
    g = b.grad
    assert not torch.isnan(g).any()
    assert not torch.isinf(g).any()


# ===========================================================================
# 5. Optimisation drives Spearman/Kendall up
# ===========================================================================

_OPTIM_CONFIGS = [
    dict(D=64, mode="hinge", diff="absolute", delta=None, n_pairs=None, c=0.1, steps=100, lr=0.05, label="D=64 exact hinge"),
    dict(D=64, mode="logistic", diff="absolute", delta=None, n_pairs=None, c=0.1, steps=100, lr=0.05, label="D=64 exact logistic"),
    dict(D=64, mode="hinge", diff="relative", delta=None, n_pairs=None, c=0.1, steps=100, lr=0.05, label="D=64 exact rel hinge"),
    dict(D=512, mode="hinge", diff="absolute", delta=None, n_pairs=None, c=0.1, steps=100, lr=0.05, label="D=512 exact hinge"),
    dict(D=512, mode="hinge", diff="absolute", delta=0.01, n_pairs=None, c=0.1, steps=100, lr=0.05, label="D=512 binned hinge"),
    dict(D=512, mode="hinge", diff="absolute", delta=0.01, n_pairs=500, c=0.1, steps=100, lr=0.05, label="D=512 bin-samp500 hinge"),
    dict(D=16384, mode="hinge", diff="absolute", delta=0.01, n_pairs=None, c=0.1, steps=30, lr=0.05, label="D=16k bin-exhaust hinge", sparse=0.5),
    dict(D=16384, mode="hinge", diff="absolute", delta=0.01, n_pairs=5000, c=0.1, steps=30, lr=0.05, label="D=16k bin-samp5k hinge", sparse=0.5),
    dict(D=16384, mode="hinge", diff="absolute", delta=0.01, n_pairs=1000, c=0.1, steps=30, lr=0.05, label="D=16k bin-samp1k hinge", sparse=0.5),
    dict(D=16384, mode="logistic", diff="absolute", delta=0.01, n_pairs=5000, c=0.1, steps=30, lr=0.05, label="D=16k bin-samp5k logistic", sparse=0.5),
]


@pytest.mark.parametrize(
    "cfg", _OPTIM_CONFIGS, ids=[c["label"] for c in _OPTIM_CONFIGS]
)
def test_optimisation_improves_rho(cfg: dict) -> None:
    """Optimising the pairwise concordance loss should improve Spearman rho."""
    d = cfg["D"]
    batch = 4 if d > 1000 else 8
    sparse = cfg.get("sparse", 0.0)

    torch.manual_seed(123)
    if sparse > 0:
        a, b_init = _make_random(batch, d, sparse_frac=sparse)
    else:
        a = torch.randn(batch, d)
        b_init = torch.randn(batch, d)

    _, traj = _run_optim(
        a,
        b_init,
        cfg["steps"],
        cfg["lr"],
        cfg["mode"],
        cfg["diff"],
        cfg["delta"],
        cfg.get("n_pairs"),
        cfg["c"],
    )

    rho_start = traj[0]["rho"]
    rho_end = traj[-1]["rho"]
    assert rho_end > rho_start + 0.05, (
        f"{cfg['label']}: rho {rho_start:.3f} -> {rho_end:.3f}"
    )


# ===========================================================================
# 5b. Hessian off-diagonal structure
# ===========================================================================


def test_hessian_offdiag_hinge_no_coupling() -> None:
    """On random data, hinge H_od should be ~0 (no coupling between dimensions)."""
    d, batch = 64, 8
    torch.manual_seed(42)
    a = torch.randn(batch, d)
    b_rand = torch.randn(batch, d)

    h_hinge = _hessian_offdiag(
        lambda a_, b_: pairwise_concordance(a_, b_, mode="hinge", diff="absolute"),
        a,
        b_rand.clone().requires_grad_(True),
    )
    assert h_hinge < 1e-4, f"H_od={h_hinge:.6f}"


def test_hessian_offdiag_hinge_leq_logistic() -> None:
    """On random data, hinge H_od should be <= logistic H_od."""
    d, batch = 64, 8
    torch.manual_seed(42)
    a = torch.randn(batch, d)
    b_rand = torch.randn(batch, d)

    h_hinge = _hessian_offdiag(
        lambda a_, b_: pairwise_concordance(a_, b_, mode="hinge", diff="absolute"),
        a,
        b_rand.clone().requires_grad_(True),
    )
    h_logistic = _hessian_offdiag(
        lambda a_, b_: pairwise_concordance(a_, b_, mode="logistic", diff="absolute"),
        a,
        b_rand.clone().requires_grad_(True),
    )
    assert h_hinge <= h_logistic + 1e-6, (
        f"hinge={h_hinge:.6f}, logistic={h_logistic:.6f}"
    )


def test_hessian_offdiag_post_optim_small() -> None:
    """After optimisation, hinge H_od should remain small (kink artifact only)."""
    d, batch = 64, 8
    torch.manual_seed(42)
    a = torch.randn(batch, d)
    b_opt = torch.randn(batch, d).clone().requires_grad_(True)
    opt = torch.optim.Adam([b_opt], lr=0.05)
    for _ in range(100):
        opt.zero_grad()
        pairwise_concordance(a, b_opt, mode="hinge", diff="absolute").sum().backward()
        opt.step()

    h_od = _hessian_offdiag(
        lambda a_, b_: pairwise_concordance(a_, b_, mode="hinge", diff="absolute"),
        a,
        b_opt,
    )
    assert h_od < 0.01, f"H_od={h_od:.6f}"


# ===========================================================================
# 6. Binned ≈ exact consistency
# ===========================================================================


@pytest.mark.parametrize(
    "d, delta, tol",
    [(64, 0.001, 0.05), (64, 0.01, 0.10), (512, 0.001, 0.05)],
    ids=["D=64 fine", "D=64 coarse", "D=512 fine"],
)
def test_binned_consistency(d: int, delta: float, tol: float) -> None:
    batch = 4
    a, b = _make_random(batch, d)

    loss_exact = pairwise_concordance(a, b, mode="hinge", diff="absolute")
    loss_binned = pairwise_concordance(
        a, b, mode="hinge", diff="absolute", delta=delta
    )

    diff_val = (loss_exact - loss_binned).abs().max().item()
    assert diff_val < tol, f"max diff={diff_val:.6f}"


# ===========================================================================
# 7. Sampled ≈ exhaustive (unbiased)
# ===========================================================================


def test_sampling_unbiased() -> None:
    d, batch = 512, 4
    a, b = _make_random(batch, d, sparse_frac=0.5)

    loss_exhaust = pairwise_concordance(
        a, b, mode="hinge", diff="absolute", delta=0.01
    )

    sampled_losses = []
    for seed in range(50):
        torch.manual_seed(seed)
        loss_s = pairwise_concordance(
            a, b, mode="hinge", diff="absolute", delta=0.01, n_pairs=500
        )
        sampled_losses.append(loss_s.detach())

    mean_sampled = torch.stack(sampled_losses).mean(dim=0)
    diff_val = (loss_exhaust - mean_sampled).abs().max().item()
    assert diff_val < 0.05, f"max diff={diff_val:.6f}"


def test_sampling_variance_decreases_with_n_pairs() -> None:
    d, batch = 512, 4
    a, b = _make_random(batch, d, sparse_frac=0.5)

    sampled_500 = []
    for seed in range(50):
        torch.manual_seed(seed)
        loss_s = pairwise_concordance(
            a, b, mode="hinge", diff="absolute", delta=0.01, n_pairs=500
        )
        sampled_500.append(loss_s.detach())
    var_500 = torch.stack(sampled_500).var(dim=0).mean().item()

    sampled_2k = []
    for seed in range(50):
        torch.manual_seed(seed + 1000)
        loss_s = pairwise_concordance(
            a, b, mode="hinge", diff="absolute", delta=0.01, n_pairs=2000
        )
        sampled_2k.append(loss_s.detach())
    var_2k = torch.stack(sampled_2k).var(dim=0).mean().item()

    assert var_2k < var_500, f"var@500={var_500:.6f}, var@2k={var_2k:.6f}"


# ===========================================================================
# 8. Large D with sparsity (gene expression scale)
# ===========================================================================


@pytest.mark.parametrize("sparse_frac", [0.5, 0.9], ids=["50% zero", "90% zero"])
def test_large_d_binned_and_sampled(sparse_frac: float) -> None:
    d = 16384
    batch = 2
    a, b = _make_random(batch, d, sparse_frac=sparse_frac)
    b = b.clone().requires_grad_(True)

    # binned exhaustive
    t0 = time.time()
    loss_be = pairwise_concordance(a, b, mode="hinge", diff="absolute", delta=0.01)
    loss_be.sum().backward()
    t_be = time.time() - t0
    g_be = b.grad.clone()
    b.grad = None

    assert not torch.isnan(g_be).any()
    assert g_be.abs().max() > 1e-10

    # binned sampled
    t0 = time.time()
    loss_bs = pairwise_concordance(
        a, b, mode="hinge", diff="absolute", delta=0.01, n_pairs=1000
    )
    loss_bs.sum().backward()
    t_bs = time.time() - t0
    g_bs = b.grad.clone()

    assert not torch.isnan(g_bs).any()
    assert g_bs.abs().max() > 1e-10

    # sampled should be faster (or both trivially fast)
    assert t_bs < t_be or t_be < 0.5, (
        f"exhaust={t_be:.2f}s, sampled={t_bs:.2f}s"
    )


# ===========================================================================
# 9. Kendall tau reference check
# ===========================================================================


def test_tau_reference_daniels_bound() -> None:
    """After optimisation, Daniels bound rho >= (3*tau - 1)/2 should hold."""
    d, batch = 64, 8
    torch.manual_seed(999)
    a = torch.randn(batch, d)
    b_param = torch.randn(batch, d, requires_grad=True)
    opt = torch.optim.Adam([b_param], lr=0.05)

    for _ in range(100):
        opt.zero_grad()
        loss = pairwise_concordance(a, b_param, mode="hinge", diff="absolute")
        loss.sum().backward()
        opt.step()

    tau_end = _scipy_tau(a, b_param)
    rho_end = _scipy_spearman(a, b_param)
    daniels_lower = (3 * tau_end - 1) / 2

    assert rho_end >= daniels_lower - 0.05, (
        f"rho={rho_end:.3f}, bound={daniels_lower:.3f}"
    )
    approx = 1.5 * tau_end - 0.5 * tau_end**3
    assert abs(rho_end - approx) < 0.15, f"rho={rho_end:.3f}, approx={approx:.3f}"


# ===========================================================================
# 10. Relative diff: c controls gradient scale
# ===========================================================================


def test_c_controls_gradient_magnitude() -> None:
    """Larger c should produce smaller gradients in relative diff mode."""
    d, batch = 64, 4
    grad_maxes: dict[float, float] = {}

    for c_val in [0.01, 0.1, 1.0]:
        a, b = _make_random(batch, d)
        b = b.clone().requires_grad_(True)
        loss = pairwise_concordance(a, b, mode="hinge", diff="relative", c=c_val)
        loss.sum().backward()
        grad_maxes[c_val] = b.grad.abs().max().item()

    assert grad_maxes[0.01] > grad_maxes[0.1], (
        f"c=0.01: {grad_maxes[0.01]:.4f}, c=0.1: {grad_maxes[0.1]:.4f}"
    )
    assert grad_maxes[0.1] > grad_maxes[1.0], (
        f"c=0.1: {grad_maxes[0.1]:.4f}, c=1.0: {grad_maxes[1.0]:.4f}"
    )


# ===========================================================================
# 11. Edge cases
# ===========================================================================


def test_all_identical_hinge_loss() -> None:
    """All-identical inputs: hinge loss = 1.0 (no concordance evidence)."""
    a = torch.ones(2, 64)
    b = torch.ones(2, 64)
    loss = pairwise_concordance(a, b, mode="hinge").mean().item()
    assert abs(loss - 1.0) < 1e-6, f"loss={loss:.6f}"


def test_all_identical_logistic_loss() -> None:
    """All-identical inputs: logistic loss = log(2)."""
    a = torch.ones(2, 64)
    b = torch.ones(2, 64)
    loss = pairwise_concordance(a, b, mode="logistic").mean().item()
    assert abs(loss - 0.6931) < 0.01, f"loss={loss:.4f}"


def test_d2_concordant_vs_discordant() -> None:
    """With D=2, discordant loss should exceed concordant loss."""
    a = torch.tensor([[1.0, 2.0]])
    b_conc = torch.tensor([[3.0, 4.0]])
    b_disc = torch.tensor([[4.0, 3.0]])

    loss_conc = pairwise_concordance(a, b_conc, mode="hinge", scale=2.0)
    loss_disc = pairwise_concordance(a, b_disc, mode="hinge", scale=2.0)

    assert loss_conc.shape == (1,)
    assert loss_disc.item() > loss_conc.item()


def test_all_zeros_gradient() -> None:
    a = torch.zeros(2, 64)
    b = torch.zeros(2, 64, requires_grad=True)
    loss = pairwise_concordance(a, b, mode="hinge", diff="relative", c=0.1)
    loss.sum().backward()
    assert not torch.isnan(b.grad).any()
    assert b.grad.abs().max().item() < 1e-8


def test_batch_size_one() -> None:
    a = torch.randn(1, 64)
    b = torch.randn(1, 64, requires_grad=True)
    loss = pairwise_concordance(a, b, mode="hinge")
    loss.sum().backward()
    assert loss.shape == (1,)
    assert not torch.isnan(b.grad).any()

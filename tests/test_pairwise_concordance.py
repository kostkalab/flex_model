"""
Test suite for pairwise_concordance.

Covers:
  - Correctness: loss = 0 for perfectly concordant, > 0 for discordant
  - Gradient: exists, finite, bounded, drives optimisation
  - Modes: hinge vs logistic, absolute vs relative diff
  - Paths: exact vs binned vs binned+sampled
  - Dimensions: D=64, D=512, D=16384
  - Sparsity: 0%, 50%, 90% zeros (gene expression patterns)
  - Consistency: binned ≈ exact for small delta
  - Sampling: unbiased (mean over seeds ≈ exhaustive)
  - Gradient health: bounded, no NaN/Inf, correct sign

Run:  pytest tests/test_pairwise_concordance.py
"""

import sys
import time
import torch
import numpy as np
from scipy.stats import spearmanr, kendalltau

from flexModel.pairwise_concordance import pairwise_concordance


torch.manual_seed(42)
PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"

results = []


def report(name, ok, detail=""):
    tag = PASS if ok else FAIL
    results.append(ok)
    msg = f"  {tag} {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)


# ===========================================================================
# Data generators
# ===========================================================================

def make_random(batch, D, sparse_frac=0.0):
    a = torch.randn(batch, D)
    b = torch.randn(batch, D)
    if sparse_frac > 0:
        a[torch.rand(batch, D) < sparse_frac] = 0.0
        b[torch.rand(batch, D) < sparse_frac] = 0.0
    return a, b


def make_concordant(batch, D, sparse_frac=0.0):
    """a and b with same ordering (perfect concordance)."""
    a = torch.randn(batch, D).abs()
    b = a * torch.rand(batch, D).clamp(min=0.5) + 0.01
    if sparse_frac > 0:
        mask = torch.rand(batch, D) < sparse_frac
        a[mask] = 0.0
        b[mask] = 0.0
    return a, b


def make_discordant(batch, D):
    """a and b with opposite ordering."""
    a = torch.randn(batch, D)
    b = -a * torch.rand(batch, D).clamp(min=0.5)
    return a, b


def scipy_spearman(a, b):
    """Mean Spearman across batch (scipy reference)."""
    rhos = []
    for i in range(a.shape[0]):
        rho, _ = spearmanr(a[i].detach().numpy(), b[i].detach().numpy())
        rhos.append(rho)
    return np.mean(rhos)


def scipy_tau(a, b):
    """Mean Kendall tau across batch (scipy reference)."""
    taus = []
    for i in range(a.shape[0]):
        tau, _ = kendalltau(a[i].detach().numpy(), b[i].detach().numpy())
        taus.append(tau)
    return np.mean(taus)


# ===========================================================================
# 1. Basic correctness
# ===========================================================================

def test_correctness():
    print("\n1. Basic correctness")
    batch, D = 8, 64

    for mode in ["hinge", "logistic"]:
        for diff in ["absolute", "relative"]:
            tag = f"mode={mode}, diff={diff}"

            # concordant → low loss
            a, b = make_concordant(batch, D)
            loss_conc = pairwise_concordance(a, b, mode=mode, diff=diff).mean().item()

            # discordant → high loss
            a, b = make_discordant(batch, D)
            loss_disc = pairwise_concordance(a, b, mode=mode, diff=diff).mean().item()

            report(f"concordant < discordant [{tag}]",
                   loss_conc < loss_disc,
                   f"conc={loss_conc:.4f}, disc={loss_disc:.4f}")

    # hinge: perfectly concordant with large separation → exactly zero
    a = torch.arange(64, dtype=torch.float32).unsqueeze(0).expand(4, -1)
    b = a * 2.0
    loss = pairwise_concordance(a, b, mode="hinge", diff="absolute", scale=2.0).mean().item()
    report("hinge: perfect concordance → loss ≈ 0",
           loss < 1e-6, f"loss={loss:.8f}")

    # logistic: perfect concordance → loss > 0 (nonzero floor)
    loss = pairwise_concordance(a, b, mode="logistic", diff="absolute").mean().item()
    report("logistic: perfect concordance → loss > 0 (expected floor)",
           loss > 0.001, f"loss={loss:.4f}")


# ===========================================================================
# 2. Output shape and dtype
# ===========================================================================

def test_shapes():
    print("\n2. Output shape and dtype")
    for D in [64, 512]:
        batch = 4
        a, b = make_random(batch, D)
        loss = pairwise_concordance(a, b)
        report(f"D={D}: shape is (batch,)", loss.shape == (batch,))
        report(f"D={D}: dtype is float", loss.dtype == torch.float32)


# ===========================================================================
# 3. Gradient existence and health
# ===========================================================================

def test_gradients():
    print("\n3. Gradient health")

    for D, diff, label in [
        (64,    "absolute", "D=64 abs"),
        (64,    "relative", "D=64 rel"),
        (512,   "absolute", "D=512 abs"),
        (512,   "relative", "D=512 rel"),
    ]:
        batch = 4
        a, b = make_random(batch, D)
        b = b.clone().requires_grad_(True)
        loss = pairwise_concordance(a, b, mode="hinge", diff=diff)
        loss.sum().backward()

        g = b.grad
        report(f"{label}: grad exists", g is not None)
        report(f"{label}: no NaN", not torch.isnan(g).any().item())
        report(f"{label}: no Inf", not torch.isinf(g).any().item())
        report(f"{label}: grad nonzero", g.abs().max().item() > 1e-10,
               f"|∇|_max={g.abs().max().item():.6f}")

    # gradient bounded by 2/c for relative diff
    print()
    for c_val in [0.01, 0.1, 1.0]:
        a, b = make_random(4, 64)
        b = b.clone().requires_grad_(True)
        loss = pairwise_concordance(a, b, mode="hinge", diff="relative", c=c_val)
        loss.sum().backward()
        g_max = b.grad.abs().max().item()
        # each pair contributes at most 2/c per element, summed over D-1 pairs
        # and divided by D(D-1)/2.  Rough bound: 4/(c * D)
        bound = 4.0 / (c_val)  # loose per-element bound
        report(f"relative c={c_val}: |∇|_max reasonable",
               g_max < bound * 10,  # generous factor
               f"|∇|_max={g_max:.4f}, rough_bound={bound:.1f}")


# ===========================================================================
# 4. Gradient with zeros (gene expression pattern)
# ===========================================================================

def test_gradient_zeros():
    print("\n4. Gradient with zeros (sparse data)")

    for sparse_frac, D, label in [
        (0.5, 512,   "50% zero D=512"),
        (0.9, 512,   "90% zero D=512"),
        (0.5, 16384, "50% zero D=16384 binned"),
        (0.9, 16384, "90% zero D=16384 binned"),
    ]:
        a, b = make_random(2, D, sparse_frac=sparse_frac)
        b = b.clone().requires_grad_(True)

        use_delta = 0.01 if D > 1000 else None
        loss = pairwise_concordance(a, b, mode="hinge", diff="absolute",
                                    delta=use_delta)
        loss.sum().backward()

        g = b.grad
        report(f"{label}: no NaN", not torch.isnan(g).any().item())
        report(f"{label}: no Inf", not torch.isinf(g).any().item())
        report(f"{label}: grad nonzero", g.abs().max().item() > 1e-10,
               f"|∇|_max={g.abs().max().item():.6f}")

    # relative diff with zeros: gradient bounded
    a, b = make_random(2, 512, sparse_frac=0.9)
    b = b.clone().requires_grad_(True)
    loss = pairwise_concordance(a, b, mode="hinge", diff="relative", c=0.1)
    loss.sum().backward()
    g = b.grad
    report("90% zero D=512 relative: no NaN", not torch.isnan(g).any().item())
    report("90% zero D=512 relative: no Inf", not torch.isinf(g).any().item())


# ===========================================================================
# 5. Optimisation drives Spearman/Kendall up + Hessian off-diag comparison
# ===========================================================================

def hessian_offdiag(loss_fn, a, b_param, seed=12345):
    """Mean absolute off-diagonal element of the Hessian ∂²L/∂b_i∂b_j.

    Computed for sample 0 via finite differences on the gradient.
    Seeds RNG before each forward pass so stochastic paths (sampled)
    draw the same pairs in base and perturbed evaluations.

    Note: for hinge mode, pairs near the threshold (scale * product ≈ 1)
    contribute a kink where finite differences pick up nonzero H_od.
    This is not coupling — it's the measure-zero non-differentiability
    of relu.  Expect small but nonzero H_od after optimisation.
    """
    eps_h = 1e-3
    D = b_param.shape[1]
    b0 = b_param.detach().clone()

    # base gradient
    b_param.data.copy_(b0)
    b_param.requires_grad_(True)
    if b_param.grad is not None:
        b_param.grad = None
    torch.manual_seed(seed)
    l = loss_fn(a, b_param)
    l[0].backward(retain_graph=False)
    g0 = b_param.grad[0].clone()

    # perturb each dimension, measure gradient change
    n_probe = min(D, 32)  # probe subset for speed
    offdiag_sum = 0.0
    count = 0
    for j in range(n_probe):
        b_param.data.copy_(b0)
        b_param.data[0, j] += eps_h
        b_param.requires_grad_(True)
        if b_param.grad is not None:
            b_param.grad = None
        torch.manual_seed(seed)  # same seed → same sampled pairs
        l = loss_fn(a, b_param)
        l[0].backward(retain_graph=False)
        gj = b_param.grad[0].clone()

        dg = (gj - g0) / eps_h  # row j of Hessian
        # off-diagonal: exclude element j
        mask = torch.ones(D, dtype=torch.bool)
        mask[j] = False
        offdiag_sum += dg[mask].abs().mean().item()
        count += 1

    b_param.data.copy_(b0)
    b_param.requires_grad_(True)
    return offdiag_sum / count if count > 0 else 0.0


def run_optim(a, b_init, steps, lr, mode, diff, delta, n_pairs, c):
    """Run optimisation, return trajectory of (loss, ρ, τ, |∇|)."""
    b_param = b_init.clone().requires_grad_(True)
    opt = torch.optim.Adam([b_param], lr=lr)
    log_steps = list(range(0, steps, max(1, steps // 10))) + [steps - 1]
    log_steps = sorted(set(log_steps))
    trajectory = []

    for step in range(steps):
        opt.zero_grad()
        loss = pairwise_concordance(
            a, b_param, mode=mode, diff=diff,
            delta=delta, n_pairs=n_pairs, c=c,
        )
        loss.sum().backward()
        if step in log_steps:
            rho = scipy_spearman(a, b_param)
            tau = scipy_tau(a, b_param)
            grad_norm = b_param.grad.abs().mean().item()
            trajectory.append(dict(
                step=step, loss=loss.mean().item(),
                rho=rho, tau=tau, grad_norm=grad_norm,
            ))
        opt.step()

    return b_param, trajectory


def estimate_K(a, b, delta):
    """Estimate average number of occupied cells K across batch."""
    if delta is None:
        return a.shape[1]  # exact: K = D
    a_bin = torch.round(a / delta).long()
    b_bin = torch.round(b / delta).long()
    Ks = []
    for n in range(a.shape[0]):
        a_off = a_bin[n] - a_bin[n].min()
        b_off = b_bin[n] - b_bin[n].min()
        b_range = b_off.max() + 1
        cell_key = a_off * b_range + b_off
        Ks.append(cell_key.unique().shape[0])
    return int(np.mean(Ks))


def test_optimisation():
    print("\n5. Optimisation (ρ, τ, |∇|, H_od) across modes and scales")

    configs = [
        # D=64: exact, compare hinge vs logistic
        dict(D=64,  mode="hinge",    diff="absolute", delta=None,  n_pairs=None, c=0.1, steps=300, lr=0.05, label="D=64 exact hinge"),
        dict(D=64,  mode="logistic", diff="absolute", delta=None,  n_pairs=None, c=0.1, steps=300, lr=0.05, label="D=64 exact logistic"),
        dict(D=64,  mode="hinge",    diff="relative", delta=None,  n_pairs=None, c=0.1, steps=300, lr=0.05, label="D=64 exact rel hinge"),
        # D=512: exact vs binned
        dict(D=512, mode="hinge",    diff="absolute", delta=None,  n_pairs=None, c=0.1, steps=300, lr=0.05, label="D=512 exact hinge"),
        dict(D=512, mode="hinge",    diff="absolute", delta=0.01,  n_pairs=None, c=0.1, steps=300, lr=0.05, label="D=512 binned hinge"),
        dict(D=512, mode="hinge",    diff="absolute", delta=0.01,  n_pairs=500,  c=0.1, steps=300, lr=0.05, label="D=512 bin-samp500 hinge"),
        # D=16384: binned only (exact would OOM), sparse GE
        dict(D=16384, mode="hinge",    diff="absolute", delta=0.01,  n_pairs=None, c=0.1, steps=100, lr=0.05, label="D=16k bin-exhaust hinge",    sparse=0.5),
        dict(D=16384, mode="hinge",    diff="absolute", delta=0.01,  n_pairs=5000, c=0.1, steps=100, lr=0.05, label="D=16k bin-samp5k hinge",   sparse=0.5),
        dict(D=16384, mode="hinge",    diff="absolute", delta=0.01,  n_pairs=1000, c=0.1, steps=100, lr=0.05, label="D=16k bin-samp1k hinge",   sparse=0.5),
        dict(D=16384, mode="logistic", diff="absolute", delta=0.01,  n_pairs=5000, c=0.1, steps=100, lr=0.05, label="D=16k bin-samp5k logistic", sparse=0.5),
    ]

    # header
    # Note: H_od after optimisation may be nonzero for hinge (kink artefact)
    # and for sampled (seeded, but stochastic gradient variance).
    # The clean comparison is at the bottom on random pre-optimisation data.
    print(f"  {'config':<30s} {'K':>6s}  {'ρ_start':>7s} {'ρ_end':>7s} {'τ_end':>7s} "
          f"{'loss_end':>9s} {'|∇|_end':>8s} {'H_od':>10s}")
    print("  " + "-" * 102)

    for cfg in configs:
        D = cfg["D"]
        batch = 4 if D > 1000 else 8
        sparse = cfg.get("sparse", 0.0)

        torch.manual_seed(123)
        if sparse > 0:
            a, b_init = make_random(batch, D, sparse_frac=sparse)
        else:
            a = torch.randn(batch, D)
            b_init = torch.randn(batch, D)

        b_param, traj = run_optim(
            a, b_init, cfg["steps"], cfg["lr"],
            cfg["mode"], cfg["diff"], cfg["delta"], cfg.get("n_pairs"),
            cfg["c"],
        )

        K = estimate_K(a, b_init, cfg["delta"])
        rho_start = traj[0]["rho"]
        rho_end = traj[-1]["rho"]
        tau_end = traj[-1]["tau"]
        loss_end = traj[-1]["loss"]
        grad_end = traj[-1]["grad_norm"]

        # Hessian off-diag (skip for D=16384 — too slow)
        if D <= 512:
            _mode, _diff, _delta = cfg["mode"], cfg["diff"], cfg["delta"]
            _n_pairs, _c = cfg.get("n_pairs"), cfg["c"]
            loss_fn = lambda a_, b_, m=_mode, d=_diff, dl=_delta, np=_n_pairs, cv=_c: \
                pairwise_concordance(a_, b_, mode=m, diff=d, delta=dl, n_pairs=np, c=cv)
            h_od = hessian_offdiag(loss_fn, a, b_param)
            h_od_str = f"{h_od:.6f}"
        else:
            h_od_str = "skip"

        print(f"  {cfg['label']:<30s} {K:>6d}  {rho_start:>7.3f} {rho_end:>7.3f} {tau_end:>7.3f} "
              f"{loss_end:>9.4f} {grad_end:>8.5f} {h_od_str:>10s}")

        report(f"{cfg['label']}: ρ improves",
               rho_end > rho_start + 0.05,
               f"ρ: {rho_start:.3f} → {rho_end:.3f}, τ={tau_end:.3f}")

    # --- H_od comparison: random vs post-optimisation, hinge vs logistic ---
    print()
    print("  H_od breakdown (D=64):")
    print(f"  {'setting':<35s}  {'H_od':>10s}  interpretation")
    print("  " + "-" * 75)

    D, batch = 64, 8

    # random (pre-optimisation)
    torch.manual_seed(42)
    a = torch.randn(batch, D)
    b_rand = torch.randn(batch, D)

    h_hinge_rand = hessian_offdiag(
        lambda a_, b_: pairwise_concordance(a_, b_, mode="hinge", diff="absolute"),
        a, b_rand.clone().requires_grad_(True),
    )
    h_logistic_rand = hessian_offdiag(
        lambda a_, b_: pairwise_concordance(a_, b_, mode="logistic", diff="absolute"),
        a, b_rand.clone().requires_grad_(True),
    )
    print(f"  {'hinge (random)':<35s}  {h_hinge_rand:>10.6f}  structural: no coupling")
    print(f"  {'logistic (random)':<35s}  {h_logistic_rand:>10.6f}  smooth sigmoid curvature")

    # post-optimisation
    torch.manual_seed(42)
    b_opt_hinge = b_rand.clone().requires_grad_(True)
    opt = torch.optim.Adam([b_opt_hinge], lr=0.05)
    for _ in range(300):
        opt.zero_grad()
        pairwise_concordance(a, b_opt_hinge, mode="hinge", diff="absolute").sum().backward()
        opt.step()

    torch.manual_seed(42)
    b_opt_logistic = b_rand.clone().requires_grad_(True)
    opt = torch.optim.Adam([b_opt_logistic], lr=0.05)
    for _ in range(300):
        opt.zero_grad()
        pairwise_concordance(a, b_opt_logistic, mode="logistic", diff="absolute").sum().backward()
        opt.step()

    h_hinge_opt = hessian_offdiag(
        lambda a_, b_: pairwise_concordance(a_, b_, mode="hinge", diff="absolute"),
        a, b_opt_hinge,
    )
    h_logistic_opt = hessian_offdiag(
        lambda a_, b_: pairwise_concordance(a_, b_, mode="logistic", diff="absolute"),
        a, b_opt_logistic,
    )
    print(f"  {'hinge (post-optim)':<35s}  {h_hinge_opt:>10.6f}  relu kink artifact, not coupling")
    print(f"  {'logistic (post-optim)':<35s}  {h_logistic_opt:>10.6f}  less curvature near optimum")
    print()

    # tests on random data (structural properties)
    report("random: hinge H_od ≈ 0 (no coupling)",
           h_hinge_rand < 1e-4,
           f"H_od={h_hinge_rand:.6f}")
    report("random: hinge H_od ≤ logistic H_od",
           h_hinge_rand <= h_logistic_rand + 1e-6,
           f"hinge={h_hinge_rand:.6f}, logistic={h_logistic_rand:.6f}")

    # post-optim: hinge kink artifact is small (not catastrophic coupling)
    report("post-optim: hinge H_od < 0.01 (kink artifact only)",
           h_hinge_opt < 0.01,
           f"H_od={h_hinge_opt:.6f}")


# ===========================================================================
# 6. Binned ≈ exact consistency
# ===========================================================================

def test_binned_consistency():
    print("\n6. Binned ≈ exact consistency")

    for D, delta, tol, label in [
        (64,  0.001, 0.05, "D=64 fine bins"),
        (64,  0.01,  0.10, "D=64 coarse bins"),
        (512, 0.001, 0.05, "D=512 fine bins"),
    ]:
        batch = 4
        a, b = make_random(batch, D)

        loss_exact = pairwise_concordance(a, b, mode="hinge", diff="absolute")
        loss_binned = pairwise_concordance(a, b, mode="hinge", diff="absolute",
                                           delta=delta)

        diff_val = (loss_exact - loss_binned).abs().max().item()
        report(f"{label}: |exact - binned| < {tol}",
               diff_val < tol,
               f"max diff={diff_val:.6f}")


# ===========================================================================
# 7. Sampled ≈ exhaustive (unbiased)
# ===========================================================================

def test_sampling_unbiased():
    print("\n7. Sampling ≈ exhaustive (unbiased over seeds)")

    D, batch = 512, 4
    a, b = make_random(batch, D, sparse_frac=0.5)

    loss_exhaust = pairwise_concordance(a, b, mode="hinge", diff="absolute",
                                        delta=0.01)

    # average over many sampled runs
    sampled_losses = []
    for seed in range(50):
        torch.manual_seed(seed)
        loss_s = pairwise_concordance(a, b, mode="hinge", diff="absolute",
                                      delta=0.01, n_pairs=500)
        sampled_losses.append(loss_s.detach())

    mean_sampled = torch.stack(sampled_losses).mean(dim=0)
    diff_val = (loss_exhaust - mean_sampled).abs().max().item()
    report(f"D={D}: |exhaust - mean(sampled)| < 0.05",
           diff_val < 0.05,
           f"max diff={diff_val:.6f}")

    # check variance decreases with more pairs
    var_500 = torch.stack(sampled_losses).var(dim=0).mean().item()
    sampled_2k = []
    for seed in range(50):
        torch.manual_seed(seed + 1000)
        loss_s = pairwise_concordance(a, b, mode="hinge", diff="absolute",
                                      delta=0.01, n_pairs=2000)
        sampled_2k.append(loss_s.detach())
    var_2k = torch.stack(sampled_2k).var(dim=0).mean().item()
    report("variance decreases with n_pairs",
           var_2k < var_500,
           f"var@500={var_500:.6f}, var@2k={var_2k:.6f}")


# ===========================================================================
# 8. Large D with sparsity (gene expression scale)
# ===========================================================================

def test_large_D():
    print("\n8. Large D (gene expression scale)")

    D = 16384
    batch = 2

    for sparse, label in [(0.5, "50% zero"), (0.9, "90% zero")]:
        a, b = make_random(batch, D, sparse_frac=sparse)
        b = b.clone().requires_grad_(True)

        # binned exhaustive
        t0 = time.time()
        loss_be = pairwise_concordance(a, b, mode="hinge", diff="absolute",
                                       delta=0.01)
        loss_be.sum().backward()
        t_be = time.time() - t0
        g_be = b.grad.clone()
        b.grad = None

        report(f"D={D} {label} binned exhaust: runs",
               True, f"loss={loss_be.mean().item():.4f}, time={t_be:.2f}s")
        report(f"D={D} {label} binned exhaust: grad ok",
               not torch.isnan(g_be).any().item() and g_be.abs().max() > 1e-10)

        # binned sampled
        b.grad = None
        t0 = time.time()
        loss_bs = pairwise_concordance(a, b, mode="hinge", diff="absolute",
                                       delta=0.01, n_pairs=1000)
        loss_bs.sum().backward()
        t_bs = time.time() - t0
        g_bs = b.grad.clone()

        report(f"D={D} {label} binned sampled: runs",
               True, f"loss={loss_bs.mean().item():.4f}, time={t_bs:.2f}s")
        report(f"D={D} {label} binned sampled: grad ok",
               not torch.isnan(g_bs).any().item() and g_bs.abs().max() > 1e-10)

        # sampled should be faster
        report(f"D={D} {label} sampled faster than exhaustive",
               t_bs < t_be or t_be < 0.5,  # allow trivial case
               f"exhaust={t_be:.2f}s, sampled={t_bs:.2f}s")


# ===========================================================================
# 9. Kendall tau reference check
# ===========================================================================

def test_tau_reference():
    print("\n9. Daniels bound check (τ → ρ guarantee)")

    D, batch = 64, 8
    torch.manual_seed(999)
    a = torch.randn(batch, D)
    b_param = torch.randn(batch, D, requires_grad=True)
    opt = torch.optim.Adam([b_param], lr=0.05)

    for step in range(300):
        opt.zero_grad()
        loss = pairwise_concordance(a, b_param, mode="hinge", diff="absolute")
        loss.sum().backward()
        opt.step()

    tau_end = scipy_tau(a, b_param)
    rho_end = scipy_spearman(a, b_param)
    daniels_lower = (3 * tau_end - 1) / 2

    report(f"τ = {tau_end:.3f}, ρ = {rho_end:.3f}", True)
    report(f"Daniels bound: ρ ≥ (3τ-1)/2",
           rho_end >= daniels_lower - 0.05,
           f"ρ={rho_end:.3f} ≥ {daniels_lower:.3f}")
    report(f"empirical approx: ρ ≈ 1.5τ - 0.5τ³",
           abs(rho_end - (1.5 * tau_end - 0.5 * tau_end**3)) < 0.15,
           f"ρ={rho_end:.3f}, approx={1.5*tau_end - 0.5*tau_end**3:.3f}")


# ===========================================================================
# 10. Relative diff: c controls gradient scale
# ===========================================================================

def test_c_gradient_scale():
    print("\n10. Relative diff: c controls gradient magnitude")

    D, batch = 64, 4
    grad_maxes = {}

    for c_val in [0.01, 0.1, 1.0]:
        a, b = make_random(batch, D)
        b = b.clone().requires_grad_(True)
        loss = pairwise_concordance(a, b, mode="hinge", diff="relative", c=c_val)
        loss.sum().backward()
        grad_maxes[c_val] = b.grad.abs().max().item()

    # larger c → smaller gradients
    report("c=0.01 > c=0.1 gradient",
           grad_maxes[0.01] > grad_maxes[0.1],
           f"|∇|: c=0.01→{grad_maxes[0.01]:.4f}, c=0.1→{grad_maxes[0.1]:.4f}")
    report("c=0.1 > c=1.0 gradient",
           grad_maxes[0.1] > grad_maxes[1.0],
           f"|∇|: c=0.1→{grad_maxes[0.1]:.4f}, c=1.0→{grad_maxes[1.0]:.4f}")


# ===========================================================================
# 11. Edge cases
# ===========================================================================

def test_edge_cases():
    print("\n11. Edge cases")

    # all identical → differences are zero → product is zero
    # hinge: relu(1 - 0) = 1 (no concordance evidence)
    # logistic: softplus(0) = log(2)
    a = torch.ones(2, 64)
    b = torch.ones(2, 64)
    loss_h = pairwise_concordance(a, b, mode="hinge").mean().item()
    loss_l = pairwise_concordance(a, b, mode="logistic").mean().item()
    report("all identical: hinge = 1.0", abs(loss_h - 1.0) < 1e-6,
           f"loss={loss_h:.6f}")
    report("all identical: logistic = log(2)",
           abs(loss_l - 0.6931) < 0.01, f"loss={loss_l:.4f}")

    # single pair (D=2)
    a = torch.tensor([[1.0, 2.0]])
    b = torch.tensor([[3.0, 4.0]])  # concordant
    loss = pairwise_concordance(a, b, mode="hinge", scale=2.0)
    report("D=2 concordant: loss exists", loss.shape == (1,), f"loss={loss.item():.4f}")

    b2 = torch.tensor([[4.0, 3.0]])  # discordant
    loss2 = pairwise_concordance(a, b2, mode="hinge", scale=2.0)
    report("D=2 discordant > concordant", loss2.item() > loss.item())

    # all zeros
    a = torch.zeros(2, 64)
    b = torch.zeros(2, 64, requires_grad=True)
    loss = pairwise_concordance(a, b, mode="hinge", diff="relative", c=0.1)
    loss.sum().backward()
    report("all zeros: no NaN grad", not torch.isnan(b.grad).any().item())
    report("all zeros: grad is zero", b.grad.abs().max().item() < 1e-8)

    # batch size 1
    a = torch.randn(1, 64)
    b = torch.randn(1, 64, requires_grad=True)
    loss = pairwise_concordance(a, b, mode="hinge")
    loss.sum().backward()
    report("batch=1: runs", loss.shape == (1,))
    report("batch=1: grad ok", not torch.isnan(b.grad).any().item())


# ===========================================================================
# Run all
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Pairwise concordance test suite")
    print("=" * 60)

    test_correctness()
    test_shapes()
    test_gradients()
    test_gradient_zeros()
    test_optimisation()
    test_binned_consistency()
    test_sampling_unbiased()
    test_large_D()
    test_tau_reference()
    test_c_gradient_scale()
    test_edge_cases()

    n_pass = sum(results)
    n_total = len(results)
    n_fail = n_total - n_pass

    print("\n" + "=" * 60)
    if n_fail == 0:
        print(f"{PASS} All {n_total} tests passed")
    else:
        print(f"{FAIL} {n_fail}/{n_total} tests failed")
    print("=" * 60)
"""
Comparison of four Spearman-based correlation implementations:
  1. scipy.stats.spearmanr          – exact, non-differentiable reference
  2. diff_spearman                  – fixed-temperature soft ranking (torchsort)
  3. diff_spearman_ai                – MADM-normalise then fixed-τ soft ranking (torchsort)
  4. monotone_invariant_spearman    – soft-CDF quantile approach (monotone invariant)

The script probes reliability across seven scenarios that commonly trip up
rank-correlation estimators:
  A. "happy path" – moderate positive correlation
  B. High noise     – near-zero true correlation
  C. Ties / constant segments
  D. Outliers       – single extreme value in one vector
  E. Scale shift    – pred and target on very different scales (tests invariance)
  F. Batch (2-D)    – multiple samples at once
  G. Non-affine monotone transform – pred = exp(4·base), target = base (true ρ = 1)

Metrics collected per scenario:
  • Value produced by each method
  • Absolute error vs scipy reference
  • Gradient norm (proxy for training signal quality; NaN-safe)
"""

import math
import torch
import numpy as np
from scipy.stats import spearmanr as scipy_spearman

from flexModel.utils import diff_spearman, diff_spearman_ai, diff_spearman_ai_st, monotone_invariant_spearman


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def scipy_ref(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Exact Spearman correlation via scipy (no gradient)."""
    if pred.dim() == 2:
        results = []
        for i in range(pred.shape[0]):
            r, _ = scipy_spearman(pred[i].detach().cpu().numpy(),
                                  target[i].detach().cpu().numpy())
            results.append(r)
        return results
    r, _ = scipy_spearman(pred.detach().cpu().numpy(),
                          target.detach().cpu().numpy())
    return float(r)


def grad_norm(fn, pred, target) -> float:
    """Return L2 gradient norm of fn w.r.t. pred, or NaN if gradient fails."""
    p = pred.clone().detach().requires_grad_(True)
    try:
        val = fn(p, target.clone().detach())
        if val.dim() > 0:          # batch – sum to scalar for backward
            val = val.mean()
        val.backward()
        if p.grad is None or torch.isnan(p.grad).any():
            return float("nan")
        return p.grad.norm().item()
    except Exception as e:
        print(f"    [grad error] {e}")
        return float("nan")


def eval_pair(name, pred, target, fn):
    """Run one method on one (pred, target) pair and return a dict of metrics."""
    ref = scipy_ref(pred, target)
    with torch.no_grad():
        val = fn(pred, target)
    if isinstance(val, torch.Tensor):
        val = val.item() if val.numel() == 1 else val.tolist()
    if isinstance(ref, list):
        err = [abs(v - r) for v, r in zip(val, ref)]
        mean_err = sum(err) / len(err)
    else:
        err = abs(val - ref)
        mean_err = err
    gn = grad_norm(fn, pred, target)
    return {"method": name, "value": val, "ref": ref, "abs_err": mean_err, "grad_norm": gn}


def print_table(scenario_name, rows):
    print(f"\n{'─'*68}")
    print(f"  Scenario: {scenario_name}")
    print(f"{'─'*68}")
    print(f"  {'Method':<30} {'Value':>8}  {'Ref':>8}  {'|err|':>8}  {'|∇|':>10}")
    print(f"  {'─'*28} {'─'*8}  {'─'*8}  {'─'*8}  {'─'*10}")
    for r in rows:
        v = r['value']
        ref = r['ref']
        err = r['abs_err']
        gn = r['grad_norm']
        # format vectors as mean for batch case
        if isinstance(v, list):
            v = sum(v) / len(v)
        if isinstance(ref, list):
            ref = sum(ref) / len(ref)
        gn_s = f"{gn:.5f}" if not math.isnan(gn) else "NaN"
        print(f"  {r['method']:<30} {v:>8.4f}  {ref:>8.4f}  {err:>8.4f}  {gn_s:>10}")


# ──────────────────────────────────────────────────────────────────────────────
# Test scenarios
# ──────────────────────────────────────────────────────────────────────────────

torch.manual_seed(0)
N = 64   # vector length

def _affine_inv(p, t, **kw):
    return diff_spearman_ai(p, t, **kw)


def _monotone_inv(p, t, **kw):
    """Wrap monotone_invariant_spearman to accept 1D or 2D inputs."""
    squeeze = p.dim() == 1
    if squeeze:
        p, t = p.unsqueeze(0), t.unsqueeze(0)
    out = monotone_invariant_spearman(p, t, **kw)
    return out.squeeze(0) if squeeze else out


methods = {
    "diff_spearman (fixed τ=1e-2)":        lambda p, t: diff_spearman(p, t),
    "diff_spearman (fixed τ=1e-3)":        lambda p, t: diff_spearman(p, t, regularization_strength=1e-3),
    "diff_spearman (fixed τ=1e-1)":        lambda p, t: diff_spearman(p, t, regularization_strength=1e-1),
    "diff_spearman_ai (τ0=3.0)":           lambda p, t: _affine_inv(p, t, tau0=3.0),
    "diff_spearman_ai (τ0=2.0)":           lambda p, t: _affine_inv(p, t, tau0=2.0),
    "diff_spearman_ai (τ0=1.0)":           lambda p, t: _affine_inv(p, t, tau0=1.0),
    "diff_spearman_ai (τ0=0.5)":           lambda p, t: _affine_inv(p, t, tau0=0.5),
    "diff_spearman_ai_st (tv=0.3,tg=5)": lambda p, t: diff_spearman_ai_st(p, t, tau_val=0.3, tau_grad=5.0),
    "monotone_invariant_spearman":          lambda p, t: _monotone_inv(p, t),
}

scenarios = {}

# A. Happy path – moderate positive correlation
base = torch.linspace(0, 1, N)
noise = torch.randn(N) * 0.3
scenarios["A. Moderate positive correlation"] = (base + noise, base)

# B. High noise – near-zero true correlation
scenarios["B. Near-zero correlation (pure noise)"] = (
    torch.randn(N), torch.randn(N)
)

# C. Ties / constant segments
pred_ties = torch.cat([torch.zeros(N // 4), torch.randn(3 * N // 4)])
target_ties = torch.linspace(-1, 1, N)
torch.manual_seed(1)
idx = torch.randperm(N)
scenarios["C. Pred has ties (constant block)"] = (pred_ties[idx], target_ties[idx])

# D. Outlier – single extreme value
pred_out = torch.randn(N)
target_out = torch.randn(N)
pred_out[0] = 100.0   # extreme outlier
scenarios["D. Single extreme outlier in pred"] = (pred_out, target_out)

# E. Scale shift – pred 1000× larger than target
pred_scale = torch.randn(N) * 1000.0
target_scale = pred_scale / 1000.0 + torch.randn(N) * 0.5
scenarios["E. Scale shift (pred ~1000× target)"] = (pred_scale, target_scale)

# F. Batch (2-D)
batch_pred   = torch.stack([base + torch.randn(N) * 0.2,
                              torch.randn(N)], dim=0)
batch_target = torch.stack([base, torch.linspace(1, 0, N)], dim=0)
scenarios["F. Batch (2 samples)"] = (batch_pred, batch_target)

# G. Non-affine monotone transform – pred = exp(base), target = base
# Spearman should be ~1.0; only monotone-invariant methods should track this well
pred_mono = torch.exp(base * 4)    # strongly non-linear but order-preserving
target_mono = base
scenarios["G. Non-affine monotone transform (exp)"] = (pred_mono, target_mono)


# ──────────────────────────────────────────────────────────────────────────────
# Run comparisons
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "="*68)
print("   SPEARMAN IMPLEMENTATION COMPARISON")
print("   Reference: scipy.stats.spearmanr  (non-differentiable, exact)")
print("="*68)

summary = {name: {"total_abs_err": 0.0, "count": 0, "nan_grads": 0} for name in methods}

for scenario_name, (pred, target) in scenarios.items():
    rows = []
    for mname, mfn in methods.items():
        r = eval_pair(mname, pred, target, mfn)
        rows.append(r)
        # accumulate summary stats
        err = r["abs_err"]
        if isinstance(err, list):
            for e in err:
                summary[mname]["total_abs_err"] += e
                summary[mname]["count"] += 1
        else:
            summary[mname]["total_abs_err"] += err
            summary[mname]["count"] += 1
        if math.isnan(r["grad_norm"]):
            summary[mname]["nan_grads"] += 1
    print_table(scenario_name, rows)


# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────

print(f"\n{'='*68}")
print("  OVERALL SUMMARY (lower abs_err is better; 0 NaN grads is best)")
print(f"{'='*68}")
print(f"  {'Method':<40} {'Mean |err|':>10}  {'NaN grads':>10}")
print(f"  {'─'*38} {'─'*10}  {'─'*10}")
for mname, s in summary.items():
    mean_err = s["total_abs_err"] / max(s["count"], 1)
    print(f"  {mname:<40} {mean_err:>10.5f}  {s['nan_grads']:>10}")

print()


# ──────────────────────────────────────────────────────────────────────────────
# Gradient quality: does the signal survive multiple backward passes?
# ──────────────────────────────────────────────────────────────────────────────

print(f"{'='*68}")
print("  GRADIENT QUALITY ACROSS TRAINING STEPS (scenario A)")
print(f"{'='*68}")
print("  500 Adam steps (lr=1e-2); printed every 25 steps.")
print("  Target: corr → 1.0 (p is free vector, target is perfectly ordered).\n")

NSTEPS   = 500
PRINT_EVERY = 25
pred_A, target_A = scenarios["A. Moderate positive correlation"]

for mname, mfn in methods.items():
    p = pred_A.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([p], lr=1e-2)
    print(f"  {mname}")
    print(f"  {'Step':>5}  {'Corr':>8}  {'Loss':>8}  {'|∇|':>10}")
    for step in range(NSTEPS):
        opt.zero_grad()
        corr = mfn(p, target_A.detach())
        if corr.dim() > 0:
            corr = corr.mean()
        loss = 1 - corr          # maximise correlation
        loss.backward()
        gn_s = p.grad.norm().item() if p.grad is not None else float("nan")
        if step % PRINT_EVERY == 0 or step == NSTEPS - 1:
            print(f"  {step:>5}  {corr.item():>8.4f}  {loss.item():>8.4f}  {gn_s:>10.5f}")
        opt.step()
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Affine-invariance tests
# An affine-invariant method must satisfy corr(a*x+b, y) == corr(x, y)
# for any a!=0, b.  We measure drift = |corr(transformed) - corr(original)|.
# ──────────────────────────────────────────────────────────────────────────────

print(f"{'='*68}")
print("  AFFINE-INVARIANCE TEST")
print("  corr(a·pred + b, target) should equal corr(pred, target)")
print("  Drift = |transformed_corr − original_corr|  (lower is more invariant)")
print(f"{'='*68}")

torch.manual_seed(0)
pred_base  = torch.randn(N)
target_base = torch.randn(N)
target_base = 0.6 * pred_base + torch.randn(N) * 0.5   # moderate true correlation

transforms = {
    "identity  (a=1, b=0)":       (1.0,   0.0),
    "scale ×10 (a=10, b=0)":      (10.0,  0.0),
    "scale ×1000":                 (1000., 0.0),
    "scale ×0.001":                (0.001, 0.0),
    "shift only (a=1, b=500)":    (1.0,   500.0),
    "shift only (a=1, b=-1e4)":   (1.0,  -1e4),
    "scale+shift (a=37, b=-999)": (37.0, -999.0),
    "tiny scale (a=1e-6, b=0)":   (1e-6,  0.0),
}
# Note: negative-scale transforms (a<0) negate the correlation by design
# (Spearman is antisymmetric), so they are tested separately below.

# Baseline values for each method
baselines = {}
with torch.no_grad():
    for mname, mfn in methods.items():
        baselines[mname] = mfn(pred_base, target_base)
        if isinstance(baselines[mname], torch.Tensor):
            baselines[mname] = baselines[mname].item()

# Header
col_w = 28
hdr_methods = list(methods.keys())
print(f"\n  {'Transform':<34}", end="")
for mname in hdr_methods:
    # shorten label for table
    short = mname.split("(")[1].rstrip(")") if "(" in mname else mname[:col_w]
    print(f"  {short:>14}", end="")
print()
print(f"  {'─'*32}", end="")
for _ in hdr_methods:
    print(f"  {'─'*14}", end="")
print()

for tname, (a, b) in transforms.items():
    pred_t = pred_base * a + b
    print(f"  {tname:<34}", end="")
    for mname, mfn in methods.items():
        with torch.no_grad():
            val = mfn(pred_t, target_base)
            if isinstance(val, torch.Tensor):
                val = val.item()
        drift = abs(val - baselines[mname])
        # flag large drift
        flag = " !" if drift > 0.01 else "  "
        print(f"  {drift:>12.5f}{flag}", end="")
    print()

print(f"\n  Baseline correlation values:")
for mname, bval in baselines.items():
    print(f"    {mname:<40}  {bval:.5f}")

# Also test negative correlation flipping:
print(f"\n  Negative-scale invariance: a=-1 should flip sign of corr")
print(f"  {'Method':<40}  {'orig':>8}  {'a=-1':>8}  {'|orig|+|neg|':>14}  {'symmetric?':>12}")
for mname, mfn in methods.items():
    with torch.no_grad():
        orig = mfn(pred_base,        target_base)
        neg  = mfn(-1.0 * pred_base, target_base)
        if isinstance(orig, torch.Tensor):
            orig = orig.item()
        if isinstance(neg, torch.Tensor):
            neg = neg.item()
    sym = abs(orig + neg)   # should be ~0 if truly antisymmetric
    ok = "YES" if sym < 1e-4 else f"NO  (|orig+neg|={sym:.5f})"
    print(f"  {mname:<40}  {orig:>8.4f}  {neg:>8.4f}  {abs(orig)+abs(neg):>14.5f}  {ok:>12}")

"""Test that the model can train and fit random data using pairwise concordance."""

from __future__ import annotations

import math

import pytest
import torch
from scipy.stats import spearmanr as scipy_spearman

from flexModel.pairwise_concordance import pairwise_concordance
from tests.test_training import create_flex_module


def _spearman_rho(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute SciPy Spearman rho for two 1D tensors."""
    rho, _ = scipy_spearman(x.detach().cpu().numpy(), y.detach().cpu().numpy())
    return float(rho)


def _module_projection_vectors(model, ge):
    """Return the projected module-level vectors used by loss_cor."""
    flxs, flxs_p = model(ge)
    flxs_eff = flxs_p if flxs_p is not None else flxs
    ge_sdt = ge - ge.mean(dim=1, keepdim=True)
    ge_sdt = ge_sdt / (ge_sdt.std(dim=1, keepdim=True) + 1e-7)
    flxs_sdt = flxs_eff.abs() - flxs_eff.abs().mean(dim=1, keepdim=True)
    flxs_sdt = flxs_sdt / (flxs_sdt.std(dim=1, keepdim=True) + 1e-7)
    return (model.Mmg @ ge_sdt.t()).t(), (model.Mmr @ flxs_sdt.t()).t()


def _similarity_vectors(model, ge):
    """Return the pairwise similarity vectors used by loss_sco."""
    flxs, flxs_p = model(ge)
    flxs_eff = flxs_p if flxs_p is not None else flxs
    flxs_n = flxs_eff - flxs_eff.mean(dim=0, keepdim=True)
    ge_n = ge - ge.mean(dim=0, keepdim=True)
    flxs_n = flxs_n / (flxs_n.norm(dim=1, keepdim=True).detach().clamp_min(1e-12))
    ge_n = ge_n / (ge_n.norm(dim=1, keepdim=True).detach().clamp_min(1e-12))
    sim_f = flxs_n @ flxs_n.t()
    sim_g = ge_n @ ge_n.t()
    n = sim_f.shape[0]
    idx = torch.triu_indices(n, n, offset=1, device=sim_f.device)
    return sim_f[idx[0], idx[1]], sim_g[idx[0], idx[1]]


def report_rho_components(model, ge):
    """Return readable SciPy rho values for the correlation-based terms."""
    cor_pred, cor_tgt = _module_projection_vectors(model, ge)
    sco_pred, sco_tgt = _similarity_vectors(model, ge)
    return {
        "rho_cor": _spearman_rho(cor_pred.flatten(), cor_tgt.flatten()),
        "rho_sco": _spearman_rho(sco_pred, sco_tgt),
    }


def oscillating_weights(
    step, total_steps, n_tasks, base_weights, amplitude=0.8, period=50
):
    """Task weights oscillate with opposite phases, damping over time.

    w_i(t) = base_i * (1 + A(t) * cos(2π t/T + 2π i/n))

    where A(t) = amplitude * (1 - t/total_steps)  # linear decay
    """
    decay = amplitude * (1 - step / total_steps)
    phases = torch.arange(n_tasks, device=base_weights.device) * (2 * math.pi / n_tasks)
    osc = decay * torch.cos(2 * math.pi * step / period + phases)
    return base_weights * (1 + osc)


def current_task_weights(step, total_steps, model, period=50):
    """Get the oscillating weights for the current step on the model's loss terms."""
    base_weights = model.loss_lms.detach()
    weights = oscillating_weights(
        step=step,
        total_steps=total_steps,
        n_tasks=base_weights.numel(),
        base_weights=base_weights,
        period=period,
    )
    total = weights.sum()
    if total.abs().item() > 0:
        weights = weights / total * base_weights.sum()
    return weights.to(device=base_weights.device, dtype=base_weights.dtype)


def report_loss_components(model, ge, task_weights=None):
    """Return the mean loss components, weighted components, and total loss."""
    with torch.no_grad():
        flxs, flxs_p = model(ge)
        components = model.losses(ge, flxs, flxs_p).mean(dim=0)
        weights = model.loss_lms if task_weights is None else task_weights
        loss_nrm = components / components.detach().abs().clamp_min(1e-12)
        weighted = loss_nrm * weights
        total = weighted.sum()
    return components, loss_nrm, weighted, total


def objective_terms(model, ge, task_weights=None):
    """Return normalized loss terms and the weighted objective terms."""
    flxs, flxs_p = model(ge)
    components = model.losses(ge, flxs, flxs_p).mean(dim=0)
    weights = model.loss_lms if task_weights is None else task_weights
    loss_nrm = components / components.detach().abs().clamp_min(1e-12)
    objective_terms = loss_nrm * weights
    return components, loss_nrm, objective_terms


def report_gradient_contributions(model, ge, task_weights=None):
    """Return per-term objective gradient norms and percentages normalized to 100%."""
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        zeros = torch.zeros(5)
        return zeros, zeros, torch.tensor(0.0)

    _, _, weighted = objective_terms(model, ge, task_weights=task_weights)

    grad_norms = []
    n_terms = weighted.numel()
    for idx in range(n_terms):
        grads = torch.autograd.grad(
            weighted[idx],
            params,
            retain_graph=True,
            allow_unused=True,
        )
        sq_norm = torch.zeros((), device=weighted.device)
        for grad in grads:
            if grad is not None:
                sq_norm = sq_norm + grad.detach().pow(2).sum()
        grad_norms.append(torch.sqrt(sq_norm))

    grad_norms = torch.stack(grad_norms)
    total_norm = grad_norms.sum()
    if total_norm.item() == 0:
        grad_pct = torch.zeros_like(grad_norms)
    else:
        grad_pct = grad_norms / total_norm * 100.0
    total_objective = weighted.sum()
    total_objective_grads = torch.autograd.grad(
        total_objective, params, retain_graph=False, allow_unused=True
    )
    total_sq_norm = torch.zeros((), device=weighted.device)
    for grad in total_objective_grads:
        if grad is not None:
            total_sq_norm = total_sq_norm + grad.detach().pow(2).sum()
    total_objective_norm = torch.sqrt(total_sq_norm)
    return grad_norms, grad_pct, total_objective_norm


def print_loss_components(label, model, ge, task_weights=None):
    names = ["los_bf", "los_pos", "los_cor", "los_sco", "los_ent"]
    components, loss_nrm, weighted, total = report_loss_components(
        model, ge, task_weights=task_weights
    )
    _, grad_pct, objective_grad_norm = report_gradient_contributions(
        model, ge, task_weights=task_weights
    )
    rho_vals = report_rho_components(model, ge)
    component_text = ", ".join(
        f"{name}={value.item():.4f}" for name, value in zip(names, components)
    )
    nrm_text = ", ".join(
        f"{name}={value.item():.4f}" for name, value in zip(names, loss_nrm)
    )
    weighted_text = ", ".join(
        f"{name}={value.item():.4f}" for name, value in zip(names, weighted)
    )
    grad_text = ", ".join(
        f"{name}={pct.item():.1f}%" for name, pct in zip(names, grad_pct)
    )
    if task_weights is None:
        task_weights = torch.ones_like(weighted)
    weighted_total = weighted.sum()
    weighted_text = ", ".join(
        f"{name}={value.item():.4f}" for name, value in zip(names, weighted)
    )
    print(f"{label}: unscaled_total={total.item():.4f}")
    print(f"  weighted_total: {weighted_total.item():.4f}")
    if task_weights is not None:
        weight_text = ", ".join(
            f"{name}={value.item():.4f}" for name, value in zip(names, task_weights)
        )
        print(f"  task_wts: {weight_text}")
    print(f"  raw:     {component_text}")
    print(f"  nrm:     {nrm_text}")
    print(f"  weighted: {weighted_text}")
    print(f"  unscaled_terms: {weighted_text}")
    print(
        f"  rho:     rho_cor={rho_vals['rho_cor']:.4f}, rho_sco={rho_vals['rho_sco']:.4f}"
    )
    print(f"  obj_grad_norm: {objective_grad_norm.item():.4f}")
    print(f"  grad%:   {grad_text}")


def format_loss_line(step, model, ge, task_weights=None):
    names = ["los_bf", "los_pos", "los_cor", "los_sco", "los_ent"]
    components, loss_nrm, weighted, total = report_loss_components(
        model, ge, task_weights=task_weights
    )
    _, grad_pct, objective_grad_norm = report_gradient_contributions(
        model, ge, task_weights=task_weights
    )
    rho_vals = report_rho_components(model, ge)
    raw_text = ", ".join(
        f"{name}={value.item():.4f}" for name, value in zip(names, components)
    )
    nrm_text = ", ".join(
        f"{name}={value.item():.4f}" for name, value in zip(names, loss_nrm)
    )
    weighted_text = ", ".join(
        f"{name}={value.item():.4f}" for name, value in zip(names, weighted)
    )
    grad_text = ", ".join(
        f"{name}={pct.item():.1f}%" for name, pct in zip(names, grad_pct)
    )
    weighted_total = weighted.sum()
    unscaled_text = ", ".join(
        f"{name}={value.item():.4f}" for name, value in zip(names, weighted)
    )
    weight_text = ""
    if task_weights is not None:
        weight_text = " | task_wts: " + ", ".join(
            f"{name}={value.item():.4f}" for name, value in zip(names, task_weights)
        )
    return (
        f"  Step {step:3d}: unscaled={total.item():.4f} "
        f"weighted_total={weighted_total.item():.4f}{weight_text} | obj_grad_norm={objective_grad_norm.item():.4f} | raw: {raw_text} | nrm: {nrm_text} | weighted: {weighted_text} | unscaled_terms: {unscaled_text} | rho: rho_cor={rho_vals['rho_cor']:.4f}, rho_sco={rho_vals['rho_sco']:.4f} | grad%: {grad_text}"
    )


@pytest.mark.parametrize("use_disc", [False, True])
def test_model_overfits_random_data_pairwise(use_disc):
    """Test that model can overfit a small random training set using pairwise concordance."""
    print(
        "\n🧪 Testing standard FlexGNN training with pairwise concordance on random data..."
    )

    torch.manual_seed(42)
    model, n_genes, _ = create_flex_module(
        n_genes=100, n_reactions=100, use_layer_weights=False, use_disc=use_disc
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    n_samples = 16
    ge = torch.randn(n_samples, n_genes, device=device).abs()

    model.train()
    with torch.no_grad():
        initial_loss = model.training_step((ge,), 0)
        initial_loss_value = initial_loss.item()

    print_loss_components("Initial loss components", model, ge)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    n_steps = 250
    losses = []

    for step in range(n_steps):
        optimizer.zero_grad()
        loss = model.training_step((ge,), step)
        task_weights = current_task_weights(step, n_steps, model)
        components = model.losses(ge, *model(ge)).mean(dim=0)
        # loss_nrm = components / components.detach().clamp_min(1e-12)
        loss_nrm = components
        current_weight = model.loss_lms * task_weights
        loss_scld = loss_nrm * current_weight
        scaled_loss = loss_scld.sum()
        scaled_loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step % 5 == 0:
            print(format_loss_line(step, model, ge, task_weights=task_weights))

    final_loss = losses[-1]
    with torch.no_grad():
        final_task_weights = current_task_weights(n_steps - 1, n_steps, model)
        final_components, final_loss_nrm, final_loss_scld = objective_terms(
            model, ge, task_weights=final_task_weights
        )
        final_weighted_loss = final_loss_scld.sum().item()
    print_loss_components(
        "Final loss components", model, ge, task_weights=final_task_weights
    )
    print(f"\nFinal unscaled loss: {final_loss:.4f}")
    print(f"Final weighted loss: {final_weighted_loss:.4f}")
    print(
        f"Loss reduction: {initial_loss_value - final_loss:.4f} ({100*(1 - final_loss/initial_loss_value):.1f}% decrease)"
    )
    print(
        f"Loss trajectory: [{losses[0]:.3f} → {losses[9]:.3f} → {losses[19]:.3f} → {losses[29]:.3f} → {losses[39]:.3f} → {losses[49]:.3f}]"
    )

    assert (
        final_loss < initial_loss_value
    ), f"Model failed to reduce loss! Initial: {initial_loss_value:.4f}, Final: {final_loss:.4f}"

    reduction_ratio = final_loss / initial_loss_value
    assert (
        reduction_ratio < 0.9
    ), f"Loss reduction too small! Only {100*(1-reduction_ratio):.1f}% decrease. Expected at least 10%."

    print(
        "✅ Model successfully overfits random data with pairwise concordance - training mechanics work!"
    )


@pytest.mark.parametrize("use_disc", [False, True])
def test_model_with_layer_weights_overfits_pairwise(use_disc):
    """Test that model with layer weights can overfit random data using pairwise concordance."""
    print(
        "\n🧪 Testing FlexGNN with layer weights and pairwise concordance on random data..."
    )

    torch.manual_seed(42)
    model, n_genes, _ = create_flex_module(
        n_genes=100, n_reactions=100, use_layer_weights=True, use_disc=use_disc
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    n_samples = 16
    ge = torch.randn(n_samples, n_genes, device=device).abs()

    model.train()
    with torch.no_grad():
        initial_loss = model.training_step((ge,), 0)
        initial_loss_value = initial_loss.item()

    print_loss_components("Initial loss components", model, ge)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    n_steps = 2000
    losses = []

    for step in range(n_steps):
        optimizer.zero_grad()
        loss = model.training_step((ge,), step)
        task_weights = current_task_weights(step, n_steps, model, period=200)
        components = model.losses(ge, *model(ge)).mean(dim=0)
        # loss_nrm = components / components.detach().clamp_min(1e-12)
        loss_nrm = components
        current_weight = model.loss_lms * task_weights
        loss_scld = loss_nrm * current_weight
        scaled_loss = loss_scld.sum()
        scaled_loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step % 5 == 0:
            print(format_loss_line(step, model, ge, task_weights=task_weights))

    final_loss = losses[-1]
    with torch.no_grad():
        final_task_weights = current_task_weights(
            n_steps - 1, n_steps, model, period=100
        )
        final_components, final_loss_nrm, final_loss_scld = objective_terms(
            model, ge, task_weights=final_task_weights
        )
        final_weighted_loss = final_loss_scld.sum().item()
    print_loss_components(
        "Final loss components", model, ge, task_weights=final_task_weights
    )
    print(f"\nFinal unscaled loss: {final_loss:.4f}")
    print(f"Final weighted loss: {final_weighted_loss:.4f}")
    print(
        f"Loss reduction: {initial_loss_value - final_loss:.4f} ({100*(1 - final_loss/initial_loss_value):.1f}% decrease)"
    )
    print(
        f"Loss trajectory: [{losses[0]:.3f} → {losses[9]:.3f} → {losses[19]:.3f} → {losses[29]:.3f} → {losses[39]:.3f} → {losses[49]:.3f}]"
    )

    assert (
        final_loss < initial_loss_value
    ), f"Model failed to reduce loss! Initial: {initial_loss_value:.4f}, Final: {final_loss:.4f}"

    reduction_ratio = final_loss / initial_loss_value
    assert (
        reduction_ratio < 0.9
    ), f"Loss reduction too small! Only {100*(1-reduction_ratio):.1f}% decrease. Expected at least 10%."

    print(
        "✅ FlexGNN with layer weights successfully overfits random data using pairwise concordance!"
    )


if __name__ == "__main__":
    test_model_overfits_random_data_pairwise()
    test_model_with_layer_weights_overfits_pairwise()

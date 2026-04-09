"""Test that the model can train and fit random data using pairwise concordance."""

from __future__ import annotations

import math

import pytest
import torch

from tests.factories import create_flex_module


def _oscillating_weights(
    step: int,
    total_steps: int,
    n_tasks: int,
    base_weights: torch.Tensor,
    amplitude: float = 0.8,
    period: int = 50,
) -> torch.Tensor:
    """Task weights oscillate with opposite phases, damping over time.

    w_i(t) = base_i * (1 + A(t) * cos(2*pi*t/T + 2*pi*i/n))
    where A(t) = amplitude * (1 - t/total_steps).
    """
    decay = amplitude * (1 - step / total_steps)
    phases = torch.arange(n_tasks, device=base_weights.device) * (2 * math.pi / n_tasks)
    osc = decay * torch.cos(2 * math.pi * step / period + phases)
    return base_weights * (1 + osc)


def _current_task_weights(
    step: int, total_steps: int, model: torch.nn.Module, period: int = 50
) -> torch.Tensor:
    """Return oscillating loss term weights for the current training step."""
    base_weights = model.loss_lms.detach()
    weights = _oscillating_weights(
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


@pytest.mark.parametrize("use_disc", [False, True])
def test_model_overfits_random_data_pairwise(use_disc: bool) -> None:
    """Test that model can overfit a small random training set using pairwise concordance."""
    torch.manual_seed(42)
    model, n_genes, _ = create_flex_module(
        n_genes=100, n_reactions=100, use_layer_weights=False, use_disc=use_disc
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    n_samples = 16
    ge = torch.randn(n_samples, n_genes, device=device).abs()

    model.train()
    with torch.no_grad():
        initial_loss = model.training_step((ge,), 0)
        initial_loss_value = initial_loss.item()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    n_steps = 50
    losses = []
    for step in range(n_steps):
        optimizer.zero_grad()
        loss = model.training_step((ge,), step)
        task_weights = _current_task_weights(step, n_steps, model)
        components = model.losses(ge, *model(ge)).mean(dim=0)
        loss_nrm = components
        current_weight = model.loss_lms * task_weights
        loss_scld = loss_nrm * current_weight
        scaled_loss = loss_scld.sum()
        scaled_loss.backward()
        optimizer.step()
        losses.append(loss.item())

    final_loss = losses[-1]
    assert (
        final_loss < initial_loss_value
    ), f"Model failed to reduce loss! Initial: {initial_loss_value:.4f}, Final: {final_loss:.4f}"

    reduction_ratio = final_loss / initial_loss_value
    assert (
        reduction_ratio < 0.9
    ), f"Loss reduction too small! Only {100*(1-reduction_ratio):.1f}% decrease."


@pytest.mark.parametrize("use_disc", [False, True])
def test_model_with_layer_weights_overfits_pairwise(use_disc: bool) -> None:
    """Test that model with layer weights can overfit random data using pairwise concordance."""
    torch.manual_seed(42)
    model, n_genes, _ = create_flex_module(
        n_genes=100, n_reactions=100, use_layer_weights=True, use_disc=use_disc
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    n_samples = 16
    ge = torch.randn(n_samples, n_genes, device=device).abs()

    model.train()
    with torch.no_grad():
        initial_loss = model.training_step((ge,), 0)
        initial_loss_value = initial_loss.item()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    n_steps = 200
    losses = []
    for step in range(n_steps):
        optimizer.zero_grad()
        loss = model.training_step((ge,), step)
        task_weights = _current_task_weights(step, n_steps, model, period=200)
        components = model.losses(ge, *model(ge)).mean(dim=0)
        loss_nrm = components
        current_weight = model.loss_lms * task_weights
        loss_scld = loss_nrm * current_weight
        scaled_loss = loss_scld.sum()
        scaled_loss.backward()
        optimizer.step()
        losses.append(loss.item())

    final_loss = losses[-1]
    assert (
        final_loss < initial_loss_value
    ), f"Model failed to reduce loss! Initial: {initial_loss_value:.4f}, Final: {final_loss:.4f}"

    reduction_ratio = final_loss / initial_loss_value
    assert (
        reduction_ratio < 0.9
    ), f"Loss reduction too small! Only {100*(1-reduction_ratio):.1f}% decrease."

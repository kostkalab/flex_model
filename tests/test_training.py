"""Test that the model can train and fit random data."""

from __future__ import annotations

import pytest
import torch

from tests.factories import create_flex_module


@pytest.mark.parametrize("use_disc", [False, True])
def test_model_overfits_random_data(use_disc: bool) -> None:
    """Test that model can overfit a small random training set (standard FlexGNN).

    If the model cannot reduce loss on random data, there's likely
    a bug in the training setup, loss computation, or backpropagation.
    """
    torch.manual_seed(42)
    model, n_genes, n_reactions = create_flex_module(
        n_genes=64, n_reactions=128, use_layer_weights=False, use_disc=use_disc
    )

    # Create fixed random gene expression data
    n_samples = 16
    ge = torch.randn(n_samples, n_genes).abs()

    # Get initial loss
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
        loss.backward()
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
def test_model_with_layer_weights_overfits(use_disc: bool) -> None:
    """Test that model with layer weights can overfit random data (FlexGNN_LW).

    Verifies that the layer-weighted variant also trains correctly.
    """
    torch.manual_seed(42)
    model, n_genes, n_reactions = create_flex_module(
        n_genes=64, n_reactions=128, use_layer_weights=True, use_disc=use_disc
    )

    # Create fixed random gene expression data
    n_samples = 16
    ge = torch.randn(n_samples, n_genes).abs()

    # Get initial loss
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
        loss.backward()
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

"""Tests for antisymmetry properties of half-space antisymmetric scorers."""

from __future__ import annotations

import pytest
import torch

from flexModel.halfSpaceAntiSymmetric import AntisymmetricFunc, BiLinAntisymmetricFunc


def _assert_antisymmetric(
    module: torch.nn.Module, x1: torch.Tensor, x2: torch.Tensor
) -> None:
    """Assert f(x1, x2) = -f(x2, x1) up to numerical tolerance."""
    y12 = module(x1, x2)
    y21 = module(x2, x1)
    torch.testing.assert_close(y12 + y21, torch.zeros_like(y12), atol=1e-6, rtol=1e-6)


def _assert_zero_on_diagonal(module: torch.nn.Module, x: torch.Tensor) -> None:
    """Assert f(x, x) = 0 for an antisymmetric function."""
    y = module(x, x)
    torch.testing.assert_close(y, torch.zeros_like(y), atol=1e-6, rtol=1e-6)


def test_bilin_antisymmetric_swap_property() -> None:
    torch.manual_seed(0)
    batch, n_reactions, d = 4, 7, 11
    x1 = torch.randn(batch, n_reactions, d)
    x2 = torch.randn(batch, n_reactions, d)

    model = BiLinAntisymmetricFunc(d=d, k=8, rank=6)
    model.eval()

    _assert_antisymmetric(model, x1, x2)


def test_bilin_antisymmetric_zero_on_diagonal() -> None:
    torch.manual_seed(1)
    batch, n_reactions, d = 3, 5, 9
    x = torch.randn(batch, n_reactions, d)

    model = BiLinAntisymmetricFunc(d=d, k=6, rank=4)
    model.eval()

    _assert_zero_on_diagonal(model, x)


def test_stacked_antisymmetric_swap_property() -> None:
    torch.manual_seed(2)
    batch, n_reactions, d = 5, 6, 13
    x1 = torch.randn(batch, n_reactions, d)
    x2 = torch.randn(batch, n_reactions, d)

    # Keep dropout off so the two calls use the same deterministic mapping.
    model = AntisymmetricFunc(
        d=d,
        k=16,
        rank=8,
        n_layers_cross=3,
        n_layers_self=0,
        activation_cross="softsign",
        dropout=0.0,
    )
    model.eval()

    _assert_antisymmetric(model, x1, x2)


def test_stacked_antisymmetric_zero_on_diagonal() -> None:
    torch.manual_seed(3)
    batch, n_reactions, d = 2, 8, 10
    x = torch.randn(batch, n_reactions, d)

    model = AntisymmetricFunc(
        d=d,
        k=12,
        rank=6,
        n_layers_cross=2,
        n_layers_self=0,
        activation_cross="tanh",
        dropout=0.0,
    )
    model.eval()

    _assert_zero_on_diagonal(model, x)


def test_bilin_output_is_nontrivial() -> None:
    torch.manual_seed(10)
    d = 11
    x1 = torch.randn(4, 7, d)
    x2 = torch.randn(4, 7, d)
    model = BiLinAntisymmetricFunc(d=d, k=8, rank=6)
    model.eval()
    y = model(x1, x2)
    assert y.abs().max() > 1e-4, "Output is trivially zero."


def test_stacked_output_is_nontrivial() -> None:
    torch.manual_seed(11)
    d = 13
    x1 = torch.randn(5, 6, d)
    x2 = torch.randn(5, 6, d)
    model = AntisymmetricFunc(
        d=d,
        k=16,
        rank=8,
        n_layers_cross=3,
        n_layers_self=0,
        dropout=0.0,
    )
    model.eval()
    y = model(x1, x2)
    assert y.abs().max() > 1e-4, "Output is trivially zero."


def test_output_shapes() -> None:
    torch.manual_seed(20)
    batch, n_reactions, d = 3, 5, 8
    x1 = torch.randn(batch, n_reactions, d)
    x2 = torch.randn(batch, n_reactions, d)

    for model in [
        BiLinAntisymmetricFunc(d=d, k=4, rank=3),
        AntisymmetricFunc(
            d=d, k=12, rank=4, n_layers_cross=2, n_layers_self=0, dropout=0.0
        ),
    ]:
        model.eval()
        assert model(x1, x2).shape == (batch, n_reactions)


def test_rejects_mismatched_shapes() -> None:
    model = BiLinAntisymmetricFunc(d=8, k=4, rank=3)
    with pytest.raises(ValueError, match="identical shape"):
        model(torch.randn(2, 5, 8), torch.randn(2, 6, 8))


def test_rejects_wrong_dimension() -> None:
    model = AntisymmetricFunc(
        d=8, k=12, rank=4, n_layers_cross=2, n_layers_self=0, dropout=0.0
    )
    with pytest.raises(ValueError, match="expected last dimension"):
        model(torch.randn(2, 5, 10), torch.randn(2, 5, 10))


def test_rejects_2d_input() -> None:
    model = BiLinAntisymmetricFunc(d=8, k=4, rank=3)
    with pytest.raises(ValueError, match="3D tensors"):
        model(torch.randn(5, 8), torch.randn(5, 8))


def test_gradients_flow_to_all_parameters() -> None:
    torch.manual_seed(30)
    d = 10
    x1 = torch.randn(2, 3, d)
    x2 = torch.randn(2, 3, d)
    model = AntisymmetricFunc(
        d=d,
        k=8,
        rank=4,
        n_layers_cross=3,
        n_layers_self=0,
        activation_cross="softsign",
        dropout=0.0,
    )
    loss = model(x1, x2).sum()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and p.grad.abs().max() > 0, f"No gradient for {name}"


def test_stacked_default_config_is_antisymmetric() -> None:
    torch.manual_seed(40)
    batch, n_reactions, d = 3, 4, 12
    x1 = torch.randn(batch, n_reactions, d)
    x2 = torch.randn(batch, n_reactions, d)

    # Default constructor includes the self pathway; subtraction must preserve antisymmetry.
    model = AntisymmetricFunc(d=d, k=10, rank=5, dropout=0.0)
    model.eval()

    _assert_antisymmetric(model, x1, x2)

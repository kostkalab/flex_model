from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import lightning as L
import numpy as np
import pytest
import torch

import flexModel
from flexModel import AlignedFlexModule, FlexModule


def _array(values: object, dtype: np.dtype | type | None = None) -> np.ndarray:
    result = np.asarray(values, dtype=dtype)
    result.setflags(write=False)
    return result


def make_bundle() -> SimpleNamespace:
    """Create a small labelled bundle satisfying the public bundle contract."""
    genes = ("g0", "g1", "g2", "g3")
    reactions = ("r0", "r1", "r2")
    compounds = ("c0", "c1")
    modules = ("m0", "m1")

    def matrix(data: object, rows: tuple, cols: tuple) -> SimpleNamespace:
        return SimpleNamespace(
            vals=_array(data, np.float64), row_ids=_array(rows), col_ids=_array(cols)
        )

    manifest = SimpleNamespace(
        format="flex-species-bundle",
        schema_version=3,
        contracts={
            "reaction_graph": "binary_shared_compound_topology_v1",
            "currency_selection": "weighted_leiden_scan_v1",
            "projection": "svd_cumulative_energy_v1",
            "module_diffusion": "binary_heat_kernel_v1",
        },
        capabilities=[
            "embeddings",
            "reaction-graph",
            "stoichiometry",
            "approximate-projection",
            "module-maps",
        ],
        dimensions={
            "genes": len(genes),
            "reactions": len(reactions),
            "compounds": len(compounds),
            "modules": len(modules),
        },
    )
    return SimpleNamespace(
        manifest=manifest,
        species="demo",
        version="1.0.0",
        content_id="a" * 64,
        genes=genes,
        reactions=reactions,
        compounds=compounds,
        modules=modules,
        emb_gen=matrix(
            [[0.1, 0.7, -0.2], [0.8, -0.4, 0.3], [-0.5, 0.2, 0.9], [0.4, 0.6, 0.1]],
            genes,
            (0, 1, 2),
        ),
        emb_rea=matrix(
            [
                [0.2, -0.4, 0.9, 0.3, 0.7, -0.8],
                [0.8, 0.1, -0.2, -0.6, 0.4, 0.5],
                [-0.3, 0.9, 0.6, 0.7, -0.1, 0.2],
            ],
            reactions,
            tuple([f"input:{i}" for i in range(3)] + [f"output:{i}" for i in range(3)]),
        ),
        stoich=matrix([[1, -1, 0], [0, 1, -1]], compounds, reactions),
        mod_gen=matrix([[1, 0, 1, 0], [0, 1, 0, 1]], modules, genes),
        mod_rea=matrix([[1, 0, 1], [0, 1, 1]], modules, reactions),
        projection=matrix(np.eye(3), tuple(range(3)), reactions),
        gr_gen_rea=SimpleNamespace(
            directed=True,
            src_type="gene",
            dst_type="reaction",
            src_ids=_array(genes),
            dst_ids=_array(reactions),
            edge_index=_array([[0, 1, 2, 3, 0], [0, 1, 2, 0, 2]], np.int64),
        ),
        gr_rea_rea=SimpleNamespace(
            directed=False,
            src_type="reaction",
            dst_type="reaction",
            src_ids=_array(reactions),
            dst_ids=_array(reactions),
            edge_index=_array([[0, 1], [1, 2]], np.int64),
            edge_attrs={
                "f_disc": SimpleNamespace(
                    vals=_array([0.2, 0.8], np.float32),
                    ids=(("r0", "r1"), ("r1", "r2")),
                )
            },
        ),
    )


@pytest.mark.parametrize("module_cls", [FlexModule, AlignedFlexModule])
def test_synthetic_bundle_forward_with_synthetic_expression(
    module_cls: type[FlexModule],
) -> None:
    """Both bundle-backed models produce finite raw and projected fluxes."""
    torch.manual_seed(31)
    bundle = make_bundle()
    model = module_cls.from_bundle(
        bundle, nlayers=2, use_disc=True, use_rea_rea_gate=True, flx_project=True
    ).eval()
    values = np.random.default_rng(31).lognormal(size=(3, len(bundle.genes)))
    expression = model.align_expression(values[:, ::-1], bundle.genes[::-1])
    with torch.inference_mode():
        raw, projected = model(expression)
    assert raw.shape == (3, len(bundle.reactions))
    assert projected is not None and projected.shape == raw.shape
    assert torch.isfinite(raw).all()
    assert torch.isfinite(projected).all()
    assert raw.abs().sum() > 0
    assert projected.abs().sum() > 0


def test_from_bundle_copies_data_and_uses_canonical_dimensions() -> None:
    bundle = make_bundle()
    model = FlexModule.from_bundle(bundle)
    assert model.Mcr.shape == (2, 3)
    assert model.Mmg.shape == (2, 4)
    assert model.Mmr.shape == (2, 3)
    assert model.gen_emb_tt.shape == (4, 3)
    assert model.rea_emb_tt.shape == (3, 6)
    assert model.eid_g2r.shape[0] == model.eid_r2r.shape[0] == 2
    assert model.bundle_identity.species == "demo"
    assert model.bundle_identity.genes == bundle.genes
    assert not torch.equal(model.Mcr, torch.zeros_like(model.Mcr))
    original = model.Mcr.clone()
    bundle.stoich.vals.setflags(write=True)
    bundle.stoich.vals[0, 0] = 99
    assert torch.equal(model.Mcr, original)
    assert model.hparams["bundle_metadata"]["content_id"] == "a" * 64


def test_bundle_training_options_are_opt_in_and_tensor_overrides_are_rejected() -> None:
    model = FlexModule.from_bundle(make_bundle())
    assert model.gnn.needs_fluxes is False
    assert model.hparams["nlayers"] == 1
    assert not model.flx_project
    for name in (
        "eid_g2r",
        "eid_r2r",
        "Mcr",
        "Mmg",
        "Mmr",
        "gen_emb",
        "rea_emb",
        "NSP",
        "f_disc_orig",
        "bundle_metadata",
    ):
        with pytest.raises(ValueError):
            FlexModule.from_bundle(make_bundle(), **{name: torch.zeros(1)})


@pytest.mark.parametrize("as_tensor", [False, True])
def test_align_expression_reorders_copies_and_accepts_extra_genes(
    as_tensor: bool,
) -> None:
    model = FlexModule.from_bundle(make_bundle())
    values = np.asarray([[30, 10, 40, 20, 999]], dtype=np.float64)
    source = torch.tensor(values) if as_tensor else values
    aligned = model.align_expression(source, ["g2", "g0", "g3", "g1", "extra"])
    assert aligned.dtype == torch.float32
    assert aligned.device == model.Mmg.device
    torch.testing.assert_close(
        aligned, torch.tensor([[10, 20, 30, 40.0]]), rtol=0, atol=0
    )
    aligned[0, 0] = -1
    assert values[0, 1] == 10


def test_align_expression_accepts_reversed_numpy_views() -> None:
    model = FlexModule.from_bundle(make_bundle())
    values = np.asarray([[1.0, 2.0, 3.0, 4.0]])
    actual = model.align_expression(values[:, ::-1], ["g3", "g2", "g1", "g0"])
    torch.testing.assert_close(
        actual, torch.tensor(values, dtype=torch.float32), rtol=0, atol=0
    )


@pytest.mark.parametrize("as_tensor", [False, True])
def test_align_expression_zero_fills_partial_columns(as_tensor: bool) -> None:
    model = FlexModule.from_bundle(make_bundle())
    values = np.asarray([[30, 999, 10], [3, 888, 1]], dtype=np.float64)
    original = values.copy()
    source = torch.tensor(values) if as_tensor else values
    with pytest.raises(flexModel.ExpressionAlignmentError):
        model.align_expression(source, ["g3", "extra", "g0"])
    aligned = model.align_expression(
        source, ["g3", "extra", "g0"], missing_genes="zero"
    )
    torch.testing.assert_close(
        aligned,
        torch.tensor([[10, 0, 0, 30], [1, 0, 0, 3]], dtype=torch.float32),
        rtol=0,
        atol=0,
    )
    np.testing.assert_array_equal(values, original)


def test_align_expression_zero_mode_preserves_gradient_for_present_values() -> None:
    model = FlexModule.from_bundle(make_bundle())
    values = torch.tensor([[2.0, 7.0]], requires_grad=True)
    aligned = model.align_expression(values, ["g2", "g0"], missing_genes="zero")
    aligned.sum().backward()
    torch.testing.assert_close(values.grad, torch.ones_like(values), rtol=0, atol=0)


@pytest.mark.parametrize("module_cls", [FlexModule, AlignedFlexModule])
def test_zero_filled_expression_matches_explicit_zeros_for_predictions_and_losses(
    module_cls: type[FlexModule],
) -> None:
    torch.manual_seed(41)
    bundle = make_bundle()
    partial_model = module_cls.from_bundle(
        bundle,
        nlayers=2,
        use_disc=True,
        use_rea_rea_gate=True,
        flx_project=True,
    ).eval()
    torch.manual_seed(41)
    full_model = module_cls.from_bundle(
        bundle,
        nlayers=2,
        use_disc=True,
        use_rea_rea_gate=True,
        flx_project=True,
    ).eval()
    partial = partial_model.align_expression(
        np.asarray([[3.0, 1.0], [6.0, 2.0], [9.0, 3.0]]),
        ["g3", "g0"],
        missing_genes="zero",
    )
    full = full_model.align_expression(
        np.asarray([[1.0, 0.0, 0.0, 3.0], [2.0, 0.0, 0.0, 6.0], [3.0, 0.0, 0.0, 9.0]]),
        bundle.genes,
    )
    with torch.no_grad():
        partial_prediction = partial_model(partial)
        full_prediction = full_model(full)
        partial_losses = partial_model.losses(partial, *partial_prediction)
        full_losses = full_model.losses(full, *full_prediction)
    for actual, expected in zip(partial_prediction, full_prediction):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(partial_losses, full_losses, rtol=0, atol=0)


def test_align_expression_zero_mode_rejects_invalid_mode_no_overlap_and_invalid_inputs() -> (
    None
):
    model = FlexModule.from_bundle(make_bundle())
    with pytest.raises(flexModel.ExpressionAlignmentError):
        model.align_expression(np.ones((1, 1)), ["g0"], missing_genes="pad")
    with pytest.raises(flexModel.ExpressionAlignmentError):
        model.align_expression(np.ones((1, 1)), ["extra"], missing_genes="zero")
    for values, ids in (
        (np.ones((1, 2)), ["g0", "g0"]),
        (np.asarray([[1, np.nan]]), ["g0", "g1"]),
        (np.asarray([[1, 2]]), ["g0"]),
    ):
        with pytest.raises(flexModel.ExpressionAlignmentError):
            model.align_expression(values, ids, missing_genes="zero")


def test_fit_expression_fill_scales_exclude_extras_and_fitted_alignment_is_frozen() -> (
    None
):
    model = FlexModule.from_bundle(make_bundle())
    training = np.asarray([[4.0, -1000.0, 2.0], [6.0, 1000.0, 8.0]])
    value = model.fit_expression_fill(
        training, ["g0", "extra", "g2"], scale="centered", percentile=25
    )
    assert value == pytest.approx(3.5)
    assert model.expression_fill.scale == "centered"
    assert model.expression_fill.percentile == 25
    assert model.expression_fill.value == value
    assert model.hparams["expression_fill"] == {
        "scale": "centered",
        "percentile": 25,
        "value": value,
    }
    with pytest.raises(FrozenInstanceError):
        model.expression_fill.value = 999
    aligned = model.align_expression(
        np.asarray([[-99.0, 20.0]]), ["g0", "g3"], missing_genes="fitted"
    )
    torch.testing.assert_close(
        aligned, torch.tensor([[-99.0, value, value, 20.0]]), rtol=0, atol=0
    )
    assert model.expression_fill.value == value
    with pytest.raises(flexModel.ExpressionAlignmentError):
        model.fit_expression_fill(training, ["g0", "extra", "g2"])
    with pytest.raises(flexModel.ExpressionAlignmentError):
        FlexModule.from_bundle(make_bundle()).align_expression(
            np.ones((1, 1)), ["g0"], missing_genes="fitted"
        )


def test_fit_expression_fill_nonnegative_policy_is_zero_and_validates_values() -> None:
    model = FlexModule.from_bundle(make_bundle())
    value = model.fit_expression_fill(
        np.asarray([[4.0, -1000.0, 2.0], [6.0, 1000.0, 8.0]]),
        ["g0", "extra", "g2"],
    )
    assert value == 0
    assert model.expression_fill.scale == "nonnegative"
    with pytest.raises(flexModel.ExpressionAlignmentError):
        model.align_expression(np.asarray([[-1.0]]), ["g0"], missing_genes="fitted")
    with pytest.raises(flexModel.ExpressionAlignmentError):
        model.fit_expression_fill(np.asarray([[-1.0]]), ["g0"])


@pytest.mark.parametrize("as_tensor", [False, True])
@pytest.mark.parametrize(("percentile", "expected"), [(None, -4.0), (1, -3.94)])
def test_fit_expression_fill_centered_uses_negative_observed_minimum_or_percentile(
    as_tensor: bool, percentile: float | None, expected: float
) -> None:
    model = FlexModule.from_bundle(make_bundle())
    values = np.asarray([[-4.0, -1000.0, -2.0], [2.0, 1000.0, 6.0]])
    source = torch.tensor(values) if as_tensor else values
    value = model.fit_expression_fill(
        source, ["g0", "extra", "g2"], scale="centered", percentile=percentile
    )
    assert value == pytest.approx(expected)


@pytest.mark.parametrize("module_cls", [FlexModule, AlignedFlexModule])
def test_fitted_expression_fill_checkpoint_matches_explicit_fill_predictions_and_losses(
    tmp_path: Path, module_cls: type[FlexModule]
) -> None:
    options = (
        {"z_dim": 4, "noise_std": 0.23, "alpha_align": 0.7}
        if module_cls is AlignedFlexModule
        else {}
    )
    torch.manual_seed(47)
    model = module_cls.from_bundle(
        make_bundle(),
        nlayers=2,
        use_disc=True,
        use_rea_rea_gate=True,
        flx_project=True,
        **options,
    ).eval()
    fit = np.asarray([[4.0, 2.0], [6.0, 8.0]])
    model.fit_expression_fill(fit, ["g0", "g2"], scale="centered")
    path = tmp_path / "fitted_fill.ckpt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": L.__version__,
        },
        path,
    )
    restored = module_cls.load_from_bundle_checkpoint(path, make_bundle()).eval()
    assert restored.expression_fill == model.expression_fill
    partial_values = np.asarray([[3.0, 1.0], [6.0, 2.0], [9.0, 3.0]])
    partial = restored.align_expression(
        partial_values, ["g3", "g0"], missing_genes="fitted"
    )
    fill = model.expression_fill.value
    complete = restored.align_expression(
        np.asarray(
            [[1.0, fill, fill, 3.0], [2.0, fill, fill, 6.0], [3.0, fill, fill, 9.0]]
        ),
        ["g0", "g1", "g2", "g3"],
    )
    with torch.no_grad():
        actual_prediction = restored(partial)
        expected_prediction = restored(complete)
        actual_losses = restored.losses(partial, *actual_prediction)
        expected_losses = restored.losses(complete, *expected_prediction)
    for actual, expected in zip(actual_prediction, expected_prediction):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_losses, expected_losses, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("values", "ids", "kwargs"),
    [
        (np.empty((0, 1)), ["g0"], {}),
        (np.ones((1, 1)), ["extra"], {}),
        (np.ones((1, 2)), ["g0", "g0"], {}),
        (np.asarray([[np.nan]]), ["g0"], {}),
        (np.asarray([[-1.0]]), ["g0"], {}),
    ],
)
def test_fit_expression_fill_rejects_invalid_training_data(
    values: np.ndarray, ids: list[str], kwargs: dict[str, object]
) -> None:
    model = FlexModule.from_bundle(make_bundle())
    with pytest.raises(flexModel.ExpressionAlignmentError):
        model.fit_expression_fill(values, ids, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scale": "unknown"},
        {"scale": "nonnegative", "percentile": 1},
        {"scale": "centered", "percentile": -1},
        {"scale": "centered", "percentile": 101},
    ],
)
def test_fit_expression_fill_rejects_invalid_scale_and_percentile(
    kwargs: dict[str, object],
) -> None:
    model = FlexModule.from_bundle(make_bundle())
    with pytest.raises(flexModel.ExpressionAlignmentError):
        model.fit_expression_fill(np.ones((1, 1)), ["g0"], **kwargs)


def test_from_bundle_rejects_malformed_expression_fill_policy() -> None:
    with pytest.raises(flexModel.ExpressionAlignmentError):
        FlexModule.from_bundle(
            make_bundle(), expression_fill={"scale": "centered", "value": "bad"}
        )


@pytest.mark.parametrize(
    ("values", "ids"),
    [
        (np.ones((1, 2)), ["g0", "g0"]),
        (np.ones((1, 2)), ["g0", "missing"]),
        (np.ones((1, 1)), ["g0"]),
        (np.ones((2, 3)), ["g0", "g1", "g2", "g3"]),
        (np.asarray([[1, 2, 3, np.nan]]), ["g0", "g1", "g2", "g3"]),
        (np.asarray([[1, 2, 3, np.inf]]), ["g0", "g1", "g2", "g3"]),
        (np.asarray([["1", "2", "3", "4"]]), ["g0", "g1", "g2", "g3"]),
        (torch.ones((2, 4), dtype=torch.bool), ["g0", "g1", "g2", "g3"]),
    ],
)
def test_align_expression_rejects_invalid_inputs(
    values: object, ids: list[str]
) -> None:
    model = FlexModule.from_bundle(make_bundle())
    with pytest.raises(flexModel.ExpressionAlignmentError):
        model.align_expression(values, ids)


def test_bundle_preserves_distinct_discordance_attributes_and_expands_edges() -> None:
    model = FlexModule.from_bundle(make_bundle(), use_disc=True, nlayers=2)
    conv = model.gnn.convs[0].convs[("R", "to", "R")]
    torch.testing.assert_close(conv.f_disc_orig, torch.tensor([0.2, 0.8, 0.2, 0.8]))
    edge = model.eid_r2r
    assert edge.shape == (2, 4)
    torch.testing.assert_close(
        edge, torch.tensor([[0, 1, 1, 2], [1, 2, 0, 1]]), rtol=0, atol=0
    )
    assert not torch.any(edge[0] == edge[1])


def test_projection_uses_labelled_basis_and_normalizes_after_projection() -> None:
    bundle = make_bundle()
    basis = np.asarray([[0.6, 0.0, 0.8], [0.0, 1.0, 0.0]])
    bundle.projection.vals = _array(basis)
    bundle.projection.row_ids = _array((0, 1))
    model = FlexModule.from_bundle(bundle, flx_project=True)
    assert model.Pi.shape == (2, 3)
    torch.testing.assert_close(
        model.Pi, torch.tensor(basis, dtype=torch.float32), rtol=0, atol=0
    )
    raw, projected = model(torch.ones(2, 4))
    unnormalized = raw @ model.Pi.T @ model.Pi
    expected = unnormalized / (unnormalized.abs().sum(dim=1, keepdim=True) + 1e-7)
    expected = expected * (model.Mmr.shape[1] ** 0.5)
    torch.testing.assert_close(projected, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("module_cls", [FlexModule, AlignedFlexModule])
def test_bundle_models_train_with_disc_and_gate(module_cls: type[FlexModule]) -> None:
    torch.manual_seed(19)
    model = module_cls.from_bundle(
        make_bundle(), use_disc=True, use_rea_rea_gate=True, nlayers=2
    )
    expression = model.align_expression(
        np.random.default_rng(3).normal(size=(6, 4)), ["g0", "g1", "g2", "g3"]
    )
    prediction, projected = model(expression)
    assert projected is None
    losses = model.losses(expression, prediction, projected)
    loss = losses.mean()
    loss.backward()
    assert torch.isfinite(loss)
    for conv in model.gnn.convs:
        disc_weight = conv.convs[("R", "to", "R")].lin_disc.weight
        assert disc_weight.grad is not None
        assert torch.isfinite(disc_weight.grad).all()
        assert disc_weight.grad.abs().sum() > 0
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    initial = float(loss.detach())
    for _ in range(3):
        optimizer.zero_grad()
        prediction, projected = model(expression)
        model.losses(expression, prediction, projected).mean().backward()
        optimizer.step()
    with torch.no_grad():
        prediction, projected = model(expression)
        final = float(model.losses(expression, prediction, projected).mean())
    assert np.isfinite(final) and final < initial


@pytest.mark.parametrize("module_cls", [FlexModule, AlignedFlexModule])
def test_bundle_checkpoint_roundtrip_and_identity_validation(
    tmp_path: Path, module_cls: type[FlexModule]
) -> None:
    options = (
        {"z_dim": 4, "noise_std": 0.23, "alpha_align": 0.7}
        if module_cls is AlignedFlexModule
        else {}
    )
    model = module_cls.from_bundle(make_bundle(), nlayers=2, use_disc=True, **options)
    path = tmp_path / "bundle.ckpt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": L.__version__,
        },
        path,
    )
    restored = module_cls.load_from_bundle_checkpoint(path, make_bundle())
    assert restored.hparams["nlayers"] == 2
    assert restored.bundle_identity == model.bundle_identity
    for key, value in options.items():
        assert restored.hparams[key] == value
    model.eval()
    restored.eval()
    x = torch.ones(2, 4)
    with torch.no_grad():
        torch.testing.assert_close(restored(x)[0], model(x)[0], rtol=1e-6, atol=1e-7)

    changed = make_bundle()
    changed.content_id = "b" * 64
    with pytest.raises(flexModel.BundleIdentityError):
        module_cls.load_from_bundle_checkpoint(path, changed)

    changed_axis = make_bundle()
    changed_axis.genes = ("g1", "g0", "g2", "g3")
    changed_axis.emb_gen.row_ids = _array(changed_axis.genes)
    changed_axis.mod_gen.col_ids = _array(changed_axis.genes)
    changed_axis.gr_gen_rea.src_ids = _array(changed_axis.genes)
    with pytest.raises(flexModel.BundleIdentityError):
        module_cls.load_from_bundle_checkpoint(path, changed_axis)

    legacy_path = tmp_path / "missing_bundle_metadata.ckpt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyper_parameters": {
                key: value
                for key, value in dict(model.hparams).items()
                if key != "bundle_metadata"
            },
            "pytorch-lightning_version": L.__version__,
        },
        legacy_path,
    )
    with pytest.raises(flexModel.BundleIdentityError):
        module_cls.load_from_bundle_checkpoint(legacy_path, make_bundle())


def test_bundle_rejects_contracts_embedding_halves_axes_and_specs() -> None:
    bundle = make_bundle()
    bundle.manifest.schema_version = 2
    with pytest.raises(flexModel.BundleCompatibilityError):
        FlexModule.from_bundle(bundle)

    bundle = make_bundle()
    bundle.manifest.contracts["projection"] = "unknown"
    with pytest.raises(flexModel.BundleCompatibilityError):
        FlexModule.from_bundle(bundle)

    bundle = make_bundle()
    bundle.emb_rea.col_ids = _array(("input:0", "input:1", "input:2", "output:0"))
    with pytest.raises(flexModel.BundleCompatibilityError):
        FlexModule.from_bundle(bundle)

    bundle = make_bundle()
    bundle.mod_gen.col_ids = _array(("g1", "g0", "g2", "g3"))
    with pytest.raises(flexModel.BundleCompatibilityError):
        FlexModule.from_bundle(bundle)

    bundle = make_bundle()
    bundle.gr_rea_rea.directed = True
    with pytest.raises(flexModel.BundleCompatibilityError):
        FlexModule.from_bundle(bundle)

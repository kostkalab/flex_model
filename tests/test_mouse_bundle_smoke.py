"""Offline smoke tests using the optional released mouse species bundle.

Run with ``python -m pytest tests/test_mouse_bundle_smoke.py -q -s``.
No data is downloaded: unavailable packages or uncached bundles cause a skip.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from flexModel import AlignedFlexModule, FlexModule


@pytest.fixture(scope="module")
def mouse_bundle() -> Any:
    """Load the released mouse 0.3.0 bundle from the existing cache only."""
    fs = pytest.importorskip("flex_species")
    try:
        return fs.load_species("mouse", version="0.3.0", offline=True)
    except fs.CacheMissError as error:
        pytest.skip(f"Mouse 0.3.0 bundle is not cached: {error}")


@pytest.mark.species_bundle
@pytest.mark.parametrize("module_cls", [FlexModule, AlignedFlexModule])
def test_mouse_bundle_forward_with_synthetic_expression(
    mouse_bundle: Any, module_cls: type[FlexModule]
) -> None:
    """Real mouse embeddings and topology support both model forward passes."""
    torch.manual_seed(31)
    model = module_cls.from_bundle(
        mouse_bundle,
        nlayers=2,
        use_disc=True,
        use_rea_rea_gate=True,
        flx_project=True,
    ).eval()
    values = np.random.default_rng(31).lognormal(size=(3, len(mouse_bundle.genes)))
    expression = model.align_expression(values[:, ::-1], mouse_bundle.genes[::-1])
    with torch.inference_mode():
        raw, projected = model(expression)
    assert expression.shape == (3, len(mouse_bundle.genes))
    assert raw.shape == (3, len(mouse_bundle.reactions))
    assert projected is not None and projected.shape == raw.shape
    assert raw.dtype == projected.dtype == torch.float32
    assert torch.isfinite(raw).all()
    assert torch.isfinite(projected).all()
    assert raw.abs().sum() > 0
    assert projected.abs().sum() > 0
    print(
        f"{module_cls.__name__}: GE {tuple(expression.shape)} -> "
        f"raw {tuple(raw.shape)}, projected {tuple(projected.shape)}, all finite; "
        f"bundle {mouse_bundle.content_id}"
    )

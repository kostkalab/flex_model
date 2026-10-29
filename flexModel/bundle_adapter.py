"""Consumer compatibility and tensor conversion for validated species bundles.

Loading, integrity verification, and biological validation belong to
``flex_species``. This adapter uses only its public labelled views and performs
no downloads, migrations, embedding transformations, or projection fitting.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import isfinite
from numbers import Real
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch

if TYPE_CHECKING:
    from flex_species import FlexSpeciesBundle


SUPPORTED_BUNDLE_SPEC_VERSIONS = frozenset({3})
SUPPORTED_BUNDLE_CONTRACTS = {
    "reaction_graph": "binary_shared_compound_topology_v1",
    "currency_selection": "weighted_leiden_scan_v1",
    "projection": "svd_cumulative_energy_v1",
    "module_diffusion": "binary_heat_kernel_v1",
}
REQUIRED_BUNDLE_CAPABILITIES = frozenset(
    {
        "embeddings",
        "reaction-graph",
        "stoichiometry",
        "approximate-projection",
        "module-maps",
    }
)
BUNDLE_INPUT_NAMES = frozenset(
    {
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
        "_expected_bundle_metadata",
        "hparams_file",
        "_instantiator",
    }
)


class BundleCompatibilityError(ValueError):
    """The bundle cannot be consumed with the supported model contracts."""


class BundleIdentityError(ValueError):
    """Checkpoint metadata is invalid or identifies a different species bundle."""


class ExpressionAlignmentError(ValueError):
    """Expression columns cannot be aligned to the model's canonical genes."""


@dataclass(frozen=True)
class ExpressionFill:
    """A scalar missing-gene floor fitted once from training expression."""

    scale: Literal["nonnegative", "centered"]
    percentile: float | None
    value: float

    def __post_init__(self) -> None:
        if self.scale not in ("nonnegative", "centered"):
            raise ExpressionAlignmentError("scale must be 'nonnegative' or 'centered'")
        if (
            isinstance(self.value, bool)
            or not isinstance(self.value, Real)
            or not isfinite(self.value)
        ):
            raise ExpressionAlignmentError(
                "Expression fill value must be finite and real"
            )
        if self.percentile is not None:
            if (
                isinstance(self.percentile, bool)
                or not isinstance(self.percentile, Real)
                or not isfinite(self.percentile)
                or not 0 <= self.percentile <= 100
            ):
                raise ExpressionAlignmentError(
                    "percentile must be finite and in [0, 100]"
                )
            if self.scale != "centered":
                raise ExpressionAlignmentError(
                    "percentile applies only to centered expression"
                )
            object.__setattr__(self, "percentile", float(self.percentile))
        if self.scale == "nonnegative" and self.value != 0:
            raise ExpressionAlignmentError(
                "Nonnegative expression requires fill value zero"
            )
        if abs(self.value) > np.finfo(np.float32).max:
            raise ExpressionAlignmentError("Expression fill value overflows float32")
        object.__setattr__(self, "value", float(self.value))

    @classmethod
    def from_mapping(cls, metadata: Mapping[str, object]) -> ExpressionFill:
        """Restore a validated floor using only built-in checkpoint types."""
        if not isinstance(metadata, Mapping) or set(metadata) != {
            "scale",
            "percentile",
            "value",
        }:
            raise ExpressionAlignmentError(
                "expression_fill must contain scale, percentile and value"
            )
        return cls(**metadata)

    def to_mapping(self) -> dict[str, object]:
        """Return plain metadata without training expression or fitted tensors."""
        return asdict(self)


@dataclass(frozen=True)
class BundleIdentity:
    """Checkpoint-safe identity and canonical axes of the source bundle."""

    species: str
    version: str
    content_id: str
    genes: tuple[str, ...]
    reactions: tuple[str, ...]
    compounds: tuple[str, ...]
    modules: tuple[str, ...]

    @classmethod
    def from_mapping(cls, metadata: Mapping[str, object]) -> BundleIdentity:
        """Validate plain checkpoint metadata before using its entity axes."""
        if not isinstance(metadata, Mapping):
            raise BundleIdentityError("bundle_metadata must be a mapping")
        fields = {
            "species",
            "version",
            "content_id",
            "genes",
            "reactions",
            "compounds",
            "modules",
        }
        if set(metadata) != fields:
            raise BundleIdentityError("bundle_metadata has missing or unknown fields")
        for key in ("species", "version", "content_id"):
            if not isinstance(metadata[key], str) or not metadata[key]:
                raise BundleIdentityError(
                    f"bundle_metadata {key} must be a nonempty string"
                )
        content_id = metadata["content_id"]
        if len(content_id) != 64 or any(
            c not in "0123456789abcdef" for c in content_id
        ):
            raise BundleIdentityError("bundle content_id must be a SHA-256 hex digest")
        axes = {}
        for key in ("genes", "reactions", "compounds", "modules"):
            ids = metadata[key]
            if not isinstance(ids, (list, tuple)) or not ids:
                raise BundleIdentityError(
                    f"bundle_metadata {key} must be a nonempty ID sequence"
                )
            if any(not isinstance(item, str) or not item for item in ids):
                raise BundleIdentityError(
                    f"bundle_metadata {key} must contain string IDs"
                )
            if len(set(ids)) != len(ids):
                raise BundleIdentityError(
                    f"bundle_metadata {key} contains duplicate IDs"
                )
            axes[key] = tuple(ids)
        return cls(
            species=metadata["species"],
            version=metadata["version"],
            content_id=content_id,
            **axes,
        )

    def to_mapping(self) -> dict[str, object]:
        """Return built-in types suitable for weights-only checkpoint loading."""
        return asdict(self)


def reject_bundle_overrides(options: Mapping[str, object]) -> None:
    """Reject replacement of tensors or identity owned by the source bundle."""
    conflicts = BUNDLE_INPUT_NAMES.intersection(options)
    if conflicts:
        raise BundleCompatibilityError(
            "Cannot override bundle-owned inputs: " + ", ".join(sorted(conflicts))
        )


def _float_tensor(values: Any, shape: tuple[int, ...], name: str) -> torch.Tensor:
    """Copy a real finite bundle array into an independent float32 tensor."""
    array = np.asarray(values)
    if array.shape != shape:
        raise BundleCompatibilityError(
            f"{name} must have shape {shape}, got {array.shape}"
        )
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise BundleCompatibilityError(f"{name} must contain finite real values")
    tensor = torch.tensor(array.copy(), dtype=torch.float32)
    if not torch.isfinite(tensor).all():
        raise BundleCompatibilityError(f"{name} values overflow float32")
    return tensor


def _matrix(view: Any, rows: tuple, cols: tuple, name: str) -> torch.Tensor:
    """Check labelled axes before copying a matrix of shape (rows, columns)."""
    if tuple(view.row_ids) != rows or tuple(view.col_ids) != cols:
        raise BundleCompatibilityError(
            f"{name} axis identities do not match canonical order"
        )
    return _float_tensor(view.vals, (len(rows), len(cols)), name)


def _edges(
    graph: Any,
    src_type: str,
    dst_type: str,
    src_ids: tuple[str, ...],
    dst_ids: tuple[str, ...],
    directed: bool,
    name: str,
) -> torch.Tensor:
    """Copy indices of shape (2, edges) after checking direction and axes."""
    if (
        graph.src_type != src_type
        or graph.dst_type != dst_type
        or graph.directed != directed
        or tuple(graph.src_ids) != src_ids
        or tuple(graph.dst_ids) != dst_ids
    ):
        raise BundleCompatibilityError(
            f"{name} direction or canonical axes are incompatible"
        )
    indices = np.asarray(graph.edge_index)
    if indices.ndim != 2 or indices.shape[0] != 2 or indices.dtype.kind not in "iu":
        raise BundleCompatibilityError(
            f"{name} must have integer edge_index shape (2, E)"
        )
    if indices.size and (
        (indices < 0).any()
        or (indices[0] >= len(src_ids)).any()
        or (indices[1] >= len(dst_ids)).any()
    ):
        raise BundleCompatibilityError(
            f"{name} edge indices are outside canonical axes"
        )
    return torch.tensor(indices.copy(), dtype=torch.long)


def bundle_model_inputs(bundle: FlexSpeciesBundle) -> dict[str, object]:
    """Convert a validated schema-3 bundle into independent constructor inputs.

    Numerical matrices use float32; edges use int64. Reaction edges are expanded
    as [original, reverse] with attributes duplicated in that same order.
    The logical projection basis (coordinates, reactions) is copied as released.
    """
    manifest = bundle.manifest
    if (
        manifest.format != "flex-species-bundle"
        or type(manifest.schema_version) is not int
        or manifest.schema_version not in SUPPORTED_BUNDLE_SPEC_VERSIONS
    ):
        raise BundleCompatibilityError(
            "Only flex-species-bundle schema_version 3 is supported"
        )
    if dict(manifest.contracts or {}) != SUPPORTED_BUNDLE_CONTRACTS:
        raise BundleCompatibilityError("Unsupported or missing bundle contracts")
    if not REQUIRED_BUNDLE_CAPABILITIES <= set(manifest.capabilities or ()):
        raise BundleCompatibilityError("Missing required bundle capabilities")
    identity = BundleIdentity.from_mapping(
        {
            "species": bundle.species,
            "version": bundle.version,
            "content_id": bundle.content_id,
            **{
                name: getattr(bundle, name)
                for name in ("genes", "reactions", "compounds", "modules")
            },
        }
    )
    for name in ("genes", "reactions", "compounds", "modules"):
        if (manifest.dimensions or {}).get(name) != len(getattr(identity, name)):
            raise BundleCompatibilityError(
                f"Manifest {name} count does not match canonical IDs"
            )

    ge_cols = tuple(bundle.emb_gen.col_ids)
    re_cols = tuple(bundle.emb_rea.col_ids)
    half = len(re_cols) // 2
    expected_halves = tuple(
        f"{side}:{i}" for side in ("input", "output") for i in range(half)
    )
    if not half or re_cols != expected_halves:
        raise BundleCompatibilityError(
            "Reaction embedding halves must be equal input/output blocks"
        )
    if not ge_cols or ge_cols != tuple(range(len(ge_cols))):
        raise BundleCompatibilityError("Gene embedding coordinate axis is incompatible")

    inputs = {
        "gen_emb": _matrix(bundle.emb_gen, identity.genes, ge_cols, "emb_gen"),
        "rea_emb": _matrix(bundle.emb_rea, identity.reactions, re_cols, "emb_rea"),
        "Mcr": _matrix(bundle.stoich, identity.compounds, identity.reactions, "stoich"),
        "Mmg": _matrix(bundle.mod_gen, identity.modules, identity.genes, "mod_gen"),
        "Mmr": _matrix(bundle.mod_rea, identity.modules, identity.reactions, "mod_rea"),
        "NSP": _matrix(
            bundle.projection,
            tuple(range(len(bundle.projection.row_ids))),
            identity.reactions,
            "projection",
        ),
        "eid_g2r": _edges(
            bundle.gr_gen_rea,
            "gene",
            "reaction",
            identity.genes,
            identity.reactions,
            True,
            "gr_gen_rea",
        ),
    }
    edges = _edges(
        bundle.gr_rea_rea,
        "reaction",
        "reaction",
        identity.reactions,
        identity.reactions,
        False,
        "gr_rea_rea",
    )
    if (edges[0] == edges[1]).any():
        raise BundleCompatibilityError(
            "Reaction bundle edges must not contain self-loops"
        )
    pairs = edges.t().tolist()
    if len({tuple(sorted(pair)) for pair in pairs}) != len(pairs):
        raise BundleCompatibilityError(
            "Reaction bundle edges must be unique undirected pairs"
        )
    fractions = bundle.gr_rea_rea.edge_attrs.get("f_disc")
    if fractions is None:
        raise BundleCompatibilityError("Reaction graph lacks f_disc")
    edge_ids = tuple(
        (identity.reactions[src], identity.reactions[dst]) for src, dst in pairs
    )
    if tuple(fractions.ids) != edge_ids:
        raise BundleCompatibilityError("f_disc IDs do not match reaction edge order")
    values = _float_tensor(fractions.vals, (edges.shape[1],), "f_disc")
    if ((values < 0) | (values > 1)).any():
        raise BundleCompatibilityError("f_disc must be in [0, 1]")
    inputs.update(
        eid_r2r=torch.cat((edges, edges.flip(0)), dim=1),
        f_disc_orig=torch.cat((values, values)),
        bundle_metadata=identity.to_mapping(),
    )
    return inputs


def _expression_inputs(
    values: torch.Tensor | np.ndarray,
    gene_ids: Sequence[str],
    canonical_genes: tuple[str, ...],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Validate (cells, input genes) values and unique, overlapping string IDs."""
    if isinstance(gene_ids, (str, bytes)):
        raise ExpressionAlignmentError("gene_ids must be a sequence of string IDs")
    ids = tuple(gene_ids)
    if any(not isinstance(item, str) or not item for item in ids):
        raise ExpressionAlignmentError("gene_ids must contain nonempty string IDs")
    if len(set(ids)) != len(ids):
        raise ExpressionAlignmentError("Expression gene IDs contain duplicates")
    if not isinstance(values, (torch.Tensor, np.ndarray)):
        raise ExpressionAlignmentError(
            "Expression must be a Torch tensor or NumPy array"
        )
    if values.ndim != 2 or values.shape[1] != len(ids):
        raise ExpressionAlignmentError(
            "Expression must have shape (cells, len(gene_ids))"
        )
    columns = {identifier: i for i, identifier in enumerate(ids)}
    if not any(identifier in columns for identifier in canonical_genes):
        raise ExpressionAlignmentError(
            "Expression gene IDs have no overlap with the model's canonical genes"
        )
    if isinstance(values, np.ndarray):
        if values.dtype.kind not in "fiu":
            raise ExpressionAlignmentError("Expression values must be real numbers")
        tensor = torch.tensor(values.copy(), dtype=torch.float32, device=device)
    else:
        if (
            values.is_complex()
            or values.dtype == torch.bool
            or values.layout != torch.strided
        ):
            raise ExpressionAlignmentError("Expression must be a dense real tensor")
        tensor = values.to(device=device, dtype=torch.float32)
    if not torch.isfinite(tensor).all():
        raise ExpressionAlignmentError(
            "Expression contains nonfinite or float32-overflow values"
        )
    return tensor, columns


def fit_expression_fill(
    values: torch.Tensor | np.ndarray,
    gene_ids: Sequence[str],
    canonical_genes: tuple[str, ...],
    *,
    scale: Literal["nonnegative", "centered"] = "nonnegative",
    percentile: float | None = None,
) -> ExpressionFill:
    """Fit one scalar floor from observed modeled genes in training data.

    Extra genes and absent-gene placeholders are excluded. Nonnegative data
    uses zero; centered data uses its observed minimum or requested percentile.
    Computation is on CPU and does not fit any other preprocessing.
    """
    ExpressionFill(scale, percentile, 0.0)
    tensor, columns = _expression_inputs(
        values, gene_ids, canonical_genes, torch.device("cpu")
    )
    if tensor.shape[0] == 0:
        raise ExpressionAlignmentError(
            "Cannot fit expression fill from empty training cells"
        )
    observed = tensor[:, [columns[item] for item in canonical_genes if item in columns]]
    minimum = float(observed.min().item())
    if scale == "nonnegative":
        if minimum < 0:
            raise ExpressionAlignmentError(
                "Nonnegative expression contains negative values"
            )
        return ExpressionFill(scale, None, 0.0)
    value = (
        minimum
        if percentile is None
        else float(np.percentile(observed.detach().numpy(), percentile))
    )
    return ExpressionFill(scale, percentile, value)


def align_expression(
    values: torch.Tensor | np.ndarray,
    gene_ids: Sequence[str],
    canonical_genes: tuple[str, ...],
    device: torch.device,
    *,
    missing_genes: Literal["error", "zero", "fitted"] = "error",
    expression_fill: ExpressionFill | None = None,
) -> torch.Tensor:
    """Return independent float32 (cells, canonical genes) values on device.

    Extra input genes are ignored. Missing genes raise by default, can be
    filled explicitly with zero, or use a frozen training-derived floor.
    No expression normalization or fitting occurs here.
    """
    if missing_genes not in ("error", "zero", "fitted"):
        raise ExpressionAlignmentError(
            "missing_genes must be 'error', 'zero' or 'fitted'"
        )
    if missing_genes == "fitted" and expression_fill is None:
        raise ExpressionAlignmentError(
            "Call fit_expression_fill on training data before using fitted fill"
        )
    tensor, columns = _expression_inputs(values, gene_ids, canonical_genes, device)
    missing = [item for item in canonical_genes if item not in columns]
    if missing and missing_genes == "error":
        raise ExpressionAlignmentError(
            f"Expression is missing required genes: {', '.join(missing[:10])}"
        )
    fill_value = 0.0
    if missing_genes == "fitted":
        fill_value = expression_fill.value
        observed_columns = [
            columns[item] for item in canonical_genes if item in columns
        ]
        if (
            expression_fill.scale == "nonnegative"
            and (tensor[:, observed_columns] < 0).any()
        ):
            raise ExpressionAlignmentError(
                "Nonnegative expression contains negative values"
            )
    if missing:
        # A shared fill column handles all absent genes while index_select
        # preserves gradients and independent storage for observed columns.
        tensor = torch.cat(
            (tensor, tensor.new_full((tensor.shape[0], 1), fill_value)), dim=1
        )
    indices = torch.tensor(
        [columns.get(item, len(columns)) for item in canonical_genes],
        dtype=torch.long,
        device=device,
    )
    return tensor.index_select(1, indices)

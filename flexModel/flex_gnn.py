"""Flexible GNN architectures for metabolic flux prediction.

Implements various GNN models combining gated graph convolutions with
antisymmetric flux prediction for metabolic network modeling.
"""

from typing import Callable

import torch
from torch_geometric.nn import HeteroConv, GCNConv
from torch.utils.checkpoint import checkpoint

from .conv_gatedGraphConv import ResGatedConv
from .halfSpaceAntiSymmetric import BiLinAntisymmetricFunc


EdgeType = tuple[str, str, str]
ConvBuilder = Callable[[int, int], torch.nn.Module]


class FluxHead(torch.nn.Module):
    """Readout head that maps reaction embeddings to scalar flux values."""

    def __init__(self, re_edim: int):
        super().__init__()
        self.re_edim = re_edim
        self.las = BiLinAntisymmetricFunc(re_edim // 2)

    def forward(self, reaction_reprs: torch.Tensor) -> torch.Tensor:
        tmp1 = reaction_reprs[:, :, : self.re_edim // 2]
        tmp2 = reaction_reprs[:, :, self.re_edim // 2 :]
        return self.las(tmp1, tmp2).squeeze()


def _default_conv_builders() -> dict[EdgeType, ConvBuilder]:
    """Default conv builders matching previous FlexGNN behavior."""

    return {
        ("G", "to", "R"): lambda ge_edim, re_edim: ResGatedConv(
            in_channels=(ge_edim, re_edim),
            out_channels=re_edim,
        ),
        ("R", "to", "R"): lambda ge_edim, re_edim: GCNConv(
            in_channels=re_edim,
            out_channels=re_edim,
        ),
    }


class FlexGNN(torch.nn.Module):
    """Generic heterogeneous GNN with pluggable per-edge convolution builders.

    Args:
        nr: Number of reaction nodes.
        re_edim: Reaction embedding dimension.
        ge_edim: Gene embedding dimension.
        nlayers: Number of hetero-conv layers.
        conv_builders: Mapping from edge type to callable that builds a conv module.
        use_layer_weights: If True, combine each layer output via learned softmax weights.
        use_layer_norm: If True, apply layer norm after each reaction update.
        use_checkpoint: If True and not using layer weights, checkpoint each conv layer.
    """

    def __init__(
        self,
        nr: int,
        re_edim: int = 1,
        ge_edim: int = 1,
        nlayers: int = 1,
        conv_builders: dict[EdgeType, ConvBuilder] | None = None,
        use_layer_weights: bool = False,
        use_layer_norm: bool = False,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.nr = nr
        self.nlayers = nlayers
        self.re_edim = re_edim
        self.ge_edim = ge_edim
        self.use_layer_weights = use_layer_weights
        self.use_layer_norm = use_layer_norm
        self.use_checkpoint = use_checkpoint

        conv_builders = conv_builders or _default_conv_builders()

        self.act = torch.nn.GELU()
        self.convs = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()

        for _ in range(nlayers):
            conv = HeteroConv(
                {
                    etype: builder(ge_edim, re_edim)
                    for etype, builder in conv_builders.items()
                }
            )
            self.convs.append(conv)
            if self.use_layer_norm:
                self.layer_norms.append(torch.nn.LayerNorm(re_edim))

        if self.use_layer_weights:
            self.layer_weights = torch.nn.Parameter(torch.zeros(nlayers + 1))
            self.layer_weights.data[-1] = 10.0

        self.flux_head = FluxHead(re_edim)
        # Backward-compatibility: keep old attribute used by existing code.
        self.las = self.flux_head.las

    def _apply_single_layer(
        self,
        conv: HeteroConv,
        x_dict: dict[str, torch.Tensor],
        ei_dict: dict[EdgeType, torch.Tensor],
        ge: torch.Tensor,
        layer_norm: torch.nn.LayerNorm | None,
    ) -> dict[str, torch.Tensor]:
        if self.use_checkpoint:
            def conv_step(x_g, x_r):
                out = conv({"G": x_g, "R": x_r}, ei_dict)
                return out["R"]

            x_r = checkpoint(conv_step, x_dict["G"], x_dict["R"], use_reentrant=False)
        else:
            x_r = conv(x_dict, ei_dict)["R"]

        x_dict["G"] = ge
        x_dict["R"] = self.act(x_r)
        if layer_norm is not None:
            x_dict["R"] = layer_norm(x_dict["R"])
        return x_dict

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        ei_dict: dict[EdgeType, torch.Tensor],
    ):
        assert "G" in x_dict, "x_dict must have key 'G'"
        assert "R" in x_dict, "x_dict must have key 'R'"

        x_dict_local = dict(x_dict) #- to keep x_dict unchanged
        ge = x_dict_local["G"].detach().clone()

        if self.use_layer_weights:
            lwts = torch.nn.functional.softmax(self.layer_weights, dim=0)
            reaction_reprs = lwts[0] * x_dict_local["R"]
            for idx, conv in enumerate(self.convs):
                ln = self.layer_norms[idx] if self.use_layer_norm else None
                # For layer-weight mode we keep the original no-checkpoint behavior.
                out = conv(x_dict_local, ei_dict)
                x_dict_local["G"] = ge
                x_dict_local["R"] = self.act(out["R"])
                if ln is not None:
                    x_dict_local["R"] = ln(x_dict_local["R"])
                reaction_reprs = reaction_reprs + lwts[idx + 1] * x_dict_local["R"]
        else:
            for idx, conv in enumerate(self.convs):
                ln = self.layer_norms[idx] if self.use_layer_norm else None
                x_dict_local = self._apply_single_layer(conv, x_dict_local, ei_dict, ge, ln)
            reaction_reprs = x_dict_local["R"]

        flx = self.flux_head(reaction_reprs)
        return flx


class FlexGNN_GCNConv_GGConv(torch.nn.Module):
    """Flexible GNN with Gated Graph Convolution layers.
    
    Uses ResGatedConv for gene-to-reaction edges and GCNConv for
    reaction-to-reaction edges, with antisymmetric flux prediction.
    """

    def __init__(self, nr: int, re_edim: int = 1, ge_edim: int = 1, nlayers: int = 1):
        super().__init__()
        self.model = FlexGNN(
            nr=nr,
            re_edim=re_edim,
            ge_edim=ge_edim,
            nlayers=nlayers,
            conv_builders=_default_conv_builders(),
            use_layer_weights=False,
            use_layer_norm=False,
            use_checkpoint=True,
        )
        self.nr = self.model.nr
        self.nlayers = self.model.nlayers
        self.re_edim = self.model.re_edim
        self.ge_edim = self.model.ge_edim
        self.las = self.model.las

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        ei_dict: dict[EdgeType, torch.Tensor],
    ):
        return self.model(x_dict, ei_dict)


class FlexGNN_GCNConv_GGConv_LW(torch.nn.Module):
    """Flexible GNN with learnable layer weighting.
    
    Similar to FlexGNN_GCNConv_GGConv but combines representations from all
    layers using learned softmax weights for final flux prediction.
    """

    def __init__(self, nr: int, re_edim: int = 1, ge_edim: int = 1, nlayers: int = 1):
        super().__init__()
        self.model = FlexGNN(
            nr=nr,
            re_edim=re_edim,
            ge_edim=ge_edim,
            nlayers=nlayers,
            conv_builders=_default_conv_builders(),
            use_layer_weights=True,
            use_layer_norm=True,
            use_checkpoint=False,
        )
        self.nr = self.model.nr
        self.nlayers = self.model.nlayers
        self.re_edim = self.model.re_edim
        self.ge_edim = self.model.ge_edim
        self.las = self.model.las
        self.layer_weights = self.model.layer_weights

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        ei_dict: dict[EdgeType, torch.Tensor],
    ):
        return self.model(x_dict, ei_dict)


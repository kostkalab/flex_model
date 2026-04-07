"""Flexible GNN architectures for metabolic flux prediction.

Implements various GNN models combining gated graph convolutions with
antisymmetric flux prediction for metabolic network modeling.
"""

from typing import Callable

import torch
from torch_geometric.nn import HeteroConv, GCNConv
from torch.utils.checkpoint import checkpoint

from .conv_reaReaConv import ReaReaConv
from .conv_gatedGraphConv import ResGatedConv
from .halfSpaceAntiSymmetric import AntisymmetricFunc


EdgeType = tuple[str, str, str]
ConvBuilder = Callable[[int, int], torch.nn.Module]


class FluxHead(torch.nn.Module):
    """Readout head that maps reaction embeddings to scalar flux values."""

    def __init__(self, re_edim: int):
        super().__init__()
        self.re_edim = re_edim
        self.las = AntisymmetricFunc(d=re_edim // 2)

    def forward(self, reaction_reprs: torch.Tensor) -> torch.Tensor:
        tmp1 = reaction_reprs[:, :, : self.re_edim // 2]
        tmp2 = reaction_reprs[:, :, self.re_edim // 2 :]
        return self.las(tmp1, tmp2).squeeze(-1)


def _default_conv_builders() -> dict[EdgeType, ConvBuilder]:
    """Default conv builders matching previous FlexGNN behavior."""
    return {
        ("G", "to", "R"): lambda ge_edim, re_edim: ResGatedConv(
            in_channels=(ge_edim, re_edim),
            out_channels=re_edim,
        ),
        ("R", "to", "R"): lambda ge_edim, re_edim: ReaReaConv(
            in_channels=re_edim,
            out_channels=re_edim,
            use_disc=False,
        ),
    }


def _disc_conv_builders(f_disc_orig: torch.Tensor) -> dict[EdgeType, ConvBuilder]:
    """Conv builders with concordant/discordant blending on R→R edges."""
    return {
        ("G", "to", "R"): lambda ge_edim, re_edim: ResGatedConv(
            in_channels=(ge_edim, re_edim),
            out_channels=re_edim,
        ),
        ("R", "to", "R"): lambda ge_edim, re_edim: ReaReaConv(
            in_channels=re_edim,
            out_channels=re_edim,
            use_disc=True,
            f_disc_orig=f_disc_orig,
        ),
    }


class FlexGNN(torch.nn.Module):
    """Generic heterogeneous GNN with pluggable per-edge convolution builders.

    When any R→R conv uses disc mode (use_disc=True), f_disc co-evolves
    with reaction representations layer by layer:
        Layer 0: uses f_disc_orig (all-positive flux assumption)
        Layer k>0: flux_head(current reprs) → updated fluxes → updated f_disc
    This creates skip connections from every layer's flux estimates through
    f_disc into all subsequent layers' R→R message passing. Single pass,
    fully differentiable, no probe iterations needed.

    Args:
        nr: Number of reaction nodes.
        re_edim: Reaction embedding dimension.
        ge_edim: Gene embedding dimension.
        nlayers: Number of hetero-conv layers.
        conv_builders: Mapping from edge type to callable that builds a conv module.
        use_layer_weights: If True, combine each layer output via learned softmax weights.
        use_layer_norm: If True, apply layer norm after each reaction update.
        use_checkpoint: If True, checkpoint the full forward logic to save memory.
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

        # Store edge type keys (tuple form) for ei_dict reconstruction
        self._ei_keys = list(conv_builders.keys())

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
            # (nlayers + 1) weights: weight[0] for input embeddings (layer 0),
            # weight[k] for output of conv layer k. Initialized to heavily
            # favor the final layer (weight[-1] = 10 → softmax ≈ 1.0).
            self.layer_weights = torch.nn.Parameter(torch.zeros(nlayers + 1))
            self.layer_weights.data[-1] = 10.0

        self.flux_head = FluxHead(re_edim)
        # Backward-compatibility: keep old attribute used by existing code.
        self.las = self.flux_head.las

        # Detect whether any R→R conv needs flux-dependent edge attributes
        # HeteroConv stores convs with "__" joined keys internally
        self.needs_fluxes = any(
            ("R", "to", "R") in conv.convs
            and isinstance(conv.convs[("R", "to", "R")], ReaReaConv)
            and conv.convs[("R", "to", "R")].use_disc
            for conv in self.convs
        )

    # ------------------------------------------------------------------
    # Internal: run conv layers with interleaved f_disc refinement
    # ------------------------------------------------------------------

    def _run_layers(
        self,
        x_r: torch.Tensor,
        x_g: torch.Tensor,
        ei_dict: dict[EdgeType, torch.Tensor],
    ) -> torch.Tensor:
        """Run all conv layers, refining f_disc at each layer from intermediate fluxes.

        Takes tensors directly (not dicts) to avoid in-place mutation issues
        with checkpointing. x_g is passed to HeteroConv each layer but never
        modified — gene embeddings stay grounded.

        When needs_fluxes=True, the flow per layer is:
            1. Compute current_fluxes from current reaction reprs via flux_head
               (layer 0: uses all-positive → f_disc = f_disc_orig)
            2. G→R conv + R→R conv (R→R uses current_fluxes internally)
            3. Activation + optional layer norm
        f_disc co-evolves with representations — no separate probe pass needed.

        When needs_fluxes=False, just runs conv + activation at each layer.

        Args:
            x_r: Reaction features, shape (batch, nr, re_edim).
            x_g: Gene features, shape (batch, ng, ge_edim). Not modified.
            ei_dict: Edge index dict with tuple keys.

        Returns:
            Reaction representations, shape (batch, nr, re_edim).
        """
        curr_x_r = x_r

        if self.use_layer_weights:
            lwts = torch.nn.functional.softmax(self.layer_weights, dim=0)
            reaction_reprs = lwts[0] * curr_x_r
        else:
            reaction_reprs = None

        for idx, conv in enumerate(self.convs):
            if self.needs_fluxes:
                # Layer 0: no flux estimates yet → all-positive → f_disc = f_disc_orig
                # Layer 1+: use flux_head on current representations
                if idx == 0:
                    current_fluxes = torch.ones(
                        curr_x_r.shape[0], self.nr, device=curr_x_r.device,
                    )
                else:
                    current_fluxes = self.flux_head(curr_x_r)

                out = conv(
                    {"G": x_g, "R": curr_x_r}, ei_dict,
                    current_fluxes_dict={("R", "to", "R"): current_fluxes},
                )
            else:
                out = conv({"G": x_g, "R": curr_x_r}, ei_dict)

            curr_x_r = self.act(out["R"])
            if self.use_layer_norm:
                curr_x_r = self.layer_norms[idx](curr_x_r)

            if self.use_layer_weights:
                reaction_reprs = reaction_reprs + lwts[idx + 1] * curr_x_r

        return reaction_reprs if self.use_layer_weights else curr_x_r

    # ------------------------------------------------------------------
    # Internal: full logic in one differentiable block (checkpoint-friendly)
    # ------------------------------------------------------------------

    def _run_layers_ckpt(self, x_r: torch.Tensor, x_g: torch.Tensor, *ei_values) -> torch.Tensor:
        """Checkpoint-compatible wrapper for _run_layers.

        Accepts flattened ei_values (tensors only) since checkpoint requires
        tensor args. Reconstructs ei_dict internally.
        """
        ei_dict = dict(zip(self._ei_keys, ei_values))
        return self._run_layers(x_r, x_g, ei_dict)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        ei_dict: dict[EdgeType, torch.Tensor],
    ):
        assert "G" in x_dict, "x_dict must have key 'G'"
        assert "R" in x_dict, "x_dict must have key 'R'"

        x_r = x_dict["R"]
        x_g = x_dict["G"]  # not detached: upstream controls gradient flow

        if self.training and self.use_checkpoint:
            # Checkpoint only _run_layers (the heavy part).
            # flux_head is cheap and stays outside — no recomputation needed.
            ei_values = [ei_dict[k] for k in self._ei_keys]
            final_reprs = checkpoint(
                self._run_layers_ckpt, x_r, x_g, *ei_values,
                use_reentrant=False,
            )
        else:
            final_reprs = self._run_layers(x_r, x_g, ei_dict)

        return self.flux_head(final_reprs)


# ------------------------------------------------------------------
# Convenience wrappers
# ------------------------------------------------------------------

class FlexGNN_GCNConv_GGConv(torch.nn.Module):
    """Flexible GNN with ReaReaConv (GCNConv-equivalent, no disc) + ResGatedConv."""

    def __init__(self, nr: int, re_edim: int = 1, ge_edim: int = 1, nlayers: int = 1):
        super().__init__()
        self.model = FlexGNN(
            nr=nr, re_edim=re_edim, ge_edim=ge_edim, nlayers=nlayers,
            conv_builders=_default_conv_builders(),
            use_layer_weights=False, use_layer_norm=False, use_checkpoint=True,
        )
        self.nr = self.model.nr
        self.nlayers = self.model.nlayers
        self.re_edim = self.model.re_edim
        self.ge_edim = self.model.ge_edim
        self.las = self.model.las

    def forward(self, x_dict, ei_dict):
        return self.model(x_dict, ei_dict)


class FlexGNN_GCNConv_GGConv_LW(torch.nn.Module):
    """Flexible GNN with learnable layer weighting (no disc)."""

    def __init__(self, nr: int, re_edim: int = 1, ge_edim: int = 1, nlayers: int = 1):
        super().__init__()
        self.model = FlexGNN(
            nr=nr, re_edim=re_edim, ge_edim=ge_edim, nlayers=nlayers,
            conv_builders=_default_conv_builders(),
            use_layer_weights=True, use_layer_norm=True, use_checkpoint=False,
        )
        self.nr = self.model.nr
        self.nlayers = self.model.nlayers
        self.re_edim = self.model.re_edim
        self.ge_edim = self.model.ge_edim
        self.las = self.model.las
        self.layer_weights = self.model.layer_weights

    def forward(self, x_dict, ei_dict):
        return self.model(x_dict, ei_dict)


class FlexGNN_Disc_GGConv(torch.nn.Module):
    """Flexible GNN with concordant/discordant ReaReaConv + ResGatedConv.

    f_disc co-evolves with representations layer by layer:
        layer 0: f_disc_orig (all-positive assumption)
        layer k: flux_head(reprs) -> updated f_disc -> R-R conv
    Single pass, fully differentiable.
    """

    def __init__(
        self, nr: int, f_disc_orig: torch.Tensor,
        re_edim: int = 1, ge_edim: int = 1, nlayers: int = 1,
    ):
        super().__init__()
        self.model = FlexGNN(
            nr=nr, re_edim=re_edim, ge_edim=ge_edim, nlayers=nlayers,
            conv_builders=_disc_conv_builders(f_disc_orig),
            use_layer_weights=False, use_layer_norm=False, use_checkpoint=True,
        )
        self.nr = self.model.nr
        self.nlayers = self.model.nlayers
        self.re_edim = self.model.re_edim
        self.ge_edim = self.model.ge_edim
        self.las = self.model.las

    def forward(self, x_dict, ei_dict):
        return self.model(x_dict, ei_dict)


class FlexGNN_Disc_GGConv_LW(torch.nn.Module):
    """Flexible GNN with concordant/discordant ReaReaConv + learnable layer weights.

    f_disc co-evolves with representations layer by layer.
    Single pass, fully differentiable.
    """

    def __init__(
        self, nr: int, f_disc_orig: torch.Tensor,
        re_edim: int = 1, ge_edim: int = 1, nlayers: int = 1,
    ):
        super().__init__()
        self.model = FlexGNN(
            nr=nr, re_edim=re_edim, ge_edim=ge_edim, nlayers=nlayers,
            conv_builders=_disc_conv_builders(f_disc_orig),
            use_layer_weights=True, use_layer_norm=True, use_checkpoint=False,
        )
        self.nr = self.model.nr
        self.nlayers = self.model.nlayers
        self.re_edim = self.model.re_edim
        self.ge_edim = self.model.ge_edim
        self.las = self.model.las
        self.layer_weights = self.model.layer_weights

    def forward(self, x_dict, ei_dict):
        return self.model(x_dict, ei_dict)
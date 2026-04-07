"""Reaction-Reaction Convolution with optional concordant/discordant blending.

f_disc_ji is a per-edge attribute quantifying the discordant strength of the
edge from reaction j to reaction i. It ranges from 0 (purely concordant: the
compound is relayed from j to i or vice versa) to 1 (purely discordant: both
reactions act on the same side of the shared compound).

    message_j→i = norm_ji * [(1 - f_disc_ji) * A_conc @ x_j  +  f_disc_ji * A_disc @ x_j]

Messages are normalized using symmetric degree normalization (D^{-1/2} A D^{-1/2})
with added self-loops, matching GCNConv behavior.

When use_disc=False, the convolution uses a single linear transform (no blending),
recovering GCNConv behavior (symmetric normalization + self-loop + bias).

When use_disc=True, f_disc is computed internally from current_fluxes passed as
a forward kwarg. Each cell may have different flux directions and therefore
different concordant/discordant assignments. The forward signature is
(x, edge_index, current_fluxes=...) which HeteroConv routes via:

    out = hetero_conv(x_dict, ei_dict,
                      current_fluxes_dict={("R", "to", "R"): current_fluxes})

Input x is always batched: shape (batch, n_nodes, channels).
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros


def compute_dynamic_f_disc(
    f_disc_orig: Tensor,
    fluxes: Tensor,
    edge_index: Tensor,
    temperature: float = 1.0,
) -> Tensor:
    """Compute per-cell discordant strength from current flux predictions.

    f_disc_orig is computed from the static metabolic graph under the assumption
    that all reactions run in the forward direction (all fluxes positive). Under
    this assumption, the concordant/discordant label of each edge is fully
    determined by the graph topology (R->C->R = concordant, R->C<-R = discordant,
    R<-C->R = discordant).

    When predicted fluxes deviate from all-positive, some edges flip their
    concordant/discordant interpretation. The trick: if the flux signs of the
    two reactions connected by an edge agree, the original topology still holds
    (keep f_disc_orig). If they disagree, one reaction has reversed, flipping
    the edge type (use 1 - f_disc_orig). The tanh provides a smooth,
    differentiable interpolation between these two cases.

    For each R-R edge (j, i) and each cell:
        sign_prod = tanh(flux_j * flux_i / temperature)
        keep = (1 + sign_prod) / 2
        f_disc_ji = keep * f_disc_orig_ji + (1 - keep) * (1 - f_disc_orig_ji)

    When fluxes agree in sign -> f_disc ~ f_disc_orig (topology holds).
    When fluxes disagree      -> f_disc ~ 1 - f_disc_orig (conc <-> disc flip).
    When either flux near zero -> f_disc ~ 0.5 (agnostic, equal blend).

    Differentiable w.r.t. fluxes.

    Args:
        f_disc_orig: Static discordant strength (all-positive-flux assumption),
            shape (n_edges,).
        fluxes: Predicted fluxes, shape (batch, n_reactions).
        edge_index: R-R edges, shape (2, n_edges).
        temperature: Controls sharpness of the sign transition.

    Returns:
        f_disc: Dynamic discordant strength, shape (batch, n_edges).
    """
    src, tgt = edge_index[0], edge_index[1]
    flux_src = fluxes[:, src]  # (batch, n_edges)
    flux_tgt = fluxes[:, tgt]  # (batch, n_edges)
    sign_prod = torch.tanh(flux_src * flux_tgt / temperature)
    keep = (1.0 + sign_prod) / 2.0
    fdo = f_disc_orig.unsqueeze(0)  # (1, n_edges)
    return keep * fdo + (1.0 - keep) * (1.0 - fdo)


def _compute_gcn_norm(
    edge_index: Tensor,
    n_nodes: int,
    add_self_loops: bool = True,
) -> tuple[Tensor, Tensor, int]:
    """Compute GCNConv-style symmetric normalization D^{-1/2} A D^{-1/2}.

    When add_self_loops=True: strips existing self-loops, adds exactly one per
    node, then normalizes. Matches GCNConv(add_self_loops=True).

    When add_self_loops=False: leaves the edge_index untouched and normalizes
    as-is. Matches GCNConv(add_self_loops=False).

    Args:
        edge_index: (2, n_edges)
        n_nodes: number of nodes
        add_self_loops: whether to manage self-loops (GCNConv default: True)

    Returns:
        edge_index_out: (2, n_edges') possibly with self-loops managed
        norm: (n_edges',) normalization coefficient per edge
        n_orig: number of non-self-loop edges in edge_index_out
            (equals n_edges' when add_self_loops=False and input has no self-loops)
    """
    if add_self_loops:
        # Strip existing self-loops, then add exactly one per node
        src, tgt = edge_index[0], edge_index[1]
        non_self = src != tgt
        edge_index_out = edge_index[:, non_self]
        n_orig = int(non_self.sum().item())

        self_loops = torch.arange(n_nodes, device=edge_index.device)
        self_loop_ei = torch.stack([self_loops, self_loops])
        edge_index_out = torch.cat([edge_index_out, self_loop_ei], dim=1)
    else:
        # Leave as-is — caller is responsible for self-loop management
        edge_index_out = edge_index
        src, tgt = edge_index_out[0], edge_index_out[1]
        n_orig = int((src != tgt).sum().item())

    src, tgt = edge_index_out[0], edge_index_out[1]
    deg = torch.zeros(n_nodes, device=edge_index_out.device, dtype=torch.float)
    deg.scatter_add_(0, tgt, torch.ones(tgt.shape[0], device=edge_index_out.device))

    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0

    norm = deg_inv_sqrt[src] * deg_inv_sqrt[tgt]
    return edge_index_out, norm, n_orig


class ReaReaConv(torch.nn.Module):
    """Reaction-reaction convolution with optional concordant/discordant message blending.

    Without concordant/discordant blending (use_disc=False):
        Adds self-loops, computes symmetric normalization, applies linear
        transform, adds bias. Equivalent to GCNConv.
        forward(x, edge_index) -> updated x

    With concordant/discordant blending (use_disc=True):
        message_j->i = norm_ji * [(1 - f_disc_ji) * A_conc @ x_j + f_disc_ji * A_disc @ x_j]
        f_disc is computed internally from current_fluxes.
        forward(x, edge_index, current_fluxes=fluxes) -> updated x

    Both modes are HeteroConv compatible. For use_disc=True, pass fluxes via:
        hetero_conv(x_dict, ei_dict,
                    current_fluxes_dict={("R", "to", "R"): current_fluxes})

    Args:
        in_channels: Input feature dimension.
        out_channels: Output feature dimension.
        use_disc: If True, create separate A_conc and A_disc matrices.
            current_fluxes must then be passed in forward().
            If False, single A matrix, current_fluxes is ignored.
        f_disc_orig: Static discordant strength per edge, shape (n_edges,),
            computed from the metabolic graph under the all-positive-flux
            assumption. Required when use_disc=True. Stored as a buffer.
        temperature: Controls sharpness of the concordant/discordant transition
            in the tanh. Only used when use_disc=True.
        add_self_loops: If True, add self-loops before normalization (GCNConv default).
        bias: If True, add learnable bias.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_disc: bool = False,
        f_disc_orig: Tensor | None = None,
        temperature: float = 1.0,
        add_self_loops: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_disc = use_disc
        self.temperature = temperature
        self.add_self_loops = add_self_loops

        if use_disc:
            if f_disc_orig is None:
                raise ValueError(
                    "f_disc_orig must be provided when use_disc=True. "
                    "Compute it from the static metabolic graph (all fluxes positive)."
                )
            self.register_buffer("f_disc_orig", f_disc_orig)
            self.lin_conc = Linear(in_channels, out_channels, bias=False)
            self.lin_disc = Linear(in_channels, out_channels, bias=False)
        else:
            self.register_buffer("f_disc_orig", None)
            self.lin = Linear(in_channels, out_channels, bias=False)

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        # Normalization cache — registered as buffers so they move with .to(device)
        # Populated lazily on first forward call
        self.register_buffer("_cached_edge_index", None, persistent=False)
        self.register_buffer("_cached_norm", None, persistent=False)
        self.register_buffer("_hash_positions", None, persistent=False)
        self._cached_n_orig: int | None = None

        self.reset_parameters()

    def reset_parameters(self):
        if self.use_disc:
            self.lin_conc.reset_parameters()
            self.lin_disc.reset_parameters()
        else:
            self.lin.reset_parameters()
        if self.bias is not None:
            zeros(self.bias)

    def _get_norm(self, edge_index: Tensor, n_nodes: int) -> tuple[Tensor, Tensor, int]:
        """Get cached normalization, recompute if edge_index changed.

        All cached tensors are non-persistent buffers, so they move with the
        model on .to(device) / .cuda() without manual device management.
        """
        n_edges = edge_index.shape[1]

        # Allocate hash positions if needed (also a buffer, moves with model)
        if self._hash_positions is None or self._hash_positions.shape[0] < n_edges:
            self._hash_positions = (
                torch.arange(n_edges, device=edge_index.device, dtype=edge_index.dtype)
                + 1
            )

        pos = self._hash_positions[:n_edges]
        hash_src = int((edge_index[0] * pos).sum().item())
        hash_tgt = int((edge_index[1] * pos).sum().item())
        key = (n_edges, n_nodes, hash_src, hash_tgt)

        # Check cache
        if self._cached_norm is not None and getattr(self, "_cache_key", None) == key:
            return self._cached_edge_index, self._cached_norm, self._cached_n_orig

        # Recompute
        edge_index_out, norm, n_orig = _compute_gcn_norm(
            edge_index, n_nodes, self.add_self_loops
        )
        self._cached_edge_index = edge_index_out
        self._cached_norm = norm
        self._cached_n_orig = n_orig
        self._cache_key = key
        return edge_index_out, norm, n_orig

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        current_fluxes: Tensor | None = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            x: Node features, shape (batch, n_nodes, in_channels).
            edge_index: Shared topology, shape (2, n_edges).
            current_fluxes: Per-cell predicted fluxes, shape (batch, n_reactions).
                Required when use_disc=True (used to compute f_disc internally).
                Ignored when use_disc=False.
                Routed by HeteroConv via current_fluxes_dict.

        Returns:
            Updated node features, shape (batch, n_nodes, out_channels).

        Raises:
            ValueError: If use_disc=True and current_fluxes is None.
        """
        batch, n_nodes, _ = x.shape

        # Normalization (with self-loops removed/re-added, cached)
        edge_index_norm, norm, n_orig = self._get_norm(edge_index, n_nodes)
        src, tgt = edge_index_norm[0], edge_index_norm[1]
        n_edges = edge_index_norm.shape[1]

        if self.use_disc:
            if current_fluxes is None:
                raise ValueError(
                    "current_fluxes must be provided when use_disc=True. "
                    "Pass via forward(x, edge_index, current_fluxes=...) or "
                    "HeteroConv(current_fluxes_dict={('R','to','R'): fluxes})."
                )

            # Validate f_disc_orig matches the non-self-loop edge count
            if self.f_disc_orig.shape[0] != n_orig:
                raise ValueError(
                    f"f_disc_orig has {self.f_disc_orig.shape[0]} entries but "
                    f"edge_index has {n_orig} non-self-loop edges. These must match."
                )

            # Compute per-cell f_disc from fluxes on non-self-loop edges only
            # (self-loops were stripped; n_orig is the count of real edges)
            edge_index_orig = edge_index_norm[:, :n_orig]
            f_disc = compute_dynamic_f_disc(
                self.f_disc_orig, current_fluxes, edge_index_orig, self.temperature
            )  # (batch, n_orig)

            # Self-loops are concordant by definition (same reaction, same sign)
            # so f_disc = 0 for self-loops -> message = A_conc @ x_self
            n_self_loops = n_edges - n_orig
            self_loop_zeros = torch.zeros(
                batch, n_self_loops, device=f_disc.device, dtype=f_disc.dtype
            )
            f_disc_full = torch.cat(
                [f_disc, self_loop_zeros], dim=1
            )  # (batch, n_edges)

            x_conc = self.lin_conc(x)  # (batch, n_nodes, out_ch)
            x_disc = self.lin_disc(x)  # (batch, n_nodes, out_ch)
            out_ch = x_conc.shape[-1]

            x_conc_j = x_conc[:, src, :]  # (batch, n_edges, out_ch)
            x_disc_j = x_disc[:, src, :]  # (batch, n_edges, out_ch)

            fd = f_disc_full.unsqueeze(-1)  # (batch, n_edges, 1)
            messages = (1.0 - fd) * x_conc_j + fd * x_disc_j
        else:
            x_lin = self.lin(x)
            out_ch = x_lin.shape[-1]
            messages = x_lin[:, src, :]  # (batch, n_edges, out_ch)

        # Apply symmetric normalization
        messages = messages * norm[None, :, None]  # (1, n_edges, 1) broadcast

        # Scatter-add messages to target nodes
        tgt_idx = tgt[None, :, None].expand(batch, n_edges, out_ch)
        out = torch.zeros(batch, n_nodes, out_ch, device=x.device, dtype=x.dtype)
        out.scatter_add_(1, tgt_idx, messages)

        if self.bias is not None:
            out = out + self.bias

        return out

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_halfspace_init(
        cls, dim: int, f_disc_orig: Tensor, **kwargs
    ) -> "ReaReaConv":
        """Initialize with halfspace geometry encoding flux coupling physics.

        A_conc (concordant) = +swap [[0, I], [I, 0]]
            Cross-half interaction with positive sign: "your product feeds
            my substrate, reinforce me." Captures relay coupling.

        A_disc (discordant) = -I
            Same-half interaction with negative sign: "you compete for my
            substrate (or flood my product pool), suppress me."
            Captures competitive/shared-pool coupling.

        Interpolation: (1 - f_disc) * (+swap) + f_disc * (-I)
            f_disc=0: pure positive cross-half coupling
            f_disc=1: pure negative same-half coupling
            f_disc=0.5: partial cancellation (ambiguous edge)

        Args:
            dim: Embedding dimension (must be even).
            f_disc_orig: Static discordant strength, shape (n_edges,).
        """
        assert dim % 2 == 0
        conv = cls(dim, dim, use_disc=True, f_disc_orig=f_disc_orig, **kwargs)
        half = dim // 2

        with torch.no_grad():
            # A_conc = +swap (cross-half, positive coupling)
            swap = torch.zeros(dim, dim)
            swap[:half, half:] = torch.eye(half)
            swap[half:, :half] = torch.eye(half)
            conv.lin_conc.weight.data.copy_(swap)

            # A_disc = -I (same-half, negative coupling)
            conv.lin_disc.weight.data.copy_(-torch.eye(dim))

        return conv

    @classmethod
    def from_diagonal_scales(
        cls,
        dim: int,
        f_disc_orig: Tensor,
        s_conc: Tensor | None = None,
        s_disc: Tensor | None = None,
        **kwargs,
    ) -> "ReaReaConv":
        """Initialize from diagonal scale vectors with flux coupling signs.

        A_conc = +swap @ diag(s_conc)   (cross-half, positive coupling)
        A_disc = -diag(s_disc)          (same-half, negative coupling)

        Args:
            dim: Embedding dimension (must be even).
            f_disc_orig: Static discordant strength, shape (n_edges,).
            s_conc: Per-dim scales for concordant (default: ones).
            s_disc: Per-dim scales for discordant (default: ones).
        """
        assert dim % 2 == 0
        half = dim // 2
        conv = cls(dim, dim, use_disc=True, f_disc_orig=f_disc_orig, **kwargs)

        if s_conc is None:
            s_conc = torch.ones(dim)
        if s_disc is None:
            s_disc = torch.ones(dim)

        with torch.no_grad():
            # A_conc = +swap @ diag(s_conc)
            diag_conc = torch.diag(s_conc)
            swap = torch.zeros(dim, dim)
            swap[:half, half:] = torch.eye(half)
            swap[half:, :half] = torch.eye(half)
            conv.lin_conc.weight.data.copy_(swap @ diag_conc)

            # A_disc = -diag(s_disc)
            conv.lin_disc.weight.data.copy_(-torch.diag(s_disc))

        return conv

    # ------------------------------------------------------------------
    # Regularization and diagnostics
    # ------------------------------------------------------------------

    def offdiag_penalty(self) -> Tensor:
        """L1 penalty on off-diagonal elements of both weight matrices.

        Encourages sparsity: most cross-dimension interactions stay zero,
        only those needed for flux prediction survive.
        """
        if not self.use_disc:
            return torch.tensor(0.0, device=self.lin.weight.device)

        def _offdiag_l1(w: Tensor) -> Tensor:
            mask = 1.0 - torch.eye(w.shape[0], w.shape[1], device=w.device)
            return (w * mask).abs().sum()

        return _offdiag_l1(self.lin_conc.weight) + _offdiag_l1(self.lin_disc.weight)

    def structure_diagnostics(self) -> dict[str, float]:
        """How far have the matrices drifted from diagonal/swap structure."""
        if not self.use_disc:
            return {}

        def _offdiag_frac(w: Tensor) -> float:
            mask = 1.0 - torch.eye(w.shape[0], w.shape[1], device=w.device)
            return ((w * mask).norm() / (w.norm() + 1e-12)).item()

        w_c = self.lin_conc.weight.data
        w_d = self.lin_disc.weight.data
        cos = torch.nn.functional.cosine_similarity(
            w_c.flatten().unsqueeze(0),
            w_d.flatten().unsqueeze(0),
        ).item()

        return {
            "offdiag_conc_frac": _offdiag_frac(w_c),
            "offdiag_disc_frac": _offdiag_frac(w_d),
            "conc_disc_cosine": cos,
        }

"""Flexible GNN architectures for metabolic flux prediction.

Implements various GNN models combining gated graph convolutions with
antisymmetric flux prediction for metabolic network modeling.
"""

import torch
from torch_geometric.nn import HeteroConv, Linear, GCNConv
from torch.utils.checkpoint import checkpoint

from .conv_gatedGraphConv import ResGatedConv
from .halfSpaceAntiSymmetric import BiLinAntisymmetricFunc


class FlexGNN_GCNConv_GGConv(torch.nn.Module):
    """Flexible GNN with Gated Graph Convolution layers.
    
    Uses ResGatedConv for gene-to-reaction edges and GCNConv for
    reaction-to-reaction edges, with antisymmetric flux prediction.
    """

    def __init__(self, nr: int, re_edim: int = 1, ge_edim: int = 1, nlayers: int = 1):
        super().__init__()
        self.nr = nr
        self.nlayers = nlayers
        self.las = BiLinAntisymmetricFunc(re_edim // 2)
        self.act = torch.nn.GELU()
        self.convs = torch.nn.ModuleList()
        self.re_edim = re_edim
        self.ge_edim = ge_edim

        for _ in range(nlayers):
            conv = HeteroConv(
                {
                    ("G", "to", "R"): ResGatedConv(
                        in_channels=(ge_edim, re_edim),
                        out_channels=re_edim,
                    ),
                    ("R", "to", "R"): GCNConv(
                        in_channels=re_edim, out_channels=re_edim
                    ),
                }
            )
            self.convs.append(conv)

    def forward(
        self, x_dict: dict[str, torch.Tensor], ei_dict: dict[str, torch.Tensor]
    ):
        assert "G" in x_dict.keys(), "x_dict must have key 'G'"
        assert "R" in x_dict.keys(), "x_dict must have key 'R'"
        ge = x_dict["G"].detach().clone().to(device=x_dict["G"].device)
        for conv in self.convs:
            def conv_step(x_g, x_r):
                out = conv({"G": x_g, "R": x_r}, ei_dict)
                return out["R"]
            x_R = checkpoint(conv_step, x_dict["G"], x_dict["R"], use_reentrant=False)
            x_dict["G"] = ge  # - clamp gene expression
            x_dict["R"] = self.act(x_R)
        
        tmp1 = x_dict["R"][:, :, : self.re_edim // 2]
        tmp2 = x_dict["R"][:, :, self.re_edim // 2 :]
        flx = self.las(tmp1, tmp2).squeeze()

        return flx


class FlexGNN_GCNConv_GGConv_LW(torch.nn.Module):
    """Flexible GNN with learnable layer weighting.
    
    Similar to FlexGNN_GCNConv_GGConv but combines representations from all
    layers using learned softmax weights for final flux prediction.
    """

    def __init__(self, nr: int, re_edim: int = 1, ge_edim: int = 1, nlayers: int = 1):
        super().__init__()
        self.nr = nr
        self.nlayers = nlayers
        self.las = BiLinAntisymmetricFunc(re_edim // 2)
        self.act = torch.nn.GELU()
        self.convs = torch.nn.ModuleList()
        self.re_edim = re_edim
        self.ge_edim = ge_edim
        self.layer_norms = torch.nn.ModuleList()

        for _ in range(nlayers):
            conv = HeteroConv(
                {
                    ("G", "to", "R"): ResGatedConv(
                        in_channels=(ge_edim, re_edim),
                        out_channels=re_edim,
                    ),
                    ("R", "to", "R"): GCNConv(
                        in_channels=re_edim, out_channels=re_edim
                    ),
                }
            )
            self.convs.append(conv)
            self.layer_norms.append(torch.nn.LayerNorm(re_edim))
        self.layer_weights = torch.nn.Parameter(torch.zeros(nlayers+1))
        self.layer_weights.data[-1] = 10.0

    def forward(
        self, x_dict: dict[str, torch.Tensor], ei_dict: dict[str, torch.Tensor]
    ):
        assert "G" in x_dict.keys(), "x_dict must have key 'G'"
        assert "R" in x_dict.keys(), "x_dict must have key 'R'"
        ge = x_dict["G"].detach().clone().to(device=x_dict["G"].device)
        convidx = 0
        lwts = torch.nn.functional.softmax(self.layer_weights, dim=0)
        reaction_reprs = lwts[convidx] * x_dict["R"]
        for conv, ln in zip(self.convs, self.layer_norms):
            convidx += 1
            x_dict = conv(x_dict, ei_dict)
            x_dict["G"] = ge  # - clamp gene expression B x nG x dG
            x_dict["R"] = self.act(x_dict["R"])  # - B x nR x dR
            x_dict["R"] = ln(x_dict["R"])
            reaction_reprs = reaction_reprs + lwts[convidx] * x_dict["R"]
            
        tmp1 = reaction_reprs[:, :, : self.re_edim // 2]  # - B x nR x dR/2
        tmp2 = reaction_reprs[:, :, self.re_edim // 2 :]  # - B x nR x dR/2
        flx = self.las(tmp1, tmp2).squeeze()  # - B x nR

        return flx


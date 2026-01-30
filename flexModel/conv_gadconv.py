
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
from torch_scatter import scatter_add, scatter_max
from typing import Optional
from torch_geometric.typing import Adj, OptTensor, PairTensor
from typing import Callable, Optional, Tuple, Union

from torch_scatter import scatter_add
import torch_scatter

import torch
from torch_geometric.utils import softmax

def batched_multi_head_softmax_chunked(
    w: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes_src: int,
    num_nodes_dst: int,
    is_sorted: bool = False,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """
    Applies multi-head softmax to batched edge weights in chunks to save memory.

    Args:
        w (torch.Tensor): A tensor of shape (B, E, H) with un-normalized edge weights.
        edge_index (torch.Tensor): A tensor of shape (2, E) with a static edge index.
        num_nodes_dst (int): The number of destination nodes in the destination graph (N).
        num_nodes_src (int): The number of source nodes in the source graph (N).
        chunk_size (int): The number of destination nodes to process at a time.

    Returns:
        torch.Tensor: A tensor of shape (B, E, H) with softmax-normalized edge weights.
    """
    B, E, H = w.shape
    
    if is_sorted:
        dst = edge_index[1]
        perm = None
        inv_perm = None
    else:
        perm = edge_index[1].argsort()
        inv_perm = perm.argsort()
        dst = edge_index[1][perm]
        w = w[:, perm, :]
        
    softmax_w = torch.empty_like(w)

    batch_offsets = torch.arange(B, device=w.device).view(-1, 1, 1) * (num_nodes_dst * H)
    head_offsets = torch.arange(H, device=w.device).view(1, 1, H) * num_nodes_dst

    for start_node in range(0, num_nodes_dst, chunk_size):
        end_node = min(start_node + chunk_size, num_nodes_dst)
        idx_start = torch.searchsorted(dst, start_node, right=False)
        idx_end = torch.searchsorted(dst, end_node, right=False)
        if idx_start == idx_end:
            continue

        w_chunk = w[:, idx_start:idx_end, :]
        dst_chunk = dst[idx_start:idx_end]

        group_index = batch_offsets + head_offsets + dst_chunk.unsqueeze(0).unsqueeze(-1)
        softmax_vals = softmax(w_chunk.reshape(-1), group_index.reshape(-1))
        softmax_w[:, idx_start:idx_end, :] = softmax_vals.view(B, -1, H)

    if inv_perm is not None:
        softmax_w = softmax_w[:, inv_perm, :]

    return softmax_w



def batched_multi_head_softmax(
    w: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """
    Applies multi-head softmax to batched edge weights, grouped by the DESTINATION node.

    Args:
        w (torch.Tensor): A tensor of shape (B, E, H) with un-normalized edge weights.
        edge_index (torch.Tensor): A tensor of shape (2, E) with a static edge index.
        num_nodes (int): The number of nodes in a single graph (N).

    Returns:
        torch.Tensor: A tensor of shape (B, E, H) with softmax-normalized edge weights.
    """
    B, E, H = w.shape
    device = w.device

    # Get the destination nodes from the static edge index
    # (2, E) -> (E)
    dst_nodes = edge_index[1].to(device)

    # --- Create a unique 1D index for each (batch, node, head) group ---
    # Create the batch offsets: `(B)` -> `(B, 1, 1)`
    # This offset ensures that each batch has a distinct set of indices.
    batch_offsets = torch.arange(B, device=device).view(-1, 1, 1) * (num_nodes * H)
    
    # Create the head offsets: `(H)` -> `(1, 1, H)`
    # This offset ensures that each head within a batch has a distinct set of indices.
    head_offsets = torch.arange(H, device=device).view(1, 1, H) * num_nodes
    
    # Add offsets and destination nodes together to create a unique index for each group.
    # `(E)` -> `(1, E, 1)`
    # The final `group_index` will have shape (B, E, H)
    group_index = batch_offsets + head_offsets + dst_nodes.unsqueeze(0).unsqueeze(-1)

    # --- Apply `torch_geometric.utils.softmax` on the flattened tensors ---
    # `softmax` requires a 1D source tensor and 1D index tensor.
    # (B, E, H) -> (B * E * H)
    flat_w = w.reshape(-1)
    
    # (B, E, H) -> (B * E * H)
    flat_index = group_index.reshape(-1)
    
    # Apply softmax.
    # The output will be a 1D tensor of shape (B * E * H)
    softmax_result_flat = softmax(src=flat_w, index=flat_index)

    # --- Reshape the result back to original shape ---
    # (B * E * H) -> (B, E, H)
    final_result = softmax_result_flat.view(B, E, H)

    return final_result



class GADConv(MessagePassing):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            heads: int = 1,
            return_heads: bool = False,
            use_residual: bool = True,
            use_gating: bool = True,
            use_attention: bool = True,
        ) -> None:
        super().__init__(aggr="add", node_dim=0)

        if isinstance(in_channels, tuple):
            in_src, in_dst = in_channels
        else:
            in_src = in_dst = in_channels
            
        self.in_channels_src = in_src
        self.in_channels_dst = in_dst
        self.out_channels = out_channels
        self.heads = heads
        self.return_heads = return_heads
        self.use_residual = use_residual
        self.use_gating = use_gating
        self.use_attention = use_attention
        
        #- register buffers for self.ei_perm, self.ei_inv_perm, self.ei_sorted
        self.register_buffer("ei_perm", None)
        self.register_buffer("ei_inv_perm", None)
        self.register_buffer("ei_sorted", None)

        # Linear projections for messages and gates (shared across heads).
        self.lin_W_src = nn.Linear(in_src, out_channels * heads, bias=False)
        self.lin_W_dst = nn.Linear(in_dst, out_channels * heads, bias=False)
        if use_gating:
            self.lin_A = nn.Linear(in_src, out_channels * heads, bias=True)
            self.lin_B = nn.Linear(in_dst, out_channels * heads, bias=True)
        else:
            self.lin_A = self.lin_B = None
        if use_residual and in_dst != out_channels * heads:
            self.res_proj = nn.Linear(in_dst, out_channels * heads, bias=False)
        else:
            self.res_proj = None

        # Learnable per-head attention vector.
        self.att = nn.Parameter(torch.empty(1, heads, out_channels))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize learnable parameters."""
        nn.init.xavier_uniform_(self.lin_W_src.weight)
        nn.init.xavier_uniform_(self.lin_W_dst.weight)
        if self.use_gating:
            nn.init.xavier_uniform_(self.lin_A.weight)
            nn.init.zeros_(self.lin_A.bias)
            nn.init.xavier_uniform_(self.lin_B.weight)
            nn.init.zeros_(self.lin_B.bias)
        nn.init.xavier_uniform_(self.att)
        if self.res_proj is not None:
            nn.init.xavier_uniform_(self.res_proj.weight)


    def forward(self, 
                x: Union[Tensor, PairTensor],
                edge_index: Adj) -> Tensor:
        """
        Run gated attention message passing.
        """
        
        #- sort the edge_index if we've not done it yes
        if self.ei_perm is None:
            self.ei_perm = edge_index[1].argsort().to(self.att.device)
            self.ei_inv_perm = self.ei_perm.argsort().to(self.att.device)
            self.ei_sorted = edge_index[:, self.ei_perm].to(self.att.device)
        elif self.ei_sorted.device != edge_index.device:
            self.ei_perm = self.ei_perm.to(edge_index.device)
            self.ei_inv_perm = self.ei_inv_perm.to(edge_index.device)
            self.ei_sorted = self.ei_sorted.to(edge_index.device)

        if not isinstance(x, tuple):
            x_src, x_dst = x, x
        else:
            x_src, x_dst = x

        assert x_src.dim() == x_dst.dim(), "Batched or unbatched tensors need to be consistent between source and destination."
        is_batched = x_src.dim() == 3
        if not is_batched:
            x_src = x_src.unsqueeze(0)  # (1, N_src, Fin)
            x_dst = x_dst.unsqueeze(0)  # (1, N_dst, Fin)

        assert x_src.shape[0] == x_dst.shape[0], "Batch size of source and destination node features must match."

        batch_size, num_nodes_src, _ = x_src.shape  # (B, N_src, Fin)
        _, num_nodes_dst, _ = x_dst.shape  # (B, N  _dst, Fin)
        ##edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)  # (2, E′)

        x_src_proj = self.lin_W_src(x_src).view(batch_size, num_nodes_src, self.heads, self.out_channels)
        x_dst_proj = self.lin_W_dst(x_dst).view(batch_size, num_nodes_dst, self.heads, self.out_channels)
        # x_dst/src_proj: (B, N_dst/N_src, H, Fout)

        if self.use_gating:
            gate_src = self.lin_A(x_src).view(batch_size, num_nodes_src, self.heads, self.out_channels)
            gate_dst = self.lin_B(x_dst).view(batch_size, num_nodes_dst, self.heads, self.out_channels)
            # gate_i/gate_j: (B, N, H, Fout)
        else:
            gate_src = gate_dst = None

        aggr = self.message_and_aggregate(
            edge_index=edge_index,
            x_src_proj=x_src_proj, 
            x_dst_proj=x_dst_proj,              
            gate_src=gate_src,
            gate_dst=gate_dst,
            batch_size=batch_size,
            size=(num_nodes_src, num_nodes_dst),
        )  # aggr: (B, N_dst, H, Fout)

        if self.use_residual:
            if self.res_proj is not None:
                residual = self.res_proj(x_dst).view(
                    batch_size, num_nodes_dst, self.heads, self.out_channels
                )  # (B, N_dst, H, Fout)
            else:
                residual = x_dst_proj  # (B, N_dst, H, Fout)
            aggr = aggr + residual  # (B, N_dst, H, Fout)

        if self.return_heads:
            out = aggr  # (B, N_dst, H, Fout)
        else:
            out = aggr.reshape(batch_size, num_nodes_dst, self.heads * self.out_channels)
            # out: (B, N_dst, H*Fout)

        return out if is_batched else out.squeeze(0)  # (B,N,*) or (N,*)

    def message_and_aggregate(
        self,
        edge_index: Tensor,
        x_src_proj: Tensor,
        x_dst_proj: Tensor,
        gate_src: Optional[Tensor],
        gate_dst: Optional[Tensor],
        batch_size: int,
        chunk_size: int = 4_096, #- 32 * 32 * 32
        #size: tuple[int, int],
        **kwargs: Tensor,
    ) -> Tensor:
        
        num_nodes_dst = x_dst_proj.shape[1]
        dst_sorted = self.ei_sorted[1]    

        #- each destinagion edge segment gets makred in the "csr" pointer
        if getattr(self, "csr_ptr", None) is None or self.csr_ptr.numel() != num_nodes_dst + 1:
            dst_counts = torch.bincount(dst_sorted, minlength=num_nodes_dst)
            self.csr_ptr = torch.cat([dst_counts.new_zeros(1), dst_counts.cumsum(0)])

        edge_ids = torch.arange(dst_sorted.numel(), device=dst_sorted.device)  # (E,)
        chunk_id = torch.div(dst_sorted, chunk_size, rounding_mode="floor") # (E,)

        unique_chunks, chunk_counts = torch.unique_consecutive(
            chunk_id, return_counts=True
        ) # (C,), (C,) ; number of unique chunks and their respective number of edges                                                                    
        
        chunk_offsets = torch.cat([chunk_counts.new_zeros(1), chunk_counts.cumsum(0)])
        chunk_edges = tuple(
            edge_ids[chunk_offsets[i] : chunk_offsets[i + 1]]
            for i in range(len(chunk_counts))
        )
        
        accum_flat = x_dst_proj.new_zeros(batch_size * self.heads, 
                                          num_nodes_dst, self.out_channels) # (BH, N_dst, Fout)
        accum = accum_flat.view(batch_size, 
                                self.heads, 
                                num_nodes_dst, 
                                self.out_channels).permute(0, 2, 1, 3) #(B, N_dst, H, Fout)

        #- csr - row pointers  for the first node of each chunk and the node just past its end
        chunk_ptr = self.csr_ptr.index_select(0, 
                                            torch.cat([unique_chunks * chunk_size, 
                                            unique_chunks * chunk_size + chunk_size]).clamp_max(num_nodes_dst))
        chunk_ptr_splits = chunk_ptr.unbind(0)
        chunk_starts = (unique_chunks * chunk_size).clamp_max(num_nodes_dst)
        chunk_ptr = self.csr_ptr
        for i, edge_idx in enumerate(chunk_edges):
            if edge_idx.numel() == 0:
                continue
            edge_index_chunk = self.ei_sorted.index_select(1, edge_idx) # (2, E_chunk)
            
            dst_chunk = edge_index_chunk[1] # (E_chunk,)
            src_chunk = edge_index_chunk[0] # (E_chunk,)

            proj_src_chunk = x_src_proj.index_select(1, src_chunk) # (B, E_chunk, H, F)
            proj_dst_chunk = x_dst_proj.index_select(1, dst_chunk) # (B, E_chunk, H, F)

            if gate_src is not None and gate_dst is not None:
                gate_src_chunk = gate_src[:, src_chunk]
                gate_dst_chunk = gate_dst[:, dst_chunk]
                messages = torch.sigmoid(gate_dst_chunk + gate_src_chunk) * proj_src_chunk
                if self.use_attention:
                    logits = (self.att * torch.relu(messages)).sum(dim=-1)
            else:
                messages = proj_src_chunk
                if self.use_attention:
                    logits = (self.att * torch.relu(proj_src_chunk + proj_dst_chunk)).sum(dim=-1)
            if self.use_attention:
                logits = (self.att * torch.relu(proj_src_chunk + proj_dst_chunk)).sum(dim=-1)  # (B, E_chunk, H)

                start_node = chunk_starts[i].item()
                end_node = min(start_node + chunk_size, num_nodes_dst)
                ptr_chunk = chunk_ptr[start_node : end_node + 1] - chunk_ptr[start_node]

                flat_logits = logits.permute(1, 0, 2).reshape(-1, batch_size * self.heads)      # (E_chunk, BH)
                
                seg_max = torch_scatter.segment_csr(flat_logits, ptr_chunk, reduce="max")
                flat_logits -= seg_max.index_select(0, torch.repeat_interleave(torch.arange(ptr_chunk.size(0) - 1, device=ptr_chunk.device), ptr_chunk[1:] - ptr_chunk[:-1]))
                flat_logits.exp_()
                seg_sum = torch_scatter.segment_csr(flat_logits, ptr_chunk, reduce="sum")
                flat_logits /= seg_sum.index_select(0, torch.repeat_interleave(torch.arange(ptr_chunk.size(0) - 1, device=ptr_chunk.device), ptr_chunk[1:] - ptr_chunk[:-1]))
                alpha = flat_logits.view(-1, batch_size, self.heads).permute(1, 0, 2).unsqueeze(-1)
                messages.mul_(alpha)
                
                # seg_max = torch_scatter.segment_csr(flat_logits, ptr_chunk, reduce="max")
                # edge_counts = ptr_chunk[1:] - ptr_chunk[:-1]
                # seg_max = torch.repeat_interleave(seg_max, edge_counts, dim=0)
                # logits_exp = (flat_logits - seg_max).exp()

                # seg_sum = torch_scatter.segment_csr(logits_exp, ptr_chunk, reduce="sum")
                # seg_sum = torch.repeat_interleave(seg_sum, edge_counts, dim=0)
                
                # alpha_flat = logits_exp / seg_sum                                               # (E_chunk, BH)
                # alpha = alpha_flat.view(-1, batch_size, self.heads).permute(1, 0, 2).unsqueeze(-1)
                # messages.mul_(alpha)
        
            messages = messages.to(accum.dtype)

            scatter_add(
                messages.permute(0, 2, 1, 3).reshape(batch_size * self.heads, -1, self.out_channels),
                dst_chunk.expand(batch_size * self.heads, -1),
                dim=1,
                out=accum_flat,
            )
           
        aggr = accum_flat.view(batch_size, self.heads, num_nodes_dst, self.out_channels).permute(0, 2, 1, 3)
        return aggr
"""Regression tests for ReaReaConv.

Checks that the layer matches GCNConv on a random graph and reports a small
timing comparison on a mid-size graph that still runs quickly on CPU.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import torch
from torch_geometric.nn import GCNConv

from flexModel.conv_reaReaConv import ReaReaConv


def _make_random_graph(num_nodes: int, num_edges: int, seed: int = 0) -> torch.Tensor:
    """Create a random edge_index without self-loops."""
    generator = torch.Generator().manual_seed(seed)
    src = torch.randint(0, num_nodes, (num_edges,), generator=generator)
    tgt = torch.randint(0, num_nodes, (num_edges,), generator=generator)
    mask = src != tgt
    return torch.stack([src[mask], tgt[mask]], dim=0)


def _time_forward(
    fn: Callable[[], torch.Tensor],
    device: torch.device,
    repeats: int = 12,
    warmup: int = 3,
) -> float:
    """Return average forward time in seconds for one timing run."""
    for _ in range(warmup):
        fn()

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    end = time.perf_counter()
    return (end - start) / repeats


def _time_forward_avg(
    fn: Callable[[], torch.Tensor],
    device: torch.device,
    runs: int = 5,
    repeats: int = 12,
    warmup: int = 3,
) -> float:
    """Return the mean forward time in seconds across several timing runs."""
    return (
        sum(
            _time_forward(fn, device=device, repeats=repeats, warmup=warmup)
            for _ in range(runs)
        )
        / runs
    )


def _run_gcn(conv: GCNConv, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Run a single GCNConv forward pass without gradients."""
    with torch.no_grad():
        return conv(x, edge_index)


def _run_rea(
    conv: ReaReaConv, x: torch.Tensor, edge_index: torch.Tensor
) -> torch.Tensor:
    """Run a single ReaReaConv forward pass without gradients."""
    with torch.no_grad():
        return conv(x.unsqueeze(0), edge_index).squeeze(0)


def _bench_on_device(device: torch.device) -> None:
    """Compare ReaReaConv and GCNConv on a single device."""
    torch.manual_seed(7)

    num_nodes = 384
    num_edges = 6144
    in_channels = 32
    out_channels = 32

    edge_index = _make_random_graph(num_nodes, num_edges, seed=7).to(device)
    x = torch.randn(num_nodes, in_channels, device=device)

    for add_self_loops in [True, False]:
        gcn = GCNConv(
            in_channels=in_channels,
            out_channels=out_channels,
            add_self_loops=add_self_loops,
            bias=True,
        ).to(device)
        rea = ReaReaConv(
            in_channels=in_channels,
            out_channels=out_channels,
            use_disc=False,
            add_self_loops=add_self_loops,
            bias=True,
        ).to(device)

        rea.load_state_dict(gcn.state_dict(), strict=False)
        gcn.eval()
        rea.eval()

        with torch.no_grad():
            gcn_out = gcn(x, edge_index)
            rea_out = rea(x.unsqueeze(0), edge_index).squeeze(0)

        torch.testing.assert_close(rea_out, gcn_out, rtol=1e-5, atol=1e-5)

        gcn_time = _time_forward_avg(
            lambda: _run_gcn(gcn, x, edge_index), device=device, runs=5
        )
        rea_time = _time_forward_avg(
            lambda: _run_rea(rea, x, edge_index), device=device, runs=5
        )
        speedup = gcn_time / rea_time if rea_time > 0 else float("inf")

        print(
            f"{device.type} add_self_loops={add_self_loops} "
            f"gcn={gcn_time * 1e3:.3f} ms "
            f"rea={rea_time * 1e3:.3f} ms "
            f"gcn/rea={speedup:.2f}x"
        )

        assert torch.isfinite(torch.tensor(gcn_time))
        assert torch.isfinite(torch.tensor(rea_time))


def test_reareaconv_matches_gcnconv_and_reports_timing() -> None:
    """ReaReaConv should match GCNConv on a random graph and stay in the same runtime class."""
    _bench_on_device(torch.device("cpu"))

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        _bench_on_device(torch.device("cuda:1"))
    elif torch.cuda.is_available():
        _bench_on_device(torch.device("cuda"))

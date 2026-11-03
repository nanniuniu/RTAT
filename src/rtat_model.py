"""RTAT hierarchical backbone used by the archived experiments.

External-conditioned regional attention adds a projected context vector to
normalized traffic tokens. See models.py for the final residual and ToD paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch
from torch import Tensor, nn
from torch.nn import functional as F

EXECUTION_OPTIMIZATION_ID = "window_layout_cache_v1"


def _window_partition(x: Tensor, window: tuple[int, int, int]) -> Tensor:
    """Partition channels-last (B,T,H,W,C) data into local windows."""
    batch, time, height, width, channels = x.shape
    wt, wh, ww = window
    return (
        x.reshape(batch, time // wt, wt, height // wh, wh, width // ww, ww, channels)
        .permute(0, 1, 3, 5, 2, 4, 6, 7)
        .reshape(-1, wt * wh * ww, channels)
    )


def _window_reverse(
    windows: Tensor, window: tuple[int, int, int], shape: tuple[int, int, int, int]
) -> Tensor:
    batch, time, height, width = shape
    wt, wh, ww = window
    channels = windows.shape[-1]
    return (
        windows.reshape(batch, time // wt, height // wh, width // ww, wt, wh, ww, channels)
        .permute(0, 1, 4, 2, 5, 3, 6, 7)
        .reshape(batch, time, height, width, channels)
    )


@dataclass(frozen=True)
class WindowLayout:
    window: tuple[int, int, int]
    shift: tuple[int, int, int]
    padded_shape: tuple[int, int, int]
    # (number of windows per sample, tokens per window, 3), unshifted coordinates
    coordinates: Tensor
    # (number of windows per sample, tokens per window)
    valid: Tensor
    # Runtime-only, non-learned tensors; never store a cached learned bias here.
    attention_masks: dict[bool, Tensor] = field(default_factory=dict, compare=False, repr=False)
    lag_index_tables: dict[int, Tensor] = field(default_factory=dict, compare=False, repr=False)


def make_window_layout(
    shape: tuple[int, int, int],
    nominal_window: tuple[int, int, int],
    shifted: bool,
    device: torch.device,
) -> WindowLayout:
    """Keep original token coordinates through cyclic shifts and padding."""
    if any(size <= 0 for size in shape) or any(size <= 0 for size in nominal_window):
        raise ValueError("spatial-temporal shape and window sizes must be positive")
    window = tuple(min(size, span) for size, span in zip(nominal_window, shape))
    shift = tuple(span // 2 if shifted and size > span else 0 for size, span in zip(shape, window))
    padded = tuple(((size + span - 1) // span) * span for size, span in zip(shape, window))
    axes = [torch.arange(size, device=device) for size in padded]
    coordinates = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    valid = torch.ones(padded, dtype=torch.bool, device=device)
    for axis, true_size in enumerate(shape):
        valid &= coordinates[..., axis] < true_size
    if any(shift):
        coordinates = torch.roll(coordinates, shifts=tuple(-s for s in shift), dims=(0, 1, 2))
        valid = torch.roll(valid, shifts=tuple(-s for s in shift), dims=(0, 1, 2))
    coordinate_windows = _window_partition(coordinates.unsqueeze(0), window).to(torch.long)
    valid_windows = _window_partition(valid.unsqueeze(0).unsqueeze(-1), window).squeeze(-1)
    return WindowLayout(window, shift, padded, coordinate_windows, valid_windows)


def allowed_attention_pairs(layout: WindowLayout, causal: bool) -> Tensor:
    """Return (window, query, key) validity without cyclic-wrap connections."""
    if causal in layout.attention_masks:
        return layout.attention_masks[causal]
    coords = layout.coordinates
    valid = layout.valid
    allowed = valid[:, :, None] & valid[:, None, :]
    for axis, (span, shift) in enumerate(zip(layout.window, layout.shift)):
        # Shifted non-cyclic partition IDs split the wrapped edge of a rolled window.
        group = (coords[..., axis] + shift) // span
        allowed &= group[:, :, None] == group[:, None, :]
    if causal:
        allowed &= coords[:, None, :, 0] <= coords[:, :, None, 0]
    return allowed


def prepare_window_layout(shape: tuple[int, int, int], window: tuple[int, int, int],
                          shifted: bool, device: torch.device,
                          max_temporal_tokens: int | None) -> WindowLayout:
    """Build constant masks/indices once on CPU, then move them to the device.

    Checking the lag table on CPU avoids a CUDA-dependent Python branch in
    every attention forward. Keep the original coordinate-based check, including
    shifted wrap edges and padding, rather than replacing it with a guessed span.
    """
    layout = make_window_layout(shape, window, shifted, torch.device("cpu"))
    tables = {}
    if max_temporal_tokens is not None:
        time = layout.coordinates[..., 0]
        lag = time[:, :, None] - time[:, None, :]
        if torch.any(lag.abs() >= max_temporal_tokens):
            raise ValueError("temporal span exceeds the learned lag-bias table")
        tables[max_temporal_tokens] = (lag + max_temporal_tokens - 1).to(device)
    masks = {causal: allowed_attention_pairs(layout, causal).to(device)
             for causal in (False, True)}
    return replace(layout, coordinates=layout.coordinates.to(device), valid=layout.valid.to(device),
                   attention_masks=masks,
                   lag_index_tables=tables)


class RegionalDelayAttention(nn.Module):
    def __init__(self, dim: int, heads: int, max_temporal_tokens: int, *,
                 use_lag_bias: bool = True, use_temporal_mask: bool = True,
                 attention_dropout: float = 0.0, projection_dropout: float = 0.0):
        super().__init__()
        if dim % heads:
            raise ValueError("embedding dimension must be divisible by attention heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.max_temporal_tokens = max_temporal_tokens
        self.use_lag_bias = use_lag_bias
        self.use_temporal_mask = use_temporal_mask
        self.qkv = nn.Linear(dim, 3 * dim)
        self.projection = nn.Linear(dim, dim)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.projection_dropout = nn.Dropout(projection_dropout)
        self.lag_bias = nn.Parameter(torch.zeros(heads, 2 * max_temporal_tokens - 1))
        nn.init.trunc_normal_(self.lag_bias, std=0.02)

    def forward(self, windows: Tensor, layout: WindowLayout) -> Tensor:
        batch_windows, token_count, dim = windows.shape
        qkv = self.qkv(windows).reshape(batch_windows, token_count, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        score = (q * self.head_dim ** -0.5) @ k.transpose(-2, -1)
        if self.use_lag_bias:
            indices = layout.lag_index_tables.get(self.max_temporal_tokens)
            if indices is None:
                # Retain the original public API for uncached/manual layouts.
                time = layout.coordinates[..., 0]
                lag = time[:, :, None] - time[:, None, :]
                if torch.any(lag.abs() >= self.max_temporal_tokens):
                    raise ValueError("temporal span exceeds the learned lag-bias table")
                indices = lag + self.max_temporal_tokens - 1
            bias = self.lag_bias[:, indices].permute(1, 0, 2, 3)
            repeats = batch_windows // layout.coordinates.shape[0]
            score = score + bias.repeat(repeats, 1, 1, 1)
        allowed = allowed_attention_pairs(layout, self.use_temporal_mask)
        repeats = batch_windows // allowed.shape[0]
        allowed = allowed.repeat(repeats, 1, 1)
        score = score.masked_fill(~allowed[:, None], torch.finfo(score.dtype).min)
        weights = self.attention_dropout(score.softmax(dim=-1))
        output = (weights @ v).transpose(1, 2).reshape(batch_windows, token_count, dim)
        return self.projection_dropout(self.projection(output))


class DropPath(nn.Module):
    """Drop an entire residual branch independently for each training sample."""

    def __init__(self, probability: float):
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("drop-path probability must be in [0, 1)")
        self.probability = probability

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.probability == 0.0:
            return x
        keep_probability = 1.0 - self.probability
        mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(mask_shape).bernoulli_(keep_probability)
        return x * mask / keep_probability


class RDFBlock(nn.Module):
    def __init__(self, dim: int, heads: int, *, window_size: tuple[int, int, int],
                 max_temporal_tokens: int, shifted: bool, use_lag_bias: bool,
                 use_temporal_mask: bool, use_external: bool,
                 dropout: float, attention_dropout: float,
                 drop_path_probability: float = 0.0, mlp_ratio: float = 4.0):
        super().__init__()
        self.window_size = window_size
        self.shifted = shifted
        self.use_external = use_external
        # One-entry cache bounds memory, accommodates a short final batch (batch
        # is not part of the key), and rebuilds after a shape/device change.
        self._layout_cache: tuple[tuple, WindowLayout] | None = None
        self.norm1 = nn.LayerNorm(dim)
        attention_kwargs = dict(
            use_lag_bias=use_lag_bias,
            use_temporal_mask=use_temporal_mask,
            attention_dropout=attention_dropout,
            projection_dropout=dropout,
        )
        self.traffic_attention = RegionalDelayAttention(dim, heads, max_temporal_tokens, **attention_kwargs)
        if use_external:
            self.external_attention = RegionalDelayAttention(dim, heads, max_temporal_tokens, **attention_kwargs)
            self.gate = nn.Linear(dim, 1)
        self.norm2 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path_probability)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, dim), nn.Dropout(dropout))

    def _apply(self, fn, recurse=True):
        self._layout_cache = None
        return super()._apply(fn, recurse=recurse)

    def _attend(self, x: Tensor, layout: WindowLayout, module: RegionalDelayAttention) -> Tensor:
        batch, time, height, width, channels = x.shape
        pt, ph, pw = layout.padded_shape
        padded = F.pad(x, (0, 0, 0, pw - width, 0, ph - height, 0, pt - time))
        if any(layout.shift):
            padded = torch.roll(padded, shifts=tuple(-s for s in layout.shift), dims=(1, 2, 3))
        windows = _window_partition(padded, layout.window)
        attended = _window_reverse(module(windows, layout), layout.window, (batch, pt, ph, pw))
        if any(layout.shift):
            attended = torch.roll(attended, shifts=layout.shift, dims=(1, 2, 3))
        return attended[:, :time, :height, :width, :].contiguous()

    def forward(self, x: Tensor, external: Tensor | None) -> Tensor:
        key = (tuple(x.shape[1:4]), x.device, self.window_size, self.shifted,
               self.traffic_attention.max_temporal_tokens, self.traffic_attention.use_lag_bias)
        if self._layout_cache is None or self._layout_cache[0] != key:
            layout = prepare_window_layout(key[0], self.window_size, self.shifted, x.device,
                                           self.traffic_attention.max_temporal_tokens
                                           if self.traffic_attention.use_lag_bias else None)
            self._layout_cache = (key, layout)
        else:
            layout = self._layout_cache[1]
        traffic = self.norm1(x)
        u = self._attend(traffic, layout, self.traffic_attention)
        if self.use_external:
            if external is None:
                raise ValueError("external features are required by this RDF block")
            v = self._attend(traffic + external[:, None, None, None, :],
                             layout, self.external_attention)
            gate = torch.sigmoid(self.gate(traffic))
            x = x + self.drop_path(gate * u + (1.0 - gate) * v)
        else:
            x = x + self.drop_path(u)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class SpatialPatchMerging(nn.Module):
    """Merge 2x2 spatial neighbours while retaining the temporal axis."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        _, _, height, width, _ = x.shape
        x = F.pad(x, (0, 0, 0, width % 2, 0, height % 2))
        joined = torch.cat((x[:, :, 0::2, 0::2], x[:, :, 1::2, 0::2],
                            x[:, :, 0::2, 1::2], x[:, :, 1::2, 1::2]), dim=-1)
        return self.reduction(self.norm(joined))


class RTATModel(nn.Module):
    """One-step 12x16 flow predictor with the manuscript's four RDF components."""

    def __init__(self, *, external_dim: int = 56, embed_dim: int = 128,
                 depths: tuple[int, ...] = (2, 18, 2, 2),
                 heads: tuple[int, ...] = (2, 4, 8, 16),
                 patch_size: tuple[int, int, int] = (3, 2, 2),
                 window_size: tuple[int, int, int] = (2, 2, 2),
                 grid_shape: tuple[int, int] = (12, 16),
                 dropout: float = 0.2, attention_dropout: float = 0.2,
                 stochastic_depth: float = 0.3,
                 use_lag_bias: bool = True, use_temporal_mask: bool = True,
                 use_shifted_windows: bool = True, use_external: bool = True):
        super().__init__()
        if len(depths) != 4 or len(heads) != 4 or any(depth <= 0 for depth in depths):
            raise ValueError("RTAT requires four non-empty encoder stages")
        if patch_size != (3, 2, 2):
            raise ValueError("this implementation currently supports the manuscript's (3,2,2) patch")
        if any(size <= 0 or size % 2 for size in grid_shape):
            raise ValueError("grid_shape dimensions must be positive and divisible by the 2x2 patch")
        self.use_external = use_external
        self.grid_shape = tuple(grid_shape)
        self.patch = nn.Conv3d(2, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_norm = nn.LayerNorm(embed_dim)
        self.position_dropout = nn.Dropout(dropout)
        self.external_embeddings = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.mergers = nn.ModuleList()
        temporal_tokens = 12 // patch_size[0]
        block_count = sum(depths)
        block_number = 0
        for stage_index, (depth, head_count) in enumerate(zip(depths, heads)):
            dim = embed_dim * 2 ** stage_index
            if use_external:
                self.external_embeddings.append(nn.Sequential(nn.Linear(external_dim, dim), nn.GELU()))
            blocks = nn.ModuleList()
            for block_index in range(depth):
                probability = stochastic_depth * block_number / max(block_count - 1, 1)
                blocks.append(RDFBlock(
                    dim, head_count, window_size=window_size,
                    max_temporal_tokens=temporal_tokens,
                    shifted=use_shifted_windows and block_index % 2 == 1,
                    use_lag_bias=use_lag_bias,
                    use_temporal_mask=use_temporal_mask,
                    use_external=use_external,
                    dropout=dropout, attention_dropout=attention_dropout,
                    drop_path_probability=probability,
                ))
                block_number += 1
            self.stages.append(blocks)
            if stage_index < 3:
                self.mergers.append(SpatialPatchMerging(dim))
        self.output_norm = nn.LayerNorm(embed_dim * 8)
        self.readout = nn.Linear(embed_dim * 8, 2 * grid_shape[0] * grid_shape[1])

    def forward(self, flow: Tensor, external: Tensor | None = None) -> Tensor:
        expected = (12, 2, *self.grid_shape)
        if flow.ndim != 5 or tuple(flow.shape[1:]) != expected:
            raise ValueError(f"expected flow shape (batch,{expected}), got {tuple(flow.shape)}")
        if self.use_external:
            if external is None:
                raise ValueError("external features are required when use_external=True")
            if external.ndim == 3 and external.shape[1] == 1:
                external = external[:, 0]
            if external.ndim != 2 or external.shape[0] != flow.shape[0]:
                raise ValueError("external features must have shape (batch, feature_count)")
        x = self.patch(flow.permute(0, 2, 1, 3, 4)).permute(0, 2, 3, 4, 1)
        x = self.position_dropout(self.patch_norm(x))
        for index, blocks in enumerate(self.stages):
            embedded = self.external_embeddings[index](external) if self.use_external else None
            for block in blocks:
                x = block(x, embedded)
            if index < len(self.mergers):
                x = self.mergers[index](x)
        if tuple(x.shape[1:4]) != (4, 1, 1):
            raise RuntimeError(f"unexpected final token grid {tuple(x.shape[1:4])}")
        final_token = self.output_norm(x[:, -1, 0, 0])
        return self.readout(final_token).reshape(flow.shape[0], 2, *self.grid_shape)

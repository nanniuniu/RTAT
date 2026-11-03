"""Shared model configuration, batching, schedule, and validation routines."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from rtat_model import RTATModel
from run_reproducible import InputScaler

@dataclass(frozen=True)
class ModelConfiguration:
    embed_dim: int
    depths: tuple[int, int, int, int]
    heads: tuple[int, int, int, int]
    patch_size: tuple[int, int, int]
    window_size: tuple[int, int, int]
    grid_shape: tuple[int, int]
    dropout: float
    attention_dropout: float
    stochastic_depth: float
    use_lag_bias: bool
    use_temporal_mask: bool
    use_shifted_windows: bool
    use_external: bool
    external_dim: int = 56

    def build(self) -> RTATModel:
        return RTATModel(**asdict(self))


class RTATDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        external: np.ndarray | None,
        start: int,
        stop: int,
        scaler: InputScaler,
        spatial_coarsening_factor: int = 1,
    ) -> None:
        self.x = x
        self.y = y
        self.external = external
        self.start = start
        self.stop = stop
        self.scaler = scaler
        self.spatial_coarsening_factor = spatial_coarsening_factor

    def __len__(self) -> int:
        return self.stop - self.start

    def __getitem__(self, index: int):
        source_index = self.start + index
        flow_array = coarsen_spatial(self.x[source_index], self.spatial_coarsening_factor)
        target_array = coarsen_spatial(self.y[source_index], self.spatial_coarsening_factor)
        flow = torch.from_numpy(self.scaler.transform(flow_array))
        target = torch.from_numpy(np.array(target_array, dtype=np.float32, copy=True))
        if self.external is None:
            return flow, target
        external = torch.from_numpy(
            np.array(self.external[source_index], dtype=np.float32, copy=True)
        )
        return flow, external, target

    def __getitems__(self, indices):
        """Vectorized fetching; preserve indices, scaling, values and order.

        PyTorch still performs its usual collation. No sample is dropped and
        no scaler is fitted here. Restrict allocation to the requested batch.
        """
        if not indices:
            return []
        source = self.start + np.asarray(indices, dtype=np.int64)
        flow_array = coarsen_spatial(self.x[source], self.spatial_coarsening_factor)
        target_array = coarsen_spatial(self.y[source], self.spatial_coarsening_factor)
        flows = torch.from_numpy(self.scaler.transform(flow_array))
        targets = torch.from_numpy(np.array(target_array, dtype=np.float32, copy=True))
        if self.external is None:
            return list(zip(flows, targets))
        external = torch.from_numpy(np.array(self.external[source], dtype=np.float32, copy=True))
        return list(zip(flows, external, targets))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def coarsen_spatial(value: np.ndarray, factor: int) -> np.ndarray:
    """Sum non-overlapping factor-by-factor index cells while preserving leading axes."""

    value = np.asarray(value)
    if factor <= 0:
        raise ValueError("spatial coarsening factor must be positive")
    if factor == 1:
        return value
    height, width = value.shape[-2:]
    if height % factor or width % factor:
        raise ValueError(f"Spatial shape {(height, width)} is not divisible by factor {factor}")
    reshaped = value.reshape(
        *value.shape[:-2], height // factor, factor, width // factor, factor
    )
    return reshaped.sum(axis=(-3, -1))


def warmup_cosine_factor(
    step: int, *, total_steps: int, warmup_steps: int, start_factor: float, min_factor: float
) -> float:
    """Learning-rate multiplier for linear warmup followed by cosine decay."""
    if total_steps <= 0 or not 0 <= step <= total_steps:
        raise ValueError("step must lie in [0,total_steps]")
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps must lie in [0,total_steps)")
    if not 0 < start_factor <= 1 or not 0 < min_factor <= 1:
        raise ValueError("learning-rate factors must lie in (0,1]")
    if warmup_steps and step < warmup_steps:
        fraction = step / warmup_steps
        return start_factor + fraction * (1.0 - start_factor)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_factor + 0.5 * (1.0 - min_factor) * (1.0 + math.cos(math.pi * progress))


def make_loader(
    dataset: Dataset,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
    workers: int,
    limit_samples: int | None,
    prefetch_factor: int = 2,
) -> DataLoader:
    if workers < 0 or prefetch_factor < 1:
        raise ValueError('workers must be nonnegative and prefetch_factor positive')
    if limit_samples is not None:
        dataset = Subset(dataset, range(min(limit_samples, len(dataset))))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        **({'prefetch_factor': prefetch_factor} if workers > 0 else {}),
        generator=generator,
    )


def unpack_batch(batch, device: torch.device, use_external: bool):
    if use_external:
        flow, external, target = batch
        return (
            flow.to(device, non_blocking=True),
            external.to(device, non_blocking=True),
            target.to(device, non_blocking=True),
        )
    flow, target = batch
    return flow.to(device, non_blocking=True), None, target.to(device, non_blocking=True)


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_external: bool,
    collect: bool,
) -> tuple[dict, np.ndarray | None, np.ndarray | None]:
    model.eval()
    square_sum = absolute_sum = 0.0
    count = 0
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            flow, external, target = unpack_batch(batch, device, use_external)
            prediction = model(flow, external)
            error = prediction - target
            square_sum += float(torch.sum(error.square()).item())
            absolute_sum += float(torch.sum(error.abs()).item())
            count += error.numel()
            if collect:
                predictions.append(prediction.cpu().numpy())
                targets.append(target.cpu().numpy())
    if count == 0:
        raise RuntimeError("Evaluation loader is empty")
    metrics = {
        "mse": square_sum / count,
        "rmse": math.sqrt(square_sum / count),
        "mae": absolute_sum / count,
        "count": count,
    }
    if not collect:
        return metrics, None, None
    return metrics, np.concatenate(predictions), np.concatenate(targets)

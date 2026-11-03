"""Residual, bounded data access, and evaluation routines from archived runs."""
from __future__ import annotations
from pathlib import Path
import hashlib
import math
import numpy as np
import torch
from torch import nn
import base_runner as reference

def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def anchor_from_scaled(flow, minimum, effective_scale):
    """Recover only the latest observed frame (t-1); never use a target."""
    require(flow.ndim == 5 and tuple(flow.shape[1:]) == (12, 2, 12, 16), "Wrong input frame shape")
    require(tuple(minimum.shape) == tuple(effective_scale.shape) == (1, 2, 12, 16), "Wrong anchor buffer shape")
    # Keep original float32 raw-count scale, also under AMP.
    frames = flow[:, [11]].float()
    raw = (frames + 1.0) * 0.5 * effective_scale.float().unsqueeze(0) + minimum.float().unsqueeze(0)
    return raw[:, 0]


class RecentResidualWrapper(nn.Module):
    """Same RTAT parameter tensors for both arms; only a fixed output skip differs."""
    def __init__(self, backbone, scaler, enabled):
        super().__init__()
        self.backbone = backbone
        self.enabled = bool(enabled)
        minimum = np.asarray(scaler.minimum, dtype=np.float32)
        scale = np.asarray(scaler.scale, dtype=np.float32)
        require(minimum.shape == scale.shape == (12, 2, 12, 16), "Fold input scaler shape changed")
        require(np.isfinite(minimum).all() and np.isfinite(scale).all() and np.all(scale > 0),
                "Invalid inverse-scaling buffers")
        self.register_buffer("anchor_minimum", torch.from_numpy(minimum[[11]].copy()))
        self.register_buffer("anchor_effective_scale", torch.from_numpy(scale[[11]].copy()))

    def anchor(self, flow):
        return anchor_from_scaled(flow, self.anchor_minimum, self.anchor_effective_scale)

    def forward(self, flow, external=None):
        correction = self.backbone(flow, external)
        return correction.float() + self.anchor(flow) if self.enabled else correction


class BoundedDataset(reference.RTATDataset):
    def __getitem__(self, index):
        require(0 <= int(index) < len(self), "Unsafe dataset index")
        return super().__getitem__(int(index))

    def __getitems__(self, indices):
        require(all(0 <= int(i) < len(self) for i in indices), "Unsafe batch indices")
        return super().__getitems__(indices)


def evaluate_detailed(model, data_loader, device, *, collect=False):
    model.eval()
    totals = [dict(sse=0.0, sae=0.0, bias=0.0, count=0) for _ in range(3)]
    predictions, targets = [], []
    with torch.no_grad():
        for batch in data_loader:
            flow, external, target = reference.unpack_batch(batch, device, True)
            prediction = model(flow, external)
            require(torch.isfinite(prediction).all().item(), "Nonfinite eval predictions")
            p = prediction.cpu().numpy().astype(np.float64)
            t = target.cpu().numpy().astype(np.float64)
            error = p - t
            for values, total in zip((error, error[:, 0], error[:, 1]), totals):
                total["sse"] += float(np.sum(values ** 2))
                total["sae"] += float(np.sum(np.abs(values)))
                total["bias"] += float(np.sum(values))
                total["count"] += values.size
            if collect:
                predictions.append(p.astype(np.float32))
                targets.append(t.astype(np.float32))
    def finish(total):
        require(total["count"] > 0, "Empty eval")
        n = total["count"]
        return dict(mse=total["sse"]/n, rmse=math.sqrt(total["sse"]/n),
                    mae=total["sae"]/n, bias=total["bias"]/n, count=n)
    metrics = finish(totals[0])
    metrics["channels"] = [finish(x) for x in totals[1:]]
    return metrics, ((np.concatenate(predictions), np.concatenate(targets)) if collect else (None, None))

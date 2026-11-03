"""Direct RTAT + zero-initialized time-of-day conditioning. No training entry point."""
from datetime import datetime, timedelta
from pathlib import Path
import csv
import hashlib
import itertools
import numpy as np
import torch
from torch import nn

EXTERNAL_WIDTH = 38
SLOTS_PER_DAY = 48
TARGET_START = datetime(2013, 1, 29)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def read_time_slots(path, *, start, stop, expected_sha256, dataset='nyctaxi', minutes=30, sample_count=51216):
    """Read known target-calendar metadata, never derive slots from flow targets.

    Returns absolute-index-aligned slots for [start, stop). Original timestamps
    remain unrecovered: source times are the previously adopted wall-clock reconstruction.
    """
    profiles={('nyctaxi',30):51216,('nyctaxi',60):25608,('nyctaxi',90):17072,
              ('nycbike',30):51264,('nycbike',60):25632,('nycbike',90):17088}
    require((dataset,minutes) in profiles and sample_count==profiles[dataset,minutes], 'Invalid calendar profile')
    require(0 <= start < stop <= sample_count, 'Invalid original sample range')
    target_start=datetime(2013 if dataset=='nyctaxi' else 2018,1,29)
    require(sha(path) == expected_sha256, 'Calendar source SHA mismatch')
    result = []
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        require({'sample_index', 'target_start_local_reconstructed'} <= set(reader.fieldnames or []),
                'Unexpected calendar schema')
        for expected_index, row in enumerate(itertools.islice(reader, stop)):
            require(int(row['sample_index']) == expected_index, 'Calendar row index mismatch')
            stamp = datetime.fromisoformat(row['target_start_local_reconstructed'])
            require(stamp.tzinfo is None and stamp == target_start + timedelta(minutes=minutes*expected_index),
                    'Wall-clock phase or chronology mismatch')
            require(stamp.minute in (0, 30) and stamp.second == stamp.microsecond == 0,
                    'Invalid half-hour boundary')
            if expected_index >= start:
                result.append(stamp.hour*2 + stamp.minute//30)
    require(len(result) == stop-start, 'Calendar has missing rows')
    return np.asarray(result, dtype=np.int64)


def pack_external(external, slots):
    """Append a routing ID after original external scaling. Do not scale this ID."""
    features = np.asarray(external, dtype=np.float32)
    slots = np.asarray(slots)
    require(features.ndim == 2 and features.shape[1] == EXTERNAL_WIDTH, 'Original external width changed')
    require(slots.shape == (len(features),) and np.issubdtype(slots.dtype, np.integer), 'Bad slot shape/type')
    require(np.isfinite(features).all() and ((slots >= 0) & (slots < SLOTS_PER_DAY)).all(), 'Invalid input')
    return np.concatenate((features, slots.astype(np.float32)[:,None]), axis=1)


class TimeOfDayExternalEmbedding(nn.Module):
    """The frozen-design external projection plus one learnable 48-slot table.

    The source projection remains trainable; 'source' denotes provenance, not
    frozen weights. Zero initialization adds no RNG consumption or output skip.
    """
    def __init__(self, original_projection, output_dim, enabled=True):
        super().__init__()
        self.original_projection = original_projection
        self.tod_embedding = nn.Embedding(SLOTS_PER_DAY, output_dim,
                                         _weight=torch.zeros(SLOTS_PER_DAY, output_dim))
        self.enabled = bool(enabled)

    def forward(self, external):
        require(external.ndim == 2 and external.shape[1] == EXTERNAL_WIDTH+1,
                'Expected 38 original features followed by one time-slot ID')
        require(bool(torch.isfinite(external).all()), 'Nonfinite external features')
        slot = external[:,-1]
        require(bool(((slot >= 0) & (slot < SLOTS_PER_DAY) & (slot == slot.floor())).all()),
                'Invalid time-slot ID')
        base = self.original_projection(external[:,:EXTERNAL_WIDTH])
        if not self.enabled:
            return base
        return base + self.tod_embedding(slot.long()).to(dtype=base.dtype)


class DirectWithTimeOfDay(nn.Module):
    """No recent/weekly/daily flow anchor, no modified loss, no geographic labels."""
    def __init__(self, backbone, enabled=True):
        super().__init__()
        require(backbone.use_external and len(backbone.external_embeddings) == 4,
                'Expected the original four-stage external-conditioned RTAT')
        self.backbone = backbone
        replacements = []
        for original, dim in zip(backbone.external_embeddings, (128,256,512,1024)):
            require(isinstance(original[0], nn.Linear) and original[0].in_features == EXTERNAL_WIDTH and
                    original[0].out_features == dim, 'Original projection architecture changed')
            replacements.append(TimeOfDayExternalEmbedding(original, dim, enabled))
        backbone.external_embeddings = nn.ModuleList(replacements)

    def forward(self, flow, external):
        return self.backbone(flow, external)

    def set_time_of_day_enabled(self, enabled):
        for layer in self.backbone.external_embeddings:
            layer.enabled = bool(enabled)

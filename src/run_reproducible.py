"""Training-only input scaling, extracted from the archived implementation."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class InputScaler:
    minimum: np.ndarray
    maximum: np.ndarray

    @property
    def scale(self) -> np.ndarray:
        value = self.maximum - self.minimum
        return np.where(value == 0, 1.0, value)

    def transform(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32)
        return ((value - self.minimum) / self.scale * 2.0 - 1.0).astype(np.float32)


def fit_input_scaler(x: np.ndarray, stop: int, chunk_size: int = 1024) -> InputScaler:
    minimum = np.full(x.shape[1:], np.inf, dtype=np.float64)
    maximum = np.full(x.shape[1:], -np.inf, dtype=np.float64)
    for start in range(0, stop, chunk_size):
        block = np.asarray(x[start : min(start + chunk_size, stop)], dtype=np.float64)
        minimum = np.minimum(minimum, block.min(axis=0))
        maximum = np.maximum(maximum, block.max(axis=0))
    return InputScaler(minimum.astype(np.float32), maximum.astype(np.float32))

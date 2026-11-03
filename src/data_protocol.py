"""Reproducible chronological split protocol for the RTAT datasets.

This module is intentionally separate from ``run.py``.  It does not change or
train the supplied model; it only constructs non-overlapping, fully covering
train/validation/test slices and validates their boundaries.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SplitRange:
    start: int
    stop: int

    @property
    def size(self) -> int:
        return self.stop - self.start

    def as_slice(self) -> slice:
        return slice(self.start, self.stop)


@dataclass(frozen=True)
class ChronologicalSplit:
    train: SplitRange
    validation: SplitRange
    test: SplitRange

    def to_dict(self) -> dict[str, dict[str, int]]:
        return {name: asdict(value) | {"size": value.size} for name, value in asdict_ranges(self)}


def asdict_ranges(split: ChronologicalSplit):
    return (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    )


def bins_per_day(aggregation_minutes: int) -> int:
    if aggregation_minutes <= 0 or 1440 % aggregation_minutes:
        raise ValueError("aggregation_minutes must be a positive divisor of 1440")
    return 1440 // aggregation_minutes


def chronological_split(
    n_samples: int,
    aggregation_minutes: int,
    *,
    validation_weeks: int = 4,
    test_weeks: int = 4,
) -> ChronologicalSplit:
    """Return chronological, non-overlapping ranges that cover all samples.

    The last ``test_weeks`` are the test set, the immediately preceding
    ``validation_weeks`` are the validation set, and all earlier samples form
    the training set.  No shuffling is performed.
    """

    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if validation_weeks <= 0 or test_weeks <= 0:
        raise ValueError("validation_weeks and test_weeks must be positive")

    samples_per_week = 7 * bins_per_day(aggregation_minutes)
    validation_size = validation_weeks * samples_per_week
    test_size = test_weeks * samples_per_week
    train_stop = n_samples - validation_size - test_size
    validation_stop = n_samples - test_size
    if train_stop <= 0:
        raise ValueError("Not enough samples for the requested validation and test windows")

    split = ChronologicalSplit(
        train=SplitRange(0, train_stop),
        validation=SplitRange(train_stop, validation_stop),
        test=SplitRange(validation_stop, n_samples),
    )
    validate_split(split, n_samples)
    return split


def validate_split(split: ChronologicalSplit, n_samples: int) -> None:
    ranges = [split.train, split.validation, split.test]
    if ranges[0].start != 0 or ranges[-1].stop != n_samples:
        raise AssertionError("Split must cover the complete sample range")
    if any(part.start < 0 or part.stop <= part.start for part in ranges):
        raise AssertionError("Every split must be non-empty and within bounds")
    if not (split.train.stop == split.validation.start and split.validation.stop == split.test.start):
        raise AssertionError("Split ranges must be contiguous and non-overlapping")
    if sum(part.size for part in ranges) != n_samples:
        raise AssertionError("Split sizes must add up to n_samples")


def split_arrays(
    x: np.ndarray,
    y: np.ndarray,
    split: ChronologicalSplit,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Apply a validated split to aligned X and Y arrays without copying them."""

    if len(x) != len(y):
        raise ValueError(f"X/Y length mismatch: {len(x)} versus {len(y)}")
    validate_split(split, len(y))
    return {
        name: (x[part.as_slice()], y[part.as_slice()])
        for name, part in asdict_ranges(split)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "实验数据与图表" / "数据集",
    )
    parser.add_argument("--validation-weeks", type=int, default=4)
    parser.add_argument("--test-weeks", type=int, default=4)
    args = parser.parse_args()

    output: dict[str, dict[str, dict[str, int]]] = {}
    for dataset in ("data_nyctaxi", "data_nycbike"):
        output[dataset] = {}
        for minutes in (30, 60, 90):
            y = np.load(args.data_root / dataset / f"data_y_{minutes}.npy", mmap_mode="r")
            split = chronological_split(
                len(y),
                minutes,
                validation_weeks=args.validation_weeks,
                test_weeks=args.test_weeks,
            )
            output[dataset][str(minutes)] = split.to_dict()
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

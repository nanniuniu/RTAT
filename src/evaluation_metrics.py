"""Auditable global, activity-stratified, and local spatial RTAT evaluation.

Activity strata are fitted from training targets only. Spatial statistics use
four-neighbour row/column adjacency on the supplied array, not reconstructed
geographic polygons. Local Moran results are descriptive index-topology
diagnostics and are never used for checkpoint selection.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _metric_block(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    if mask is not None:
        error = error[np.broadcast_to(mask, error.shape)]
    else:
        error = error.reshape(-1)
    if error.size == 0:
        return {"count": 0, "mae": None, "rmse": None, "bias": None}
    return {
        "count": int(error.size),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(np.mean(error)),
    }


def fit_activity_strata(train_target: np.ndarray, high_activity_count: int = 19) -> dict:
    """Fit per-channel high/lower/inactive cell strata on training targets."""

    train_target = np.asarray(train_target, dtype=np.float64)
    if train_target.ndim != 4:
        raise ValueError("Expected training targets with shape (sample,channel,row,column)")
    if high_activity_count <= 0:
        raise ValueError("high_activity_count must be positive")
    cell_mean = train_target.mean(axis=0)
    active = cell_mean > 0
    high = np.zeros_like(active)
    for channel in range(cell_mean.shape[0]):
        active_indices = np.flatnonzero(active[channel].reshape(-1))
        if active_indices.size < high_activity_count:
            raise ValueError(
                f"Channel {channel} has {active_indices.size} active cells; "
                f"cannot select {high_activity_count} high-activity cells"
            )
        means = cell_mean[channel].reshape(-1)[active_indices]
        order = np.argsort(-means, kind="stable")
        chosen = active_indices[order[:high_activity_count]]
        high[channel].reshape(-1)[chosen] = True
    lower_active = active & ~high
    inactive = ~active
    return {
        "definition": (
            f"Per channel: top {high_activity_count} positive training-mean cells are high; "
            "remaining positive cells are lower_active; zero-mean cells are inactive."
        ),
        "high_activity_count": high_activity_count,
        "cell_mean": cell_mean,
        "active": active,
        "high": high,
        "lower_active": lower_active,
        "inactive": inactive,
    }


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted q-values with shape preserved."""

    p_values = np.asarray(p_values, dtype=np.float64)
    if np.any(~np.isfinite(p_values)) or np.any((p_values < 0) | (p_values > 1)):
        raise ValueError("p-values must be finite and lie in [0,1]")
    flat = p_values.reshape(-1)
    order = np.argsort(flat, kind="stable")
    ranked = flat[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output.reshape(p_values.shape)


def _rook_neighbours(index: int, height: int, width: int) -> np.ndarray:
    row, column = divmod(index, width)
    neighbours: list[int] = []
    if row > 0:
        neighbours.append((row - 1) * width + column)
    if row + 1 < height:
        neighbours.append((row + 1) * width + column)
    if column > 0:
        neighbours.append(row * width + column - 1)
    if column + 1 < width:
        neighbours.append(row * width + column + 1)
    return np.asarray(neighbours, dtype=np.int64)


def local_moran(
    values: np.ndarray,
    *,
    permutations: int = 999,
    seed: int = 2024,
    alpha: float = 0.05,
) -> dict[str, np.ndarray]:
    """Compute binary-rook Local Moran statistics and conditional pseudo-p values."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("values must be a finite two-dimensional array")
    if permutations <= 0:
        raise ValueError("permutations must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    height, width = values.shape
    centred = values.reshape(-1) - float(values.mean())
    variance = float(np.mean(np.square(centred)))
    statistic = np.zeros_like(centred)
    spatial_lag = np.zeros_like(centred)
    p_value = np.ones_like(centred)
    if variance > 0:
        rng = np.random.default_rng(seed)
        indices = np.arange(centred.size)
        for index in indices:
            neighbours = _rook_neighbours(index, height, width)
            spatial_lag[index] = float(np.sum(centred[neighbours]))
            statistic[index] = centred[index] * spatial_lag[index] / variance
            pool = centred[indices != index]
            exceedances = 0
            for _ in range(permutations):
                sampled = rng.choice(pool, size=len(neighbours), replace=False)
                permuted = centred[index] * float(np.sum(sampled)) / variance
                exceedances += int(abs(permuted) >= abs(statistic[index]))
            p_value[index] = (exceedances + 1) / (permutations + 1)
    q_value = benjamini_hochberg(p_value)
    cluster = np.full(centred.shape, "not_significant", dtype=object)
    significant = q_value <= alpha
    cluster[significant & (centred >= 0) & (spatial_lag >= 0)] = "high-high"
    cluster[significant & (centred < 0) & (spatial_lag < 0)] = "low-low"
    cluster[significant & (centred >= 0) & (spatial_lag < 0)] = "high-low"
    cluster[significant & (centred < 0) & (spatial_lag >= 0)] = "low-high"
    shape = (height, width)
    return {
        "statistic": statistic.reshape(shape),
        "spatial_lag": spatial_lag.reshape(shape),
        "p_value": p_value.reshape(shape),
        "q_value": q_value.reshape(shape),
        "cluster": cluster.reshape(shape),
    }


def evaluate_predictions(
    prediction: np.ndarray,
    target: np.ndarray,
    train_target: np.ndarray | None = None,
    *,
    high_activity_count: int = 19,
    spatial_permutations: int = 999,
    spatial_seed: int = 2024,
) -> tuple[dict, list[dict]]:
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {prediction.shape} versus {target.shape}")
    if prediction.ndim != 4:
        raise ValueError("Expected arrays with shape (sample, channel, row, column)")

    error = prediction.astype(np.float64) - target.astype(np.float64)
    mae_map = np.mean(np.abs(error), axis=0)
    bias_map = np.mean(error, axis=0)
    summary: dict[str, object] = {
        "shape": list(target.shape),
        "overall": _metric_block(prediction, target),
        "channels": {
            f"channel_{channel}": _metric_block(prediction[:, channel], target[:, channel])
            for channel in range(target.shape[1])
        },
        "target_value_groups": {
            "zero_target": _metric_block(prediction, target, target == 0),
            "positive_target": _metric_block(prediction, target, target > 0),
        },
    }

    strata = None
    local_results: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    if train_target is not None:
        strata = fit_activity_strata(train_target, high_activity_count=high_activity_count)
        summary["activity_strata"] = {
            "definition": strata["definition"],
            "fit_subset": "training targets only",
            **{
                name: _metric_block(prediction, target, mask[np.newaxis, ...])
                for name, mask in (
                    ("high", strata["high"]),
                    ("lower_active", strata["lower_active"]),
                    ("inactive", strata["inactive"]),
                )
            },
        }
        for channel in range(target.shape[1]):
            local_results[channel] = {
                "mae": local_moran(
                    mae_map[channel], permutations=spatial_permutations,
                    seed=spatial_seed + channel * 2,
                ),
                "bias": local_moran(
                    bias_map[channel], permutations=spatial_permutations,
                    seed=spatial_seed + channel * 2 + 1,
                ),
            }
        summary["local_spatial_error"] = {
            "support": "binary four-neighbour row/column adjacency on the supplied array",
            "permutations": spatial_permutations,
            "seed": spatial_seed,
            "multiple_testing": "Benjamini-Hochberg FDR within each dataset-channel-variable map",
            "interpretation": "index-topology diagnostic; not verified geographic contiguity",
        }

    cell_rows: list[dict] = []
    for channel in range(target.shape[1]):
        for row in range(target.shape[2]):
            for column in range(target.shape[3]):
                block = _metric_block(
                    prediction[:, channel, row, column], target[:, channel, row, column]
                )
                cell: dict[str, object] = {
                    "channel": channel,
                    "grid_row": row,
                    "grid_column": column,
                    "test_target_mean": float(np.mean(target[:, channel, row, column])),
                    "test_target_zero_rate": float(np.mean(target[:, channel, row, column] == 0)),
                    **block,
                }
                if strata is not None:
                    if strata["high"][channel, row, column]:
                        activity = "high"
                    elif strata["lower_active"][channel, row, column]:
                        activity = "lower_active"
                    else:
                        activity = "inactive"
                    cell.update({
                        "train_target_mean": float(strata["cell_mean"][channel, row, column]),
                        "activity_stratum": activity,
                        "local_moran_mae": float(local_results[channel]["mae"]["statistic"][row, column]),
                        "local_moran_mae_p": float(local_results[channel]["mae"]["p_value"][row, column]),
                        "local_moran_mae_q": float(local_results[channel]["mae"]["q_value"][row, column]),
                        "local_moran_mae_cluster": str(local_results[channel]["mae"]["cluster"][row, column]),
                        "local_moran_bias": float(local_results[channel]["bias"]["statistic"][row, column]),
                        "local_moran_bias_p": float(local_results[channel]["bias"]["p_value"][row, column]),
                        "local_moran_bias_q": float(local_results[channel]["bias"]["q_value"][row, column]),
                        "local_moran_bias_cluster": str(local_results[channel]["bias"]["cluster"][row, column]),
                    })
                cell_rows.append(cell)
    return summary, cell_rows


def save_evaluation(output_dir: Path, summary: dict, cell_rows: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "per_cell_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cell_rows[0]))
        writer.writeheader()
        writer.writerows(cell_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="NPZ containing prediction and target arrays")
    parser.add_argument("--train-target", type=Path, help="Training target .npy for spatial strata")
    parser.add_argument("--train-stop", type=int, help="Exclusive training stop index")
    parser.add_argument("--high-activity-count", type=int, default=19)
    parser.add_argument("--spatial-permutations", type=int, default=999)
    parser.add_argument("--spatial-seed", type=int, default=2024)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    arrays = np.load(args.predictions)
    train_target = None
    if args.train_target is not None:
        if args.train_stop is None:
            parser.error("--train-stop is required with --train-target")
        train_target = np.load(args.train_target, mmap_mode="r")[: args.train_stop]
    summary, cell_rows = evaluate_predictions(
        arrays["prediction"], arrays["target"], train_target,
        high_activity_count=args.high_activity_count,
        spatial_permutations=args.spatial_permutations,
        spatial_seed=args.spatial_seed,
    )
    output_dir = args.output_dir or args.predictions.parent
    save_evaluation(output_dir, summary, cell_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

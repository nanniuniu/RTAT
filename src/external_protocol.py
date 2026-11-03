"""Validate and standardize a hashed, explicitly reconstructed external bundle."""
import json
from pathlib import Path

import numpy as np

from rebuild_external_features import FEATURE_NAMES, PERIODS, file_hash


def load_external_bundle(metadata_path: Path, *, dataset: str, minutes: int,
                         sample_count: int, x_path: Path, y_path: Path):
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {"schema_version": 1, "dataset": dataset, "minutes": minutes,
                "sample_count": sample_count, "external_width": len(FEATURE_NAMES),
                "feature_names": FEATURE_NAMES, "weather_model": "ERA5",
                "timeline_status": "published_period_local_wallclock_reconstruction_not_original_timestamps"}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"External bundle mismatch for {key}")
    first, last = PERIODS[dataset]
    audit = metadata.get("flow_audit", {})
    if (audit.get("published_start") != f"{first}-01-01"
            or audit.get("published_end") != f"{last}-12-31"):
        raise ValueError("Published flow period mismatch")
    scale = next((s for s in audit.get("scales", []) if s.get("minutes") == minutes), {})
    if not scale.get("all_history_frames_match") or not scale.get("consecutive_30min_sum_matches"):
        raise ValueError("Incomplete relative timeline audit")
    directory = metadata_path.resolve().parent
    feature_path = directory / metadata["feature_file"]
    timestamp_path = directory / metadata["timestamp_file"]
    if feature_path.resolve().parent != directory or timestamp_path.resolve().parent != directory:
        raise ValueError("Bundle paths must remain in their own directory")
    for key, path in (("feature_sha256", feature_path), ("timestamp_sha256", timestamp_path),
                      ("x_sha256", x_path), ("y_sha256", y_path)):
        if metadata.get(key) != file_hash(path):
            raise ValueError(f"External bundle hash mismatch for {key}")
    features = np.load(feature_path, allow_pickle=False)
    if features.shape != (sample_count, len(FEATURE_NAMES)) or not np.isfinite(features).all():
        raise ValueError("Invalid external features")
    import csv
    with timestamp_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != sample_count:
        raise ValueError("Timestamp row count mismatch")
    targets = np.array([r["target_start_local_reconstructed"] for r in rows], dtype="datetime64[m]")
    weather = np.array([r["weather_valid_local"] for r in rows], dtype="datetime64[m]")
    expected_start = np.datetime64(f"{first}-01-29T00:00", "m")
    expected_targets = expected_start + np.arange(sample_count)*np.timedelta64(minutes, "m")
    ages = (targets-weather)/np.timedelta64(1, "m")
    if not np.array_equal(targets, expected_targets) or np.any(ages < 60) or np.any(ages > 180):
        raise ValueError("Invalid reconstructed target grid or future/stale weather")
    weekdays = np.array([t.astype(object).weekday() for t in targets])
    if (not np.array_equal(features[:, 2], weekdays >= 5)
            or not np.array_equal(features[:, -7:], np.eye(7)[weekdays])
            or not np.all(features[:, 3:-7].sum(1) == 1)):
        raise ValueError("Invalid reconstructed calendar/weather encoding")
    return np.asarray(features, dtype=np.float32), metadata


def standardize_external(features: np.ndarray, train_stop: int):
    """Fit only two continuous columns on training samples; do not scale one-hot."""
    if not 0 < train_stop <= len(features):
        raise ValueError("Invalid external training range")
    mean = features[:train_stop, :2].mean(0, dtype=np.float64)
    scale = features[:train_stop, :2].std(0, dtype=np.float64)
    scale = np.where(scale > 1e-8, scale, 1.0)
    result = features.copy()
    result[:, :2] = (result[:, :2]-mean)/scale
    return result, {"columns": [0, 1], "mean": mean.tolist(), "std": scale.tolist(),
                    "fit_range": [0, train_stop], "method": "training-only population z-score"}

"""Audit the released time-series and build a separate retrospective ERA5 protocol.

Absolute dates and local wall-clock bins are reconstructed conventions, not
recovered original timestamps. No original array is overwritten. Reanalysis
has retrospective availability: lagging valid times does not make it an
operational weather forecast or an exact reproduction of the released weather.
"""
from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
from zoneinfo import ZoneInfo

import numpy as np

PERIODS = {"nyctaxi": (2013, 2015), "nycbike": (2018, 2020)}
WMO_CODES = (0, 1, 2, 3, 45, 48, 51, 53, 55, 56, 57, 61, 63, 65,
             66, 67, 71, 73, 75, 77, 80, 81, 82, 85, 86, 95, 96, 99)
FEATURE_NAMES = (["temperature_2m_celsius", "wind_speed_10m_m_s", "weekend"]
                 + [f"weather_wmo_{code}" for code in WMO_CODES]
                 + [f"weekday_{day}" for day in range(7)])
PAPER = "https://arxiv.org/html/2111.03459v1#S4.SS1"
WEATHER_DOC = "https://open-meteo.com/en/docs/historical-weather-api"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def lags(minutes: int) -> list[int]:
    daily = 1440 // minutes
    return [28*daily, 21*daily, 14*daily, 7*daily,
            4*daily, 3*daily, 2*daily, daily, 4, 3, 2, 1]


def reconstruct_raw(x: np.ndarray, y: np.ndarray, minutes: int) -> np.ndarray:
    """Exact relative-index consistency test over every released historical frame."""
    offsets = lags(minutes)
    prefix = offsets[0]
    if len(x) < prefix:
        raise ValueError("Insufficient samples to reconstruct complete historical prefix")
    if len(x) != len(y) or x.shape[1:] != (12, *y.shape[1:]):
        raise ValueError("Unexpected flow shapes")
    raw = np.concatenate((x[:prefix, 0], y), axis=0)
    for start in range(0, len(y), 256):
        stop = min(start + 256, len(y))
        for frame, lag in enumerate(offsets):
            expected = raw[prefix + start - lag:prefix + stop - lag]
            if not np.array_equal(x[start:stop, frame], expected):
                raise ValueError(f"History mismatch at sample block {start}, frame {frame}, lag {lag}")
    if not np.isfinite(raw).all() or np.any(raw < 0):
        raise ValueError("Non-finite or negative flow counts")
    return raw


def audit_flow(data_root: Path, dataset: str) -> dict:
    first, last = PERIODS[dataset]
    expected_days = sum(366 if calendar.isleap(year) else 365 for year in range(first, last+1))
    scales = []
    raw30 = None
    for minutes in (30, 60, 90):
        directory = data_root / f"data_{dataset}"
        xp, yp = directory / f"data_x_{minutes}.npy", directory / f"data_y_{minutes}.npy"
        x, y = np.load(xp, mmap_mode="r"), np.load(yp, mmap_mode="r")
        raw = reconstruct_raw(x, y, minutes)
        if len(raw) != expected_days * (1440 // minutes):
            raise ValueError(f"{dataset}/{minutes}: rows do not cover the published period")
        if minutes == 30:
            raw30 = raw
        else:
            factor = minutes // 30
            expected = raw30.reshape(-1, factor, *raw30.shape[1:]).sum(axis=1)
            if not np.array_equal(raw, expected):
                raise ValueError(f"{dataset}/{minutes}: aggregation differs from 30-min consecutive sums")
        scales.append({"minutes": minutes, "sample_count": len(y), "raw_rows": len(raw),
                       "lags": lags(minutes), "all_history_frames_match": True,
                       "consecutive_30min_sum_matches": True,
                       "x_sha256": file_hash(xp), "y_sha256": file_hash(yp)})
    return {"dataset": dataset, "published_start": f"{first}-01-01",
            "published_end": f"{last}-12-31", "published_days": expected_days,
            "date_source": PAPER, "absolute_timestamp_status": "reconstructed_from_published_period",
            "bin_convention": "fixed 24h local wall-clock day; bins labelled by start",
            "original_timezone_and_missing_bin_policy_verified": False, "scales": scales}


def audit_legacy_calendar(data_root: Path) -> dict:
    path = data_root / "data_nycbike" / "ext_features_30.npy"
    values = np.asarray(np.load(path, allow_pickle=True), dtype=np.float64)
    start = datetime(2018, 1, 1)
    dates = [start + timedelta(minutes=30*i) for i in range(len(values))]
    weekdays = np.array([t.weekday() for t in dates])
    incorrect_flag = np.array([int(t.day > 4) for t in dates])
    correct_weekend = np.array([int(t.weekday() >= 5) for t in dates])
    return {"external_sha256": file_hash(path), "rows": len(values),
            "weekday_match_2018_2020_monday_zero": bool(np.array_equal(values[:, -7:].argmax(1), weekdays)),
            "third_column_equals_day_of_month_greater_than_four": bool(np.array_equal(values[:, 2], incorrect_flag)),
            "third_column_weekend_mismatched_rows": int(np.count_nonzero(values[:, 2] != correct_weekend)),
            "taxi_same_external_hash": file_hash(data_root / "data_nyctaxi" / "ext_features_30.npy") == file_hash(path)}


def download_year(year: int, cache: Path) -> Path:
    path = cache / f"era5_nyc_{year}_utc.json"
    if path.exists():
        print(f"Cached weather {year}", flush=True)
        return path
    query = {"latitude": 40.7128, "longitude": -74.0060,
             "start_date": f"{year}-01-01", "end_date": f"{year+1}-01-02",
             "hourly": "temperature_2m,wind_speed_10m,weather_code", "models": "era5",
             "timezone": "GMT", "wind_speed_unit": "ms"}
    url = "https://archive-api.open-meteo.com/v1/archive?" + urlencode(query)
    print(f"Downloading weather {year}", flush=True)
    payload = None
    for attempt in range(3):
        try:
            with urlopen(url, timeout=40) as response:
                payload = json.load(response)
            break
        except (TimeoutError, OSError) as error:
            print(f"Weather {year}, attempt {attempt+1}: {type(error).__name__}", flush=True)
            if attempt == 2:
                raise
            time.sleep(1)
    if "hourly" not in payload:
        raise ValueError(f"Weather API error: {payload}")
    record = {"request_url": url, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
              "response": payload}
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    return path


def weather_lookup(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Convert explicit UTC hours to NY wall time; choose earlier UTC on DST folds."""
    zone = ZoneInfo("America/New_York")
    rows = {}
    for path in sorted(paths):
        hourly = json.loads(path.read_text(encoding="utf-8"))["response"]["hourly"]
        for stamp, temp, wind, code in zip(hourly["time"], hourly["temperature_2m"],
                                           hourly["wind_speed_10m"], hourly["weather_code"]):
            utc = datetime.fromisoformat(stamp).replace(tzinfo=timezone.utc)
            local = utc.astimezone(zone).replace(tzinfo=None)
            if local not in rows:
                rows[local] = (temp, wind, code)
    times = np.array(sorted(rows), dtype="datetime64[m]")
    values = np.asarray([rows[t.astype(object)] for t in times], dtype=np.float64)
    if not np.isfinite(values).all() or not np.isin(values[:, 2], WMO_CODES).all():
        raise ValueError("Missing weather or unsupported WMO code; no silent filling allowed")
    return times, values


def make_features(targets: np.ndarray, times: np.ndarray, weather: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use latest hour at least 60 min before target start, never target-period weather."""
    cutoff = targets - np.timedelta64(60, "m")
    indices = np.searchsorted(times, cutoff, side="right") - 1
    if np.any(indices < 0):
        raise ValueError("Missing historical weather before first prediction")
    selected = times[indices]
    ages = (targets - selected) / np.timedelta64(1, "m")
    if np.any(ages < 60) or np.any(ages > 180):
        raise ValueError("Weather valid-time age exceeds explicit permitted range 60..180 min")
    result = np.zeros((len(targets), len(FEATURE_NAMES)), dtype=np.float32)
    result[:, :2] = weather[indices, :2]
    weekdays = np.array([t.astype(object).weekday() for t in targets], dtype=np.int64)
    result[:, 2] = weekdays >= 5
    mapping = {code: column for column, code in enumerate(WMO_CODES)}
    codes = np.array([mapping[int(code)] for code in weather[indices, 2]])
    result[np.arange(len(targets)), 3 + codes] = 1
    result[np.arange(len(targets)), 3 + len(WMO_CODES) + weekdays] = 1
    return result, selected


def build_bundle(output: Path, audit: dict, paths: list[Path]) -> None:
    times, values = weather_lookup(paths)
    dataset = audit["dataset"]
    for scale in audit["scales"]:
        minutes = scale["minutes"]
        start = np.datetime64(audit["published_start"], "m") + np.timedelta64(scale["lags"][0] * minutes, "m")
        targets = start + np.arange(scale["sample_count"]) * np.timedelta64(minutes, "m")
        features, selected = make_features(targets, times, values)
        directory = output / f"data_{dataset}"
        directory.mkdir(parents=True, exist_ok=True)
        feature_path = directory / f"external_{minutes}.npy"
        stamp_path = directory / f"timestamps_{minutes}.csv"
        metadata_path = directory / f"external_{minutes}.json"
        if any(p.exists() for p in (feature_path, stamp_path, metadata_path)):
            raise FileExistsError(f"Refusing to overwrite bundle for {dataset}/{minutes}")
        np.save(feature_path, features, allow_pickle=False)
        with stamp_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["sample_index", "target_start_local_reconstructed", "weather_valid_local"])
            writer.writerows((i, str(target), str(hour)) for i, (target, hour) in enumerate(zip(targets, selected)))
        meta = {"schema_version": 1, "dataset": dataset, "minutes": minutes,
                "sample_count": len(targets), "external_width": len(FEATURE_NAMES),
                "feature_names": FEATURE_NAMES, "feature_file": feature_path.name,
                "feature_sha256": file_hash(feature_path), "timestamp_file": stamp_path.name,
                "timestamp_sha256": file_hash(stamp_path), "x_sha256": scale["x_sha256"],
                "y_sha256": scale["y_sha256"], "flow_audit": audit,
                "timeline_status": "published_period_local_wallclock_reconstruction_not_original_timestamps",
                "weather_source": WEATHER_DOC, "weather_model": "ERA5",
                "representative_requested_lat_lon": [40.7128, -74.0060],
                "weather_spatial_scope": "single city-level point; not a grid georeferencing claim",
                "weather_requests": [json.loads(p.read_text(encoding="utf-8"))["request_url"] for p in paths],
                "weather_response_locations": [
                    {k: json.loads(p.read_text(encoding="utf-8"))["response"].get(k)
                     for k in ("latitude", "longitude", "elevation", "hourly_units")} for p in paths],
                "weather_raw_sha256": {p.name: file_hash(p) for p in paths},
                "weather_valid_time_policy": "backward as-of at target start minus 60 minutes",
                "weather_valid_age_minutes_min_max": [float(((targets-selected)/np.timedelta64(1,"m")).min()),
                                                       float(((targets-selected)/np.timedelta64(1,"m")).max())],
                "dst_policy": "earlier UTC on repeated wall hour; backward hold across absent spring hour",
                "availability": "retrospective reanalysis; not operationally available at forecast origin",
                "calendar_policy": "target weekday Monday=0; weekend Saturday/Sunday",
                "target_first": str(targets[0]), "target_last": str(targets[-1]),
                "original_weather_reproduced": False}
        metadata_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Built {dataset}/{minutes}: {features.shape}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--download-weather", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    audits = []
    for dataset in PERIODS:
        print(f"Auditing every historical frame and aggregation: {dataset}", flush=True)
        audits.append(audit_flow(args.data_root, dataset))
    report = {"flow": audits, "legacy_external": audit_legacy_calendar(args.data_root)}
    (args.output_root / "timeline_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["legacy_external"], indent=2), flush=True)
    if args.download_weather:
        cache = args.output_root / "weather_raw"
        cache.mkdir(exist_ok=True)
        # Each annual request includes two next-year days for local year-end coverage.
        years = (2013, 2014, 2015, 2018, 2019, 2020)
        with ThreadPoolExecutor(max_workers=2) as pool:
            paths = dict(zip(years, pool.map(lambda year: download_year(year, cache), years)))
        for audit in audits:
            first, last = PERIODS[audit["dataset"]]
            build_bundle(args.output_root, audit, [paths[y] for y in range(first, last+1)])


if __name__ == "__main__":
    main()

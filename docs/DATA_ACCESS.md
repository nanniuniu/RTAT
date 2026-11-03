# Data acquisition and preparation

## Flow benchmarks: obtain from the original release

Source: https://github.com/yanxiao1930/ProSTformer_code

The source repository README links the processed datasets:

- NYCBike: https://pan.baidu.com/s/1i4N3xuqFrppRgg0Gd6yHUQ (public extraction code: `1234`).
- NYCTaxi: https://pan.baidu.com/s/1dW3UyYJKL-YLgw7a6ujuLw (public extraction code: `1234`).

These download links are maintained by the benchmark provider; consult the source repository for updates. The processed arrays are derived from NYC TLC and Citi Bike records. Use the processed benchmark arrays to match the supplied sample counts, layouts and experimental input definitions.

Required per dataset: `data_x_30.npy`, `data_y_30.npy`, `data_x_60.npy`, `data_y_60.npy`, `data_x_90.npy`, and `data_y_90.npy`. Expected hashes and sample counts are in `data_metadata/expected_data_sha256.json`. Use `prepare_data.py` to verify these before training.

## External features and calendars: included

`external_data/data_nyctaxi/` and `external_data/data_nycbike/` contain matching `external_30/60/90.npy`, `timestamps_30/60/90.csv`, and schema metadata. No API call is required for normal preparation or training. Attribution, modifications and upstream license links are in `external_data/ATTRIBUTION.md`.

```bash
python prepare_data.py --flow-root benchmark --external-root external_data --output-root data
```

The output must be a new directory. `--link-flow` optionally creates hard links on the same filesystem; use these inputs as read-only. The copied metadata is the portable version in `data_metadata/`.

## Optional feature reconstruction

The original reconstruction implementation is retained:

```bash
python src/rebuild_external_features.py --data-root benchmark --output-root rebuilt_external --download-weather
```

This audits both datasets and all intervals, and requires the original `ext_features_30.npy` files for its legacy-calendar diagnostic in addition to the flow arrays. It requests retrospective ERA5 weather through Open-Meteo; it does not recover original calendar labels or released weather. For the documented input version, use the matched arrays in `external_data/`. When regenerating inputs, compare them with the expected values in `data_metadata/` and record the feature version used for the experiment. API access remains subject to Open-Meteo's service terms and limits.

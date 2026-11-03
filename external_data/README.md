# External features and reconstructed calendars

Included: two datasets × three observation intervals, with one 38-column `.npy` array, one reconstructed calendar `.csv`, and portable `.json` schema per profile. Values are unstandardized; training fits continuous-feature scaling using training observations only.

The first two columns are temperature and wind; the remaining columns encode weekend, 28 weather codes and seven weekdays. Exact feature names, sample counts, target periods and source descriptions are in the metadata. The preparation and loading code validates these inputs against the expected values in `data_metadata/`.

Weather data by [Open-Meteo.com](https://open-meteo.com/), based on ERA5 reanalysis. See [attribution and data terms](ATTRIBUTION.md). The arrays are derived using retrospective city-level weather and reconstructed wall-clock labels. They are not raw trip data, spatially varying grid-cell weather, operational forecasts or recovered original timestamps.

To prepare these alongside the original processed flow benchmark, run from repository root:

```bash
python prepare_data.py --flow-root benchmark --external-root external_data --output-root data
```

No trained weights or run records are included in this directory.

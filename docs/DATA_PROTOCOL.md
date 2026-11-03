# Data and experimental protocol

## Spatial and temporal layout

All six benchmark settings use the supplied 12 × 16 grid and two endpoint-count channels. No geographic coordinates are inferred from array indices. Inputs contain 12 selected maps, ordered from earliest to most recent. The lag sets, measured in observation intervals, are:

- 30 minutes: 1344, 1008, 672, 336, 192, 144, 96, 48, 4, 3, 2, 1.
- 60 minutes: 672, 504, 336, 168, 96, 72, 48, 24, 4, 3, 2, 1.
- 90 minutes: 448, 336, 224, 112, 64, 48, 32, 16, 4, 3, 2, 1.

The latest input is position 11. Positions 8 through 11 drive the recent-observation gate. The temporal patch size is three selected maps; relative patch positions do not represent uniformly spaced elapsed times.

## Splits and scaling

The final four weeks form the test set; the preceding four weeks form validation; all earlier supervised samples form training. Exact boundaries and sample counts are in each `configs/*.json`. For Taxi30, these are `[0,48528)`, `[48528,49872)`, and `[49872,51216)`.

Input min/max values are fitted separately for each history-position/channel/cell using only training observations. Inputs are mapped to [-1,1] using these fitted values, without clipping held-out values. A constant training dimension uses an effective scale of one. Targets remain raw counts. The external temperature and wind columns use training-only population mean and SD; binary and one-hot columns are unchanged.

The output residual inverse-transforms the latest observation with the matching training scaler. Its gate is `2 * sigmoid(Conv1x1(GELU(Conv1x1(recent))))`, with channel widths 8→16→2. Zero-initializing the final convolution starts the multiplier at one. Each stage has a zero-initialized 48-slot ToD embedding. There is no output clipping.

## Calendar and weather

Absolute timestamps are reconstructed from published periods and a fixed 24-hour local wall-clock convention. The first supervised target starts on 29 January 2013 for NYCTaxi and 29 January 2018 for NYCBike. Slot IDs are `hour * 2 + minute // 30`; they are not standardized. These are reconstructed calendar labels, not recovered original timestamps.

The 38 external columns are temperature, wind, weekend, 28 weather-code indicators, and seven weekday indicators. Weather features use retrospective ERA5 reanalysis through the historical-weather API. Backward alignment uses the latest weather valid time at least 60 minutes before the reconstructed target start. On repeated local hours the earlier UTC observation is retained; absent spring hours use backward alignment. A lagged reanalysis value is not a verified operational weather forecast.

The preferred reproduction inputs are the exact hashed external arrays and calendar files. `src/rebuild_external_features.py` retains the original audit and rebuilding implementation:

```bash
python src/rebuild_external_features.py --data-root /path/to/original_benchmark --output-root rebuilt_external --download-weather
```

This command audits all six flow arrays, also reads the original `ext_features_30.npy` files for the legacy-calendar diagnostic, and retrieves weather when requested. A separately retained original weather cache can be placed in `rebuilt_external/weather_raw/` to reuse preserved responses. Newly retrieved API data can differ from the archived inputs; compare hashes before treating them as the same experiment. No network request occurs during ordinary training.

## Training and selection

Training uses raw-count MSE, AdamW, batch size 128, weight decay 0.08, gradient clipping at 1.0, five warmup epochs and a 500-epoch cosine schedule. The default learning rate is 0.0002. Dropout and attention dropout are 0.4 at 30 minutes and 0.2 at 60/90 minutes; stochastic depth is 0.3. Taxi30 stops by 300 epochs; the other five profiles stop by 500. Validation MSE selects the checkpoint, with patience 60.

CUDA runs use mixed precision with initial gradient scale 256 and fused AdamW. Validation and final testing run without autocast. Overflowed updates skip the scheduler step. The final test loader is created after checkpoint selection and validation reload verification. Hashing source files and arrays does not fit parameters or use test metrics for checkpoint selection.

A complete three-seed aggregate requires seeds 0, 1 and 2. Direct runs, fine-tuned runs and structural ablations are grouped separately. Historical run snapshots and prediction records are not distributed. Use the included training and aggregation interfaces to produce and summarize new runs.

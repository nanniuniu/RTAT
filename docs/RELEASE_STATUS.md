# Project guide

Repository: https://github.com/nanniuniu/RTAT

## Model and experiment workflows

The implementation provides the RTAT backbone, time-of-day conditioning and gated recent-observation residual. Six dataset/interval profiles support NYCTaxi and NYCBike at 30, 60 and 90 minutes. Command-line entry points cover training, explicit weight-only fine-tuning, path/component ablations, prediction, spatial evaluation and multi-seed aggregation.

## Data preparation

Obtain the processed flow benchmark from the ProSTformer data release linked in `DATA_ACCESS.md`. Combine it with the matched external/calendar inputs in `external_data/` using `prepare_data.py`. The preparation and loading code validates input integrity, sample alignment and feature schemas.

## Training outputs

Each training run generates its configuration, fitted scalers, epoch history, validation-selected model checkpoint, validation predictions and final test evaluation. Use the checkpoint and scalers from the same run for prediction. `summarize.py` aggregates completed runs and reports sample standard deviations, seed coverage and matched-seed comparisons.

The main README contains installation and command examples. `VERIFICATION.md` describes CPU functional checks; `DATA_PROTOCOL.md` describes the input and experimental protocol. Source attribution and applicable third-party terms are recorded in `THIRD_PARTY_NOTICES.md` and `external_data/ATTRIBUTION.md`.

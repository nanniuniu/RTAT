# Verification checks

The repository includes CPU functional tests, a synthetic forward example and input-validation entry points.

## CPU protocol and model tests

```bash
python -m unittest discover -s tests -v
```

The tests check:

- Chronological split boundaries and historical lag sets for six dataset/interval profiles.
- Training-only input/external scaling and unscaled time-slot IDs.
- Parameter counts and forward execution for all five path variants, plus gradients through the full model's gate and time-of-day embeddings.
- Weight transfer while preserving target-interval anchor buffers.
- Seed coverage, sample standard deviations and duplicate-seed handling in aggregation.
- Validation-based checkpoint selection, strict reload and one final test evaluation, using a small synthetic training model to exercise the lifecycle.

## Synthetic forward example

```bash
python examples/forward.py
```

This example constructs the full 64,812,306-parameter model with random weights and synthetic inputs, then checks a finite output of shape `[1,2,12,16]`. It checks model execution; benchmark accuracy is evaluated separately through the training/evaluation workflow.

## Input validation

After data preparation, validate one profile with:

```bash
python train.py --data-root data --dataset nyctaxi --minutes 30 --validate-only
```

The preparation and loading code checks expected input values, shapes, reconstructed calendar alignment and external-feature encodings. `docs/DATA_PROTOCOL.md` describes the reconstructed labels and lagged retrospective weather protocol.

During training, validation selects the checkpoint; the final test evaluation follows checkpoint reload and validation verification. Each run records its explicit initialization and configuration, and completed runs can be summarized with matched seed groups.

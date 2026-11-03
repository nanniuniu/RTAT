# Validation scope

The model, training, prediction, data preparation and aggregation implementations are retained from the checked RTAT review package. This public tree adds documentation, a batch command runner and a synthetic forward example. Existing model/training code is not rewritten for publication. The original private engineering records and experiment files remain archived separately.

The included CPU tests cover six chronological split/lag profiles, training-only scalers, slot encoding, all five path model counts and forward passes, gradients through gate/ToD, weight-transfer buffer handling, duplicate/missing seeds, and the validation-selection/final-test lifecycle using a tiny synthetic training model. These checks are functional checks, not new benchmark training runs.

The checks described here establish code functionality and input integrity. They do not constitute new benchmark training runs or certify exact numerical recreation of every historical manuscript result. Trained weights and historical prediction records are not distributed in this repository.

New training generates its own validation-selected checkpoint and one final test evaluation. The weight-transfer interface records the explicitly supplied source; it does not infer historical fine-tuning provenance. Component-removal flags define fresh experiments and do not promise equality with unspecified historical single-run ablations. Independent baseline implementations are not bundled.

The preparation and loading code validates included feature/calendar data against the expected values in `data_metadata/`. Reconstructed labels and lagged retrospective weather describe the adopted input protocol; they do not certify recovered original timestamps or real-time weather availability.

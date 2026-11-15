# Data dictionary

## Real signal sample

`data/sample/real_signal_sample.npz` can be opened with
`numpy.load(path, allow_pickle=False)`. There are 144 rows. Each row is the
first cached window of a selected acquisition run. At most ten distinct runs
are selected per power/class stratum using selection seed 20261114; the two
source healthy strata contain only five and nine available parent runs.

| Field | Meaning |
|---|---|
| `X` | Float32 array `[144,3,128]`, before source normalization; channel order rpm, iq, id |
| `rpm_raw` | Float32 `[144,128]`, unstandardized mechanical speed in r/min for phase integration |
| `fs_hz` | Per-window sampling rate, 323 or 333 Hz, as recorded in the parent cache |
| `y`, `class_names` | Labels 0 health/zero offset, 1 up, 2 down, 3 left, 4 right |
| `power` | Nominal motor rating in W: 200, 400, 1500 |
| `speed` | Recorded nominal speed condition in r/min |
| `offset_mm` | Prescribed support-offset magnitude in mm |
| `domain` | Source (200/400 W) or target (1500 W) |
| `trial`, `fold` | Original protocol grouping fields, preserved unchanged |
| `run_id` | Release-local acquisition-run identifier; each appears once |
| `workbook_id` | Release-local container group, preserving shared-container membership |
| `window_start`, `window_index` | Original cached window location metadata |
| `source_row_index` | Zero-based row in the parent derived cache |

The iq/id values retain the controller units stored in the parent cache; this
release does not add a physical-current calibration. File paths and original
workbook names are omitted. `sample_index.csv` provides readable row metadata.
`selection.json` documents the counts and selection rule.

The role rules mirror the original trainer: all 200-W samples are source
training; 400-W trial `fold+1` is source validation; the other 400-W trials are
source training. At 1500 W, folds `fold`, `(fold+1)%3`, `(fold+2)%3` are adaptation,
test and probe respectively. Runs are disjoint between these target roles,
but different roles may still share workbook containers. The original result
tables include a separate workbook-level sensitivity analysis.

The sample labels are included for inspection. The demo masks target-adaptation
labels and fits normalization only on its source-training subset. This is an
illustrative subset, not a replacement for the complete experimental protocol.

## Recorded result tables

All CSV score values are fractions in [0,1]; multiply by 100 for percentages.

- `extended_metric_summary.csv`: nine methods by three endpoints; means, sample
  SDs, run-based confidence intervals and workbook-based sensitivity intervals.
- `oof_seed_metrics.csv`: 81 rows, with three model seeds per method/endpoint.
  Three folds have already been combined within each seed.
- `extended_matched_paired_tests_24.csv`: EPCDA minus eight controls for three
  endpoints; stored paired differences, intervals and adjusted p-values.
- `epcda_five_class_confusion_summary.csv`: 25 confusion cells, averaged across
  three model seeds. Mean counts can be fractional; they are not pooled counts.
- `target_supervised_summary.json`: a separate target-supervised calibration
  with three seed-level outputs and metric summaries.

Endpoints: `five_class`; `fault_only_four_direction` (conditional on nonzero
offset); and `binary_fault` (zero versus nonzero offset). The target-supervised
JSON uses `five_class_bacc`, `four_direction_bacc`, and `binary_bacc`.

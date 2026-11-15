# Source and version correspondence

Prepared on 2026-11-14 from the existing `EPCDA_failure_analysis_revision_20260812`
project and its related reproducibility archive.

| Released item | Existing source within those projects | Packaging change |
|---|---|---|
| `src/epcda_model.py` | `experiments/controlled_calibration/epcda_model.py` | None; model equations and defaults retained |
| `tests/test_epcda_model.py` | `03_reproducibility/code/test_epcda_model.py` | Added a relative import path for standalone use |
| `src/data_utils.py` | Normalizer, build_splits and make_batch in `experiments/controlled_calibration/target_supervised/train_epcda_suite.py` | Extracted portable NumPy utilities and added input checks |
| Signal sample | `experiments/controlled_calibration/real_run_dataset_l128.npz` | Selected rows; remapped run/container names; retained signal values and grouping fields |
| Three comparison CSVs | `experiments/dmaster_import/factorization_addon/` | Files copied unchanged |
| Confusion CSV | `experiments/dmaster_import/original_suite/` | File copied unchanged |
| Target-supervised JSON | `experiments/controlled_calibration/target_supervised/summary.json` | File copied unchanged |
| Demo and summary scripts | Created for this release | Execution example and arithmetic checks of recorded results |

The imported comparison tables trace back to the existing reproducibility
archive's factorization and original-suite analysis outputs. The 19.87% and
54.34% headline values also appear in the provided `1115.pdf` manuscript.

Both the related MSSP manuscript and `1115.pdf` use this compact-model evaluation
line. The package does not assert that every equation or experiment in the
revised 26-page PDF has been independently reproduced. It includes selected
existing materials and explicitly scoped checks.

The separate TCE manuscript describes a larger EPCDA implementation and a
different data/evaluation protocol. This release must not be described as the
code reproducing its 84.6% mean accuracy result.

No model fitting or new scientific measurement was performed to assemble this
release. It includes no manuscript PDFs, submission correspondence, raw workbook
archive, pretrained weights, or unverified bibliographic metadata.

Repository naming: the root directory and README heading use the title of the
provided PR-targeted manuscript. This naming update does not change the code,
sample values, recorded results, or stated release scope.

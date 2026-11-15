# Vision-Inspired Order Maps for Cross-Drive Misalignment Recognition

Selected code and data for **Vision-Inspired Order Maps for Cross-Drive
Misalignment Recognition**. This package contains the existing
compact EPCDA implementation and recorded results from the related cross-motor
evaluation project. It is a partial release, not a complete training archive.

## Contents

- `src/epcda_model.py`: original PyTorch model, including the phase-indexed
  order front end, complex coherency, global/local encoder, two projections,
  classifier and auxiliary heads. 82,989 total parameters and 77,077 in the
  prediction path.
- `src/data_utils.py`: portable normalization, role selection and label-masking
  utilities adapted from the original trainer.
- `data/sample/`: 144 real 128-point windows from 144 distinct acquisition
  runs, covering 200/400/1500 W and all five state labels.
- `data/results/`: original nine-method comparison, per-seed metrics, paired
  comparisons, confusion summary and target-supervised reference summary.
- `scripts/`: a real-input model demonstration and numerical result-table
  recomputation.
- `tests/`: existing model tests, including phase integration and gradient
  reversal checks. Test-generated tensors are test fixtures, not study data.

## Run

Use a Python environment compatible with the selected PyTorch version. From
the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/reproduce_results.py
python scripts/run_demo.py
python -m unittest discover -s tests
```

For the test command, the model is imported through a small local test loader.
No GPU or network access is required after installing dependencies.

`reproduce_results.py` uses only the Python standard library. It recomputes
all 27 method/endpoint means and sample standard deviations from the per-seed
tables, checks 24 paired mean differences, and verifies the five-class EPCDA
score from the confusion matrix. Expected headline values are 19.87% balanced
accuracy for cross-drive adaptation and 54.34% for the separate target-supervised
reference. Confidence intervals and p-values remain the stored original outputs.

`run_demo.py` uses randomly initialized weights and saves feature tensors under
`results/demo/`. It verifies interfaces, finite outputs and parameter counts;
it does not reproduce trained-model accuracy. The example normalization is
fitted only on the released source-training subset. Physical rpm is kept
separate from standardized input, and target-adaptation labels are masked.

## Data and version scope

The released signals are a deterministic stratified selection from an existing
8,540-window derived cache. They are not synthetic signals or a new acquisition.
This small selection is for inspecting the interface, not for obtaining the
paper's performance estimates. Full raw workbooks, trained checkpoints and the
complete training/analysis pipeline are not included.

This is the **77,077-parameter compact model and 323/333 Hz evaluation line**.
It is not the 2.91-million-parameter, 10 kHz implementation described by the
separate TCE manuscript. See `docs/PROVENANCE.md` and `docs/DATA.md` for details.

The manuscript is not bundled. No publication status, DOI, or final author
list is assigned by this code release. No reuse license has been selected.

中文说明见 `README_中文.md`。

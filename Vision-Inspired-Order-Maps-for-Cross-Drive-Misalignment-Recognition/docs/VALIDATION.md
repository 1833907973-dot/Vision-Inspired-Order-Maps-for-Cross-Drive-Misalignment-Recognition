# Release verification

Verified on 2026-11-14 using Python 3.14.4, NumPy 2.4.4 and PyTorch 2.12.1+cu126.
The demonstration ran on CPU. These are the packaging-test versions, not a claim
about the original training environment.

- Ten existing model tests passed.
- Real-input forward demonstration passed: finite outputs; 82,989 total and
  77,077 deployment-path parameters.
- Sample contains 144 unchanged real signal windows from distinct parent runs.
- All 27 method/endpoint means and sample SDs matched the recorded tables.
- All 24 paired mean differences matched the recorded comparison table.
- Confusion-derived five-class BAcc matched the EPCDA summary.
- Target-supervised means and SDs matched the three stored seed-level results.
- Reference values recovered: 19.86651762183568% and 54.342758051469154%.

Training was not rerun. No new performance estimates were generated. The
random-weight demonstration does not evaluate diagnostic accuracy.

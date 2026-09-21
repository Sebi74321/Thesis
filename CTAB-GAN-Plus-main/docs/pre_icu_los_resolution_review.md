# pre_icu_los_days: distribution and implementation review

## Dataset correction adopted after this review

On 2026-09-21, the user selected the explicit assumption that negative durations
represent administrative timing errors and should be treated as immediate ICU
admission (0 days). Both WiDS.csv and WiDS_cleaned.csv now replace only negative
pre_icu_los_days cells with zero. No rows are deleted, and no other features,
positive durations, generated datasets, or generator settings are changed.

The original files are preserved under Real_Datasets/.backups/pre_icu_los_nonnegative/.
Real_Datasets/pre_icu_los_nonnegative_correction.json records counts, before/after
SHA-256 hashes and the backup filenames. That folder is excluded from Git.
Upload the corrected CSVs separately when using a remote cluster. Existing runs
retain their previous data hashes: do not bypass their mismatch checks. Use the
matching original backup for old-run reproduction or start a new corrected-data
experiment. This is a documented assumption, not reconstruction of true timestamps.

## Minute-grid postprocessing adopted after this review

At the user's request, newly fitted CTAB-GAN+, CTGAN and DP-CGAN adapters now
round generated pre_icu_los_days to `round(days * 1440) / 1440`. The values remain
in days, not integer minutes or a fixed number of decimal places. Half-minute
ties use nearest-even rounding, consistent with existing numeric postprocessing.
This is a sampling-time correction only: training representation is unchanged.

The shared support guard still leaves a value unchanged if rounding would move
it farther beyond the fitted bounds. A fitted endpoint within 0.001 seconds of
the minute grid is recognized as that grid endpoint to allow CSV serialization
error. No blanket clipping is introduced. Raw samples remain available through
the adapters' last_raw_sample, and postprocessing diagnostics count changes,
guarded rows and raw support violations. Existing CSVs are not automatically
rewritten, and old checkpoints without the new constraint metadata retain their
original decimal-rounding behavior.

The measurements and recommendations below describe the **pre-correction data
and implementation**, before the dataset and minute-grid changes above.

Inspected 2026-09-21 using the local WiDS_cleaned.csv and its seed-42
60/20/10/10 mortality-stratified split. No GAN or postprocessing settings changed.
Synthetic WiDS results from the cluster were not available locally.

## Training-data evidence

All 51,233 training values (including nonzero values alone) align to whole minutes
within 0.001 seconds. Maximum observed distance from the minute grid is 0.000384
seconds, consistent with decimal serialization. Only 19.71% of nonzero values
align to five-minute steps. The defensible grid is minutes, not whole days.
In the original, uncleaned WiDS.csv, 91,712 of 91,713 rows align within this tolerance.

| Training interval | Rows | Fraction |
|---|---:|---:|
| Negative | 210 | 0.410% |
| Exactly zero | 2,016 | 3.935% |
| (0, 5] minutes | 2,721 | 5.311% |
| (5, 60] minutes | 8,549 | 16.687% |
| (1, 24] hours | 29,274 | 57.139% |
| Over 24 hours | 8,463 | 16.519% |

Median: 0.140972222 days (~3.38 hours); 99th percentile: 10.6866 days;
range: -0.244444444 to 84.36736111 days. Negative values already exist in real
data; this inspection does not establish why and does not justify clipping them.

## Implementation finding

configs/wids_ctabgan.json lists this feature under general_columns. In
model/synthesizer/transformer.py, general continuous columns use linear min-max
encoding into [-1, 1], rather than the mixture encoding used by other continuous
columns. With this training range, a one-minute change spans ~0.0000164 encoded
units; the first hour spans ~0.000985. This is a plausible difficulty for learning
short-stay structure, not proof that it causes the observed synthetic mismatch.

The shared generator adapter infers decimal-place precision, not a rational
1/1440-day grid. Decimal rounding therefore does not restore minute-aligned values.

## Decision

Implement standalone evaluation diagnostics first. Report exact zeros, grid
alignment, signed negative mass and disjoint interval frequencies. Compare every
available saved variant against the matching real validation/test split. Also
show a counterfactual nearest-minute view of BOTH real and synthetic values,
clearly marked diagnostic-only. Keep authoritative metrics and CSVs unchanged.

If interval frequencies agree but exact-value frequencies do not, minute-grid
restoration belongs in shared adapter postprocessing (with training-fitted
support and unit-aware metadata), not categorical rare-value pooling or
integer-day rounding. Such a correction is NOT enabled by this change.

If short-stay or negative mass remains wrong after the diagnostic rounding,
rounding cannot fix it. Test a separate validation configuration using more
suitable CTAB encoding (e.g. mixture encoding and explicit zero mode), keeping
other settings fixed. Do not silently change the scientific baseline or inject
noise into real durations before confirming this distributional mismatch.

## Existing-run command

```bash
python -m xai_reweighting.run_resolution_diagnostics --run-dir results/YOUR_WIDS_RUN
```

This writes duration_resolution.csv, duration_interval_frequencies.csv and
duration_diagnostics_manifest.json. Missing variants are skipped. No classifiers
are fitted; no training, source data, generated CSVs, or existing metrics change.
It is deliberately not wired into generation. The only supported duration in
this initial diagnostic is pre_icu_los_days, whose unit is explicitly days.

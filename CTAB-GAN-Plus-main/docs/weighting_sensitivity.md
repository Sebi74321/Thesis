# Weighting sensitivity study

The study uses the normal `run_ablation` training/evaluation implementation, not a
reduced evaluator. It works with any of the existing CTAB-GAN+, CTGAN, and
DP-CGAN ablation configurations. Enabled evaluations and their hyperparameters
are inherited unchanged. Test data are not evaluated: this is a validation-only
configuration study, not a final test-set comparison.

## Design

By default, vary one parameter at a time around the supplied configuration:

| Parameter | Reference | Higher setting |
|---|---|---|
| alpha | configured | reference × 2 |
| gamma | configured | reference × 1.5 |
| w_max | configured | 1 + (reference − 1) × 2 |
| top_k | configured | reference × 2 |

This normally produces five settings, each with seeds 42, 43, 44 and all six
variants A0–A5. That is **15 full ablation runs / 90 GAN fits**. All authoritative
settings should retain at least three seeds. This is a **one-sided** local study:
it tests stronger settings, not lower values or curvature. A0 is repeated
for isolation; settings within a seed use the same baseline protocol and seed.
A1 controls augmentation size when gamma changes. Do not interpret A0/A1 changes
as evidence of sensitivity to parameters they do not use. The retained top-k may
be smaller than requested when there are fewer eligible features. The defaults
are a local sensitivity range, not an optimized or clinically validated grid.
OAT does not estimate parameter interactions; it is not a factorial experiment.

At the measured MIMIC CTAB runtime of 60 minutes for all six variants including
evaluation, the estimate is **15 hours**, or **18.75 hours** with a 25% margin.
The default `--budget-hours 20` rejects plans whose buffered estimate exceeds
20 hours before training. Override `--baseline-run-minutes` for a different
dataset, device, or generator: the MIMIC timing is not valid for WiDS or CPU.
`--runtime-margin` defaults to 1.25. These operational settings may be changed
on resume without changing the scientific study plan.

The runtime guard checks elapsed time and the larger of the predicted child
duration and the longest observed child duration before launching each remaining
run. It saves `status=budget_paused` and exits if there is insufficient time;
continue later with `--resume`. The budget is per invocation, not a hard kill:
an unexpectedly slow in-flight child may exceed it. Completion within 20 hours
cannot be guaranteed without interrupting training. No epochs/evaluations are
silently reduced to fit the budget.

The data split is fixed with `--split-seed 42` across all settings and seeds.
Generator and evaluation randomness vary with `--seeds`, so the seed spread is
end-to-end experimental variability, not exclusively generator variability.
Standalone ablation runs retain their old split behavior unless `split_seed` is
explicitly supplied. No weighting feature exclusion or SHAP coefficient changes
are introduced by this study.

## Commands

From `CTAB-GAN-Plus-main`, preview the exact settings and compute cost:

```bash
python -m xai_reweighting.run_weighting_sensitivity \
  --config configs/mimic_ctabgan.json \
  --output-dir results/mimic_weighting_sensitivity \
  --device cuda:0 --dry-run
```

Remove `--dry-run` to train. Use `configs/wids_ctabgan.json` and a different output
directory for WiDS. Choose the corresponding CTGAN/DP-CGAN config for those models.
CPU uses `--device cpu` plus a measured `--baseline-run-minutes`. No experiment is launched
by a dry run and no files are written.

For a smoke check, add `--smoke` and use a new directory. It runs only the reference
and first perturbation, using the first seed and the normal runner's reduced
training/evaluation settings. Artifacts are explicitly non-thesis results.

An explicit alternative grid is supplied with
`--grid configs/weighting_sensitivity_grid.json`. This is a parameter-to-values
JSON map; each value is tested independently, not as a Cartesian product. The
configured reference is always included. The supplied grid is an explicit
five-setting example for a reference of alpha=3, gamma=0.6, w_max=3, top_k=5;
omitting `--grid` derives higher settings from the actual configuration. Custom
larger grids remain possible but must pass the runtime budget check. Three seeds
are the minimum for authoritative studies; smoke runs may use one seed.

Long runs can be detached using the usual `nohup` pattern:

```bash
mkdir -p logs
nohup python -u -m xai_reweighting.run_weighting_sensitivity \
  --config configs/mimic_ctabgan.json --device cuda:0 \
  --output-dir results/mimic_weighting_sensitivity --progress on \
  --seeds 42,43,44 --baseline-run-minutes 60 --budget-hours 20 \
  > logs/mimic_weighting_sensitivity.log 2>&1 < /dev/null &
echo $! > logs/mimic_weighting_sensitivity.pid
```

Resume with the same command/settings plus `--resume`. Completed child runs are
skipped; interrupted children use normal variant-level resume. Changing the
study settings, source data or code is rejected. `nohup` does not protect against
pod termination; output/log directories must be on persistent storage. Normal
resume limitations still apply, including interrupted in-memory snapshot work.

Use the same command with `--summarize-only` to rebuild reports from completed
children without training. Partial studies report available observations and
missing seed counts; do not treat them as complete evidence.

## Results and spread

Each `SETTING/seedSEED/` contains the normal full artifacts: synthetic data,
fidelity and rare-event measures, mortality and balanced-mortality utility,
additive/replacement utility, real-only references, privacy proxies, detectors,
feature diagnostics and discriminator snapshot evaluations when enabled.
Existing result directories are not modified. Exact-value frequency diagnostics
retain their original values; no rounding or bin merging is applied by reporting.

At the study root:

| File | Meaning |
|---|---|
| `study_plan.json` | Frozen settings, seeds, split seed, config and source/code hashes |
| `sensitivity_run_metrics.csv` | One measurement per setting/seed/context; classifier repeats averaged within run, with within-run SD and counts |
| `sensitivity_seed_spread.csv` | Mean, sample SD, median, quartiles, min/max, range and IQR across seeds for each setting |
| `sensitivity_ablation_summary.csv` | Main ablation metrics with explicit variant, parameter settings and across-seed spreads |
| `sensitivity_setting_spread.csv` | Spread across seed-averaged settings (descriptive, not an uncertainty interval) |
| `sensitivity_parameter_spread.csv` | Same setting spread, separately for each parameter plus reference |
| `sensitivity_paired_deltas.csv` | Matched-seed changes versus the reference weighting setting, holding variant/context fixed |
| `sensitivity_delta_spread.csv` | Across-seed spread of those paired setting changes |
| `sensitivity_artifact_coverage.csv` | Evaluation tables present and row counts for each completed child |
| `sensitivity_*.png` | Aggregate fidelity, utility, detector and privacy means with seed SD error bars |

The reference weighting setting is not the A0 variant. Normal A0 and real-baseline
deltas are still produced by every child and are included in the long reports.
`artifact`, `context` (JSON identity fields), and `metric` identify every measure.
For example, utility contexts separate task, additive/replacement protocol and
synthetic fraction; SHAP contexts separate feature, population and epoch.
Raw synthetic rows and individual detector prediction rows are deliberately not
pooled into spread statistics. All original detailed files remain in each child.
NaN metrics remain missing; sample SD with fewer than two observations stays NaN,
not zero. These are descriptive spreads, **not confidence intervals**. Rare
category/value-specific contexts may have fewer matching observations, which
must be considered when interpreting their spread.

Notebook display example (no additional training):

```python
from pathlib import Path
import pandas as pd
from IPython.display import display, Image
study = Path('../results/mimic_weighting_sensitivity').resolve()
spread = pd.read_csv(study / 'sensitivity_seed_spread.csv')
display(spread[spread.artifact == 'ablation_summary.csv'])
display(Image(filename=str(study / 'sensitivity_mean_wasserstein_scaled.png')))
```

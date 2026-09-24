# Does SHAP add value?

This focused study tests the feature-prioritisation policy, not merely whether
resampling changes the generated distribution. It uses the existing generator,
preprocessing, A0 audit, region deficits, augmentation and evaluation pipeline.
The normal A0–A5 command and its default variants are unchanged.

| Variant | Feature priority before top-k normalisation | Question |
| --- | --- | --- |
| A0 | No augmentation | How does the method compare with the original generator? |
| A5_NO_SHAP | 0.6 mismatch + 0.4 tail | Does A5 outperform a distribution-only policy? |
| A5_SHUFFLED_SHAP | 0.5 permuted SHAP + 0.3 mismatch + 0.2 tail | Does the assignment of SHAP importance to features matter? |
| A5 | 0.5 SHAP + 0.3 mismatch + 0.2 tail | Full proposed method |

SHAP changes priorities and potentially the top-k features, which changes row
weights and the augmentation sampling distribution. **All weighted controls still
use the same distributional region deficits.** SHAP does not supply those deficits.
The no-SHAP control renormalises the remaining terms to preserve the 3:2 ratio.
It compares two policies; it does not isolate an additive causal coefficient.

Every seed trains A0 once and derives a common set of audit signals. All three
weighted fits retain every original training row and use the same augmentation
size, alpha, cap, top-k, preprocessing policy and generator settings. The fixed
split seed is independent of generator seeds. The existing backend-specific
transformer policy is retained, including within-run DP-CGANS transformer reuse
when enabled. No validation/test rows enter fitting or weighting.

The shuffled control deterministically permutes the existing SHAP magnitudes
over feature names. Its seed and source-feature mapping are saved. Fixed points,
ties and all-zero signals are retained, not rerolled to manufacture a difference.
It preserves the SHAP magnitude distribution, **not necessarily the final weight
distribution**. Inspect weight SD, cap fraction, maximum sampling probability and
sampling ESS (`1 / sum(p_i**2)`, with `p_i = w_i / sum(w)`). ESS describes the
augmentation probabilities, not an independent-patient sample size or a privacy
guarantee. Original row weights, selection counts and feature scores remain saved.

## Run and resume

From `CTAB-GAN-Plus-main`, with the project environment activated:

```bash
python -m xai_reweighting.run_shap_contribution \
  --config configs/mimic_ctabgan.json \
  --output-dir results/shap_contribution_mimic_ctabgan \
  --device cuda:0 --seeds 42,43,44 --stage val --dry-run
```

Remove `--dry-run` to train. Use `--device cpu` locally, or choose an existing
CTGAN, DP-CGANS or WiDS config; use a different output directory for each study.
No scientific generator or evaluation settings are reduced for authoritative runs.
The three-seed minimum means **12 GAN fits**, not a full sensitivity grid. At the
reported 60 minutes per six-variant MIMIC run, linear planning gives approximately
2 hours, or 2.5 hours with the default 25% margin. This is not a runtime guarantee;
WiDS needs its own measured `--baseline-run-minutes`. The launch budget defaults
to 20 hours and does not terminate an in-flight seed run.

Detached GPU execution:

```bash
mkdir -p logs
nohup python -u -m xai_reweighting.run_shap_contribution \
  --config configs/mimic_ctabgan.json \
  --output-dir results/shap_contribution_mimic_ctabgan \
  --device cuda:0 --seeds 42,43,44 --stage val --progress on \
  > logs/shap_contribution_mimic_ctabgan.log 2>&1 < /dev/null &
echo $! > logs/shap_contribution_mimic_ctabgan.pid
tail -f logs/shap_contribution_mimic_ctabgan.log
```

Add `--resume` to the same command after an interruption. Completed seeds are
validated and skipped; incomplete seeds use the existing variant-level resume.
An interrupted GAN fit may need to restart: this is not epoch-level checkpointing.
Data, code, config, endpoint, seeds and device must match the saved study plan.
Keep logs and results on persistent storage. `nohup` cannot survive node/pod death.

`--smoke` runs seed 42 only with the usual reduced smoke settings and marks the
study as non-thesis output. It cannot be combined with `--stage test`.
Final test studies require the input config's `frozen: true` and a separate output
directory. Do not select configurations on test results.

## Endpoints and reports

The default primary endpoint is `mean_wasserstein_scaled` (lower is better).
It is saved **before training**, not selected from favourable outcomes. Change it
before launching via `--primary-metric NAME --primary-direction higher|lower`,
using a numeric column from `ablation_summary.csv`. If downstream utility is the
central thesis claim, select its endpoint deliberately instead of relying on the
fidelity default. This local frozen plan is not a public preregistration.

All regular configured evaluations still run: global and tail/rare fidelity,
detector, both utility tasks, additive/replacement utility, privacy proxies and
enabled diagnostics, including discriminator snapshots where supported. Each
`seed42/`, `seed43/`, `seed44/` directory has the usual complete artifacts.

Study-level outputs:

- `study_primary_pairs.csv`, `study_primary_summary.csv`, `study_primary_pairs.png`:
  paired-seed evidence for A5 versus each control; positive primary improvement
  always means better. The figure shows the two SHAP-specific contrasts.
- `study_paired_deltas.csv`, `study_delta_spread.csv`: all available matched
  variant metrics, including A0 contrasts. Raw deltas are **variant minus control**;
  negative is favourable for discrepancy metrics, positive for utility scores.
  Detector and privacy-proxy changes require contextual interpretation.
- `study_run_metrics.csv`, `study_seed_spread.csv`: absolute measurements and
  mean, sample SD, min/max, quartiles and availability across generator seeds.
  Utility repeats are averaged within each generator seed before computing spread.
  Task, protocol, mixture fraction, feature and diagnostic scope stay separate.
  Real-only baselines remain in the raw inventory and standard child artifacts.
- `study_weight_diagnostics.csv`: concentration and magnitude of sampling weights.
- `study_mechanism_diagnostics.csv`: selected-feature overlap, priority changes
  and total variation between augmentation probabilities and A5. Zero SHAP signals
  or identical sampling probabilities make a shuffled comparison uninformative;
  this is reported, not hidden by rerolling the permutation.
- `study_run_coverage.csv`, `study_artifact_coverage.csv`, `study_interpretation.json`:
  completion, artifact coverage and interpretation limits. Undefined results stay
  missing; a missing comparison is not treated as zero.

`--summarize-only` refreshes these reports from completed children without GAN
training. It does not rerun their evaluations. To reevaluate a child, use the
existing `run_evaluation --run-dir results/.../seed42` command (the saved control
variants are supported), then summarize the study with the original arguments.
Never pool incompatible software/evaluation protocols as if they were replicates.

Open `notebooks/shap_contribution_study.ipynb` to launch/resume, inspect absolute
values, plot paired deltas and inspect weight concentration. Existing A4 versus A2
results can provide complementary evidence, but this focused run does not retrain
A2/A4 or silently pool earlier runs with different configurations/code versions.

## What the study can support

Consistent A5 improvements over **both** no-SHAP and shuffled-SHAP, without material
utility/tail/privacy-proxy regressions, support the usefulness of the SHAP-guided
prioritisation policy in the tested setting. Mixed or null results should narrow
the claim to auditability or metric-specific benefits, not be hidden in an average.
Distribution movement alone is not evidence of additional SHAP benefit.

Three generator seeds give descriptive training stability on one fixed split,
not strong significance evidence or population-level uncertainty. One permutation
per generator seed is a negative control, not a permutation test. It cannot show
that SHAP is superior to every alternative feature-importance method. The study
reports observations and spread; it does not automatically declare a winner.

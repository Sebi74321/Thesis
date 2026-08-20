# Thesis

Explainability and Auditability in GAN-based Tabular Health Data Generation.

## RQ1 multi-GAN baseline comparison

RQ1 compares CTAB-GAN+, CTGAN 0.12.1, and DP-CGANS 0.2.0 on the same
persisted MIMIC or WiDS split. Generator seeds change across runs, while the
patient-row split stays fixed at seed 42.

```bash
cd CTAB-GAN-Plus-main

# Fast, explicitly non-thesis CPU check
python -m xai_reweighting.run_model_comparison \
  --config configs/rq1_mimic.json --stage val --device cpu --smoke

# Authoritative three-seed GPU run
python -m xai_reweighting.run_model_comparison \
  --config configs/rq1_mimic.json --stage val --device cuda:0 \
  --models ctabgan_plus,ctgan,dp_cgan --seeds 42,43,44
```

For WiDS, use `configs/rq1_wids.json`. The runner writes a checkpoint and
synthetic dataset after every completed fit, then evaluates that artifact. Use
`--resume` after a disconnect. To detach the whole job from Jupyter or VPN:

```bash
nohup python -u -m xai_reweighting.run_model_comparison \
  --config configs/rq1_wids.json --stage val --device cuda:0 \
  --models ctabgan_plus,ctgan,dp_cgan --seeds 42,43,44 \
  > rq1_wids.log 2>&1 &
```

Open `CTAB-GAN-Plus-main/notebooks/rq1_model_comparison.ipynb` to launch or
resume the CLI and display `rq1_results.csv`, the mean/std summary, training
history, DP accounting, and fidelity/utility/tail/privacy-proxy plots.

DP-CGAN is always invoked with its upstream `private=True` mode. Its package
hard-codes noise multiplier 1 and delta `2e-6`; the saved epsilon values are
labeled `upstream_privacy_estimate_unverified`, because this implementation
does not use conventional per-example DP-SGD clipping. They must not be
presented as independently verified formal privacy guarantees.

## XAI-guided weighted retraining

The runner implements the MIMIC and WiDS variants `A0,A1,A2,A3,A4,A5` for
CTAB-GAN+, CTGAN, and DP-CGAN. Run it from `CTAB-GAN-Plus-main` and select the
framework through the configuration file.

```bash
# Local CPU smoke test
python -m xai_reweighting.run_ablation \
  --config configs/mimic_ctabgan.json --stage val --device cpu --smoke

# Full run, automatically selecting CPU or CUDA
python -m xai_reweighting.run_ablation \
  --config configs/mimic_ctabgan.json --stage val --device auto

# The identical weighted protocol with CTGAN or DP-CGAN
python -m xai_reweighting.run_ablation \
  --config configs/mimic_ctgan.json --stage val --device auto
python -m xai_reweighting.run_ablation \
  --config configs/mimic_dpcgan.json --stage val --device auto

# WiDS H100 smoke test, then remove --smoke for the full run
python -m xai_reweighting.run_ablation \
  --config configs/wids_ctabgan.json --stage val --device cuda:0 --smoke
```

WiDS alternatives are `configs/wids_ctgan.json` and
`configs/wids_dpcgan.json`. Every variant gets a fresh generator, isolated
backend work directory, checkpoint, convergence diagnostics, and identical
evaluation. DP-CGAN additionally saves per-variant upstream privacy estimates;
these do not constitute an end-to-end DP guarantee for the XAI weighting
pipeline.

GAN fits show epoch progress with elapsed time and ETA. In non-interactive
cluster logs, `--progress auto` writes periodic progress lines instead. Use
`--progress on` to force a bar or `--progress off` to disable progress output.

For JupyterLab, open
`CTAB-GAN-Plus-main/notebooks/xai_retraining_orchestrator.ipynb`. It performs
GPU/data preflight checks, launches or resumes the CLI, and displays the saved
ablation tables, deltas, feature priorities, row weights, and diagnostic plots.

Continuous-column mixture preprocessing retries non-converged fits and fails
clearly if the retry also fails. Each GAN fit saves `training_history_A*.csv`,
`training_diagnostics_A*.json`, and `mixture_diagnostics_A*.json`; non-finite
losses or parameters stop the experiment rather than producing invalid output.

WiDS uses its larger-model settings (`batch_size=1024`, `epochs=200`,
`random_dim=128`, `num_channels=96`) and a 5,000-row smoke test. Its exact-match
privacy check remains exhaustive; nearest-neighbor privacy distances use
deterministic 20,000-row reference and 10,000-row query caps to avoid a
quadratic full-dataset evaluation.

Submit WiDS through the shared SLURM script with:

```bash
sbatch --export=ALL,CONFIG=configs/wids_ctabgan.json run_h100.slurm
```

Use `--device cuda` inside a one-GPU H100 job. `run_h100.slurm` is an example
SLURM submission. Validation configs use `frozen=false`; after selecting the
final settings, copy the config, set `frozen=true`, and run it once with
`--stage test`.

The single setup entry point installs the shared packages, selects the matching
PyTorch wheel, registers the Jupyter kernel, and activates the environment in
the current shell:

```bash
source setup_env.sh auto   # Detect CPU or CUDA
source setup_env.sh cpu    # Force the CPU environment
source setup_env.sh cuda   # Require a visible NVIDIA GPU
```

CPU and CUDA use separate Python 3.10 environments and kernels, so switching
does not overwrite the PyTorch backend. Subsequent calls skip installation when
the script and requirements are unchanged. Use `FORCE_SETUP=1` to reinstall.

Artifacts are written atomically under `CTAB-GAN-Plus-main/results/`. Use
`--resume` only when the configuration, source data, and code are unchanged.

Each new run also diagnoses a high A0 detector AUC by ranking features by
TreeSHAP importance and comparing the top ten features separately within each
outcome class. `baseline_conditional_detector_metrics.csv` reports fresh
real-vs-synthetic AUCs within each class with the outcome column removed;
`baseline_conditional_feature_diagnostics.csv` contains the corresponding
continuous/categorical gaps. To add these artifacts to an existing run without
retraining the GAN, run:

```bash
python -m xai_reweighting.run_diagnostics \
  --config configs/mimic_ctabgan.json --run-dir results/<existing_run>
```

## Mixed real/synthetic utility

Full MIMIC and WiDS runs now include two downstream utility curves. The
`additive` protocol retains all real training rows and adds 0%, 25%, 50%, or
100% synthetic rows. The `replacement` protocol fixes the training size while
replacing 0%, 25%, 50%, 75%, or 100% of real rows with synthetic rows. MIMIC
uses five deterministic classifier repeats; the larger WiDS evaluation uses
three repeats and 100-tree forests to control runtime.

Utility classification is data agnostic: the class-balanced Random Forest uses
its fixed argmax prediction rule, with no threshold tuning on audit,
validation, or test data. Mortality and the more balanced gender task are
reported separately alongside a real-only reference.

The main artifacts are `utility_mixture_results.csv`,
`utility_mixture_summary.csv`, and `utility_real_only_baseline.csv`. Add the
evaluation to an existing non-smoke run without retraining any GAN with:

```bash
python -m xai_reweighting.run_mixed_utility \
  --run-dir results/<existing_run>
```

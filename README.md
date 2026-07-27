# Thesis

Explainability and Auditability in GAN-based Tabular Health Data Generation.

## XAI-guided weighted retraining

The runner implements the MIMIC and WiDS CTAB-GAN+ variants `A0,A1,A2,A4,A5`.
Run it from `CTAB-GAN-Plus-main`.

```bash
# Local CPU smoke test
python -m xai_reweighting.run_ablation \
  --config configs/mimic_ctabgan.json --stage val --device cpu --smoke

# Full run, automatically selecting CPU or CUDA
python -m xai_reweighting.run_ablation \
  --config configs/mimic_ctabgan.json --stage val --device auto

# WiDS H100 smoke test, then remove --smoke for the full run
python -m xai_reweighting.run_ablation \
  --config configs/wids_ctabgan.json --stage val --device cuda:0 --smoke
```

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

Install shared packages from `requirements-base.txt` and install PyTorch
separately using either its CPU wheel or a cluster-compatible CUDA wheel.
`setup_cpu_env.sh` prepares a Linux/macOS CPU environment,
`setup_cpu_env.ps1` provides the Windows equivalent, and `setup_gpu_env.sh`
prepares the existing CUDA 12.8 environment.

Artifacts are written atomically under `CTAB-GAN-Plus-main/results/`. Use
`--resume` only when the configuration, source data, and code are unchanged.

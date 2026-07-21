# Thesis

Explainability and Auditability in GAN-based Tabular Health Data Generation.

## XAI-guided weighted retraining

The runner implements the MIMIC/CTAB-GAN+ variants `A0,A1,A2,A4,A5`.
Run it from `CTAB-GAN-Plus-main`.

```bash
# Local CPU smoke test
python -m xai_reweighting.run_ablation \
  --config configs/mimic_ctabgan.json --stage val --device cpu --smoke

# Full run, automatically selecting CPU or CUDA
python -m xai_reweighting.run_ablation \
  --config configs/mimic_ctabgan.json --stage val --device auto
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

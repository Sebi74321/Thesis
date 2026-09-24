# Explainability and auditability of synthetic health data

Research code and thesis material for comparing tabular GANs and studying
XAI-guided weighted retraining on MIMIC and WiDS.

The implementation supports CTAB-GAN+, CTGAN, and the DP-CGANS architecture.
Current DP-CGANS configurations run **without differential privacy**.

## Get started

From the repository root, set up and activate a Python 3.10 environment:

```bash
source setup_env.sh cpu       # Local CPU
# source setup_env.sh cuda    # NVIDIA GPU
```

Then run a small end-to-end check:

```bash
cd CTAB-GAN-Plus-main
python -m xai_reweighting.run_ablation \
  --config configs/mimic_ctabgan.json --stage val --device cpu --smoke
```

Smoke outputs are for checking the pipeline, not thesis results. Full CPU and
CUDA experiments use the same scientific configuration.

## Repository layout

| Path | Contents |
|---|---|
| `CTAB-GAN-Plus-main/xai_reweighting/` | Training runners, weighting, evaluation, and reporting |
| `CTAB-GAN-Plus-main/model/` | Adapted CTAB-GAN+ implementation |
| `CTAB-GAN-Plus-main/configs/` | Dataset and model settings |
| `CTAB-GAN-Plus-main/notebooks/` | Experiment orchestration and result inspection |
| `CTAB-GAN-Plus-main/tests/` | Unit and integration tests |
| `CTAB-GAN-Plus-main/docs/` | Protocol details and operating instructions |
| `*.tex`, `*.bib` | Thesis text, references, and result tables |
| `setup_env.sh`, `run_h100.slurm` | Environment setup and example GPU job |

The existing project directory name is retained so saved configurations,
notebooks, and cluster commands keep working. Results, logs, caches, and local
tooling files are not intended for version control.

## Experiments

- [Commands and notebooks](CTAB-GAN-Plus-main/README.md)
- [Running and resuming experiments](CTAB-GAN-Plus-main/docs/running_experiments.md)
- [Evaluation, SHAP, and preprocessing](CTAB-GAN-Plus-main/docs/evaluation.md)
- [Weighting sensitivity and runtime budget](CTAB-GAN-Plus-main/docs/weighting_sensitivity.md)

The sensitivity defaults use five settings, three seeds, and all six ablation
variants. At the measured MIMIC runtime, the buffered estimate is 18.75 hours;
use `--dry-run` and a runtime estimate appropriate to your hardware before
launching a study.

## Development

With the environment activated:

```bash
cd CTAB-GAN-Plus-main
python -m pytest tests
```

Keep scientific settings in JSON configurations, preserve raw outputs, and test
changes to preprocessing or evaluation. See [development notes](CONTRIBUTING.md)
for repository conventions.

## Upstream work

The generator code is adapted from [CTAB-GAN+](https://github.com/Team-TUD/CTAB-GAN-Plus).
Upstream references and historical setup notes are preserved in
[the attribution document](CTAB-GAN-Plus-main/docs/upstream.md).

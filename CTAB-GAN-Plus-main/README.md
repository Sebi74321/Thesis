# Experiment code

Run Python commands from this directory after activating the environment with
`source ../setup_env.sh cpu` or `source ../setup_env.sh cuda`.

## Entry points

| Task | Module |
|---|---|
| A0–A5 weighted retraining | `xai_reweighting.run_ablation` |
| RQ1 model comparison | `xai_reweighting.run_model_comparison` |
| Weighting sensitivity | `xai_reweighting.run_weighting_sensitivity` |
| Reevaluate saved synthetic data | `xai_reweighting.run_evaluation` |
| Inspect feature artifacts | `xai_reweighting.run_diagnostics` |
| Refresh SHAP comparison reports | `xai_reweighting.run_shap_comparison` |

Each module exposes `--help`. For example:

```bash
python -m xai_reweighting.run_ablation \
  --config configs/mimic_ctabgan.json --stage val --device cuda:0
```

Use the matching WiDS, CTGAN, or DP-CGAN configuration for other experiments.
The current DP-CGAN configurations select the non-private baseline.

## Notebooks

- [Ablation orchestration](notebooks/xai_retraining_orchestrator.ipynb)
- [RQ1 model comparison](notebooks/rq1_model_comparison.ipynb)
- [Discriminator and detector SHAP](notebooks/discriminator_snapshot_analysis.ipynb)

## Documentation

- [Setup, training, and resume](docs/running_experiments.md)
- [Evaluation and diagnostics](docs/evaluation.md)
- [Weighting sensitivity](docs/weighting_sensitivity.md)
- [Duration-resolution review](docs/pre_icu_los_resolution_review.md)
- [Upstream attribution and legacy examples](docs/upstream.md)

Results are stored under `results/`. A run's saved `config.json`, manifest,
split indices, and source hash describe the experiment; do not edit them to
bypass resume checks. Code changes, including refactoring, change run fingerprints.

# NEWS! - 19/11/2023
Our new paper [TabuLa: Harnessing Language Models for Tabular Data Synthesis](https://arxiv.org/abs/2310.12746) is on arxiv now! The code is published 
[here](https://github.com/zhao-zilong/Tabula). Tabula improves tabular data synthesis by leveraging language model structures without the burden of pre-trained model weights. It offers a faster training process by preprocessing tabular data to shorten token sequence, which sharply reducing training time while consistently delivering higher-quality synthetic data. Its training time is longer than CTAB-GAN+, but the synthetic data fidelity is amazing! **It also works for high-dimentional categorical columns!**

# CTAB-GAN+
This is the official git paper [CTAB-GAN+: Enhancing Tabular Data Synthesis](https://arxiv.org/abs/2204.00401). Current code is **WITHOUT** differential privacy part. The code with differential privacy is in this [github](https://github.com/Team-TUD/CTAB-GAN-Plus-DP). 
If you have any question, please contact `z.zhao-8@tudelft.nl` for more information.

## Thesis evaluation utility task

The thesis runners evaluate mortality under two controlled training regimes.
`mortality` preserves the real-training death prevalence, while
`mortality_balanced` uses 50/50 death/survival training rows. The real
validation or test split is never rebalanced, so both tasks are evaluated on
the same clinically realistic patients. Results use separate
`utility_mortality_*` and `utility_mortality_balanced_*` metric namespaces.

Utility evaluation is deliberately data agnostic: a class-balanced Random
Forest uses its fixed argmax prediction rule. No threshold is selected on the
audit, validation, or test split. Headline classification metrics are macro F1,
macro precision/recall, balanced accuracy, accuracy, ROC-AUC, and PR-AUC. The
configured positive label `1` is fixed before evaluation and is used only for
ROC/PR and class-specific supplementary metrics. Additive and replacement
mixture curves enforce their target mortality prevalence at every fraction and
record the achieved positive/negative counts in their CSV artifacts. The same
artifacts also retain the raw real and synthetic mortality rates, the
counterfactual unadjusted mixture rate, and the applied prevalence adjustment.

### Measurement-aware generation and priority sensitivity

CTAB-GAN+ sampling restores numeric measurement precision inferred exclusively
from the fitted training rows. Raw pre-restoration samples are saved as
`synthetic_raw_<variant>.csv`; authoritative samples retain the usual
`synthetic_<variant>.csv` names. The inferred grids are recorded in
`measurement_precision_<variant>.json`. SpO2 maximum is configured as a mixed
column with an explicit value at 100 so its common measurement ceiling can be
modelled rather than approximated by arbitrary values just below 100.
MIMIC WBC minimum and maximum are modelled in log space to reduce the excessive
synthetic upper tail; the upstream positive-log inverse-transform assignment
has been repaired so generated values return to the original clinical scale.

Weighted variants select at most one feature from each connected correlation
group by default (`|r| >= 0.65`). This prevents top-k selection from spending
most of its priority on redundant minimum/mean/maximum measurements from one
physiological family. The mapping is saved in
`priority_correlation_groups.json`.

`feature_family_detector_sensitivity.csv` and
`feature_family_utility_sensitivity.csv` report what happens when each selected
family, or every selected feature, is ignored. The latter covers both mortality
and gender utility. These are evaluation-only sensitivity analyses; they do not
remove the columns from GAN training.

`top_shap_feature_variant_metrics.csv` follows the A0 detector's top SHAP
features across every ablation variant on the frozen evaluation split. It uses
IQR-scaled Wasserstein distance for continuous features and Jensen-Shannon
distance for categorical features, with absolute and relative changes versus
A0. Negative deltas indicate improved feature-level fidelity.

The SHAP component used by A3/A4/A5 is calculated exclusively from
misclassified rows in the A0 detector's held-out audit partition. Correctly
classified detector rows still contribute to AUC, average precision, and
accuracy, but never to feature priorities. The eligible error count, explained
row count, and both error directions are recorded in
`baseline_detector_metrics.json`. If the holdout has no errors, the SHAP signal
is explicitly zero rather than falling back to correctly classified rows.

### Controlled ablation deltas

The primary ablation reference is always A0. `ablation_deltas.csv` therefore
contains `A1-A0`, `A2-A0`, `A3-A0`, `A4-A0`, and `A5-A0` comparisons (plus the zero
`A0-A0` reference) for fidelity, tail, detector, utility, and privacy-proxy
metrics. It also contains one `<variant>-REAL` row per variant for downstream
utility metrics. The real reference is trained on `real_train` and evaluated on
the same frozen validation or test split as the synthetic-data utility models.

`ablation_summary.csv` exposes the same values as
`delta_<metric>_vs_A0`, and utility-only `delta_<metric>_vs_real` columns along
with the corresponding `real_baseline_<metric>` value. No variant-to-variant
contrasts other than each variant versus A0 are produced.

### Weighted retraining across GAN frameworks

The same A0/A1/A2/A3/A4/A5 runner supports `ctabgan_plus`, `ctgan`, and the
`dp_cgan` architecture in non-private baseline mode. Select the framework through its config;
the split, A0-derived audit signals, augmentation size, evaluation protocol,
and controlled deltas remain identical:

```bash
python -m xai_reweighting.run_ablation --config configs/mimic_ctgan.json  --stage val --device cuda:0
python -m xai_reweighting.run_ablation --config configs/mimic_dpcgan.json --stage val --device cuda:0
python -m xai_reweighting.run_ablation --config configs/wids_ctgan.json   --stage val --device cuda:0
python -m xai_reweighting.run_ablation --config configs/wids_dpcgan.json  --stage val --device cuda:0
```

Each variant trains a fresh instance in an isolated backend directory and
saves a model-specific checkpoint, loss history, convergence warnings, and
environment/configuration metadata. The `dp_cgan` backend always uses
`private=false`; no noise multiplier, delta, epsilon, clipping, or privacy
accounting is applied or reported. It is therefore a comparison of the
DP-CGANS package's generator architecture, not a differentially private model
or an end-to-end DP pipeline.

### Reevaluate a completed run without GAN training

All evaluation artifacts for a completed, non-smoke ablation run can be
recomputed from its persisted split indices and saved synthetic CSV files:

```bash
python -m xai_reweighting.run_evaluation \
  --run-dir results/<completed-run-directory> \
  --variants A0,A1,A2,A3,A4,A5 \
  --progress on
```

This never fits a generator. It verifies the source-data hash and saved
schemas, then refreshes per-variant fidelity, tail, utility, privacy and
detector metrics; real-only and mixed utility; A0/real controlled deltas; and
diagnostic analyses. Progress and completion are recorded in
`evaluation_rerun_manifest.json`. Requested variants without a corresponding
`synthetic_<variant>.csv` are reported and skipped, which allows older runs
without A3 to be reevaluated using the current default variant list. The
manifest records both evaluated and skipped variants. In the orchestration notebook, select the
completed seed and set `RERUN_EVALUATION=True` to run the same command.


## Prerequisite

The required package version
```
numpy==1.21.0
torch==1.9.1
pandas==1.2.4
sklearn==0.24.1
dython==0.6.4.post1
scipy==1.4.1
```
The sklean package in newer version has updated its function for `sklearn.mixture.BayesianGaussianMixture`. Therefore, user should use this proposed sklearn version to successfully run the code!

## Example
`Experiment_Script_Adult.ipynb`  `Experiment_Script_king.ipynb` are two example notebooks for training CTAB-GAN+ with Adult (classification) and king (regression) datasets. The datasets are alread under `Real_Datasets` folder.
The evaluation code is also provided.

## Problem type

You can either indicate your dataset problem type as Classification, Regression. If there is no problem type, you can leave the problem type as None as follows:
```
problem_type= {None: None}
```

## For large dataset

If your dataset has large number of column, you may encounter the problem that our currnet code cannot encode all of your data since CTAB-GAN+ will wrap the encoded data into an image-like format. What you can do is changing the line 378 and 385 in `model/synthesizer/ctabgan_synthesizer.py`. The number in the `slide` list
```
sides = [4, 8, 16, 24, 32]
```
is the side size of image. You can enlarge the list to [4, 8, 16, 24, 32, 64] or [4, 8, 16, 24, 32, 64, 128] for accepting larger dataset.

## Bibtex

To cite this paper, you could use this bibtex

```
@article{zhao2023ctab,
  title={Ctab-gan+: Enhancing tabular data synthesis},
  author={Zhao, Zilong and Kunar, Aditya and Birke, Robert and Van der Scheer, Hiek and Chen, Lydia Y},
  journal={Frontiers in big Data},
  volume={6},
  year={2023},
  publisher={Frontiers Media SA}
}
```

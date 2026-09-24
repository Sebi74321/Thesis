# Thesis version (6): methods and metric consistency review

Reviewed against the supplied PDF on 24 September 2026 and the current local implementation. Page numbers below are **printed thesis pages** (add two for PDF viewer page numbers).

## Deliverables and scope

- `replacement_passages.tex` contains paste-ready replacements keyed to version-(6) sections, including chapters whose sources are missing locally. It is **not** a chapter to input wholesale.
- Matching local sources were corrected: `../methodology.tex`, `../evaluation.tex`, the three `../results_*_table.tex` files, and `../additional_references.bib`.
- The local methodology chapter has an older structure than version (6). Transfer its relevant corrections rather than replacing the version-(6) chapter wholesale.
- No training code, saved numerical results, dataset, or cohort query was changed. No corrected full-thesis PDF was generated: the complete version-(6) LaTeX project and figure sources are not available here.
- Mathematical expressions and control flow were checked against source code. This is not a rerun of the experiments or verification that each published number came from the current code revision.

## 1. What SHAP contributes

The implemented path is:

`A0 detector → correctly identified synthetic holdout rows → per-feature mean absolute TreeSHAP → max-normalised SHAP signal → variant feature score → top-k selection → selected priorities summing to one → region-deficit row score → capped weights → normalised sampling probabilities`.

SHAP can change **feature selection, feature priorities, and indirectly the sampling distribution**. It does not directly weight a row using that row's SHAP values or modify the GAN loss. Signed one-hot contributions are summed per original feature before taking their absolute value. With the real class encoded as 1, positive SHAP contributions point towards real and negative contributions towards synthetic; the magnitude-based priority discards this direction. The explanation concerns the fitted detector, not a causal feature-quality effect.

For A3, the feature score is SHAP alone. Nevertheless, **A3 still uses real-minus-A0-synthetic distributional region deficits** when forming row weights. All A2–A5 variants share that mechanism. A3 should be called “SHAP-only feature prioritisation”, not an entirely distribution-free or explanation-only weighting method. Table 5.4's coefficients belong to feature scores, not directly to sampling probabilities. Its last heading should say “Feature-priority rule / augmentation”.

Code: `detector.py` (`train_detector`, original-feature SHAP aggregation), `scoring.py` (`normalize_signal`, `compute_feature_priority`, region modelling and row weights) under `CTAB-GAN-Plus-main/xai_reweighting/`.

## 2. Jensen–Shannon terminology

Equation (6.5), p. 28, is already the correct **distance**: the square root of Jensen–Shannon divergence. Both scoring and evaluation call SciPy `jensenshannon(..., base=2)`. Keep the square root and use **Jensen–Shannon distance** consistently. The values need no squaring or recalculation.

Correct these version-(6) locations:

- Section 5.7, p. 25: “categorical mismatch … divergence” → “distance”.
- Table 7.2 and accompanying paragraph, p. 36: explicitly label “Jensen–Shannon distance”.
- Table 7.4 and interpretation, p. 39; CTGAN interpretation, p. 40; Table 7.5, p. 41: change divergence to distance.
- Discussion, p. 44: change the corresponding result descriptions to distance.
- Generic “Jensen–Shannon values” in the limitations and conclusion, pp. 46–47: make “distance” explicit.
- Keep “divergence” when referring to the unsquared expression **inside** the square root, or to unrelated metrics from other literature.

The categorical summary is the arithmetic mean of **feature distances**. Squaring that mean would not recover the mean feature divergence. Reference: [SciPy's definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html).

## 3. Equation audit

### Chapter 3

| Version-(6) equations | Finding / correction |
|---|---|
| (3.1) partition; (3.2) detector output | Consistent. Clarify that the evaluation partition means the selected validation or test split, never training input. |
| (3.3) mean absolute SHAP | Consistent for a non-empty explained subset. Specify correctly identified synthetic holdout rows and zero signal when no such rows exist; sum signed one-hot contributions before absolute value. |
| (3.4) scaled Wasserstein mismatch | Add the audit-data population-standard-deviation and 1.0 fallbacks when the IQR is unusable. |
| (3.5) tail deficit | Consistent; lower/upper tests are inclusive. Categorical scores sum positive deficits, including the minority target even if its frequency exceeds 5%. Do not replace one-sided deficits with two-sided absolute error. |
| (3.6) normalisation | Consistent: each signal family is divided by its **maximum**, not by its sum; all-zero vectors stay zero. |
| (3.7) combined score | Consistent with A2=(0,1,0), A3=(1,0,0), A4=(.625,.375,0), A5=(.5,.3,.2). |
| (3.8) selected priorities | Add a zero-denominator branch and zero priority outside the selected set. Positive selected priorities sum to one; a zero-signal vector does not. |
| (3.9) positive region deficit; (3.10) within-feature maximum normalisation | Consistent. This is the shared distributional component used by A3 as well as the other weighted variants. |
| (3.11) row score; (3.12) cap; (3.13) sampling probabilities | Consistent. No extra division by top-k or row-score maximum. The cap bounds weights, not realised duplicate counts. All-zero row scores imply uniform augmentation. |
| (3.14) multiset augmentation | Consistent. Use m=floor(gamma*N) and N+m, rather than treating .6N and 1.6N as exact integer row counts. All original GAN-training rows are retained. This is distinct from class-balanced utility resampling. |

### Chapter 6

| Version-(6) equations | Finding / correction |
|---|---|
| (6.1) Wasserstein | Integral is correct. Describe transport cost (mass times distance), not merely the amount of mass moved. |
| (6.2) scale | Correct with finite positive IQR, population standard deviation (`ddof=0`), then 1.0. Main fidelity uses the real evaluation sample; weighting uses the real audit sample. |
| (6.3) KS | Correct ECDF supremum over the full support. |
| (6.4) correlation | Correct Frobenius norm divided by the number of continuous features, not its square. Undefined correlation entries are filled with zero in code; document this convention. |
| (6.5) JS | Correct square-root distance with base-2 logs. Fix labels and interpretation, not the equation or numbers. |
| (6.6) CDF-tail | Correct **200-point grid approximation** to the largest full-ECDF gap between the real 90th percentile and real maximum. Prefer “upper-tail ECDF gap”. It is neither a KL/JS divergence nor a conditional-tail CDF comparison. Degenerate interval returns zero by convention. |
| (6.7)–(6.8) tail mass | Correct inclusive threshold frequencies and absolute frequency difference. Under ties, actual real tail mass need not equal .05 or .01. |
| (6.9) coverage | Correct code definition: synthetic mass divided by nominal tail probability, capped at one. A score of one neither proves real-tail agreement nor penalises overrepresentation. |
| (6.10) quantile error | Correct absolute quantile-location difference divided by scale. RQ1 uses train-derived scale but still compares synthetic and evaluation quantiles in the numerator. Distinguish this from its train-fixed mass thresholds. |
| (6.11) rare-category set | Restrict to **observed real categories** with positive frequency at most .05. Otherwise an unseen category with zero frequency would incorrectly satisfy the written condition. |
| (6.12) rare-category error | Correct sum within each feature. The common aggregate averages feature sums; the separate RQ1 aggregate instead averages errors over individual train-defined rare categories. These are different summaries. |
| (6.13) rare outcome | Correct absolute frequency error. Common evaluator selects the minority class from real evaluation; RQ1 fixes it from real training. Do not silently conflate those protocols. |
| (6.14)–(6.15) joint rarity | Correct mass and absolute mass error. Each feature contributes at most one Boolean flag: its lower OR upper tail, or rare-category membership. “At least two” means two distinct features. |
| (6.16) support coverage | Correct only for non-empty rare-tuple sets. State the denominator condition; excluded subsets must not be treated as zero coverage. This is presence/absence, not frequency fidelity, and remains a supplementary standalone analysis. |
| (6.17)–(6.21) accuracy, precision, recall, F1, balanced accuracy | Correct usual binary formulas. Specify zero-division handling. Macro F1 is the mean of class-wise F1 values, not F1 formed from macro precision/recall. |
| ROC-AUC / PR summary prose (no numbered equation) | Add half credit for AUC ties. The implemented PR summary is **average precision**, not trapezoidal PR-AUC. Added AP=sum(recall increment * precision), retaining old machine keys. |
| (6.22) exact matches | Definition is appropriate as a descriptive equality rate; it does not prove copying or privacy. Implementation compares row hashes after categorical normalisation, across all columns, before NN caps. |
| (6.23) nearest distance | Formula is correct but use D_ref, the actual possibly capped training reference, rather than unconditionally the entire D_train. The scale is sqrt(encoded dimension), including the outcome in this full-record analysis. |
| (6.24) percentile ratio | Use the actual query subsamples and the same reference in numerator/denominator. A zero denominator is undefined. Core evaluator stores median ratio and all six percentiles; reporting derives p5/p95 ratios. |
| (6.25) ablation difference | Correct raw variant-minus-A0 difference. Direction-adjusted heatmap colours are a separate display operation. A raw negative delta is not universally an improvement. |

### Additional prose corrections

- Section 5.3: not all preprocessing fits real training data. Generator, detector, utility, and privacy preprocessors have different legitimate fitting samples. The replacements describe each without allowing evaluation leakage.
- Section 6 opening: privacy references real training data, not just validation/test data.
- Section 6.3.1: `average_precision_score` computes non-interpolated AP. Use “AP (PR)” in all relevant table/figure headings and “average precision (AP)” in prose, including Table 7.1, utility figures 7.2–7.6, discussion, and conclusion. Retain `utility_pr_auc`/`pr_auc` as documented historical machine keys. [Official scikit-learn definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).
- Section 6.3.2: additive utility preserves the real-source **row count**, not necessarily all distinct real records after balancing. Class-count sampling can omit and duplicate rows. Additive f=1 is 1:1 real/synthetic source size. The natural real evaluation prevalence remains unchanged, including in the balanced utility task.
- Section 6.3.3: AP uses ranked probability scores and does **not** depend on a fixed .5 decision threshold. Accuracy does. Balanced-probe prevalence gives an approximate AP chance reference of .5, not an exact finite-sample guarantee.
- Section 6.2.4: the real–real bootstrap is context for finite-sample variability, not a universal achievable lower bound or independent-patient validation. The two bootstrap samples can share records because they are drawn from the same evaluation pool.

### Canonical names and saved fields

| Thesis label | Saved field / implementation | Meaning |
|---|---|---|
| Mean scaled Wasserstein distance | `mean_wasserstein_scaled` | Mean real-scale-normalised W1 across continuous features |
| Mean Jensen–Shannon distance | `mean_jensen_shannon` | Mean square-root JS, base 2, across categorical features |
| Correlation distance | `correlation_distance` | Frobenius matrix difference / continuous-feature count |
| Mean upper-tail ECDF gap | `mean_cdf_tail_divergence` | Mean 200-point upper-tail full-ECDF gap; legacy field name |
| Fixed-tail mass error | `rq1_mean_fixed_tail_mass_error` | Mean absolute mass difference at train-fixed .05/.95/.99 thresholds |
| AP (PR) | `utility_pr_auc`, mixed-utility `pr_auc` | Non-interpolated average precision |
| Exact-match rate | `privacy_exact_match_rate` | Fraction of synthetic row hashes found in real train |
| Median NN distance ratio | `privacy_median_distance_ratio` | Synthetic median NN distance / held-out-real median NN distance |

No stored fields or old results were renamed, preserving compatibility. Figure exports embedded in the PDF still need relabelling/regeneration from their original sources.

## 4. Privacy interpretation

Delete the general claim on p. 32 that GAN exact matches should always be zero. The appropriate statements are:

- Non-zero matches may reflect chance collisions or common discrete/rounded profiles, not necessarily memorisation.
- Zero matches only excludes full-record equality in the evaluated release; it does not exclude partial/approximate copying or inference attacks.
- Nearest-neighbour distances are representation- and sample-dependent diagnostics. A small ratio motivates investigation, but neither small nor large values establish a privacy guarantee.
- The p5/median/p95 aggregates can conceal vulnerable individuals. Distinguish these proxies from explicit attack-based audits and verified DP.

The replacement and corrected evaluation chapter include a supporting citation to [Stadler, Oprisanu and Troncoso (USENIX Security 2022)](https://www.usenix.org/conference/usenixsecurity22/presentation/stadler). Its BibTeX entry was added to `additional_references.bib`. This paper supports the broader limitation that synthetic data do not automatically prevent inference attacks; the chance-match explanation also follows directly from equality on finite/discretised support.

## 5. Transformer reuse

This asymmetry exists in the implementation:

| Architecture | A1–A5 transformer policy | Consequence |
|---|---|---|
| DP-CGANS, reuse enabled | Load the hash-validated transformer fitted on A0 real training rows; transform variant rows; rebuild sampler and networks | Fixed representation, changed row/conditional frequencies, fresh GAN |
| CTAB-GAN+ | Fit a new transformer on each variant's training table | Resampling can also change mixture components/representation |
| CTGAN | Native `fit` fits a new transformer on each variant's training table | Same representational caveat |

It is not a DP requirement and does not imply that the other architectures are mathematically unable to reuse a transformer. The saved-transformer interface provides a practical reuse path for expensive DP-CGANS preprocessing. Describe it as a runtime choice and within-DP-CGANS control, and acknowledge that the cross-architecture intervention is not held fixed at the representation level. A claim that *all* generator comparisons isolate only row sampling would be too strong.

Code: `run_ablation.py` manages and validates reuse; `generator_adapters.py` implements each model's fit path and passes `saved_transformer` to the external `dp_cgans.DP_CGAN` constructor. The [upstream Python API](https://github.com/sunchang0124/dp_cgans) exposes this argument. The runner setting is `generator.reuse_transformer_across_variants`, not a universal guarantee for every past DP-CGANS run.

## 6. Cohort query

Your reading is correct. Appendix A's order is:

1. `icu_cohort`: filter by **initial** care unit (MICU or MICU/SICU).
2. `ranked`: rank that restricted set within `subject_id` by `intime`.
3. `cohort`: retain `rn=1`, then require age 18–89.
4. Join first-day measurements.

It selects the **earliest stay in the selected initial care units**, not necessarily the first ICU stay overall. Nor does it select the first age-eligible selected-unit stay: if the earliest one fails the age restriction, the patient is excluded even if a later stay might qualify. At most one selected stay remains per patient, so the patient-level deduplication rationale survives, but the stated cohort definition must change.

Correct Section 4.2.1, its following exclusion paragraph, Figure 4.1, Table 4.1's first operation, and the introduction to Appendix A. Keep the executed SQL and cohort counts unchanged. Retrospectively changing the query would define a different experiment and require re-extraction and reruns. `ORDER BY intime` alone also leaves ties unspecified; a deterministic tie-breaker could be considered for future extractions, not silently added to describe this one.

## 7. Historical claims still requiring run provenance

Current implementation is evidence for what the software does now, not proof of the settings used to obtain published tables. Before final submission, reconcile these against each reported run's frozen configuration, manifest, diagnostics, and saved metric CSV:

- Version (6) states alpha=4; some current configurations use alpha=3. Do not globally replace a historical value without checking the actual runs.
- Snapshot frequency/count and whether WiDS discriminator evaluation was enabled differ between the thesis description and current defaults. Keep a run-specific description.
- Section 5.8 describes aborting after failed mixture convergence. The current intended policy distinguishes non-convergence warnings from an unusable transformation; match the text to the revision used for each reported run.
- Confirm the reported stage, software/hardware, seeds and nested classifier repetitions from manifests. The local table fragments differ slightly in rounding/layout from the PDF; numerical values were not overwritten to make these versions look identical.
- The current code establishes how JS and AP are calculated. Check the original table-export inputs if any numbers were imported from another evaluator or older code path.

These checks are deliberately not resolved by inventing provenance or silently changing scientific settings.

## Integration checklist

1. Transfer the keyed passages into the complete version-(6) LaTeX project.
2. Apply the equation/prose corrections from the corrected local evaluation chapter, preserving the version-(6) labels and chapter organisation.
3. Relabel JS, AP, and upper-tail ECDF-gap headings throughout results, figures, discussion and conclusion. Numerical values remain unchanged.
4. Add the supplied privacy citation to the thesis bibliography.
5. Reconcile the historical run settings above; do not substitute current defaults.
6. Compile the full thesis and inspect table widths, references and figures. Only static source checks were possible locally; a full-thesis compilation was not possible without its complete source project.

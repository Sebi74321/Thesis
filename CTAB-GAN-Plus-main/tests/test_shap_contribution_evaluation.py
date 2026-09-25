"""Reporting uses generator seeds as the unit of replication."""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from xai_reweighting.shap_contribution_evaluation import (
    evaluation_records, evaluation_deltas, seed_summary, write_evaluation_reports,
    plot_metric_panels, plot_mixture_curves, plot_utility_heatmap,
)


def row(seed, variant, metric, value, artifact="ablation_summary.csv", **context):
    if variant is not None:
        context["variant"] = variant
    return dict(seed=seed, artifact=artifact, context=json.dumps(context), metric=metric,
                value=value, within_run_std=.9, observations=3)


def inventory():
    rows = []
    for i, seed in enumerate([42, 43, 44]):
        for variant, offset in [("A0", .1), ("A5_NO_SHAP", .05), ("A5_SHUFFLED_SHAP", .02), ("A5", 0.)]:
            rows.append(row(seed, variant, "mean_wasserstein_scaled", .2+i*.1+offset))
            for task in ["mortality", "mortality_balanced"]:
                for protocol in ["additive", "replacement"]:
                    for fraction in [0., 1.]:
                        value = .4+i*.1 if fraction == 0 else .5+i*.1-offset
                        rows.append(row(seed, variant, "pr_auc", value, "utility_mixture_results.csv",
                                        utility_task=task, protocol=protocol, synthetic_fraction=fraction))
        for task in ["mortality", "mortality_balanced"]:
            rows.append(row(seed, None, "pr_auc", .4+i*.1, "utility_real_only_baseline.csv",
                            utility_task=task, protocol="real_only", synthetic_fraction=0.))
    return pd.DataFrame(rows)


def test_seed_sd_not_rf_sd_and_paired_covariance():
    records=evaluation_records(inventory())
    summary=seed_summary(records,[42,43,44])
    a5=summary[(summary.domain=="Global fidelity") & (summary.variant=="A5")].iloc[0]
    assert a5["mean"]==pytest.approx(.3)
    assert a5["std"]==pytest.approx(.1)  # not within-run .9, or SD/sqrt(n)
    assert a5.n==3
    pairs=evaluation_deltas(records)
    ds=seed_summary(pairs,[42,43,44],paired=True)
    real=ds[(ds.reference=="REAL") & (ds.variant=="A5") & (ds.synthetic_fraction==1)]
    assert len(real)==4  # two tasks x two protocols, never pooled
    np.testing.assert_allclose(real["mean"], .1)
    np.testing.assert_allclose(real["std"], 0., atol=1e-15)
    assert (real.n==3).all()


def test_missing_seeds_pairs_and_infinities_are_not_zero():
    raw=inventory()
    raw=raw[raw.seed==42].copy()
    raw=raw[~raw.context.str.contains('A5_NO_SHAP')]
    raw.loc[(raw.metric=="mean_wasserstein_scaled") & raw.context.str.contains('A0'),"value"]=np.inf
    records=evaluation_records(raw)
    summary=seed_summary(records,[42,43,44])
    assert summary["std"].isna().all()
    assert summary.unavailable_seeds.min()==2
    ds=seed_summary(evaluation_deltas(records),[42,43,44],paired=True)
    missing=ds[ds.reference=="A5_NO_SHAP"]
    assert (missing.n==0).all()
    assert missing["mean"].isna().all()
    assert (missing.unavailable_seeds==3).all()


def test_legacy_balanced_accuracy_collision_is_not_reproduced():
    raw=pd.DataFrame([
        row(42,"A5","utility_mortality_roc_auc",.8),
        row(42,"A5","utility_mortality_recall_macro",.53),
        row(42,"A5","utility_mortality_balanced_accuracy",.91),
        row(42,"A5","utility_mortality_balanced_roc_auc",.82),
        row(42,"A5","utility_mortality_balanced_recall_macro",.63),
        row(42,"A5","utility_mortality_balanced_balanced_accuracy",.63),
    ])
    records=evaluation_records(raw).set_index(["utility_task","metric"])
    assert records.loc[("mortality","balanced_accuracy"),"value"]==.53
    assert records.loc[("mortality_balanced","balanced_accuracy"),"value"]==.63
    assert records.loc[("mortality_balanced","accuracy"),"value"]==.91


def test_duplicate_seed_context_rejected():
    raw=inventory()
    with pytest.raises(ValueError,match="Ambiguous"):
        evaluation_records(pd.concat([raw,raw.iloc[:1]]))


def test_no_pair_between_incompatible_main_utility_and_real_baseline():
    raw=pd.concat([inventory(),pd.DataFrame([row(42,"A5","utility_mortality_roc_auc",.8)])])
    pairs=evaluation_deltas(evaluation_records(raw))
    assert pairs[(pairs.protocol=="synthetic_only") & (pairs.reference=="REAL")].empty


def test_atomic_reporting_refresh_old_inventory_and_plots(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    raw=inventory()
    raw.to_csv(tmp_path/"study_run_metrics.csv",index=False)
    (tmp_path/"study_plan.json").write_text(json.dumps({"seeds":[42,43,44],"stage":"val","smoke":False}))
    before=(tmp_path/"study_run_metrics.csv").read_bytes()
    tables=write_evaluation_reports(tmp_path)
    assert (tmp_path/"study_run_metrics.csv").read_bytes()==before
    assert all((tmp_path/name).is_file() for name in tables)
    assert not list(tmp_path.glob("*.tmp"))
    summary=tables['study_evaluation_summary.csv']
    deltas=tables['study_evaluation_delta_summary.csv']
    g=summary[summary.domain=="Global fidelity"]
    fig=plot_metric_panels(g,"test")
    assert all(ax.get_xlabel() and ax.get_ylabel() for ax in fig.axes)
    fig.savefig(tmp_path/"seed_spread.png")
    plt.close(fig)
    fig=plot_mixture_curves(deltas,"mortality","additive")
    assert all(ax.get_xlabel() and ax.get_ylabel() for ax in fig.axes)
    plt.close(fig)
    g=summary[(summary.domain=="Utility") & (summary.utility_task=="mortality")
              & (summary.protocol=="additive") & (summary.synthetic_fraction==1)]
    fig=plot_utility_heatmap(g,"test")
    assert any('±' in t.get_text() and 'n=3' in t.get_text() for t in fig.axes[0].texts)
    plt.close(fig)
    with pytest.raises(ValueError,match="one task"):
        plot_metric_panels(summary,"mixed contexts")
    # An unavailable seed SD remains missing and is safe to plot.
    g=g.assign(std=np.nan,n=1)
    plt.close(plot_metric_panels(g,"one seed"))
    plt.close(plot_utility_heatmap(g,"one seed"))


def test_notebook_has_separate_tasks_real_deltas_and_seed_sd():
    book=json.loads((Path(__file__).resolve().parents[1]/"notebooks/shap_contribution_study.ipynb").read_text(encoding="utf-8"))
    codes=["".join(c["source"]) for c in book["cells"] if c["cell_type"]=="code"]
    for code in codes:
        ast.parse(code)
    joined="\n".join(codes)
    for token in ["write_evaluation_reports(STUDY_DIR)","positive_precision","positive_f1",
                  "plot_mixture_curves","plot_utility_heatmap","utility.utility_task.drop_duplicates()"]:
        assert token in joined
    assert "LAUNCH = False" in joined

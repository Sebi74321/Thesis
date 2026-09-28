"""Seed-level thesis figures from completed SHAP studies; no training or evaluation.

Accepts a study directory or an exported ZIP. Inputs are read without extraction
or modification. Each plotted observation is one generator-seed measurement.
"""

import argparse
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import re
import tempfile
import zipfile

import numpy as np
import pandas as pd

from .io_utils import atomic_write_csv, atomic_write_json
from .shap_contribution_evaluation import DIMENSIONS, LABELS, VARIANTS


CONTROLS = ("A5_NO_SHAP", "A5_SHUFFLED_SHAP")
SHORT_NAMES = {"A0": "A0", "A5": "A5", "A5_NO_SHAP": "No SHAP",
               "A5_SHUFFLED_SHAP": "Shuffled SHAP"}
SCORE_LABELS = {**LABELS, "mean_wasserstein_scaled": "Mean scaled Wasserstein distance",
                "positive_recall": "Positive recall", "positive_f1": "Positive F1",
                "wasserstein_scaled": "Scaled Wasserstein distance"}
UTILITY_SCORES = ("pr_auc", "roc_auc", "positive_recall", "positive_f1")


@dataclass
class Study:
    label: str
    source: Path
    plan: dict
    records: pd.DataFrame


def load_study(source, label=None):
    """Read only the frozen plan and focused seed export, including inside ZIPs."""
    source = Path(source).expanduser().resolve()
    names = ("study_plan.json", "study_evaluation_seed_metrics.csv")
    if source.is_dir():
        payload = {name: (source / name).read_bytes() for name in names}
    else:
        with zipfile.ZipFile(source) as archive:
            payload = {}
            for name in names:
                matches = [item for item in archive.namelist()
                           if not item.endswith("/") and
                           re.sub(r" \(\d+\)(?=\.[^.]+$)", "", Path(item).name) == name]
                if len(matches) != 1:
                    raise ValueError(f"Expected one {name} in {source}, found {len(matches)}")
                payload[name] = archive.read(matches[0])
    plan = json.loads(payload[names[0]])
    records = pd.read_csv(io.BytesIO(payload[names[1]]))
    required = {"seed", "variant", "value", *DIMENSIONS}
    if not required.issubset(records):
        raise ValueError(f"Missing seed-export columns: {sorted(required - set(records))}")
    for key in ["domain", "utility_task", "protocol", "feature", "metric"]:
        records[key] = records[key].fillna("")
    records["value"] = pd.to_numeric(records.value, errors="raise").replace([np.inf, -np.inf], np.nan)
    if records.duplicated(["seed", "variant", *DIMENSIONS]).any():
        raise ValueError("Duplicate measurements within a generator seed and context")
    if not set(records.seed).issubset(set(plan["seeds"])):
        raise ValueError("Export contains seeds outside the saved study plan")
    name = plan.get("config", {}).get("generator_name", source.stem)
    model_label = {"ctabgan_plus": "CTAB-GAN+", "ctgan": "CTGAN"}.get(name, name)
    if name == "dp_cgan":
        private = plan.get("config", {}).get("generator", {}).get("private")
        model_label = "DP-CGANS (non-private)" if private is False else "DP-CGANS"
    return Study(label or model_label,
                 source, plan, records)


def _series(study, metric, **context):
    rows = study.records[study.records.metric == metric]
    for key, value in context.items():
        rows = rows[rows[key] == value]
    if rows.duplicated(["seed", "variant"]).any():
        raise ValueError(f"Select a single evaluation context for {metric}")
    return rows[["seed", "variant", "value"]].copy()


def paired_values(study, metric, reference, **context):
    """Subtract within seed, retaining missing pairs as NaN, never zero."""
    rows = _series(study, metric, **context)
    left = rows[rows.variant == "A5"].set_index("seed").value
    right = rows[rows.variant == reference].set_index("seed").value
    frame = pd.DataFrame({"value": left, "reference_value": right}).reindex(study.plan["seeds"])
    frame["delta"] = frame.value - frame.reference_value
    frame.index.name = "seed"
    return frame.reset_index().assign(study=study.label, variant="A5", reference=reference,
                                     metric=metric, **context)


def _compatible(studies):
    if not studies:
        raise ValueError("Select at least one study")
    for study in studies[1:]:
        for key in ("data_sha256", "split_seed", "stage", "primary_metric", "primary_direction", "smoke"):
            if studies[0].plan.get(key) != study.plan.get(key):
                raise ValueError(f"Cannot combine studies with different {key}")


def _seed_style(studies):
    import matplotlib.pyplot as plt

    seeds = sorted({seed for study in studies for seed in study.plan["seeds"]})
    cmap = plt.get_cmap("tab10" if len(seeds) <= 10 else "tab20")
    colors = {seed: cmap(i % cmap.N) for i, seed in enumerate(seeds)}
    jitter = dict(zip(seeds, np.linspace(-.16, .16, len(seeds))))
    return seeds, colors, jitter


def _points(ax, data, groups, value, styles, expected):
    seeds, colors, jitter = styles
    ticks = []
    for x, (key, label) in enumerate(groups):
        values = data[data["group"] == key].dropna(subset=[value])
        for row in values.itertuples():
            ax.scatter(x + jitter[row.seed], getattr(row, value), s=32,
                       color=colors[row.seed], alpha=.85, zorder=3)
        n = len(values)
        ticks.append(f"{label}\nn={n}/{expected}")
        if n:
            mean, sd = values[value].mean(), values[value].std(ddof=1)
            ax.scatter(x, mean, color="black", marker="D", s=45, zorder=5)
            if n > 1:
                ax.errorbar(x, mean, yerr=sd, color="black", capsize=5, linewidth=1.3, zorder=4)
    ax.set_xticks(range(len(groups)), ticks)
    ax.set_xlim(-.5, len(groups) - .5)
    ax.grid(axis="y", alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)


def _finish(fig, studies, styles, title):
    from matplotlib.lines import Line2D

    seeds, colors, _ = styles
    handles = [Line2D([], [], marker="o", linestyle="none", color=colors[s], label=str(s)) for s in seeds]
    handles.append(Line2D([], [], marker="D", linestyle="none", color="black", label="Mean ± seed SD"))
    fig.legend(handles=handles, title="Generator seed", loc="lower center", ncol=min(6, len(handles)),
               fontsize=9, title_fontsize=9, frameon=False)
    stage = studies[0].plan.get("stage", "unknown")
    dataset = studies[0].plan.get("config", {}).get("dataset_name", "dataset")
    smoke = " — SMOKE / non-thesis" if any(s.plan.get("smoke") for s in studies) else ""
    fig.suptitle(f"{title}\n{dataset} · {stage} · fixed split {studies[0].plan.get('split_seed', '?')}{smoke}", fontsize=13)
    fig.tight_layout(rect=(0, .14, 1, .91))
    return fig


def plot_primary(studies, *, include_a0=True):
    """The predeclared metric: raw A5-minus-control deltas, one panel per study."""
    import matplotlib.pyplot as plt

    studies = list(studies)
    _compatible(studies)
    styles = _seed_style(studies)
    metric = studies[0].plan["primary_metric"]
    direction = studies[0].plan["primary_direction"]
    if direction not in ("lower", "higher"):
        raise ValueError("Unknown primary metric direction")
    refs = ("A0", *CONTROLS) if include_a0 else CONTROLS
    fig, axes = plt.subplots(1, len(studies), figsize=(6 * len(studies), 5.5), squeeze=False)
    tables = []
    for ax, study in zip(axes.flat, studies):
        data = pd.concat([paired_values(study, metric, ref).assign(group=ref) for ref in refs])
        tables.append(data)
        _points(ax, data, [(r, SHORT_NAMES[r]) for r in refs], "delta", styles, len(study.plan["seeds"]))
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        sign = "Negative" if direction == "lower" else "Positive"
        ax.set(title=study.label, xlabel="Reference policy", ylabel=f"A5 − reference: {SCORE_LABELS.get(metric, metric)}\n{sign} values favour A5")
    return _finish(fig, studies, styles, "Primary fidelity: paired differences"), pd.concat(tables, ignore_index=True)


def plot_utility_controls(study, *, task="mortality", protocol="replacement", fraction=1., styles=None):
    """One task and mixture setting; ranking scores and fixed-decision scores."""
    import matplotlib.pyplot as plt

    if protocol not in ("replacement", "additive"):
        raise ValueError("Choose a fixed-protocol additive or replacement evaluation")
    styles = styles or _seed_style([study])
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), squeeze=False)
    tables = []
    for ax, metric in zip(axes.flat, UTILITY_SCORES):
        context = dict(domain="Utility", utility_task=task, protocol=protocol, synthetic_fraction=fraction)
        data = pd.concat([paired_values(study, metric, ref, **context).assign(group=ref) for ref in CONTROLS])
        tables.append(data)
        _points(ax, data, [(r, SHORT_NAMES[r]) for r in CONTROLS], "delta", styles, len(study.plan["seeds"]))
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set(title=SCORE_LABELS[metric], xlabel="Reference policy",
               ylabel="A5 − reference\nPositive values favour A5")
    share = fraction / (1 + fraction) if protocol == "additive" else fraction
    title = f"{study.label}: {task} — {protocol} f={fraction:g} ({share:.0%} synthetic rows)"
    return _finish(fig, [study], styles, title), pd.concat(tables, ignore_index=True)


def plot_fixed_features(study, features, *, styles=None):
    """Fixed continuous features from the full export, not conditional top-k rows."""
    import matplotlib.pyplot as plt

    features = list(features)
    if not features or len(set(features)) != len(features):
        raise ValueError("Provide distinct feature names")
    styles = styles or _seed_style([study])
    cols = min(3, len(features))
    rows = int(np.ceil(len(features) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows + 1), squeeze=False)
    tables = []
    for ax, feature in zip(axes.flat, features):
        data = _series(study, "wasserstein_scaled", domain="Per-feature fidelity", feature=feature)
        if data.empty or not data.value.notna().any():
            raise ValueError(f"No continuous per-feature fidelity measurements for {feature} in {study.label}")
        data = data[data.variant.isin(VARIANTS)].assign(study=study.label, feature=feature,
                    metric="wasserstein_scaled", group=lambda d: d.variant)
        tables.append(data)
        _points(ax, data, [(v, SHORT_NAMES[v].replace(" SHAP", "\nSHAP")) for v in VARIANTS],
                "value", styles, len(study.plan["seeds"]))
        ax.set(title=feature, xlabel="Training policy", ylabel="Scaled Wasserstein distance\nLower values indicate closer agreement")
    for ax in axes.flat[len(features):]:
        ax.remove()
    return _finish(fig, [study], styles, f"{study.label}: fixed-feature fidelity across seeds"), pd.concat(tables, ignore_index=True)


def _save_figure(fig, path, fmt):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        fig.savefig(temporary, format=fmt, dpi=300, bbox_inches="tight", facecolor="white")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def export_thesis_figures(studies, output_dir, *, features=("temperature_min", "creatinine_min", "creatinine_max"),
                          tasks=("mortality", "mortality_balanced"), protocol="replacement", fraction=1.):
    """Export figures and their exact plotted seed values; source artifacts unchanged."""
    import matplotlib.pyplot as plt

    studies = list(studies)
    _compatible(studies)
    if not np.isfinite(fraction) or fraction < 0 or (protocol == "replacement" and fraction > 1):
        raise ValueError("Invalid mixture fraction")
    if len({s.label for s in studies}) != len(studies):
        raise ValueError("Study labels must be unique")
    slugs = [re.sub(r"[^a-z0-9]+", "_", s.label.lower()).strip("_") for s in studies]
    if len(set(slugs)) != len(slugs):
        raise ValueError("Study labels produce colliding export filenames")
    output_dir = Path(output_dir).expanduser().resolve()
    styles = _seed_style(studies)
    exported = {}

    def save(name, result):
        fig, data = result
        try:
            for fmt in ("png", "pdf"):
                _save_figure(fig, output_dir / f"{name}.{fmt}", fmt)
            atomic_write_csv(output_dir / f"{name}_points.csv", data.drop(columns="group", errors="ignore"))
            exported[name] = output_dir / f"{name}.png"
        finally:
            plt.close(fig)

    # A local style context leaves the user's other notebook figures untouched.
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42}):
        save("primary_paired_fidelity", plot_primary(studies))
        for study, slug in zip(studies, slugs):
            for task in tasks:
                if study.records[(study.records.domain == "Utility") & (study.records.utility_task == task)
                                 & (study.records.protocol == protocol)
                                 & (study.records.synthetic_fraction == fraction)].empty:
                    raise ValueError(f"No {task}/{protocol}/f={fraction} measurements for {study.label}")
                save(f"utility_{slug}_{task}_{protocol}_f{fraction:g}",
                     plot_utility_controls(study, task=task, protocol=protocol, fraction=fraction, styles=styles))
            if features:
                save(f"fixed_features_{slug}", plot_fixed_features(study, features, styles=styles))
    atomic_write_json(output_dir / "thesis_figures_manifest.json", {
        "sources": [{"label": s.label, "path": s.source, "seeds": s.plan["seeds"],
                     "data_sha256": s.plan.get("data_sha256"), "code_sha256": s.plan.get("code_sha256"),
                     "stage": s.plan.get("stage"), "split_seed": s.plan.get("split_seed")} for s in studies],
        "figures": exported, "features": list(features), "utility_tasks": list(tasks),
        "protocol": protocol, "fraction": fraction,
        "spread": "Black diamond: mean; whiskers: sample SD across generator seeds, not CI/SE",
        "delta": "A5 minus reference within seed; negative favours fidelity, positive favours utility",
        "coverage": "n labels count finite observations or finite matched pairs; missing is not zero",
        "feature_scope": "Full per-feature export, not conditional on top-SHAP selection",
        "limitations": "Illustrative feature selection; secondary utility is not the primary endpoint. No retraining or evaluation.",
    })
    return exported


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", action="append", required=True, help="Study directory or ZIP; repeat for another architecture")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--features", nargs="*", default=["temperature_min", "creatinine_min", "creatinine_max"])
    parser.add_argument("--protocol", choices=["replacement", "additive"], default="replacement")
    parser.add_argument("--fraction", type=float, default=1.)
    args = parser.parse_args(argv)
    studies = [load_study(path) for path in args.study]
    exports = export_thesis_figures(studies, args.output_dir, features=args.features,
                                    protocol=args.protocol, fraction=args.fraction)
    print(f"Exported {len(exports)} figures (PNG/PDF and seed CSVs) to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

"""Collect evaluation artifacts and summarize sensitivity across settings and seeds."""

import json
import numpy as np
import pandas as pd

from .io_utils import atomic_write_csv, atomic_write_json, file_sha256
from .run_ablation import _fingerprint

# Only aggregate evaluation/diagnostic tables, never patient or prediction rows.
CSV_PATTERNS = (
    'ablation_summary.csv', 'ablation_deltas.csv', 'feature_metrics_*.csv',
    'utility_mixture_results.csv', 'utility_real_only_baseline.csv',
    'baseline_detector_*.csv', 'baseline_conditional_*.csv',
    'prioritized_feature_*.csv', 'feature_exclusion_*.csv', 'feature_family_*_sensitivity.csv',
    'top_shap_feature_variant_metrics.csv', 'discriminator_shap_*.csv',
    'discriminator_snapshot_metrics_*.csv',
    'discriminator_snapshot_epoch_matched_metrics_*.csv',
    'discriminator_snapshot_late_window_*.csv',
    'discriminator_detector_shap_comparison_*.csv', 'feature_scores_*.csv',
)
JSON_PATTERNS = ('metrics_*.json', 'baseline_detector_metrics.json',
                 'row_weight_summary_*.json', 'training_diagnostics_*.json',
                 'discriminator_detector_shap_comparison_*.json')
DIMENSIONS = {'variant', 'feature', 'other_feature', 'correlated_feature', 'kind',
              'utility_task', 'target_col', 'target_balance', 'protocol',
              'synthetic_fraction', 'epoch', 'group', 'population', 'scope',
              'comparison', 'control', 'reference', 'source', 'category', 'value',
              'target_value', 'subset', 'model', 'metric', 'feature_group',
              'removed_features', 'condition', 'evaluation', 'task', 'analysis',
              'epoch_from', 'epoch_to', 'outcome_group', 'experiment', 'scenario',
              'feature_set', 'mode', 'selection', 'pair', 'class_label',
              'exclusion', 'excluded_features', 'condition_value', 'target', 'positive_label', 'outcome'}
REPEAT_FIELDS = {'repeat', 'seed', 'generator_seed', 'random_state'}


def _json_numbers(obj, prefix=''):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from _json_numbers(value, f'{prefix}.{key}' if prefix else key)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield prefix, obj


def collect_run(run_dir, setting, seed):
    """Reduce repeated evaluations within one completed generator run."""
    records, coverage = [], []
    for pattern in CSV_PATTERNS:
        for path in sorted(run_dir.glob(pattern)):
            if path.name in {entry['artifact'] for entry in coverage}:
                continue
            try:
                frame = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                frame = pd.DataFrame()
            coverage.append({'artifact': path.name, 'rows': len(frame)})
            dimensions = [c for c in frame if c in DIMENSIONS]
            metrics = [c for c in frame.select_dtypes(include=[np.number])
                       if c not in {*dimensions, *REPEAT_FIELDS}]
            for row in frame.to_dict('records'):
                context = json.dumps({c: (None if pd.isna(row[c]) else row[c])
                                      for c in dimensions}, sort_keys=True)
                for metric in metrics:
                    records.append((path.name, context, metric, row[metric]))
    for pattern in JSON_PATTERNS:
        for path in sorted(run_dir.glob(pattern)):
            entries = list(_json_numbers(json.loads(path.read_text(encoding='utf-8'))))
            coverage.append({'artifact': path.name, 'rows': len(entries)})
            records.extend((path.name, '{}', metric, value) for metric, value in entries)
    raw = pd.DataFrame(records, columns=['artifact', 'context', 'metric', 'value'])
    if raw.empty:
        raise ValueError(f'No evaluation measurements in {run_dir}')
    raw['value'] = raw['value'].replace([np.inf, -np.inf], np.nan)
    # RF repeats are nested within a generator seed, not independent GAN runs.
    per_run = raw.groupby(['artifact', 'context', 'metric'], dropna=False)['value'].agg(
        value='mean', within_run_std='std', observations='count', attempted='size'
    ).reset_index()
    per_run['setting'], per_run['seed'] = setting, seed
    coverage = pd.DataFrame(coverage).assign(setting=setting, seed=seed)
    return per_run, coverage


def spread(frame, keys, column='value'):
    result = frame.groupby(keys, dropna=False)[column].agg(
        n='count', attempted='size', mean='mean', std='std', minimum='min',
        q25=lambda x: x.quantile(.25), median='median',
        q75=lambda x: x.quantile(.75), maximum='max',
    ).reset_index()
    result['range'] = result['maximum'] - result['minimum']
    result['iqr'] = result['q75'] - result['q25']
    result['missing'] = result['attempted'] - result['n']
    return result  # n=1 has undefined SD, deliberately not zero.


def summarize_study(output_dir, plan):
    rows, coverage, split_hashes = [], [], set()
    run_coverage = []
    for setting in plan['settings']:
        for seed in plan['seeds']:
            child = output_dir / setting['setting'] / f'seed{seed}'
            manifest_path = child / 'manifest.json'
            if not manifest_path.exists():
                run_coverage.append(dict(setting=setting['setting'], seed=seed, status='not_started'))
                continue
            manifest = json.loads(manifest_path.read_text())
            run_coverage.append(dict(setting=setting['setting'], seed=seed,
                                     status=manifest.get('status', 'unknown')))
            if manifest.get('status') != 'complete':
                continue
            child_config = json.loads((child / 'config.json').read_text())
            if (manifest.get('fingerprint') != _fingerprint(
                    child_config, plan['data_sha256'], plan['code_sha256'])
                    or child_config.get('weighting') != setting['weighting']
                    or child_config.get('seed') != seed
                    or child_config.get('split_seed') != plan['split_seed']):
                raise ValueError(f'Child run identity does not match study: {child}')
            if set(manifest.get('variants_completed', [])) != set(plan['variants']):
                raise ValueError(f'Incomplete variant coverage: {child}')
            split_hashes.add(file_sha256(child / 'split_indices.json'))
            values, artifacts = collect_run(child, setting['setting'], seed)
            rows.append(values)
            coverage.append(artifacts)
    if len(split_hashes) > 1:
        raise ValueError('Sensitivity child runs do not share identical splits')
    atomic_write_csv(output_dir / 'sensitivity_run_coverage.csv', pd.DataFrame(run_coverage))
    if not rows:
        return
    per_run = pd.concat(rows, ignore_index=True)
    keys = ['artifact', 'context', 'metric']
    seeds = spread(per_run, ['setting', *keys])
    seeds['expected_seeds'] = len(plan['seeds'])
    seeds['unavailable_seeds'] = len(plan['seeds']) - seeds['n']
    order = {s['setting']: i for i, s in enumerate(plan['settings'])}
    seeds = seeds.assign(_order=seeds.setting.map(order)).sort_values('_order', kind='stable').drop(columns='_order')
    # Parameter spread uses seed-averaged setting means, keeping noise separate.
    settings = spread(seeds, keys, 'mean')
    settings['expected_settings'] = len(plan['settings'])
    settings['unavailable_settings'] = len(plan['settings']) - settings['n']
    parameter_spreads = []
    for parameter in dict.fromkeys(
        setting['parameter'] for setting in plan['settings']
        if setting['parameter'] != 'reference'
    ):
        members = [s['setting'] for s in plan['settings']
                   if s['parameter'] in {'reference', parameter}]
        if len(members) > 1:
            parameter_spreads.append(spread(seeds[seeds.setting.isin(members)], keys, 'mean')
                                     .assign(parameter=parameter))
    reference = per_run[per_run.setting == 'reference'][['seed', *keys, 'value']]
    paired = per_run.merge(reference.rename(columns={'value': 'reference_value'}),
                           on=['seed', *keys], how='left', validate='many_to_one')
    paired['delta_vs_reference_setting'] = paired['value'] - paired['reference_value']
    atomic_write_csv(output_dir / 'sensitivity_run_metrics.csv', per_run)
    atomic_write_csv(output_dir / 'sensitivity_seed_spread.csv', seeds)
    headline = seeds[seeds.artifact == 'ablation_summary.csv'].copy()
    if not headline.empty:
        headline['variant'] = headline.context.map(lambda c: json.loads(c).get('variant'))
        settings_table = pd.DataFrame([
            {'setting': s['setting'], 'parameter': s['parameter'], **s['weighting']}
            for s in plan['settings']])
        atomic_write_csv(output_dir / 'sensitivity_ablation_summary.csv',
                         headline.merge(settings_table, on='setting', validate='many_to_one'))
    atomic_write_csv(output_dir / 'sensitivity_setting_spread.csv', settings)
    if parameter_spreads:
        atomic_write_csv(output_dir / 'sensitivity_parameter_spread.csv', pd.concat(parameter_spreads))
    atomic_write_csv(output_dir / 'sensitivity_paired_deltas.csv', paired)
    atomic_write_csv(output_dir / 'sensitivity_delta_spread.csv',
                     spread(paired, ['setting', *keys], 'delta_vs_reference_setting'))
    atomic_write_csv(output_dir / 'sensitivity_artifact_coverage.csv', pd.concat(coverage))
    _plot_summary(output_dir, seeds)


def _plot_summary(output_dir, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected = summary[summary.artifact == 'ablation_summary.csv'].copy()
    chosen = ['mean_wasserstein_scaled', 'correlation_distance', 'mean_cdf_tail_divergence',
              'detector_auc', 'privacy_median_distance_ratio', 'privacy_exact_match_rate']
    chosen += [m for m in selected.metric.unique()
               if m.startswith('utility_') and any(m.endswith(s) for s in
                  ('roc_auc', 'pr_auc', 'precision', 'recall', 'f1', 'f1_macro', 'balanced_accuracy'))]
    for metric in dict.fromkeys(chosen):
        subset = selected[selected.metric == metric].copy()
        if subset.empty:
            continue
        subset['variant'] = subset.context.map(lambda x: json.loads(x).get('variant', 'unknown'))
        fig, ax = plt.subplots(figsize=(12, 5))
        labels = subset.setting.drop_duplicates().tolist()
        for variant, group in subset.groupby('variant'):
            group = group.set_index('setting').reindex(labels)
            # A marker with no error bar is not evidence of zero variability.
            ax.errorbar(range(len(labels)), group['mean'], yerr=group['std'],
                        marker='o', capsize=3, label=variant)
        ax.set_xticks(range(len(labels)), labels, rotation=35, ha='right')
        ax.set(xlabel='Weighting setting (see study_plan.json)', ylabel=metric,
               title='Validation mean ± sample SD across seeds (SD unavailable for n=1)')
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f'sensitivity_{metric}.png', dpi=160)
        plt.close(fig)

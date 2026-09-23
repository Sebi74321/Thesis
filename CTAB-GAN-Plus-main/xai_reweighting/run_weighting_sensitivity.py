"""Validation-only, one-factor-at-a-time weighting study using full ablation runs."""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .io_utils import atomic_write_csv, atomic_write_json, file_sha256
from .run_ablation import VALID_VARIANTS, _code_hash, _fingerprint, _load_config, run_experiment


PARAMETERS = ('alpha', 'gamma', 'w_max', 'top_k')
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


def make_settings(config, grid=None):
    baseline = copy.deepcopy(config.get('weighting', {}))
    for key, value in dict(alpha=4.0, gamma=0.6, w_max=3.0, top_k=5).items():
        baseline.setdefault(key, value)
    if grid is None:
        grid = {
            'alpha': [baseline['alpha'] * 2],
            'gamma': [baseline['gamma'] * 1.5],
            'w_max': [1 + (baseline['w_max'] - 1) * 2],
            'top_k': [baseline['top_k'] * 2],
        }
    if set(grid) - set(PARAMETERS):
        raise ValueError(f'Grid keys must be in {PARAMETERS}')

    def validate(weights):
        for key in PARAMETERS:
            val = weights[key]
            if not np.isfinite(val) or val < (1 if key in {'w_max', 'top_k'} else 0):
                raise ValueError(f'Invalid weighting value {key}={val}')
            if key == 'top_k' and int(val) != val:
                raise ValueError('top_k must be an integer')
        weights['top_k'] = int(weights['top_k'])

    validate(baseline)
    settings = [{'setting': 'reference', 'parameter': 'reference', 'value': None,
                 'weighting': baseline}]
    for key in PARAMETERS:
        for value in dict.fromkeys(grid.get(key, [])):
            if value == baseline[key]:
                continue
            weights = dict(baseline, **{key: value})
            validate(weights)
            settings.append({'setting': f'{key}_{len(settings):02d}', 'parameter': key,
                             'value': value, 'weighting': weights})
    return settings


def _json_numbers(obj, prefix=''):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from _json_numbers(value, f'{prefix}.{key}' if prefix else key)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield prefix, obj


def collect_run(run_dir, setting, seed):
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
    for parameter in PARAMETERS:
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


def runtime_estimate(settings, seeds, variants, baseline_run_minutes=60, runtime_margin=1.25):
    if (not np.isfinite(baseline_run_minutes) or baseline_run_minutes <= 0
            or not np.isfinite(runtime_margin) or runtime_margin < 1):
        raise ValueError('Baseline runtime must be positive and runtime margin must be >= 1')
    runs = len(settings) * len(seeds)
    hours = runs * len(variants) / len(VALID_VARIANTS) * baseline_run_minutes / 60
    return {'ablation_runs': runs, 'gan_fits': runs * len(variants),
            'estimated_hours': hours, 'buffered_hours': hours * runtime_margin,
            'baseline_full_ablation_minutes': baseline_run_minutes,
            'runtime_margin': runtime_margin,
            'assumption': 'linear scaling from measured six-variant training plus evaluation; not a deadline guarantee'}


def run_study(config, project_root, output_dir, *, grid=None, seeds=(42, 43, 44),
              split_seed=42, variants=VALID_VARIANTS, device='auto', smoke=False,
              resume=False, dry_run=False, summarize_only=False, progress='auto',
              experiment_runner=run_experiment, baseline_run_minutes=60,
              budget_hours=20, runtime_margin=1.25):
    if not np.isfinite(budget_hours) or budget_hours <= 0:
        raise ValueError('budget_hours must be positive')
    output_dir = Path(output_dir).resolve()
    seeds = list(dict.fromkeys(int(s) for s in seeds))
    variants = list(dict.fromkeys(variants))
    if not seeds or not variants or set(variants) - set(VALID_VARIANTS):
        raise ValueError('Provide seeds and valid ablation variants')
    if not smoke and len(seeds) < 3:
        raise ValueError('Authoritative sensitivity studies require at least three distinct seeds; use --smoke for a pilot')
    if 'A0' not in variants:
        raise ValueError('A0 is required as the ablation control')
    settings = make_settings(config, grid)
    if smoke:
        seeds, settings = [seeds[0]], settings[:2]
    estimate = runtime_estimate(settings, seeds, variants, baseline_run_minutes, runtime_margin)
    print(f"Planned: {estimate['gan_fits']} GAN fits, {estimate['estimated_hours']:.1f} hours "
          f"estimated; {estimate['buffered_hours']:.1f} hours with margin; budget {budget_hours:g} hours.", flush=True)
    plan = dict(config=config, settings=settings, seeds=seeds, variants=variants,
                split_seed=int(split_seed), device=device, smoke=smoke, stage='val',
                data_sha256=file_sha256(project_root / config['data_path']),
                code_sha256=_code_hash(project_root),
                design='one_factor_at_a_time', thesis_results=not smoke)
    if dry_run:
        print(json.dumps(plan, indent=2))
        print(f'{len(settings) * len(seeds)} full ablation runs; '
              f'{len(settings) * len(seeds) * len(variants)} GAN fits (A0 included).')
        return output_dir
    if not summarize_only and not smoke and estimate['buffered_hours'] > budget_hours:
        raise ValueError('Estimated study exceeds runtime budget. Reduce the grid/seeds/variants, '
                         'or explicitly change the budget after reviewing --dry-run.')
    plan_path = output_dir / 'study_plan.json'
    if output_dir.exists():
        if not (resume or summarize_only):
            raise FileExistsError('Study exists; use --resume or a new output directory')
        saved = json.loads(plan_path.read_text())
        if saved != plan:
            raise ValueError('Study fingerprint mismatch: settings, config, seeds, device, data or code changed')
    else:
        if resume or summarize_only:
            raise FileNotFoundError('No existing sensitivity study to resume/summarize')
        atomic_write_json(plan_path, plan)
    if summarize_only:
        summarize_study(output_dir, plan)
        return output_dir
    atomic_write_json(output_dir / 'runtime_budget.json', {**estimate, 'budget_hours': budget_hours,
                      'scope': 'graceful launch budget per invocation; in-flight runs are not killed'})
    atomic_write_json(output_dir / 'manifest.json', {'status': 'running', 'thesis_results': not smoke})
    completed = []
    session_start = time.monotonic()
    longest_child_seconds = 0.0
    predicted_child_seconds = baseline_run_minutes * 60 * len(variants) / len(VALID_VARIANTS)
    try:
        for setting in settings:
            for seed in seeds:
                child = output_dir / setting['setting'] / f'seed{seed}'
                child_manifest = child / 'manifest.json'
                cfg = copy.deepcopy(config)
                cfg.update(seed=seed, split_seed=split_seed, weighting=setting['weighting'])
                if not (child_manifest.exists() and
                        json.loads(child_manifest.read_text()).get('status') == 'complete'):
                    elapsed = time.monotonic() - session_start
                    needed = max(predicted_child_seconds, longest_child_seconds) * runtime_margin
                    if elapsed + needed > budget_hours * 3600:
                        summarize_study(output_dir, plan)
                        atomic_write_json(output_dir / 'manifest.json', {
                            'status': 'budget_paused', 'completed_runs': completed,
                            'elapsed_seconds': elapsed, 'next_run': str(child.relative_to(output_dir)),
                            'estimated_next_run_seconds_with_margin': needed, 'thesis_results': not smoke})
                        print('Runtime budget reached: saved completed work. Continue with --resume.', flush=True)
                        return output_dir
                    print(f"Sensitivity {setting['setting']}, seed {seed}", flush=True)
                    child_start = time.monotonic()
                    experiment_runner(cfg, project_root, 'val', device, variants,
                                      output_override=child, resume=child_manifest.exists(),
                                      smoke=smoke, progress=progress)
                    longest_child_seconds = max(longest_child_seconds, time.monotonic() - child_start)
                completed.append(str(child.relative_to(output_dir)))
                summarize_study(output_dir, plan)
    except BaseException as exc:
        atomic_write_json(output_dir / 'manifest.json', {
            'status': 'interrupted', 'completed_runs': completed,
            'error': str(exc), 'thesis_results': not smoke})
        raise
    atomic_write_json(output_dir / 'manifest.json', {
        'status': 'complete', 'completed_runs': completed, 'thesis_results': not smoke})
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--grid', type=Path, help='JSON parameter-to-values map; OAT, not Cartesian product')
    parser.add_argument('--seeds', default='42,43,44')
    parser.add_argument('--baseline-run-minutes', type=float, default=60,
                        help='Measured total runtime for all six variants including evaluation on this dataset/device')
    parser.add_argument('--budget-hours', type=float, default=20)
    parser.add_argument('--runtime-margin', type=float, default=1.25)
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--variants', default=','.join(VALID_VARIANTS))
    parser.add_argument('--device', default='auto')
    parser.add_argument('--progress', choices=['auto', 'on', 'off'], default='auto')
    for flag in ('smoke', 'resume', 'dry-run', 'summarize-only'):
        parser.add_argument('--' + flag, action='store_true')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    output = run_study(_load_config(args.config.resolve()), root, args.output_dir,
                       grid=json.loads(args.grid.read_text()) if args.grid else None,
                       seeds=[int(s) for s in args.seeds.split(',')], split_seed=args.split_seed,
                       variants=[s.strip().upper() for s in args.variants.split(',')],
                       device=args.device, smoke=args.smoke, resume=args.resume,
                       dry_run=args.dry_run, summarize_only=args.summarize_only, progress=args.progress,
                       baseline_run_minutes=args.baseline_run_minutes, budget_hours=args.budget_hours,
                       runtime_margin=args.runtime_margin)
    print(output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

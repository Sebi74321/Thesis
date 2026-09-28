"""Validation-only, one-factor-at-a-time weighting study using full ablation runs."""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np

from .io_utils import atomic_write_json, file_sha256
from .run_ablation import VALID_VARIANTS, _code_hash, _load_config, run_experiment
# Re-export the reporting helpers for existing imports.
from .sensitivity_reporting import collect_run, spread, summarize_study


PARAMETERS = ('alpha', 'gamma', 'w_max', 'top_k')


def make_settings(config, grid=None):
    """Build independent perturbations without changing unrelated settings."""
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


def runtime_estimate(settings, seeds, variants, baseline_run_minutes=60, runtime_margin=1.25):
    """Scale a measured six-variant runtime for planning, not deadline enforcement."""
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

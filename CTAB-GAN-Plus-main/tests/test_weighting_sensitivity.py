import json

import numpy as np
import pandas as pd
import pytest

from xai_reweighting.io_utils import atomic_write_csv, atomic_write_json
from xai_reweighting.run_ablation import _code_hash, _fingerprint
from xai_reweighting.io_utils import file_sha256
from xai_reweighting.run_weighting_sensitivity import (
    collect_run, make_settings, run_study, spread, runtime_estimate,
)


def test_oat_settings_change_exactly_one_parameter():
    settings = make_settings({'weighting': {'alpha': 3., 'gamma': .6, 'w_max': 3., 'top_k': 5}})
    assert len(settings) == 5
    base = settings[0]['weighting']
    for setting in settings[1:]:
        changed = [k for k in base if base[k] != setting['weighting'][k]]
        assert changed == [setting['parameter']]
    with pytest.raises(ValueError):
        make_settings({}, {'top_k': [1.5]})
    with pytest.raises(ValueError):
        make_settings({}, {'alpha': [-1]})
    assert len(make_settings({'weighting': {'alpha': 3}}, {'alpha': [3, 3, 6, 6]})) == 2


def test_spread_is_sample_sd_and_does_not_invent_single_seed_certainty():
    frame = pd.DataFrame({'metric': ['a', 'a', 'b'], 'value': [1., 3., 5.]})
    result = spread(frame, ['metric']).set_index('metric')
    assert result.loc['a', 'mean'] == 2
    assert result.loc['a', 'std'] == pytest.approx(np.sqrt(2))
    assert result.loc['a', 'range'] == 2
    assert np.isnan(result.loc['b', 'std'])


def test_repeat_averaging_preserves_tasks_fractions_and_exact_spikes(tmp_path):
    atomic_write_csv(tmp_path / 'utility_mixture_results.csv', pd.DataFrame({
        'variant': ['A5'] * 4, 'utility_task': ['mortality'] * 4,
        'protocol': ['additive'] * 4, 'synthetic_fraction': [0., 0., 1., 1.],
        'repeat': [0, 1, 0, 1], 'seed': [10, 11, 10, 11],
        'recall': [.1, .3, .6, .8]}))
    atomic_write_csv(tmp_path / 'prioritized_feature_value_spikes.csv', pd.DataFrame({
        'feature': ['pre_icu_los_days'] * 2, 'value': [0.000694444, 1 / 1440],
        'real_frequency': [.1, 0.], 'synthetic_frequency': [0., .1]}))
    atomic_write_csv(tmp_path / 'synthetic_A5.csv', pd.DataFrame({'patient': [123]}))
    rows, coverage = collect_run(tmp_path, 'reference', 42)
    utility = rows[rows.metric == 'recall']
    assert len(utility) == 2
    assert sorted(utility.value) == pytest.approx([.2, .7])
    assert utility.observations.tolist() == [2, 2]
    assert 'synthetic_A5.csv' not in coverage.artifact.tolist()
    assert rows[rows.artifact == 'prioritized_feature_value_spikes.csv'].context.nunique() == 2


def test_full_study_resume_spreads_and_identity(tmp_path, monkeypatch):
    root = tmp_path / 'project'
    root.mkdir()
    (root / 'data.csv').write_text('x,y\n1,0\n2,1\n')
    config = {'data_path': 'data.csv', 'weighting': {'alpha': 3}}
    calls = []

    def fake_experiment(cfg, project, stage, device, variants, **kwargs):
        calls.append((cfg['weighting']['alpha'], cfg['seed'], cfg['split_seed']))
        path = kwargs['output_override']
        assert stage == 'val'
        atomic_write_json(path / 'split_indices.json', {'train': [0], 'val': [1]})
        atomic_write_json(path / 'config.json', cfg)
        atomic_write_json(path / 'manifest.json', {
            'status': 'complete', 'variants_completed': variants,
            'fingerprint': _fingerprint(cfg, file_sha256(project / 'data.csv'), _code_hash(project))})
        atomic_write_csv(path / 'ablation_summary.csv', pd.DataFrame({
            'variant': variants,
            'mean_wasserstein_scaled': [cfg['weighting']['alpha'] + cfg['seed']] * len(variants)}))
        atomic_write_csv(path / 'utility_mixture_results.csv', pd.DataFrame({
            'variant': ['A5'], 'protocol': ['additive'], 'synthetic_fraction': [1.],
            'utility_task': ['mortality_balanced'], 'recall': [.8]}))
        atomic_write_json(path / 'metrics_A5.json', {'privacy': {'median': .1}})
        return path

    output = root / 'study'
    options = dict(grid={'alpha': [1., 3., 6.]}, seeds=[42, 43, 44],
                   variants=['A0', 'A5'], experiment_runner=fake_experiment)
    run_study(config, root, output, **options)
    assert len(calls) == 9
    assert {call[2] for call in calls} == {42}
    report = pd.read_csv(output / 'sensitivity_seed_spread.csv')
    values = report[(report.artifact == 'ablation_summary.csv') & (report.setting == 'reference')]
    assert set(values['n']) == {3}
    assert values['std'].iloc[0] == pytest.approx(1.0)
    assert (output / 'sensitivity_mean_wasserstein_scaled.png').exists()
    deltas = pd.read_csv(output / 'sensitivity_paired_deltas.csv')
    values = deltas[(deltas.setting == 'alpha_01') & (deltas.artifact == 'ablation_summary.csv')]
    assert set(values.delta_vs_reference_setting) == {-2.}
    run_study(config, root, output, resume=True, **options)
    assert len(calls) == 9
    (root / 'data.csv').write_text('x,y\n3,0\n2,1\n')
    with pytest.raises(ValueError, match='fingerprint mismatch'):
        run_study(config, root, output, resume=True, **options)


def test_smoke_and_dry_run_do_not_launch_authoritative_study(tmp_path):
    (tmp_path / 'data.csv').write_text('x\n1\n')
    output = tmp_path / 'study'
    run_study({'data_path': 'data.csv'}, tmp_path, output, smoke=True, dry_run=True)
    assert not output.exists()


def test_ablation_uses_split_seed_independently_of_training_seed(tmp_path, monkeypatch):
    from xai_reweighting.run_ablation import run_experiment
    pd.DataFrame({'x': range(100), 'target': [0, 1] * 50}).to_csv(tmp_path / 'data.csv', index=False)
    captured = []

    def capture(data, target, *, seed, **kwargs):
        captured.append(seed)
        raise RuntimeError('captured split before any model training')

    monkeypatch.setattr('xai_reweighting.run_ablation.create_data_splits', capture)
    for seed in (43, 44):
        config = {'data_path': 'data.csv', 'target_col': 'target',
                  'categorical_cols': ['target'], 'generator': {},
                  'seed': seed, 'split_seed': 42}
        with pytest.raises(RuntimeError, match='captured split'):
            run_experiment(config, tmp_path, 'val', 'cpu', ['A0'],
                           output_override=tmp_path / str(seed), adapter_factory=object)
    assert captured == [42, 42]


def test_interrupted_study_resumes_without_repeating_completed_children(tmp_path):
    (tmp_path / 'data.csv').write_text('x,y\n1,0\n2,1\n')
    calls = []

    def fake(cfg, root, stage, device, variants, **kwargs):
        calls.append(cfg['weighting']['alpha'])
        if len(calls) == 2:
            raise RuntimeError('simulated interruption')
        child = kwargs['output_override']
        atomic_write_json(child / 'config.json', cfg)
        atomic_write_json(child / 'manifest.json', {
            'status': 'complete', 'variants_completed': variants,
            'fingerprint': _fingerprint(cfg, file_sha256(root / 'data.csv'), _code_hash(root))})
        atomic_write_json(child / 'split_indices.json', {'train': [0]})
        atomic_write_csv(child / 'ablation_summary.csv', pd.DataFrame({
            'variant': variants, 'mean_wasserstein_scaled': [.1] * len(variants)}))

    cfg = {'data_path': 'data.csv', 'weighting': {'alpha': 3}}
    options = dict(grid={'alpha': [6]}, seeds=[42, 43, 44], variants=['A0', 'A5'], experiment_runner=fake)
    output = tmp_path / 'study'
    with pytest.raises(RuntimeError, match='simulated interruption'):
        run_study(cfg, tmp_path, output, **options)
    assert json.loads((output / 'manifest.json').read_text())['status'] == 'interrupted'
    run_study(cfg, tmp_path, output, resume=True, **options)
    assert calls == [3, 3, 3, 3, 6, 6, 6]
    assert json.loads((output / 'manifest.json').read_text())['status'] == 'complete'
    assert (pd.read_csv(output / 'sensitivity_seed_spread.csv')['n'] == 3).all()


def test_default_budget_retains_three_seeds_and_fits_under_twenty_hours():
    estimate = runtime_estimate(make_settings({}), [42, 43, 44], ['A0', 'A1', 'A2', 'A3', 'A4', 'A5'])
    assert estimate['gan_fits'] == 90
    assert estimate['estimated_hours'] == 15
    assert estimate['buffered_hours'] == 18.75


def test_excess_budget_and_insufficient_seeds_are_rejected_before_training(tmp_path):
    (tmp_path / 'data.csv').write_text('x\n1\n')
    output = tmp_path / 'study'
    with pytest.raises(ValueError, match='exceeds runtime budget'):
        run_study({'data_path': 'data.csv'}, tmp_path, output, baseline_run_minutes=120)
    assert not output.exists()
    with pytest.raises(ValueError, match='three distinct seeds'):
        run_study({'data_path': 'data.csv'}, tmp_path, output, seeds=[42, 43, 43])
    assert not output.exists()


def test_budget_guard_pauses_before_launching_new_child(tmp_path, monkeypatch):
    (tmp_path / 'data.csv').write_text('x\n1\n')
    ticks = iter([0., 72000.])
    monkeypatch.setattr('xai_reweighting.run_weighting_sensitivity.time.monotonic', lambda: next(ticks))
    def must_not_launch(*args, **kwargs):
        raise AssertionError('No new training is allowed beyond the budget')
    output = tmp_path / 'study'
    run_study({'data_path': 'data.csv'}, tmp_path, output, experiment_runner=must_not_launch)
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['status'] == 'budget_paused'
    assert manifest['completed_runs'] == []
    assert manifest['next_run']

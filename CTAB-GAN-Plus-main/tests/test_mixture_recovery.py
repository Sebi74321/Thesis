import json

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning

from xai_reweighting.run_ablation import _fingerprint, _validate_resume, build_parser


def test_nonconverged_valid_mixture_is_accepted(monkeypatch):
    from model.synthesizer import transformer as module
    original = module.BayesianGaussianMixture.fit

    def fit(self, values):
        original(self, values)
        self.converged_ = False
        return self

    monkeypatch.setattr(module.BayesianGaussianMixture, 'fit', fit)
    data = pd.DataFrame({'x': np.linspace(90, 100, 60)})
    transformer = module.DataTransformer(train_data=data, n_clusters=2)
    with pytest.warns(ConvergenceWarning, match='numerically validated'):
        transformer.fit()
    record = transformer.mixture_diagnostics[0]
    assert len(record['attempts']) == 2
    assert record['accepted_nonconverged'] and record['usable']
    assert not record['converged']
    assert np.isfinite(transformer.transform(data.values)).all()


@pytest.mark.parametrize('corruption', ['means', 'variance', 'probabilities'])
def test_invalid_mixtures_still_fail(monkeypatch, corruption):
    from model.synthesizer import transformer as module
    original = module.BayesianGaussianMixture.fit

    def fit(self, values):
        original(self, values)
        if corruption == 'means':
            self.means_[:] = np.nan
        elif corruption == 'variance':
            self.covariances_[:] = 0
        else:
            self.predict_proba = lambda x: np.full((len(x), 2), 0.1)
        return self

    monkeypatch.setattr(module.BayesianGaussianMixture, 'fit', fit)
    transformer = module.DataTransformer(n_clusters=2)
    with pytest.raises(RuntimeError, match='Unusable'):
        transformer._fit_mixture(np.linspace(0, 1, 60), 0, 'continuous')
    assert not transformer.mixture_diagnostics[-1]['usable']


def saved_run(path):
    config = {'seed': 42, 'stage': 'val', 'variants': ['A0', 'A5']}
    manifest = {'fingerprint': _fingerprint(config, 'data', 'old'),
                'data_sha256': 'data', 'code_sha256': 'old'}
    (path / 'config.json').write_text(json.dumps(config))
    (path / 'manifest.json').write_text(json.dumps(manifest))
    (path / '.A0.complete').write_text('complete')
    (path / 'metrics_A0.json').write_text('{}')
    return config


def test_resume_requires_explicit_code_change_acceptance(tmp_path):
    config = saved_run(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(ValueError, match='fingerprint mismatch'):
        _validate_resume(tmp_path, config, 'data', 'new')
    previous, event = _validate_resume(tmp_path, config, 'data', 'new', True)
    assert event['completed_variants_preserved'] == ['A0']
    assert event['previous_manifest'] == previous
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
    assert _validate_resume(tmp_path, config, 'data', 'old')[1] is None


@pytest.mark.parametrize('change', ['config', 'data', 'fingerprint'])
def test_recovery_never_bypasses_scientific_identity(tmp_path, change):
    config = saved_run(tmp_path)
    data_hash = 'data'
    if change == 'config':
        config['seed'] = 43
    elif change == 'data':
        data_hash = 'different'
    else:
        (tmp_path / 'manifest.json').write_text(json.dumps({
            'fingerprint': 'corrupt', 'data_sha256': 'data', 'code_sha256': 'old'}))
    with pytest.raises(ValueError, match='identical saved config and data'):
        _validate_resume(tmp_path, config, data_hash, 'new', True)


def test_recovery_cli_is_opt_in():
    parser = build_parser()
    assert not parser.parse_args(['--config', 'config.json']).resume_allow_code_change
    assert parser.parse_args(['--config', 'config.json', '--resume',
                              '--resume-allow-code-change']).resume_allow_code_change

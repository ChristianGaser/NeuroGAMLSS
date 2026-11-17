"""Test of scripts/simulate_brainage.py on a small design."""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import simulate_brainage as SB  # noqa: E402


def test_age_distributions():
    rng = np.random.default_rng(0)
    for dist in SB.DISTRIBUTIONS:
        age = SB.draw_ages(dist, 4000, rng)
        assert age.min() >= SB.AGE_RANGE[0] and age.max() <= SB.AGE_RANGE[1]
    mid = np.mean(SB.AGE_RANGE)
    assert np.mean(SB.draw_ages('young', 4000, rng)) < mid < np.mean(SB.draw_ages('old', 4000, rng))
    bim = SB.draw_ages('bimodal', 4000, rng)
    assert np.mean(np.abs(bim - mid) < 10) < 0.15


def test_summarize_finds_the_bias():
    rng = np.random.default_rng(1)
    age = rng.uniform(*SB.AGE_RANGE, 5000)
    edges = np.linspace(*SB.AGE_RANGE, SB.N_BINS + 1)
    s = SB.summarize(-0.2 * (age - 53) + rng.standard_normal(5000), age, edges)
    assert abs(s['slope_all'] + 0.2) < 0.01 and s['bias'][0] > 5 and s['bias'][-1] < -5
    assert np.allclose(s['slope'], -0.2, atol=0.1)


def test_simulation_runs(tmp_path):
    out = str(tmp_path / 'sim')
    cond = SB.main(['--scenarios', 'gm', '--distributions', 'uniform', 'young', '--sizes', '150',
                    '--reps', '2', '--n-test', '300', '--features', '200', '--pca', '10',
                    '--kfold', '3', '--bias-replicates', '20', '--out', out])
    assert set(cond) == {('gm', 'uniform', 150), ('gm', 'young', 150)}
    agg = cond[('gm', 'uniform', 150)]
    for s in SB.SETS:
        assert {'ndm_loglik', 'ndm_mae', 'gpr_raw', 'gpr_trend'} <= set(agg['summary'][s])
    test = agg['summary']['test']
    assert 'ndm_bc' in test and test['ndm_loglik']['bias']['mean'].shape == (SB.N_BINS,)
    # the uncorrected regression is shrunk toward the mean age, its trend correction is not
    assert test['gpr_raw']['slope_all']['mean'] < -0.05
    assert abs(test['gpr_trend']['slope_all']['mean']) < 0.05
    assert 0.5 < agg['cover_lr']['mean'] <= 1 and sum(agg['df_loglik'].values()) == 2
    with open(out + '.json') as f:
        saved = json.load(f)
    assert len(saved['runs']) == 4 and 'gm/uniform/150' in saved['conditions']
    with open(out + '.md') as f:
        assert 'training ages young' in f.read()

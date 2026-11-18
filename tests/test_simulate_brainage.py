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
        assert {'ndm', 'ndm_trend', 'gpr_raw', 'gpr_trend'} <= set(agg['summary'][s])
    test = agg['summary']['test']
    assert set(test) == set(SB.METHODS)
    assert test['ndm']['bias']['mean'].shape == (SB.N_BINS,)
    # the uncorrected regression is shrunk toward the mean age, its trend correction is not
    assert test['gpr_raw']['slope_all']['mean'] < -0.05
    assert abs(test['gpr_trend']['slope_all']['mean']) < 0.05
    assert abs(test['ndm_trend']['slope_all']['mean']) < 0.05
    # and it recovers less of a true difference in brain age than the likelihood
    rec = agg['recovery']
    assert set(rec) == {'ndm', 'ndm_model_hat', 'gpr_raw'}
    assert 0.2 < rec['gpr_raw']['mean'] < rec['ndm']['mean'] < 1.3
    assert 0.5 < agg['cover_lr']['mean'] <= 1 and sum(agg['df'].values()) == 2
    with open(out + '.json') as f:
        saved = json.load(f)
    assert len(saved['runs']) == 4 and 'gm/uniform/150' in saved['conditions']
    with open(out + '.md') as f:
        assert 'training ages young' in f.read()


def test_model_bias_is_the_curve_of_the_sex():
    cur = dict(age=np.array([20.0, 40.0, 60.0]), male=np.array([0.0, 1.0]),
               bias=np.array([[1.0, 2.0], [0.0, 0.0], [-1.0, -4.0]]))
    b = SB.model_bias(cur, np.array([30.0, 30.0, 50.0, 70.0]), np.array([0.0, 1.0, 1.0, 0.0]))
    assert np.allclose(b, [0.5, 1.0, -2.0, -1.0])
    one = dict(age=cur['age'], male=np.array([0.0]), bias=cur['bias'][:, :1])
    assert np.allclose(SB.model_bias(one, np.array([30.0]), np.array([1.0])), [0.5])


def test_variant_reproduces_the_estimator():
    """The axes of pca_axes are those of the estimator, and a variant with the training
    data and these axes gives the same brain ages."""
    world = SB.World('gm', p=200)
    rng = np.random.default_rng(2)
    d = world.sample(SB.draw_ages('uniform', 500, rng), rng)
    est = SB.NG.NDMBrainAge(df_mu=2, df_sigma=1, pca=8).fit(d)
    centre, axes = SB.pca_axes(d, 8)
    part = est.parts[0]
    cos = np.sum(axes * part.Vt, axis=1) / (np.linalg.norm(axes, axis=1)
                                            * np.linalg.norm(part.Vt, axis=1))
    assert np.allclose(centre, part.center, atol=1e-6) and np.allclose(np.abs(cos), 1)
    same = SB.variant(est, d, (centre, axes))
    assert np.allclose(same.predict(d)['age'], est.predict(d)['age'], atol=1e-3)
    assert same.parts[0].model is not part.model and est.parts[0] is part


def test_sources_of_the_bias(tmp_path):
    out = str(tmp_path / 'sim')
    cond = SB.main(['--sources', '--scenarios', 'gm', '--distributions', 'uniform', '--sizes',
                    '200', '--reps', '2', '--n-test', '400', '--features', '150', '--pca', '10',
                    '--components', '5', '--n-big', '3000', '--out', out])
    res = cond[('gm', 'uniform', 200)]
    assert {'pipeline', 'new_scores', 'big_models', 'big_axes', 'big_both', 'big_flexible',
            'no PCA', '5 components'} <= set(res)
    # test subjects are distributed as models of many new subjects say, and their scores
    # are less dispersed than the training scores that took part in the PCA
    assert abs(res['big_both']['var_z']['mean'] - 1) < 0.1
    assert res['pipeline']['var_z']['mean'] < res['new_scores']['var_z']['mean']
    assert sum(res['pipeline']['df'].values()) == 2 and 'df' not in res['big_both']
    with open(out + '_sources.md') as f:
        assert '| big_axes |' in f.read()
    with open(out + '_sources.json') as f:
        assert 'gm/uniform/200' in json.load(f)['conditions']

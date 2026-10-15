"""Tests of mat2image.py on synthetic data: python -m pytest tests"""
import os
import sys

import numpy as np
import pytest
from scipy.io import loadmat, savemat

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mat2image as M2I  # noqa: E402
import neurogamlss as NG  # noqa: E402

nib = pytest.importorskip('nibabel')


def write_sample(path, n, p, seed, **extra):
    """Input mat-file of BA_data2mat.m: data that fall with age, feature 0 near zero."""
    rng = np.random.default_rng(seed)
    age = rng.uniform(20, 80, n)
    Y = 0.5 + 0.004 * (80 - age)[:, None] * rng.uniform(0.5, 1.5, p)
    Y += 0.05 * rng.standard_normal((n, p))
    Y[:, 0] = 0.01 * rng.random(n)                           # outside the mask
    savemat(path, dict(Y=Y.astype(np.float32), age=age[:, None],
                       male=rng.integers(0, 2, n).astype(float)[:, None], **extra))
    return path


def zmaps(tmp_path, model, p, **extra):
    """z-maps of 30 test subjects and diagnostics of NeuroGAMLSS: prefix of the files."""
    tr = write_sample(str(tmp_path / f'{model}_TR_CAT12.9.mat'), 200, p, 1, **extra)
    te = write_sample(str(tmp_path / f'{model}_TE_CAT12.9.mat'), 30, p, 2, **extra)
    out = str(tmp_path / 'r')
    NG.main(['--normative-only', '--train', tr, '--test', te, '--family', 'normal',
             '--df-mu', '2', '--df-sigma', '1', '--diagnostics', '--out', out])
    return out


def write_mesh(path, n_vertices):
    rng = np.random.default_rng(0)
    nib.save(nib.gifti.GiftiImage(darrays=[
        nib.gifti.GiftiDataArray(rng.random((n_vertices, 3)).astype(np.float32),
                                 intent='NIFTI_INTENT_POINTSET'),
        nib.gifti.GiftiDataArray(np.array([[0, 1, 2]], np.int32),
                                 intent='NIFTI_INTENT_TRIANGLE')]), path)
    return path


def test_grid_of_cat_io_data2mat():
    for res, dim in (('8', (23, 27, 23)), ('4', (45, 54, 45)), ('2', (91, 109, 91))):
        shape, affine = M2I.volume_grid(res)           # V.dim and V.mat of the mat-files
        r = float(res)
        spm_mat = np.diag([r, r, r, 1])                 # 1-based voxels of SPM
        spm_mat[:3, 3] = np.array([-90, -126, -72]) - r
        shift = np.eye(4)
        shift[:3, 3] = 1                                # 0-based voxels of NIfTI
        assert shape == dim and np.allclose(affine, spm_mat @ shift)
    for res, n in (('4', 29852), ('8', 3747)):         # indices in the atlas folder
        _, shape, _, ind = M2I.volume_space(f's4rp1_{res}mm_X_CAT12.9.mat', n)
        assert ind.size == n and ind.max() <= np.prod(shape)


def test_volume_zmaps_and_diagnostics(tmp_path):
    shape, _ = M2I.volume_grid('8')
    ind = np.sort(np.random.default_rng(0).choice(np.prod(shape), 40, replace=False)) + 1
    savemat(str(tmp_path / 'ind_8mm.mat'), dict(ind=ind[:, None].astype(float)))
    out = zmaps(tmp_path, 's4rp1_8mm', 40, ind=np.zeros((0, 1)), dim=np.array([[23, 27, 23]]))
    zfile = out + '_zmaps_s4rp1_8mm_TE_CAT12.9.mat'
    with pytest.raises(SystemExit):                     # 3747 indices of atlases/ind_8mm.mat
        M2I.main([zfile])
    M2I.main([zfile, '--ind', str(tmp_path)])           # folder with ind_8mm.mat
    img = nib.load(out + '_zmaps_s4rp1_8mm_TE_CAT12.9.nii')
    Z = loadmat(zfile, squeeze_me=True, struct_as_record=False)['NDMzmap'].Z
    flat = img.get_fdata().reshape(-1, 30, order='F')   # MATLAB order of find(mask)
    assert img.shape == shape + (30,) and np.allclose(img.affine, M2I.volume_grid('8')[1])
    assert np.allclose(flat[ind - 1].T, Z, equal_nan=True) and np.isnan(Z[:, 0]).all()
    assert np.isnan(np.delete(flat, ind - 1, axis=0)).all()
    assert img.header['sform_code'] == 2 and img.header.get_intent()[0] == 'z score'
    M2I.main([out + '_diagnostics_s4rp1_8mm_TR_CAT12.9.mat',
              '--ind', str(tmp_path / 'ind_8mm.mat')])
    q = nib.load(out + '_diagnostics_s4rp1_8mm_TR_CAT12.9_Q.nii')
    diag = loadmat(out + '_diagnostics_s4rp1_8mm_TR_CAT12.9.mat', squeeze_me=True,
                   struct_as_record=False)['NDMdiag']
    assert q.shape == shape + (4,)                      # mean, variance, skewness, kurtosis
    assert np.allclose(q.get_fdata().reshape(-1, 4, order='F')[ind - 1].T, diag.Q, equal_nan=True)
    conv = nib.load(out + '_diagnostics_s4rp1_8mm_TR_CAT12.9_converged.nii')
    assert conv.shape == shape


def test_input_data_needs_a_resolution(tmp_path):
    shape, _ = M2I.volume_grid('8')
    ind = np.sort(np.random.default_rng(1).choice(np.prod(shape), 40, replace=False)) + 1
    savemat(str(tmp_path / 'ind.mat'), dict(ind=ind[:, None].astype(float)))
    f = write_sample(str(tmp_path / 'rp1_X.mat'), 5, 40, 3)
    with pytest.raises(SystemExit):                     # no resolution such as _8mm_ in the name
        M2I.main([f, '--ind', str(tmp_path / 'ind.mat')])
    M2I.main([f, '--ind', str(tmp_path / 'ind.mat'), '--res', '8', '--split'])
    img = nib.load(str(tmp_path / 'rp1_X_3.nii'))
    assert img.shape == shape
    assert np.allclose(img.get_fdata().ravel(order='F')[ind - 1], loadmat(f)['Y'][2])


def test_surface_zmaps(tmp_path):
    ind = np.sort(np.random.default_rng(2).choice(100, 60, replace=False)) + 1   # mesh of 100
    out = zmaps(tmp_path, 's12.mesh.thickness', 60, ind=ind[None].astype(float),
                dim=np.array([[100.0]]))
    zfile = out + '_zmaps_s12.mesh.thickness_TE_CAT12.9.mat'
    M2I.main([zfile])
    g = nib.load(out + '_zmaps_s12.mesh.thickness_TE_CAT12.9.gii')
    Z = loadmat(zfile, squeeze_me=True, struct_as_record=False)['NDMzmap'].Z
    data = np.array([d.data for d in g.darrays])
    assert data.shape == (30, 100) and np.allclose(data[:, ind - 1], Z, equal_nan=True)
    assert np.isnan(np.delete(data, ind - 1, axis=1)).all() and 'SurfaceID' not in g.meta
    mesh = write_mesh(str(tmp_path / 'mesh.gii'), 100)
    M2I.main([zfile, '--subjects', '2,5', '--split', '--mesh', mesh,
              '--outdir', str(tmp_path / 'split')])
    g5 = nib.load(str(tmp_path / 'split' / 'r_zmaps_s12.mesh.thickness_TE_CAT12.9_05.gii'))
    assert len(os.listdir(str(tmp_path / 'split'))) == 2 and g5.meta['SurfaceID'] == mesh
    assert np.allclose(g5.darrays[0].data[ind - 1], Z[4], equal_nan=True)
    with pytest.raises(SystemExit):                     # mesh with other vertices
        M2I.main([zfile, '--mesh', write_mesh(str(tmp_path / 'mesh80.gii'), 80)])
    M2I.main([out + '_diagnostics_s12.mesh.thickness_TR_CAT12.9.mat'])
    q = nib.load(out + '_diagnostics_s12.mesh.thickness_TR_CAT12.9_p_Q.gii')
    assert [d.meta['Name'] for d in q.darrays] == list(M2I.MOMENTS)


def test_number_of_vertices_without_dim():
    n_vertices, ind = M2I.surface_space('s12.mesh.thickness_X.mat', 3, np.array([1., 5., 40000.]))
    assert n_vertices == 64984 and ind.dtype == np.int64     # both hemispheres of the 32k mesh
    with pytest.raises(ValueError):
        M2I.surface_space('s12.mesh.thickness_X.mat', 4, np.array([1., 5., 40000.]))


def test_volume_indices_stored_in_the_mat_files(tmp_path):
    shape, _ = M2I.volume_grid('8')
    ind = np.sort(np.random.default_rng(3).choice(np.prod(shape), 40, replace=False)) + 1
    out = zmaps(tmp_path, 's4rp1_8mm', 40, ind=ind[:, None].astype(float))
    M2I.main([out + '_zmaps_s4rp1_8mm_TE_CAT12.9.mat'])       # stored ind, no --ind
    img = nib.load(out + '_zmaps_s4rp1_8mm_TE_CAT12.9.nii')
    flat = img.get_fdata().reshape(-1, 30, order='F')
    assert np.isfinite(flat[ind[1:] - 1]).all() and np.isnan(flat[ind[0] - 1]).all()
    assert np.isnan(np.delete(flat, ind - 1, axis=0)).all()

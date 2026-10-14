#!/usr/bin/env python3
"""
mat2image.py - NIfTI and GIFTI images of the maps of NeuroGAMLSS.

NeuroGAMLSS stores maps with one value per voxel or vertex, in the order of the
columns of Y in its input mat-files (BA_data2mat.m, cat_io_data2mat.m of CAT12).
mat2image.py writes them as images that can be overlaid onto a template:

  <out>_zmaps_<model>.mat        z-maps Z of all subjects -> <out>_zmaps_<model>.nii
  <out>_diagnostics_<model>.mat  Q statistics Q and their p-values p_Q, each as four
                                 maps (mean, variance, skewness, kurtosis), and the
                                 convergence -> <name>_Q.nii, <name>_p_Q.nii and
                                 <name>_converged.nii
  input mat-files                data Y of all subjects -> <name>.nii

Surface data give .gii instead of .nii.  Voxels and vertices without a value, such
as those outside the mask or of the medial wall, are NaN.

Volume data
-----------
cat_io_data2mat.m resamples the images to r mm on a grid of round([181 217 181]/r)
voxels whose first voxel lies at (-90, -126, -72) mm in MNI space.  The columns of Y
are the voxels inside the brain mask in MATLAB order.  The mat-files
store no indices of these voxels (ind is empty for volume data), so they are read
from ind_<r>mm.mat: from --ind, or else from the atlas folder of NeuroGAMLSS, which
contains them for 4 and 8 mm.  The resolution r comes from the model name, as in
s4rp1_8mm_..., or from --res.  The maps are written as float32 NIfTI-1 files with
this grid as sform and qform: one 4D file with a volume per subject or map, or with
--split one 3D file each.

Surface data
------------
Surface data are resampled CAT12 meshes, such as the 64984 vertices of both
hemispheres of the 32k mesh (mesh.*, left hemisphere first).  Input mat-files,
z-maps and diagnostics store the 1-based indices ind of the vertices with values
and the number of vertices dim.  The maps are written as GIFTI files with a data
array per subject or map, or with --split one file each.  CAT12 displays them on
its template surface with the same number of vertices (cat_surf_display,
cat_surf_results).  SPM finds the surface of a data file in its metadata
(SurfaceID), which --mesh sets.

Examples
--------
z-maps of 4 mm gray matter, with the indices of atlases/ind_4mm.mat:

    python mat2image.py adni_zmaps_s4rp1_4mm_ADNI_CAT12.9.mat

z-maps of the first 10 subjects at 8 mm as 3D files, with the indices of another
folder:

    python mat2image.py adni_zmaps_s4rp1_8mm_ADNI_CAT12.9.mat --subjects 1:10 --split \\
        --ind /Volumes/UltraMax/BrainAGE

z-maps of cortical thickness on the 32k mesh, with the template surface for SPM:

    python mat2image.py adni_zmaps_s12.mesh.thickness_ADNI_CAT12.9.mat \\
        --mesh <CAT12>/templates_surfaces_32k/mesh.central.freesurfer.gii

Requirements: numpy, scipy, nibabel; h5py for input mat-files of MATLAB v7.3
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass

import numpy as np

import neurogamlss as NG

# grid of cat_io_data2mat.m and BA_atlas2mat.m: the 1 mm grid of REF_DIM voxels whose
# first voxel lies at REF_ORIGIN (mm), resampled to r mm as round(REF_DIM / r) voxels
# with the same first voxel
REF_DIM = (181, 217, 181)
REF_ORIGIN = (-90.0, -126.0, -72.0)

# vertices of the resampled CAT12 meshes: one hemisphere and both (mesh.*) of the 32k
# and the 164k mesh
MESH_SIZES = (32492, 64984, 163842, 327684)

MOMENTS = ('mean', 'variance', 'skewness', 'kurtosis')


@dataclass
class Maps:
    """Maps of one variable: values (k, n_features), with the names of the k maps or
    None for one map per subject, the NIfTI intent and a title."""
    suffix: str
    values: np.ndarray
    labels: tuple | None
    intent: str
    title: str


def _vector(x):
    """A numeric variable as a 1D array, None if it is missing or empty."""
    if x is None:
        return None
    x = np.asarray(x, dtype=np.float64).ravel()
    return x if x.size else None


def read_maps(path):
    """Maps of a mat-file: z-maps (NDMzmap) or diagnostics (NDMdiag) of NeuroGAMLSS, or
    data Y of an input mat-file.  Returns the model name (the names of the input
    files), the list of Maps and the stored ind and dim (None if not stored)."""
    from scipy.io import loadmat, whosmat
    try:
        names = {v[0] for v in whosmat(path)}
    except NotImplementedError:          # MATLAB v7.3: input data of BA_data2mat.m
        names = set()
    key = next((k for k in ('NDMzmap', 'NDMdiag') if k in names), None)
    if key is None:
        d = NG.load_data(path)
        return d.name, [Maps('', d.Y, None, 'NIFTI_INTENT_NONE', 'data')], d.ind, d.dim
    s = loadmat(path, variable_names=[key], squeeze_me=True, struct_as_record=False)[key]
    if key == 'NDMzmap':
        maps = [Maps('', np.atleast_2d(s.Z), None, 'NIFTI_INTENT_ZSCORE', 'z-maps')]
    else:
        maps = [Maps('_Q', np.atleast_2d(s.Q), MOMENTS, 'NIFTI_INTENT_NONE', 'Q statistics'),
                Maps('_p_Q', np.atleast_2d(s.p_Q), MOMENTS, 'NIFTI_INTENT_PVAL',
                     'p-values of Q'),
                Maps('_converged', np.atleast_2d(s.converged), ('converged',),
                     'NIFTI_INTENT_NONE', 'convergence')]
    return (str(s.model), maps, _vector(getattr(s, 'ind', None)),
            _vector(getattr(s, 'dim', None)))


def load_index(path):
    """1-based indices ind of a mat-file (MATLAB v5 or v7.3) and its dim, if stored."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"index file {path} not found: give it with --ind.")
    from scipy.io import loadmat
    try:
        m = loadmat(path, variable_names=['ind', 'dim'])
    except NotImplementedError:          # v7.3 is HDF5
        import h5py
        with h5py.File(path, 'r') as h:
            m = {k: h[k][()] for k in ('ind', 'dim')
                 if k in h and not h[k].attrs.get('MATLAB_empty', 0)}
    ind = _vector(m.get('ind'))
    if ind is None:
        raise ValueError(f"{path} contains no indices ind.")
    return ind.astype(np.int64), _vector(m.get('dim'))


def _check_index(ind, n, size, source, unit):
    """Raise an error unless ind are n distinct 1-based indices of size voxels/vertices."""
    if ind.size != n:
        raise ValueError(f"{source} has {ind.size} indices for {n} {unit} of the maps.")
    if ind.min() < 1 or ind.max() > size or np.unique(ind).size != ind.size:
        raise ValueError(f"the indices of {source} are repeated or outside the {size} {unit} "
                         "of the data.")


def volume_grid(res):
    """Dimensions and affine (0-based voxels -> mm) of the data of cat_io_data2mat.m
    resampled to res mm."""
    r = float(res)
    dim = tuple(int(np.floor(k / r + 0.5)) for k in REF_DIM)      # round() of MATLAB
    affine = np.diag([r, r, r, 1.0])
    affine[:3, 3] = REF_ORIGIN
    return dim, affine


def volume_space(model, n, ind=None, dim=None, ind_spec=None, res=None):
    """Resolution, grid (dim, affine) and 1-based voxel indices of the n columns of a
    volume model.  The indices come from ind_spec (a file or a folder with
    ind_<res>mm.mat), else from ind stored with the maps, else from the atlas folder."""
    if res is None:
        m = re.search(r'_(\d+(?:\.\d+)?)mm_', os.path.basename(model.split('+')[0]))
        if m is None:
            raise ValueError(f"no resolution such as _8mm_ in the name {model}: give --res.")
        res = m.group(1)
    res = f'{float(res):g}'
    shape, affine = volume_grid(res)
    if dim is not None and tuple(int(k) for k in dim) != shape:
        raise ValueError(f"the volume dimensions {[int(k) for k in dim]} of {model} are not "
                         f"those of cat_io_data2mat.m at {res} mm, {list(shape)}.")
    source = 'the mat-file'
    if ind_spec is not None or ind is None:
        source = ind_spec or NG.ATLAS_DIR
        if os.path.isdir(source):
            source = os.path.join(source, f'ind_{res}mm.mat')
        ind = load_index(source)[0]
    ind = ind.astype(np.int64)
    _check_index(ind, n, int(np.prod(shape)), source, 'voxels')
    return res, shape, affine, ind


def mesh_vertices(path):
    """Number of vertices of a GIFTI surface."""
    import nibabel as nib
    pointset = nib.nifti1.intent_codes['NIFTI_INTENT_POINTSET']
    sizes = [d.data.shape[0] for d in nib.load(path).darrays if d.intent == pointset]
    if not sizes:
        raise ValueError(f"{path} contains no surface (vertices).")
    return sizes[0]


def surface_space(model, n, ind=None, dim=None, ind_spec=None, mesh=None):
    """Number of vertices of the mesh and 1-based vertex indices of the n columns of a
    surface model.  The indices come from ind_spec (the input mat-file of the model)
    or from ind stored with the maps."""
    if ind_spec is not None:
        ind, dim_spec = load_index(ind_spec)
        dim = dim if dim is not None else dim_spec
    if ind is None:
        raise ValueError(f"no vertex indices ind stored with the maps: give the input "
                         f"mat-file of {model} with --ind.")
    n_mesh = None if mesh is None else mesh_vertices(mesh)
    if dim is not None:
        n_vertices = int(dim[0])
    elif n_mesh is not None:
        n_vertices = n_mesh
    else:
        n_vertices = next((k for k in MESH_SIZES if k >= ind.max()), int(ind.max()))
        print(f"Note: no number of vertices stored with the maps of {model}; a mesh of "
              f"{n_vertices} vertices is assumed.")
    if n_mesh is not None and n_mesh != n_vertices:
        raise ValueError(f"{mesh} has {n_mesh} vertices, but the data have {n_vertices}.")
    ind = ind.astype(np.int64)
    _check_index(ind, n, n_vertices, ind_spec or 'the mat-file', 'vertices')
    return n_vertices, ind


def _meta(d):
    """GIFTI metadata of a dict (nibabel 3 and newer)."""
    from nibabel.gifti import GiftiMetaData
    return GiftiMetaData(d) if hasattr(GiftiMetaData, 'keys') else GiftiMetaData.from_dict(d)


def write_nifti(path, values, shape, affine, ind, intent, title):
    """values (k, n) at the 1-based voxel indices ind (MATLAB order) of a grid as a 3D
    (k = 1) or 4D float32 NIfTI-1 file, NaN elsewhere."""
    import nibabel as nib
    k = values.shape[0]
    data = np.full((k, int(np.prod(shape))), np.nan, dtype=np.float32)
    data[:, ind - 1] = values
    # every row is in MATLAB (column-major) order: (k, z, y, x) in C order -> (x, y, z, k)
    data = data.reshape((k,) + shape[::-1]).transpose(3, 2, 1, 0)
    img = nib.Nifti1Image(data[..., 0] if k == 1 else data, affine)
    img.set_sform(affine, code=2)              # 'aligned', as written by SPM
    img.set_qform(affine, code=2)
    img.header.set_xyzt_units('mm')
    img.header.set_intent(intent)
    img.header['descrip'] = f'NeuroGAMLSS {title}'[:79]
    nib.save(img, path)


def write_gifti(path, values, n_vertices, ind, names, intent, mesh=None):
    """values (k, n) at the 1-based vertex indices ind as a GIFTI file with k data
    arrays, NaN elsewhere; mesh, if given, is the surface of the data (SurfaceID)."""
    import nibabel as nib
    from nibabel.gifti import GiftiDataArray, GiftiImage
    full = np.full((values.shape[0], n_vertices), np.nan, dtype=np.float32)
    full[:, ind - 1] = values
    darrays = [GiftiDataArray(v, intent=intent, datatype='NIFTI_TYPE_FLOAT32',
                              encoding='GIFTI_ENCODING_B64BIN', meta=_meta({'Name': name}))
               for v, name in zip(full, names)]
    meta = _meta({'SurfaceID': os.path.abspath(mesh)} if mesh else {})
    nib.save(GiftiImage(darrays=darrays, meta=meta), path)


def convert(path, ind=None, res=None, mesh=None, subjects=None, split=False, outdir=None):
    """Write the maps of the mat-file path as NIfTI (volume data) or GIFTI (surface
    data) files, named like the mat-file, and return their names.

    ind      : mat-file with the 1-based indices ind of the voxels or vertices, instead
               of those stored with the maps; for volume data also a folder with
               ind_<res>mm.mat (default: the atlas folder)
    res      : resolution of volume data in mm, if the model name contains none
    mesh     : GIFTI surface of surface data, stored as their SurfaceID
    subjects : 1-based indices of the subjects such as '1:10,15' (default: all)
    split    : one file per subject or map instead of one file with all maps
    outdir   : output folder (default: the folder of path)
    """
    model, maps, ind_stored, dim = read_maps(path)
    n = maps[0].values.shape[1]
    surf = 'mesh' in os.path.basename(model.split('+')[0])
    if surf:
        n_vertices, index = surface_space(model, n, ind_stored, dim, ind, mesh)
        print(f"{os.path.basename(path)}: surface data, {n} of {n_vertices} vertices")
    else:
        res, shape, affine, index = volume_space(model, n, ind_stored, dim, ind, res)
        print(f"{os.path.basename(path)}: volume data, {n} voxels of a {res} mm grid of "
              f"{' x '.join(map(str, shape))} voxels")
    base = os.path.join(outdir or os.path.dirname(path),
                        os.path.splitext(os.path.basename(path))[0])
    files = []
    for m in maps:
        values, labels, names = m.values, m.labels, m.labels
        if labels is None:                     # one map per subject
            sel = (np.arange(len(values)) if subjects is None
                   else NG._parse_index(subjects, len(values)))
            values = values[sel]
            labels = tuple(str(i + 1).zfill(len(str(len(m.values)))) for i in sel)
            names = tuple(f'subject {i + 1}' for i in sel)
        if split and (m.labels is None or len(labels) > 1):
            parts = [(f'_{lab}', values[j:j + 1], names[j:j + 1]) for j, lab in enumerate(labels)]
        else:
            parts = [('', values, names)]
        for tag, v, nm in parts:
            out = base + m.suffix + tag + ('.gii' if surf else '.nii')
            if surf:
                write_gifti(out, v, n_vertices, index, nm, m.intent, mesh)
            else:
                write_nifti(out, v, shape, affine, index, m.intent, m.title)
            files.append(out)
    return files


def main(argv=None):
    p = argparse.ArgumentParser(
        description="NIfTI (volume data) and GIFTI (surface data) images of the maps of "
                    "NeuroGAMLSS (z-maps, diagnostics) and of its input mat-files.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument('files', nargs='+', metavar='FILE',
                   help="mat-files: z-maps (<out>_zmaps_<model>.mat), diagnostics "
                        "(<out>_diagnostics_<model>.mat) or input data of BA_data2mat.m")
    p.add_argument('--ind', help="mat-file with the 1-based indices ind of the voxels or "
                                 "vertices of the maps, instead of those stored with them: "
                                 "for volume data ind_<res>mm.mat or a folder that contains it "
                                 "(default: ind_<res>mm.mat of the atlas folder), for surface "
                                 "data the input mat-file of the model")
    p.add_argument('--res', help="volume data: resolution in mm, if the model name contains "
                                 "none such as _8mm_")
    p.add_argument('--mesh', help="surface data: GIFTI surface of the maps, stored for SPM as "
                                  "their SurfaceID, e.g. templates_surfaces_32k/"
                                  "mesh.central.freesurfer.gii of CAT12")
    p.add_argument('--subjects', help="1-based indices of the subjects, e.g. 1:10,15 "
                                      "(default: all)")
    p.add_argument('--split', action='store_true',
                   help="one file per subject or map instead of one 4D NIfTI file or one "
                        "GIFTI file with all maps")
    p.add_argument('--outdir', help="output folder (default: the folder of each mat-file)")
    a = p.parse_args(argv)
    try:
        if a.outdir:
            os.makedirs(a.outdir, exist_ok=True)
        for path in a.files:
            files = convert(path, a.ind, a.res, a.mesh, a.subjects, a.split, a.outdir)
            print("  Saved " + (', '.join(files) if len(files) <= 3 else
                                f"{files[0]} ... {files[-1]} ({len(files)} files)"))
    except ImportError as err:
        p.error(f"{err}: pip install {err.name}" if err.name else str(err))
    except (OSError, ValueError) as err:
        p.error(str(err))


if __name__ == '__main__':
    main()

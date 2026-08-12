import numpy as np

from pymv3d import MatViz3DResult, Col


def test_set_discovery_ignores_last_set(synthetic_hdf5):
    r = MatViz3DResult(synthetic_hdf5["path"])
    assert len(r) == 2
    assert r.set_ids == ["1", "2"]           # 1-based names, 'last_set' skipped


def test_scalars_and_loadstep_counts(synthetic_hdf5):
    r = MatViz3DResult(synthetic_hdf5["path"])
    assert r.cube_size(0) == synthetic_hdf5["S"]
    assert r.num_points(0) == synthetic_hdf5["n_grains"]
    assert r.n_loadsteps(0) == synthetic_hdf5["n_ls"]
    assert r.n_loadsteps(1) == 0             # stiffness-mode set has no ls_ groups


def test_microstructure_arrays(synthetic_hdf5):
    r = MatViz3DResult(synthetic_hdf5["path"])
    vox = r.voxels(0)
    assert vox.shape == (5, 5, 5) and vox.dtype == np.int32
    assert r.local_cs(0).shape == (6, 3)


def test_effective_tensors(synthetic_hdf5):
    r = MatViz3DResult(synthetic_hdf5["path"])
    C, S = r.stiffness(0), r.compliance(0)
    assert C.shape == (6, 6) and S.shape == (6, 6)
    assert np.allclose(C, synthetic_hdf5["Cmat"], atol=1e-4)
    assert np.allclose(C @ S, np.eye(6), atol=1e-3)   # C is the inverse of S
    assert r.effective_moduli(0).shape == (6,)
    assert r.poisson_matrix(0).shape == (6, 6)
    # stiffness-only set also exposes the tensors
    assert r.stiffness(1) is not None


def test_results_table_and_named_columns(synthetic_hdf5):
    r = MatViz3DResult(synthetic_hdf5["path"])
    tab = r.results(0, 1)
    assert tab.shape == (125, 22)
    assert r.column(0, 1, Col.SX).shape == (125,)
    assert r.eps_as_loading(0, 2)[0] == np.float32(2e-3)


def test_field_scatter_is_coordinate_correct(synthetic_hdf5):
    r = MatViz3DResult(synthetic_hdf5["path"])
    sf = r.stress_field(0, 3)
    assert sf.shape == (5, 5, 5, 6)
    ix, iy, iz = 2, 1, 3
    idx = iz * 25 + iy * 5 + ix
    assert sf[ix, iy, iz, 0] == np.float32(100.0 * 3 + idx)
    vm = r.scalar_field(0, 3, Col.SEQV)
    assert vm.shape == (5, 5, 5)
    assert vm[ix, iy, iz] == np.float32(10.0 * 3 + idx)


def test_macro_response(synthetic_hdf5):
    r = MatViz3DResult(synthetic_hdf5["path"])
    m = r.macro_response(0)
    assert m["eps_applied"].shape == (4, 6)
    assert m["stress_avg"].shape == (4, 6)
    assert m["strain_avg"].shape == (4, 6)
    assert m["seqv_avg"].shape == (4,)


def test_missing_set_raises(synthetic_hdf5):
    r = MatViz3DResult(synthetic_hdf5["path"])
    import pytest
    with pytest.raises(IndexError):
        r.voxels(99)

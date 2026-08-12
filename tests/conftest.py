"""Shared fixtures: a schema-accurate synthetic HDF5 file and a fake MatViz3D exe.

The synthetic file reproduces MatViz3D's exact on-disk layout so the reader can be
tested without a real solver run:
  * set "1" — full `dataset` mode: ls_ groups + effective S/C/P/moduli
  * set "2" — `stiffness` mode: only the effective tensors, no ls_ groups
  * /last_set — a root scalar that must be ignored by the reader
"""

import os
import stat

import numpy as np
import h5py
import pytest

from pymv3d import Col

NCOLS = 22


@pytest.fixture(scope="session")
def synthetic_hdf5(tmp_path_factory):
    S, n_ls, n_grains = 5, 4, 6
    path = tmp_path_factory.mktemp("data") / "dataset.hdf5"
    rng = np.random.default_rng(0)

    Cmat = (rng.random((6, 6)) + np.eye(6) * 5).astype("f4")
    Smat = np.linalg.inv(Cmat).astype("f4")
    Pmat = np.zeros((6, 6), dtype="f4")
    for i in range(6):
        for j in range(6):
            Pmat[i, j] = -Smat[i, j] / Smat[j, j] if abs(Smat[j, j]) > 1e-30 else 0.0
    moduli = np.array([1.0 / Smat[i, i] for i in range(6)], dtype="f4")
    vox = rng.integers(1, n_grains + 1, size=(S, S, S)).astype("i4")
    local_cs = rng.random((n_grains, 3)).astype("f4")

    def write_tensors(grp):
        grp.create_dataset("S_matrix", data=Smat)
        grp.create_dataset("C_matrix", data=Cmat)
        grp.create_dataset("P_matrix", data=Pmat)
        grp.create_dataset("Effective_Moduli", data=moduli)

    with h5py.File(path, "w") as f:
        f.create_dataset("last_set", data=2, dtype="i4")

        g1 = f.create_group("1")
        g1.create_dataset("voxels", data=vox, dtype="i4")
        g1.create_dataset("cubeSize", data=S, dtype="i4")
        g1.create_dataset("numPoints", data=n_grains, dtype="i4")
        g1.create_dataset("local_cs", data=local_cs)
        for k in range(1, n_ls + 1):
            ls = g1.create_group(f"ls_{k}")
            tab = np.zeros((S * S * S, NCOLS), dtype="f4")
            idx = 0
            for iz in range(S):
                for iy in range(S):
                    for ix in range(S):
                        tab[idx, Col.ID] = idx + 1
                        tab[idx, Col.X] = ix
                        tab[idx, Col.Y] = iy
                        tab[idx, Col.Z] = iz
                        tab[idx, Col.SX] = 100.0 * k + idx
                        tab[idx, Col.SEQV] = 10.0 * k + idx
                        idx += 1
            ls.create_dataset("results", data=tab)
            ls.create_dataset("results_avg", data=tab.mean(0))
            ls.create_dataset("results_max", data=tab.max(0))
            ls.create_dataset("results_min", data=tab.min(0))
            ls.create_dataset("eps_as_loading",
                              data=np.array([1e-3 * k, 0, 0, 0, 0, 0], dtype="f4"))
        write_tensors(g1)

        g2 = f.create_group("2")
        g2.create_dataset("voxels", data=vox, dtype="i4")
        g2.create_dataset("cubeSize", data=S, dtype="i4")
        g2.create_dataset("numPoints", data=n_grains, dtype="i4")
        g2.create_dataset("local_cs", data=local_cs)
        write_tensors(g2)

    return {"path": str(path), "S": S, "n_ls": n_ls, "n_grains": n_grains,
            "Cmat": Cmat, "Smat": Smat}


@pytest.fixture(scope="session")
def fake_exe(tmp_path_factory):
    """A stand-in for the MatViz3D binary: echoes argv, touches --output."""
    path = tmp_path_factory.mktemp("bin") / "FakeMatViz3D.sh"
    path.write_text(
        '#!/usr/bin/env bash\n'
        'echo "ARGV: $@"\n'
        'out=""\n'
        'while [ $# -gt 0 ]; do if [ "$1" = "--output" ]; then out="$2"; fi; shift; done\n'
        '[ -n "$out" ] && : > "$out"\n'
        'exit 0\n'
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)

"""
Read MatViz3D HDF5 output into NumPy arrays.

On-disk schema (written by stressanalysis_fft.cpp / stressanalysis.cpp /
stressanalysiscontroller.cpp, all through hdf5wrapper.cpp)
--------------------------------------------------------------------------
/last_set                     int   run counter (NOT a data set -> ignored)
/{n}/                         group one run, n = 1,2,3,... (1-based!)
    voxels            int32  [S,S,S]   grain-id per voxel
    cubeSize          int    scalar    S
    numPoints         int    scalar    number of seed points / grains
    local_cs          float  [G, k]    per-grain orientation rows
    S_matrix          float  [6,6]     effective compliance      (stiffness/dataset mode)
    C_matrix          float  [6,6]     effective stiffness       (stiffness/dataset mode)
    P_matrix          float  [6,6]     nu_ij = -S_ij/S_jj        (stiffness/dataset mode)
    Effective_Moduli  float  [6]       1/S_ii                    (stiffness/dataset mode)
    ls_{k}/                   group one load step, k = 1..num_loads
        results        float [nvox, 22]  per-voxel table (see ResCol)
        results_avg    float [22]        column-wise mean
        results_max    float [22]        column-wise max
        results_min    float [22]        column-wise min
        eps_as_loading float [6]         applied macro strain exx,eyy,ezz,exy,eyz,exz

22-column `results` layout (fft_solver_session.hpp :: enum ResCol)
    0  ID          1..3  X,Y,Z (0-based voxel coords)   4..6  UX,UY,UZ (0 for FFT)
    7..12  SX,SY,SZ,SXY,SYZ,SXZ            (Cauchy stress, Pa)
    13..18 EpsX,EpsY,EpsZ,EpsXY,EpsYZ,EpsXZ (ENGINEERING strain; shear = 2*eps_tensor)
    19  USUM        20  SEQV (von Mises stress)   21  EpsEQV (equivalent strain)
"""

from __future__ import annotations

from enum import IntEnum
import warnings
import numpy as np
import h5py


class Col(IntEnum):
    """Column indices into the `results` table."""
    ID = 0
    X = 1; Y = 2; Z = 3
    UX = 4; UY = 5; UZ = 6
    SX = 7; SY = 8; SZ = 9; SXY = 10; SYZ = 11; SXZ = 12
    EX = 13; EY = 14; EZ = 15; EXY = 16; EYZ = 17; EXZ = 18
    USUM = 19; SEQV = 20; EEQV = 21
    NCOLS = 22


STRESS_COLS = [Col.SX, Col.SY, Col.SZ, Col.SXY, Col.SYZ, Col.SXZ]
STRAIN_COLS = [Col.EX, Col.EY, Col.EZ, Col.EXY, Col.EYZ, Col.EXZ]
_IGNORED_ROOT = {"last_set"}


class MatViz3DResult:
    """Lazy reader over one MatViz3D HDF5 file.

    Sets are addressed by 0-based position `g` (g=0 -> HDF5 group "1").
    Use `.set_ids` to see the underlying 1-based group names.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._loaded = False
        self._set_ids: list[str] = []          # e.g. ["1","2",...]
        self._sets: list[dict] = []            # cached scalar/small metadata per set

    # -- structure discovery --------------------------------------------------
    def _load(self):
        if self._loaded:
            return
        with h5py.File(self.file_path, "r") as f:
            keys = [k for k in f.keys()
                    if k not in _IGNORED_ROOT and isinstance(f.get(k), h5py.Group)]
            keys.sort(key=lambda s: int(s) if s.isdigit() else float("inf"))
            self._set_ids = keys
            for name in keys:
                g = f[name]
                self._sets.append({
                    "cubeSize":  self._scalar(g, "cubeSize"),
                    "numPoints": self._scalar(g, "numPoints"),
                    "n_loadsteps": len([k for k in g.keys() if k.startswith("ls_")]),
                    "has_stiffness": "C_matrix" in g,
                })
        self._loaded = True

    @staticmethod
    def _scalar(group, name):
        if name not in group:
            return None
        v = group[name][()]
        if isinstance(v, np.ndarray):
            v = v.reshape(-1)[0] if v.size else None
        return None if v is None else v.item() if hasattr(v, "item") else v

    def _group_name(self, g: int) -> str:
        self._load()
        if not (0 <= g < len(self._set_ids)):
            raise IndexError(f"set g={g} does not exist (have {len(self._set_ids)})")
        return self._set_ids[g]

    # -- counts ---------------------------------------------------------------
    @property
    def set_ids(self) -> list[str]:
        self._load(); return list(self._set_ids)

    def __len__(self) -> int:
        self._load(); return len(self._set_ids)

    def n_loadsteps(self, g: int) -> int:
        self._load(); return self._sets[g]["n_loadsteps"]

    def cube_size(self, g: int) -> int:
        self._load(); return self._sets[g]["cubeSize"]

    def num_points(self, g: int) -> int:
        self._load(); return self._sets[g]["numPoints"]

    # -- microstructure -------------------------------------------------------
    def voxels(self, g: int) -> np.ndarray:
        """Grain-id array, shape (S,S,S), int32."""
        name = self._group_name(g)
        with h5py.File(self.file_path, "r") as f:
            ds = f[name].get("voxels")
            if ds is None:
                raise ValueError(f"'voxels' missing for set g={g}")
            return ds[()]

    def local_cs(self, g: int) -> np.ndarray | None:
        """Per-grain orientation rows, shape (G,k), or None."""
        name = self._group_name(g)
        with h5py.File(self.file_path, "r") as f:
            ds = f[name].get("local_cs")
            return None if ds is None else np.atleast_2d(ds[()])

    # -- effective (homogenized) tensors --------------------------------------
    def _matrix(self, g: int, key: str) -> np.ndarray | None:
        name = self._group_name(g)
        with h5py.File(self.file_path, "r") as f:
            ds = f[name].get(key)
            return None if ds is None else np.asarray(ds[()]).reshape(6, 6)

    def compliance(self, g: int) -> np.ndarray | None:
        """Effective compliance S, (6,6). None if not computed."""
        return self._matrix(g, "S_matrix")

    def stiffness(self, g: int) -> np.ndarray | None:
        """Effective stiffness C, (6,6). None if not computed."""
        return self._matrix(g, "C_matrix")

    def poisson_matrix(self, g: int) -> np.ndarray | None:
        """nu_ij = -S_ij/S_jj, (6,6). None if not computed."""
        return self._matrix(g, "P_matrix")

    def effective_moduli(self, g: int) -> np.ndarray | None:
        """[Ex,Ey,Ez, 2Gxy,2Gyz,2Gxz]-style vector (1/S_ii), (6,). None if absent."""
        name = self._group_name(g)
        with h5py.File(self.file_path, "r") as f:
            ds = f[name].get("Effective_Moduli")
            return None if ds is None else np.asarray(ds[()]).reshape(-1)

    # -- per-load-step tables -------------------------------------------------
    def _ls_dataset(self, g: int, ls: int, dset: str) -> np.ndarray | None:
        name = self._group_name(g)
        with h5py.File(self.file_path, "r") as f:
            grp = f[name].get(f"ls_{ls}")
            if grp is None:
                raise IndexError(f"set g={g} has no ls_{ls}")
            ds = grp.get(dset)
            if ds is None:
                return None
            arr = ds[()]
            return np.atleast_2d(arr).T if arr.ndim == 1 and dset == "results" else arr

    def results(self, g: int, ls: int) -> np.ndarray:
        """Raw per-voxel table, shape (nvox, 22)."""
        arr = self._ls_dataset(g, ls, "results")
        if arr is None:
            raise ValueError(f"'results' missing for g={g}, ls_{ls}")
        return np.asarray(arr, dtype=np.float32)

    def results_avg(self, g: int, ls: int) -> np.ndarray:
        return np.asarray(self._ls_dataset(g, ls, "results_avg")).reshape(-1)

    def results_max(self, g: int, ls: int) -> np.ndarray:
        return np.asarray(self._ls_dataset(g, ls, "results_max")).reshape(-1)

    def results_min(self, g: int, ls: int) -> np.ndarray:
        return np.asarray(self._ls_dataset(g, ls, "results_min")).reshape(-1)

    def eps_as_loading(self, g: int, ls: int) -> np.ndarray:
        """Applied macro strain (6,): exx,eyy,ezz,exy,eyz,exz."""
        return np.asarray(self._ls_dataset(g, ls, "eps_as_loading")).reshape(-1)

    def column(self, g: int, ls: int, col: Col) -> np.ndarray:
        """One named column from the results table, shape (nvox,)."""
        return self.results(g, ls)[:, int(col)]

    # -- fields reshaped to the voxel grid (Level-2 DeepONet targets) ---------
    def _scatter_to_grid(self, g: int, ls: int, cols) -> np.ndarray:
        """Scatter selected result columns back into an (S,S,S,len(cols)) grid,
        using the X,Y,Z columns so it is robust to row ordering / partial voxels."""
        S = self.cube_size(g)
        tab = self.results(g, ls)
        ix = tab[:, int(Col.X)].astype(np.intp)
        iy = tab[:, int(Col.Y)].astype(np.intp)
        iz = tab[:, int(Col.Z)].astype(np.intp)
        ncomp = len(cols)
        grid = np.zeros((S, S, S, ncomp), dtype=np.float32)
        vals = tab[:, [int(c) for c in cols]]
        grid[ix, iy, iz, :] = vals
        return grid

    def stress_field(self, g: int, ls: int) -> np.ndarray:
        """Per-voxel Cauchy stress, shape (S,S,S,6): SX,SY,SZ,SXY,SYZ,SXZ [Pa]."""
        return self._scatter_to_grid(g, ls, STRESS_COLS)

    def strain_field(self, g: int, ls: int) -> np.ndarray:
        """Per-voxel engineering strain, shape (S,S,S,6): EX,EY,EZ,EXY,EYZ,EXZ."""
        return self._scatter_to_grid(g, ls, STRAIN_COLS)

    def scalar_field(self, g: int, ls: int, col: Col) -> np.ndarray:
        """Any single column as an (S,S,S) grid, e.g. Col.SEQV for von Mises."""
        return self._scatter_to_grid(g, ls, [col])[..., 0]

    # -- convenience: whole dataset as one dict-of-arrays ---------------------
    def load_step_stack(self, g: int, dset: str = "results_avg") -> np.ndarray:
        """Stack a per-loadstep vector across all load steps, shape (n_ls, ...)."""
        n = self.n_loadsteps(g)
        rows = [np.asarray(self._ls_dataset(g, k, dset)).reshape(-1) for k in range(1, n + 1)]
        return np.stack(rows) if rows else np.empty((0,))

    def macro_response(self, g: int) -> dict:
        """All (eps_applied -> avg stress/strain) pairs for a set — the raw
        material for macro (Level-1) fitting. Returns arrays keyed by name."""
        n = self.n_loadsteps(g)
        eps = np.stack([self.eps_as_loading(g, k) for k in range(1, n + 1)]) if n else np.empty((0, 6))
        avg = self.load_step_stack(g, "results_avg")
        return {
            "eps_applied": eps,                       # (n_ls, 6)
            "stress_avg": avg[:, [int(c) for c in STRESS_COLS]] if avg.size else avg,
            "strain_avg": avg[:, [int(c) for c in STRAIN_COLS]] if avg.size else avg,
            "seqv_avg":   avg[:, int(Col.SEQV)] if avg.size else avg,
        }

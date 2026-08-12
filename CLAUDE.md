# CLAUDE.md

Guidance for AI coding sessions in this repository. Read this before changing code.

## What this is

Repository `matviz3d-py`; import package and PyPI distribution `pymv3d`.

`pymv3d` is a thin Python bridge to **MatViz3D** (C++/Qt6 polycrystalline
microstructure generator + FFT/ANSYS homogenizer, repo `MME-NTU-KhPI/MatViz3D`,
branch `qml-development`). It does two things and nothing else:

1. **`launcher.py`** — build a MatViz3D command line and run the binary headless
   (`subprocess`), for microstructure generation and stress/homogenization.
2. **`reader.py`** — load MatViz3D's HDF5 output into NumPy: voxel grids,
   per-grain orientations, effective S/C/P tensors, per-voxel stress/strain
   fields, and macro (applied-strain → avg-stress) response.

It is the CLI-wrapper half of the project's PIML data pipeline. It does **not**
bind any C++ (no pybind11); the interchange format is HDF5 on disk. If in-loop,
zero-copy access is ever needed, that is a separate future effort binding the
header-only FFT core — do not fold it into this package.

## Layout

```
pymv3d/
  launcher.py   MatViz3DLauncher, GenParams, StressParams, RunResult
  reader.py     MatViz3DResult, Col (IntEnum for the 22-col results table)
  __init__.py   public API
tests/
  conftest.py   synthetic HDF5 fixture + fake-exe fixture
  test_reader.py, test_launcher.py
examples/
```

## The one rule that matters: mirror the C++ contract

Two pieces of this package are hand-mirrors of MatViz3D source. When the C++
changes, these must be updated in lockstep — there is no automatic sync.

1. **CLI option registry** in `launcher.py` (`_VALUE_OPTIONS`, `_FLAG_OPTIONS`,
   `_REMOVED_OPTIONS`) mirrors
   [`commandline_parser.cpp::setupParser`](https://github.com/MME-NTU-KhPI/MatViz3D/blob/qml-development/commandline_parser.cpp).
   If you add/rename/remove a `parser.addOption(...)` there, update the registry
   here. Removed flags should be moved into `_REMOVED_OPTIONS` with a reason, not
   silently deleted, so old scripts get a clear error.

2. **HDF5 schema + column layout** in `reader.py` mirrors the writers in
   `stressanalysis_fft.cpp`, `stressanalysis.cpp`, `stressanalysiscontroller.cpp`
   (all via `hdf5wrapper.cpp`), and the `ResCol` enum in `fft_solver_session.hpp`.
   If a dataset name or a results column changes there, update `Col` and the
   accessor here.

When you touch either, add/adjust a test in `tests/` and note the upstream commit
you synced against.

## MatViz3D CLI (current, qml-development)

Generation: `--size`, `--points` **or** `--concentration` (%; wins over points),
`--algorithm` (e.g. `"Probability Algorithm"`), `--seed`, `--np`,
`--wave_coefficient`, `--halfaxis_a|b|c`, `--orientation_angle_a|b|c`,
`--ellipse_order`.
Texture: `--texture random|extrusion|rolling|recrystallization|shear|scattered_cube`,
`--lattice fcc|bcc`, `--scatter <deg>` (last two need `--texture`).
Stress: `--run_stress_calc`, `--solver ansys|fft`,
`--stress_mode single|dataset|stiffness`, `--eps exx,eyy,ezz,exy,eyz,exz`
(mode=single), `--num_rnd_loads`, `--working_directory`.
Flow: `--autostart` (generate), `--nogui`, `--output <file>`.

Gotchas baked into the wrapper:
- `--hasProbParameters` no longer exists; it is derived true when any
  `halfaxis_*`/`orientation_angle_*` is set. Do not pass it.
- `--wave_spread`, `--initial_nuclei_count`, `--stefan_number` were dropped from
  the CLI. If a future build reintroduces them, move them out of `_REMOVED_OPTIONS`
  back into `_VALUE_OPTIONS`.
- `stiffness` mode writes only the set-level tensors, **no `ls_` groups**.

## HDF5 schema (as written by the C++)

```
/last_set                       int    run counter — NOT data, reader ignores it
/{n}/                           group  one run, n = 1,2,3,... (1-based)
    voxels            int32 [S,S,S]     grain id per voxel
    cubeSize          int   scalar      S
    numPoints         int   scalar      seed points / grains
    local_cs          float [G,k]       per-grain orientation rows
    S_matrix|C_matrix|P_matrix  float [6,6]  effective compliance/stiffness/nu
    Effective_Moduli  float [6]         1/S_ii
    ls_{k}/                     group    load step k = 1..num_loads
        results        float [nvox,22]  per-voxel table (see below)
        results_avg|max|min  float [22] column-wise reductions
        eps_as_loading float [6]         applied macro strain
```

`results` 22 columns — `enum ResCol` in `fft_solver_session.hpp`:

```
0 ID | 1..3 X,Y,Z (0-based voxel coords) | 4..6 UX,UY,UZ (0 for FFT)
7..12  SX,SY,SZ,SXY,SYZ,SXZ              Cauchy stress [Pa]
13..18 EpsX,EpsY,EpsZ,EpsXY,EpsYZ,EpsXZ  ENGINEERING strain (shear = 2·eps_tensor)
19 USUM | 20 SEQV (von Mises) | 21 EpsEQV
```

Row index for a voxel is `idx = iz*S*S + iy*S + ix`; `ID = idx+1`. The reader
never relies on row order — it scatters by the X,Y,Z columns.

Interpretation notes for downstream work:
- `local_cs` rows are Bunge ZXZ Euler angles in MatViz3D's own convention. The
  ANSYS path uses ZXY Tait-Bryan; do not assume the two are interchangeable
  without the rotation-matrix intermediary. (This package just returns the raw
  array; conversion is the caller's responsibility.)
- Strain columns are engineering shear, whereas `eps_as_loading` is the applied
  macro strain tensor. Keep the factor-of-2 in mind when relating them.

## Conventions

- Sets addressed by 0-based position `g` (g=0 → HDF5 group `"1"`).
  `MatViz3DResult.set_ids` exposes the 1-based names.
- Dataclass fields left `None` are omitted from the command line (→ C++ default).
  Preserve this; do not emit `--opt None`.
- Prose/code comments in English; the wider MatViz3D project mixes English and
  Ukrainian, code comments are English.
- Reader opens the file per call (context-managed) and caches only small scalar
  metadata. Keep large arrays out of the cache.

## Testing

```bash
pip install -e ".[dev]"
pytest
```

`tests/conftest.py` fabricates a schema-accurate HDF5 (a `dataset`-mode set with
`ls_` groups + tensors, and a `stiffness`-mode set with only tensors) plus a fake
executable that echoes argv. No real MatViz3D binary is required. Any schema or
CLI change should come with a matching test update.

## Non-goals

- No C++ bindings, no Qt in the import path, no OpenGL.
- No plotting/analysis beyond returning clean NumPy — keep those in notebooks or
  the ML repo that consumes this.

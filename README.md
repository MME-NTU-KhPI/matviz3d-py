# pymv3d

> GitHub repository: **`matviz3d-py`** · import package: **`pymv3d`** · PyPI: `pymv3d`

Headless launcher and HDF5→NumPy reader for **[MatViz3D](https://github.com/MME-NTU-KhPI/MatViz3D)** —
the polycrystalline microstructure generator/solver developed at the Department of
Materials Mechanics, NTU "Kharkiv Polytechnic Institute".

`pymv3d` drives MatViz3D's command-line interface from Python to generate
microstructures and run FFT/ANSYS homogenization, then loads the resulting HDF5
into NumPy arrays — including per-voxel stress/strain fields and effective
stiffness tensors — ready for the PIML pipeline (Level-1 macro fitting,
Level-2 DeepONet localization).

## Install

```bash
pip install -e .          # from a clone
# or, with the test tooling
pip install -e ".[dev]"
```

Requires Python ≥ 3.10, `numpy`, `h5py`. A working MatViz3D binary is needed only
to *run* the solver; the reader works on any MatViz3D HDF5 file.

## Launch

```python
from pymv3d import MatViz3DLauncher, GenParams, StressParams

mv = MatViz3DLauncher("/path/to/MatViz3D")

# generate + full FFT dataset (Hill-sampled load cases)
mv.run(GenParams(size=40, concentration=2.0, algorithm="Probability Algorithm"),
       StressParams(run=True, solver="fft", mode="dataset", num_rnd_loads=300),
       output="dataset.hdf5", check=True)

# effective stiffness only (6 unit-strain solves -> S/C/P/moduli)
mv.run(GenParams(size=40, concentration=2.0),
       StressParams(run=True, solver="fft", mode="stiffness"),
       output="stiffness.hdf5")
```

Only the fields you set land on the command line; the rest fall back to MatViz3D's
own defaults. For bulk dataset generation, spawn one `mv.run(...)` per process
across your cores — each writes its own HDF5, keeping provenance clean.

### Why this replaces the old `MatViz3DLauncher`

The earlier draft passed flags the current CLI no longer defines, which makes
`QCommandLineParser::process()` abort before generation. `pymv3d` keeps an
option registry that mirrors `commandline_parser.cpp` and rejects them in Python
with a clear message:

| Old flag | Status now |
|---|---|
| `--wave_spread` | removed |
| `--initial_nuclei_count` | removed |
| `--stefan_number` | removed |
| `--hasProbParameters` | auto-derived from `halfaxis_*` / `orientation_angle_*` |

## Read

```python
from pymv3d import MatViz3DResult, Col

r = MatViz3DResult("dataset.hdf5")
r.set_ids                      # ['1','2',...]  (1-based; '/last_set' is ignored)

vox = r.voxels(0)              # (S,S,S) int32 grain ids
C   = r.stiffness(0)           # (6,6) effective C  (None if not computed)
S   = r.compliance(0)          # (6,6) effective S

sig = r.stress_field(0, ls=1)  # (S,S,S,6)  Level-2 DeepONet target
eps = r.strain_field(0, ls=1)  # (S,S,S,6)
vm  = r.scalar_field(0, 1, Col.SEQV)   # (S,S,S) von Mises

m = r.macro_response(0)        # applied strain -> volume-avg stress (Level-1)
```

Fields are scattered back onto the grid via the table's X,Y,Z columns, so results
are correct regardless of row order or partially-populated voxel lists.

## HDF5 schema

See [CLAUDE.md](CLAUDE.md) for the full on-disk schema, the 22-column `results`
layout, and the maintenance rules that keep this package in sync with the C++ CLI.

## Tests

```bash
pytest
```

Tests build a schema-accurate synthetic HDF5 (both `dataset` and `stiffness`
layouts) and a fake executable, so no real MatViz3D binary is required.

## License

MIT (placeholder) — **confirm and align with the MatViz3D project's licensing
policy before publishing**, since the upstream repository does not currently ship
an explicit license file.

"""Generate several microstructures + FFT datasets in parallel across cores.

Each run writes its own HDF5, so provenance stays one-file-per-sample. This is the
bulk-generation pattern the CLI wrapper is built for.

    python examples/01_generate_dataset.py /path/to/MatViz3D ./out 8
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from pymv3d import MatViz3DLauncher, GenParams, StressParams


def one_run(args):
    exe, out_dir, i = args
    mv = MatViz3DLauncher(exe)
    out = str(Path(out_dir) / f"sample_{i:04d}.hdf5")
    res = mv.run(
        GenParams(size=40, concentration=2.0, algorithm="Probability Algorithm",
                  seed=i, texture="rolling", lattice="fcc", scatter=11.0),
        StressParams(run=True, solver="fft", mode="dataset", num_rnd_loads=300),
        output=out, stream=False,       # quiet: many parallel runs
    )
    return i, res.ok, out


def main():
    exe = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "./out"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # NOTE: MatViz3D itself is multithreaded (--np). If you let each run use all
    # cores, keep max_workers small; if you pin each run to few threads, raise it.
    with ProcessPoolExecutor(max_workers=4) as ex:
        for i, ok, out in ex.map(one_run, [(exe, out_dir, i) for i in range(n)]):
            print(f"sample {i}: {'ok' if ok else 'FAILED'} -> {out}")


if __name__ == "__main__":
    main()

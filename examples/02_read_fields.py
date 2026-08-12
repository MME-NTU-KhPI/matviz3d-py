"""Read one MatViz3D HDF5 and assemble arrays for the PIML levels.

    python examples/02_read_fields.py dataset.hdf5
"""

import sys
import numpy as np

from pymv3d import MatViz3DResult, Col


def main():
    path = sys.argv[1]
    r = MatViz3DResult(path)
    print(f"{path}: {len(r)} set(s), ids={r.set_ids}")

    g = 0
    print(f"\nset g={g}: S={r.cube_size(g)}, grains={r.num_points(g)}, "
          f"load steps={r.n_loadsteps(g)}")

    # ---- Level 1: effective stiffness + macro response --------------------
    C = r.stiffness(g)
    if C is not None:
        print("\nEffective C (GPa):")
        print(np.array2string(C / 1e9, precision=2, suppress_small=True))
    macro = r.macro_response(g)
    if macro["eps_applied"].size:
        print(f"\nmacro pairs: eps {macro['eps_applied'].shape}, "
              f"stress {macro['stress_avg'].shape}")

    # ---- Level 2: per-voxel fields as a training tensor -------------------
    if r.n_loadsteps(g) > 0:
        n_ls = r.n_loadsteps(g)
        S = r.cube_size(g)
        # branch input: microstructure (grain ids); could be one-hot per grain
        micro = r.voxels(g)                                  # (S,S,S)
        # per-load-step targets
        sig = np.stack([r.stress_field(g, k) for k in range(1, n_ls + 1)])  # (n_ls,S,S,S,6)
        eps_macro = np.stack([r.eps_as_loading(g, k) for k in range(1, n_ls + 1)])  # (n_ls,6)
        vm = np.stack([r.scalar_field(g, k, Col.SEQV) for k in range(1, n_ls + 1)]) # (n_ls,S,S,S)
        print(f"\nLevel-2 tensors: micro {micro.shape}, "
              f"stress_field {sig.shape}, eps_macro {eps_macro.shape}, "
              f"vonMises {vm.shape}")


if __name__ == "__main__":
    main()

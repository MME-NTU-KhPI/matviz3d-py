"""
Thin subprocess wrapper around the MatViz3D command-line interface.

The option registry below is a 1:1 mirror of MatViz3D's `commandline_parser.cpp`
(branch qml-development).  Keeping it in one place means:

  * unknown / removed flags are rejected in Python with a clear message,
    instead of the Qt app aborting on `Unknown option` deep inside `QCommandLineParser::process()`;
  * "value" options vs bare boolean flags are handled correctly;
  * syncing with a future C++ change is a single-table edit.

Typical use
-----------
    from pymv3d import MatViz3DLauncher, GenParams, StressParams

    mv = MatViz3DLauncher("/path/to/MatViz3D")

    # 1) generate a microstructure only
    mv.run(GenParams(size=40, concentration=2.0, algorithm="Probability Algorithm"),
           output="cube.hdf5")

    # 2) generate + full FFT dataset (300 Hill-sampled load cases)
    mv.run(GenParams(size=40, concentration=2.0),
           StressParams(run=True, solver="fft", mode="dataset", num_rnd_loads=300),
           output="dataset.hdf5")

    # 3) effective stiffness only (6 unit-strain solves -> S/C/P/moduli)
    mv.run(GenParams(size=40, concentration=2.0),
           StressParams(run=True, solver="fft", mode="stiffness"),
           output="stiffness.hdf5")
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, fields
from typing import Optional, Sequence

log = logging.getLogger("pymv3d")


# ---------------------------------------------------------------------------
#  Option registry — mirror of commandline_parser.cpp::setupParser
# ---------------------------------------------------------------------------
# name -> takes a value?  (False => bare boolean flag)
# This is the *only* place that has to change when the C++ CLI changes.
_VALUE_OPTIONS = {
    # cube geometry
    "size", "points", "concentration", "seed", "np",
    # algorithm
    "algorithm", "wave_coefficient",
    "halfaxis_a", "halfaxis_b", "halfaxis_c",
    "orientation_angle_a", "orientation_angle_b", "orientation_angle_c",
    "ellipse_order",
    # crystallographic texture
    "texture", "lattice", "scatter",
    # stress analysis
    "solver", "stress_mode", "eps", "num_rnd_loads", "working_directory",
    # output
    "output",
}
_FLAG_OPTIONS = {"autostart", "nogui", "run_stress_calc"}
_KNOWN_OPTIONS = _VALUE_OPTIONS | _FLAG_OPTIONS

# Flags the *old* launcher used that no longer exist in the parser.
# Passing them would make the Qt app abort; we catch them early and explain.
_REMOVED_OPTIONS = {
    "wave_spread": "removed from the CLI",
    "initial_nuclei_count": "removed from the CLI",
    "stefan_number": "removed from the CLI",
    "hasProbParameters": "auto-derived now: it is set true automatically when any "
                         "halfaxis_* / orientation_angle_* option is given",
}


@dataclass
class GenParams:
    """Microstructure-generation options. Unset (None) fields are omitted from
    the command line, so MatViz3D falls back to its own defaults."""
    size: int
    points: Optional[int] = None            # mutually exclusive with concentration
    concentration: Optional[float] = None   # % of cube volume; wins over points
    algorithm: str = "Probability Algorithm"
    seed: Optional[int] = None              # omit -> MatViz3D seeds from time()
    np: Optional[int] = None                # threads; omit -> physical core count

    wave_coefficient: Optional[float] = None
    halfaxis_a: Optional[float] = None
    halfaxis_b: Optional[float] = None
    halfaxis_c: Optional[float] = None
    orientation_angle_a: Optional[float] = None
    orientation_angle_b: Optional[float] = None
    orientation_angle_c: Optional[float] = None
    ellipse_order: Optional[float] = None

    # texture preset (fills the same Parameters::textureComponents as the editor)
    texture: Optional[str] = None    # random|extrusion|rolling|recrystallization|shear|scattered_cube
    lattice: Optional[str] = None    # fcc|bcc (needs texture)
    scatter: Optional[float] = None  # degrees (needs texture)

    def to_cli(self) -> list[str]:
        args: list[str] = []
        for f in fields(self):
            v = getattr(self, f.name)
            if v is None:
                continue
            args += [f"--{f.name}", str(v)]
        return args


@dataclass
class StressParams:
    """Stress / homogenization options."""
    run: bool = False                       # -> --run_stress_calc
    solver: str = "fft"                     # ansys|fft
    mode: str = "dataset"                   # single|dataset|stiffness
    eps: Optional[Sequence[float]] = None   # 6 comps for mode="single": exx,eyy,ezz,exy,eyz,exz
    num_rnd_loads: Optional[int] = None
    working_directory: Optional[str] = None

    def to_cli(self) -> list[str]:
        if not self.run:
            return []
        args = ["--run_stress_calc", "--solver", self.solver, "--stress_mode", self.mode]
        if self.mode == "single":
            if self.eps is None or len(self.eps) != 6:
                raise ValueError("stress_mode='single' requires eps with exactly 6 components")
            args += ["--eps", ",".join(str(x) for x in self.eps)]
        if self.num_rnd_loads is not None:
            args += ["--num_rnd_loads", str(self.num_rnd_loads)]
        if self.working_directory is not None:
            args += ["--working_directory", self.working_directory]
        return args


@dataclass
class RunResult:
    returncode: int
    output_file: Optional[str]
    argv: list[str]
    stdout: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and (
            self.output_file is None or os.path.exists(self.output_file)
        )


class MatViz3DLauncher:
    def __init__(self, exe_path: str):
        if not os.path.exists(exe_path):
            log.warning("Executable path does not exist yet: %s", exe_path)
        self.exe_path = exe_path

    # -- high-level -----------------------------------------------------------
    def run(
        self,
        gen: GenParams,
        stress: Optional[StressParams] = None,
        *,
        output: Optional[str] = None,
        nogui: bool = True,
        autostart: bool = True,
        extra: Optional[dict] = None,
        timeout: Optional[float] = None,
        stream: bool = True,
        check: bool = False,
    ) -> RunResult:
        """Build the command line from dataclasses and run MatViz3D once.

        `extra` lets you pass options not yet modelled here (validated against
        the registry). `check=True` raises on a non-zero exit or missing output.
        """
        argv = [self.exe_path] + gen.to_cli()
        if stress is not None:
            argv += stress.to_cli()
        if output is not None:
            argv += ["--output", output]
        if autostart:
            argv += ["--autostart"]
        if nogui:
            argv += ["--nogui"]
        if extra:
            argv += self._extra_to_cli(extra)

        self._validate(argv)
        return self._exec(argv, output, timeout=timeout, stream=stream, check=check)

    # -- low-level (dict passthrough, still validated) ------------------------
    def run_raw(self, options: dict, *, output: Optional[str] = None,
                timeout: Optional[float] = None, stream: bool = True,
                check: bool = False) -> RunResult:
        argv = [self.exe_path] + self._extra_to_cli(options)
        if output is not None and "output" not in options:
            argv += ["--output", output]
        self._validate(argv)
        out = output or options.get("output")
        return self._exec(argv, out, timeout=timeout, stream=stream, check=check)

    # -- helpers --------------------------------------------------------------
    @staticmethod
    def _extra_to_cli(options: dict) -> list[str]:
        args: list[str] = []
        for key, val in options.items():
            if key in _FLAG_OPTIONS:
                if val:
                    args.append(f"--{key}")
            else:
                args += [f"--{key}", str(val)]
        return args

    @staticmethod
    def _validate(argv: Sequence[str]) -> None:
        for tok in argv:
            if not tok.startswith("--"):
                continue
            name = tok[2:]
            if name in _REMOVED_OPTIONS:
                raise ValueError(
                    f"--{name} is not accepted by the current MatViz3D CLI "
                    f"({_REMOVED_OPTIONS[name]}). Remove it from your call."
                )
            if name not in _KNOWN_OPTIONS:
                raise ValueError(
                    f"--{name} is not a known MatViz3D option. Known value options: "
                    f"{sorted(_VALUE_OPTIONS)}; flags: {sorted(_FLAG_OPTIONS)}."
                )

    def _exec(self, argv, output_file, *, timeout, stream, check) -> RunResult:
        log.info("Running: %s", " ".join(argv))
        captured: list[str] = []
        try:
            proc = subprocess.Popen(
                argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in proc.stdout:            # live streaming, like the original
                captured.append(line)
                if stream:
                    print(line, end="", flush=True)
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            log.error("MatViz3D timed out after %ss", timeout)
            return RunResult(-1, None, list(argv), "".join(captured))
        except Exception as e:  # noqa: BLE001  (surface any launch failure)
            log.error("Failed to launch MatViz3D: %s", e)
            return RunResult(-1, None, list(argv), "".join(captured))

        res = RunResult(proc.returncode, output_file, list(argv), "".join(captured))
        if proc.returncode != 0:
            log.error("MatViz3D exited with code %s", proc.returncode)
        if check and not res.ok:
            raise RuntimeError(
                f"MatViz3D run failed (rc={proc.returncode}, "
                f"output_exists={output_file and os.path.exists(output_file)})"
            )
        return res

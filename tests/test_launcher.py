import os

import pytest

from pymv3d import MatViz3DLauncher, GenParams, StressParams


def test_dataset_mode_argv(fake_exe, tmp_path):
    mv = MatViz3DLauncher(fake_exe)
    out = str(tmp_path / "dataset.hdf5")
    res = mv.run(
        GenParams(size=40, concentration=2.0, algorithm="Probability Algorithm"),
        StressParams(run=True, solver="fft", mode="dataset", num_rnd_loads=300),
        output=out, stream=False,
    )
    assert res.ok
    argv = " ".join(res.argv)
    for must in ["--size 40", "--concentration 2.0", "--algorithm Probability Algorithm",
                 "--run_stress_calc", "--solver fft", "--stress_mode dataset",
                 "--num_rnd_loads 300", "--output", "--autostart", "--nogui"]:
        assert must in argv, f"missing {must!r}"
    assert os.path.exists(out)


def test_none_fields_are_omitted(fake_exe, tmp_path):
    mv = MatViz3DLauncher(fake_exe)
    res = mv.run(GenParams(size=20), output=str(tmp_path / "c.hdf5"), stream=False)
    argv = " ".join(res.argv)
    assert "--points" not in argv          # never set -> omitted
    assert "--concentration" not in argv
    assert "--seed" not in argv


def test_single_mode_eps_serialization(fake_exe, tmp_path):
    mv = MatViz3DLauncher(fake_exe)
    res = mv.run(
        GenParams(size=20),
        StressParams(run=True, mode="single", solver="fft", eps=[1e-3, 0, 0, 0, 0, 0]),
        output=str(tmp_path / "single.hdf5"), stream=False,
    )
    assert "--eps 0.001,0,0,0,0,0" in " ".join(res.argv)


def test_single_mode_rejects_bad_eps():
    with pytest.raises(ValueError):
        StressParams(run=True, mode="single", eps=[1, 2, 3]).to_cli()


@pytest.mark.parametrize("bad", ["wave_spread", "initial_nuclei_count",
                                 "stefan_number", "hasProbParameters"])
def test_removed_flags_rejected(fake_exe, tmp_path, bad):
    mv = MatViz3DLauncher(fake_exe)
    with pytest.raises(ValueError) as e:
        mv.run_raw({"size": 10, bad: 1}, output=str(tmp_path / "x.hdf5"), stream=False)
    assert bad in str(e.value)


def test_unknown_flag_rejected(fake_exe):
    mv = MatViz3DLauncher(fake_exe)
    with pytest.raises(ValueError):
        mv.run_raw({"size": 10, "totally_made_up": 1}, stream=False)


def test_texture_argv(fake_exe, tmp_path):
    mv = MatViz3DLauncher(fake_exe)
    res = mv.run(GenParams(size=30, texture="rolling", lattice="bcc", scatter=8.0),
                 output=str(tmp_path / "tex.hdf5"), stream=False)
    a = " ".join(res.argv)
    assert "--texture rolling" in a and "--lattice bcc" in a and "--scatter 8.0" in a

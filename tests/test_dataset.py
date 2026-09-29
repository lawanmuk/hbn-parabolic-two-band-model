import json

import numpy as np

import twoband as tb
from twoband.dataset import COL_A, COL_E, COL_J, COL_TIME, FILES


def small_config():
    return tb.DatasetConfig(
        n_k=201,
        k_max=0.4,
        dt=0.2,
        t_end=200.0,
        pump_center_fs=2.4,
        pump_duration_fs=3.0,
        delays_fs=[-0.2, 0.0, 0.2],
    )


def test_generate_dataset_writes_all_files(tmp_path):
    metadata = tb.generate_dataset(tmp_path, small_config(), verbose=False)
    n_t = metadata["block_length"]
    for name in FILES.values():
        assert (tmp_path / name).exists()
    probe = np.loadtxt(tmp_path / FILES["probe"])
    assert probe.shape == (n_t, 8)
    scan = np.loadtxt(tmp_path / FILES["delay_scan"])
    assert scan.shape == (3 * n_t, 8)
    blocks = tb.split_delays(scan, n_t)
    np.testing.assert_allclose(blocks[2, :, COL_TIME], probe[:, COL_TIME])


def test_metadata_describes_the_run(tmp_path):
    tb.generate_dataset(tmp_path, small_config(), verbose=False)
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert "not measured data" in metadata["description"]
    assert metadata["config"]["delays_fs"] == [-0.2, 0.0, 0.2]
    assert metadata["columns"] == {
        "time": COL_TIME,
        "vector_potential": COL_A,
        "current": COL_J,
        "field": COL_E,
    }


def test_pump_probe_minus_pump_isolates_probe(tmp_path):
    tb.generate_dataset(tmp_path, small_config(), verbose=False)
    pump = np.loadtxt(tmp_path / FILES["pump"])
    probe = np.loadtxt(tmp_path / FILES["probe"])
    pump_probe = np.loadtxt(tmp_path / FILES["pump_probe"])
    np.testing.assert_allclose(pump_probe[:, COL_A] - pump[:, COL_A], probe[:, COL_A], atol=1e-12)


def test_load_or_generate_reuses_existing_data(tmp_path):
    first = tb.load_or_generate(tmp_path, small_config())
    stamp = (tmp_path / FILES["probe"]).stat().st_mtime_ns
    second = tb.load_or_generate(tmp_path, small_config())
    assert second == first
    assert (tmp_path / FILES["probe"]).stat().st_mtime_ns == stamp

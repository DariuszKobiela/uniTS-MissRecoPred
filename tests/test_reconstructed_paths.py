from pathlib import Path

import pytest

from utils.experiment_naming import (
    build_reconstructed_relative_path,
    iter_reconstructed_csv_files,
    parse_flat_reconstructed_stem,
    parse_reconstructed_metadata,
)


def test_build_and_parse_hierarchical_path():
    relative = build_reconstructed_relative_path(
        "boiler_outlet_temp_univ",
        "MCAR",
        "scattered",
        2,
        3,
        "impute_mean",
    )
    assert relative.as_posix() == "boiler_outlet_temp_univ/MCAR-scattered/2p/impute_mean/3.csv"

    fixed = Path("/data/4_fixed_data")
    entry = fixed / relative
    meta = parse_reconstructed_metadata(fixed, entry)
    assert meta["dataset_name"] == "boiler_outlet_temp_univ"
    assert meta["technique"] == "MCAR"
    assert meta["structure"] == "scattered"
    assert meta["rate_percent"] == 2
    assert meta["iteration"] == 3
    assert meta["model"] == "impute_mean"
    assert meta["reconstruction_model"] == "impute_mean"
    assert meta["reconstruction_iteration"] == 3


def test_parse_flat_legacy_filename():
    stem = "pump_sensor_28_univ_MNAR-contiguous_20p_7_stable_diffusion_2_gaf"
    meta = parse_flat_reconstructed_stem(stem)
    assert meta["dataset_name"] == "pump_sensor_28_univ"
    assert meta["model"] == "stable_diffusion_2_gaf"
    assert meta["rate_percent"] == 20
    assert meta["iteration"] == 7


def test_parse_flat_file_in_fixed_root(tmp_path: Path):
    flat = tmp_path / "sensor_MCAR-scattered_5p_1_knn.csv"
    flat.write_text("index,value\n0,1.0\n", encoding="utf-8")
    meta = parse_reconstructed_metadata(tmp_path, flat)
    assert meta["model"] == "knn"
    assert meta["iteration"] == 1


def test_iter_reconstructed_csv_files_finds_nested(tmp_path: Path):
    nested = (
        tmp_path
        / "sensor"
        / "MAR-mixed"
        / "20p"
        / "sarimax"
        / "2.csv"
    )
    nested.parent.mkdir(parents=True)
    nested.write_text("index,value\n0,1.0\n", encoding="utf-8")
    assert list(iter_reconstructed_csv_files(tmp_path)) == [nested]


def test_invalid_path_raises():
    with pytest.raises(ValueError, match="Unrecognized"):
        parse_reconstructed_metadata(
            Path("/fixed"),
            Path("/fixed/a/b/c.csv"),
        )

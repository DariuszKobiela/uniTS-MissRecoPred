"""RunConfig / Config.from_dict smoke tests."""

from __future__ import annotations

import textwrap

import yaml

from framework.config_models import RunConfig


def test_run_config_from_yaml_to_config_missing_dir(tmp_path):
    cfg = {
        "data": {
            "source_dir": "data/2_splitted_data/train",
            "missing_dir": "custom/missing/path",
            "fixed_dir": "data/4_fixed_data",
        }
    }
    p = tmp_path / "minimal.yaml"
    p.write_text(yaml.dump(cfg), encoding="utf-8")

    rc = RunConfig.from_yaml(str(p))
    assert rc.paths.missing_dir == "custom/missing/path"

    c = rc.to_config()
    assert c.get_missing_dir() == "custom/missing/path"


def test_config_from_dict_after_safe_load(tmp_path):
    from utils.config_loader import Config

    raw = yaml.safe_load(
        textwrap.dedent(
            """
            data:
              source_dir: s
              missing_dir: m2
              fixed_dir: f
            """
        )
    )
    c = Config.from_dict(raw, config_path=str(tmp_path / "virtual.yaml"))
    assert c.get_missing_dir() == "m2"


def test_dataset_split_override_is_isolated_to_vibration():
    from utils.config_loader import Config

    config = Config.from_dict(
        {
            "data": {},
            "split": {
                "sd2_validation": {"share": 0.10, "min_samples": 64, "max_samples": 500},
                "horizons": {"max_holdout_share": 0.20, "min_train_length": 200},
                "dataset_overrides": {
                    "vibration_sensor_S1.csv": {
                        "min_reconstruction_length": 144,
                        "sd2_validation": {"min_samples": 24, "max_samples": 24},
                    }
                },
            },
        }
    )

    vibration = config.get_dataset_split_settings("vibration_sensor_S1")
    assert vibration["min_reconstruction_length"] == 144
    assert vibration["validation_min_samples"] == 24
    assert vibration["validation_max_samples"] == 24
    assert vibration["override_applied"] is True

    boiler = config.get_dataset_split_settings("boiler_outlet_temp_univ.csv")
    assert boiler["min_reconstruction_length"] == 200
    assert boiler["validation_min_samples"] == 64
    assert boiler["override_applied"] is False

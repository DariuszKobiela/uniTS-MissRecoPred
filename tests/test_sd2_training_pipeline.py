from pathlib import Path

import numpy as np
from PIL import Image

from training.finetune_sd2_windowed import WindowedInpaintingDataset, group_split
from training.generate_sd2_windowed_dataset import image_mask


def test_image_masks_match_representation_geometry():
    missing = np.array([False, True, False, False, True])

    matrix_mask = image_mask(missing, "gaf", image_size=9)
    spec_mask = image_mask(missing, "spec", image_size=9)

    for pixel in (2, 8):
        assert np.all(matrix_mask[pixel, :] == 255)
        assert np.all(matrix_mask[:, pixel] == 255)
        assert np.all(spec_mask[:, pixel] == 255)
    assert np.all(spec_mask[0, :] == spec_mask[-1, :])
    assert spec_mask[:, 0].sum() == 0


def test_group_split_keeps_all_encodings_of_a_series_together():
    records = [
        {"series_id": series_id, "encoding": encoding}
        for series_id in range(10)
        for encoding in ("gaf", "mtf", "rp", "spec")
    ]

    train, validation = group_split(records, validation_share=0.2, seed=7)

    train_ids = {record["series_id"] for record in train}
    validation_ids = {record["series_id"] for record in validation}
    assert train_ids.isdisjoint(validation_ids)
    assert len(validation_ids) == 2
    assert len(train) + len(validation) == len(records)


def test_training_dataset_uses_explicit_mask(tmp_path: Path):
    for directory in ("clean", "conditioning", "masks"):
        (tmp_path / directory).mkdir()

    clean = np.full((8, 8, 3), 200, dtype=np.uint8)
    condition = np.full((8, 8, 3), 100, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[:, 3:5] = 255
    Image.fromarray(clean).save(tmp_path / "clean" / "sample.png")
    Image.fromarray(condition).save(tmp_path / "conditioning" / "sample.png")
    Image.fromarray(mask).save(tmp_path / "masks" / "sample.png")

    record = {
        "clean": "clean/sample.png",
        "conditioning": "conditioning/sample.png",
        "mask": "masks/sample.png",
        "prompt": "time-series image",
        "series_id": 3,
        "encoding": "rp",
    }
    item = WindowedInpaintingDataset(tmp_path, [record])[0]

    assert tuple(item["clean"].shape) == (3, 8, 8)
    assert tuple(item["mask"].shape) == (1, 8, 8)
    assert np.allclose(item["masked"][:, :, 3:5].numpy(), 0.0)
    assert not np.allclose(item["masked"][:, :, :3].numpy(), 0.0)
    assert set(np.unique(item["mask"].numpy())) == {0.0, 1.0}

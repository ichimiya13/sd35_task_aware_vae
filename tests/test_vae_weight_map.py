from __future__ import annotations

import torch

from src.sd35_task_aware_vae.vae.losses import build_spatial_weight_map


def test_center_gaussian_weight_map_peaks_at_center() -> None:
    batch = torch.zeros(2, 3, 32, 32, dtype=torch.float32)
    weight_map = build_spatial_weight_map(
        batch,
        {
            "mode": "center_gaussian",
            "min_weight": 1.0,
            "max_weight": 3.0,
            "center_x": 0.5,
            "center_y": 0.5,
            "sigma": 0.2,
            "apply_retina_mask": False,
        },
    )
    assert weight_map is not None
    center = float(weight_map[0, 0, 16, 16])
    corner = float(weight_map[0, 0, 0, 0])
    assert center > corner
    assert center <= 3.0 + 1e-5
    assert corner >= 1.0 - 1e-5


def test_peripheral_weight_map_is_larger_toward_border() -> None:
    batch = torch.zeros(1, 3, 32, 32, dtype=torch.float32)
    weight_map = build_spatial_weight_map(
        batch,
        {
            "mode": "peripheral",
            "min_weight": 1.0,
            "max_weight": 3.0,
            "gamma": 1.0,
            "apply_retina_mask": False,
        },
    )
    assert weight_map is not None
    center = float(weight_map[0, 0, 16, 16])
    corner = float(weight_map[0, 0, 0, 0])
    assert corner > center

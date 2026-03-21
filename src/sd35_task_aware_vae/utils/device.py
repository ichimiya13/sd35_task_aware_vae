from __future__ import annotations

import os
from typing import Any


_GPU_KEY_PATHS: tuple[tuple[str, str], ...] = (
    ("runtime", "gpu_ids"),
    ("model", "gpu_ids"),
    ("sd3", "gpu_ids"),
    ("train", "gpu_ids"),
    ("classifier", "gpu_ids"),
    ("vae", "gpu_ids"),
)


def get_gpu_ids(cfg: dict[str, Any], default: list[int] | None = None) -> list[int]:
    """Resolve visible GPU ids from a config.

    Priority order mirrors the legacy SDXL scripts so existing habits keep
    working. The returned ids are the *physical* CUDA ids that will be written
    to ``CUDA_VISIBLE_DEVICES``.
    """
    for scope, key in _GPU_KEY_PATHS:
        value = (cfg.get(scope, {}) or {}).get(key, None)
        if isinstance(value, (list, tuple)) and len(value) > 0:
            return [int(x) for x in value]
        if isinstance(value, str) and value.strip():
            return [int(x.strip()) for x in value.split(",") if x.strip()]
    if default is not None:
        return [int(x) for x in default]
    return [0]



def set_visible_gpus(gpu_ids: list[int] | tuple[int, ...] | None) -> None:
    if gpu_ids is None:
        return
    ids = [int(x) for x in gpu_ids]
    if not ids:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in ids)



def describe_visible_gpus() -> str:
    return os.environ.get("CUDA_VISIBLE_DEVICES", "")

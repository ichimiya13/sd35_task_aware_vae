from __future__ import annotations

import random
from typing import Optional

import numpy as np


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except Exception:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass


def build_generator(seed: int | None, device: str | None = None):
    if seed is None:
        return None
    import torch

    g = torch.Generator(device=device) if device else torch.Generator()
    g.manual_seed(int(seed))
    return g

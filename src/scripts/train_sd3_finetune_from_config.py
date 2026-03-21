from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from src.sd35_task_aware_vae.utils.device import get_gpu_ids, set_visible_gpus



def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    return obj or {}



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train SD3.5 Medium/Large components from YAML config (full DiT, LoRA, and optional VAE joint training)."
    )
    parser.add_argument("--config", required=True, type=str, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    gpu_ids = get_gpu_ids(cfg)
    set_visible_gpus(gpu_ids)

    from src.sd35_task_aware_vae.sd3.finetune import train_sd35_system_from_config

    out_dir = train_sd35_system_from_config(cfg, args.config)
    print(f"[done] SD3.5 finetuning finished: {out_dir}")


if __name__ == "__main__":
    main()

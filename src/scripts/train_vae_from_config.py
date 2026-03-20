from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    return obj or {}



def main() -> None:
    parser = argparse.ArgumentParser(description="Train a VAE from YAML config.")
    parser.add_argument("--config", required=True, type=str, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    model_cfg = cfg.get("model", {}) or {}
    vae_cfg = cfg.get("vae", {}) or {}

    backend = str(vae_cfg.get("backend", model_cfg.get("family", "sdxl"))).lower()
    if backend in {"sd35", "sd3", "stable_diffusion_3", "stable-diffusion-3"}:
        from src.sd35_task_aware_vae.vae.trainer import train_sd35_vae_from_config

        out_dir = train_sd35_vae_from_config(cfg, args.config)
        print(f"[done] SD3.5 VAE training finished: {out_dir}")
        return

    if backend in {"sdxl", "legacy"}:
        from src.scripts.train_vae_legacy_sdxl_from_config import train_from_config

        train_from_config(cfg, args.config)
        return

    raise ValueError(f"Unsupported vae backend: {backend}")


if __name__ == "__main__":
    main()

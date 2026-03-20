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
    parser = argparse.ArgumentParser(description="Reconstruct images from a VAE config.")
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    model_cfg = cfg.get("model", {}) or {}
    vae_cfg = cfg.get("vae", {}) or {}
    backend = str(vae_cfg.get("backend", model_cfg.get("family", "sdxl"))).lower()

    if backend in {"sd35", "sd3", "stable_diffusion_3", "stable-diffusion-3"}:
        from src.scripts.export_recon_dataset_from_config import reconstruct_sd35_from_config

        reconstruct_sd35_from_config(cfg, args.config)
        return

    if backend in {"sdxl", "legacy"}:
        from src.scripts.reconstruct_legacy_sdxl_from_config import reconstruct_from_config

        reconstruct_from_config(cfg, args.config)
        return

    raise ValueError(f"Unsupported vae backend: {backend}")


if __name__ == "__main__":
    main()

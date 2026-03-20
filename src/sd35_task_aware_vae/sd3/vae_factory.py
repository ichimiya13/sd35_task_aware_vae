from __future__ import annotations

from pathlib import Path
from typing import Any



def _resolve_vae_source(model_cfg: dict[str, Any], vae_cfg: dict[str, Any]) -> tuple[str, str]:
    checkpoint = vae_cfg.get("checkpoint", None)
    if checkpoint:
        return str(checkpoint), str(vae_cfg.get("subfolder", ""))
    repo_id = str(vae_cfg.get("model_repo_id", model_cfg.get("repo_id")))
    subfolder = str(vae_cfg.get("subfolder", "vae"))
    return repo_id, subfolder



def _load_state_dict_file(path: Path):
    import torch

    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except Exception as e:  # pragma: no cover - optional dependency
            raise RuntimeError("safetensors is required to load .safetensors VAE checkpoints") from e
        return load_file(str(path))

    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        for key in ["state_dict", "model", "vae", "module"]:
            if key in state and isinstance(state[key], dict):
                return state[key]
    return state



def build_sd3_vae(model_cfg: dict[str, Any], vae_cfg: dict[str, Any], torch_dtype=None, device=None):
    """Load an SD3/SD3.5-compatible ``AutoencoderKL``.

    Supported sources:
      - official VAE from ``repo_id/subfolder='vae'``
      - custom local / Hub directory saved with ``save_pretrained``
      - local ``.pt/.pth/.bin/.safetensors`` state dict, loaded onto the official SD3.5 VAE architecture
    """
    try:
        from diffusers import AutoencoderKL
    except Exception as e:  # pragma: no cover - env dependent
        raise RuntimeError("diffusers is required to load the SD3.5 VAE") from e

    source, subfolder = _resolve_vae_source(model_cfg, vae_cfg)
    source_path = Path(source)

    def _maybe_to_eval(vae):
        if device is not None:
            try:
                vae.to(device)
            except Exception:
                pass
        if bool(vae_cfg.get("eval_mode", True)):
            vae.eval()
        return vae

    # State-dict file: initialize from the official SD3.5 VAE config/weights first, then load the custom weights.
    if source_path.is_file() and source_path.suffix.lower() in {".pt", ".pth", ".bin", ".safetensors"}:
        base_repo = str(vae_cfg.get("model_repo_id", model_cfg.get("repo_id")))
        base_subfolder = str(vae_cfg.get("base_subfolder", vae_cfg.get("subfolder", "vae")))
        vae = AutoencoderKL.from_pretrained(base_repo, subfolder=base_subfolder, torch_dtype=torch_dtype)
        state_dict = _load_state_dict_file(source_path)
        missing, unexpected = vae.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[warn] VAE state_dict load: missing={len(missing)} unexpected={len(unexpected)}")
        return _maybe_to_eval(vae)

    load_kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
    if subfolder:
        load_kwargs["subfolder"] = subfolder

    vae = AutoencoderKL.from_pretrained(source, **load_kwargs)
    return _maybe_to_eval(vae)



def apply_freeze_patterns(module, freeze_patterns: list[str] | None = None, unfreeze_patterns: list[str] | None = None) -> None:
    import re

    freeze_patterns = freeze_patterns or []
    unfreeze_patterns = unfreeze_patterns or []
    for _, p in module.named_parameters():
        p.requires_grad_(True)

    if freeze_patterns:
        compiled = [re.compile(p) for p in freeze_patterns]
        for name, p in module.named_parameters():
            if any(rx.search(name) for rx in compiled):
                p.requires_grad_(False)

    if unfreeze_patterns:
        compiled = [re.compile(p) for p in unfreeze_patterns]
        for name, p in module.named_parameters():
            if any(rx.search(name) for rx in compiled):
                p.requires_grad_(True)

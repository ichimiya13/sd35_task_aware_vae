from __future__ import annotations

from pathlib import Path
from typing import Any


def _load_state_dict_file(path: Path):
    import torch

    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except Exception as e:  # pragma: no cover - optional dependency
            raise RuntimeError("safetensors is required to load .safetensors transformer checkpoints") from e
        return load_file(str(path))

    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        for key in ["state_dict", "model", "transformer", "module"]:
            if key in state and isinstance(state[key], dict):
                return state[key]
    return state


def _normalize_transformer_source(path_like: str | None) -> str | None:
    if not path_like:
        return None
    source = Path(str(path_like))
    if source.is_dir() and not (source / "config.json").is_file() and (source / "transformer").is_dir():
        return str(source / "transformer")
    return str(source)


def build_sd3_transformer(
    model_cfg: dict[str, Any],
    transformer_cfg: dict[str, Any] | None,
    *,
    torch_dtype=None,
):
    """Build an SD3/SD3.5 transformer from config.

    Supported sources:
      - official transformer from repo_id/subfolder='transformer'
      - local or Hub directory saved with ``save_pretrained``
      - local ``.pt/.pth/.bin/.safetensors`` state dict loaded onto the official SD3 architecture
    """
    transformer_cfg = transformer_cfg or {}
    ckpt = transformer_cfg.get("checkpoint", None) or model_cfg.get("transformer_checkpoint", None)
    normalized = _normalize_transformer_source(ckpt)
    if not normalized:
        return None

    try:
        from diffusers import SD3Transformer2DModel
    except Exception as e:  # pragma: no cover - env dependent
        raise RuntimeError("diffusers is required to load the SD3 transformer") from e

    source = Path(normalized)
    if source.is_file() and source.suffix.lower() in {".pt", ".pth", ".bin", ".safetensors"}:
        base_repo = str(transformer_cfg.get("model_repo_id", model_cfg.get("repo_id")))
        base_subfolder = str(transformer_cfg.get("base_subfolder", transformer_cfg.get("subfolder", "transformer")))
        transformer = SD3Transformer2DModel.from_pretrained(base_repo, subfolder=base_subfolder, torch_dtype=torch_dtype)
        state_dict = _load_state_dict_file(source)
        missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[warn] transformer state_dict load: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        return transformer

    load_kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
    subfolder = str(transformer_cfg.get("subfolder", ""))
    if subfolder and not source.is_dir():
        load_kwargs["subfolder"] = subfolder
    return SD3Transformer2DModel.from_pretrained(normalized, **load_kwargs)


def _resolve_lora_source(transformer_cfg: dict[str, Any] | None, model_cfg: dict[str, Any] | None = None) -> str | None:
    transformer_cfg = transformer_cfg or {}
    model_cfg = model_cfg or {}
    source = transformer_cfg.get("lora_path", None) or model_cfg.get("lora_path", None)
    if not source:
        return None
    path = Path(str(source))
    if path.is_dir() and not any(path.glob("*.safetensors")) and not any(path.glob("*.bin")) and (path / "lora").is_dir():
        return str(path / "lora")
    return str(path)


def maybe_load_sd3_lora(pipe, transformer_cfg: dict[str, Any] | None = None, model_cfg: dict[str, Any] | None = None):
    """Optionally load a saved LoRA adapter into an SD3 pipeline."""
    source = _resolve_lora_source(transformer_cfg, model_cfg)
    if not source:
        return pipe

    adapter_name = str((transformer_cfg or {}).get("adapter_name", "default"))
    scale = float((transformer_cfg or {}).get("lora_scale", 1.0))
    fuse = bool((transformer_cfg or {}).get("fuse_lora", False))

    if not hasattr(pipe, "load_lora_weights"):
        raise RuntimeError("The current diffusers pipeline does not expose load_lora_weights for SD3 LoRA inference.")

    pipe.load_lora_weights(source, adapter_name=adapter_name)

    if scale != 1.0:
        try:
            if hasattr(pipe, "set_adapters"):
                pipe.set_adapters(adapter_name, adapter_weights=scale)
            elif hasattr(pipe, "fuse_lora"):
                pipe.fuse_lora(adapter_names=[adapter_name], lora_scale=scale)
        except Exception as e:
            print(f"[warn] failed to apply explicit LoRA scale={scale}: {e}", flush=True)

    if fuse:
        try:
            if hasattr(pipe, "fuse_lora"):
                if scale != 1.0:
                    pipe.fuse_lora(adapter_names=[adapter_name], lora_scale=scale)
                else:
                    pipe.fuse_lora(adapter_names=[adapter_name])
        except Exception as e:
            print(f"[warn] failed to fuse LoRA weights: {e}", flush=True)

    return pipe

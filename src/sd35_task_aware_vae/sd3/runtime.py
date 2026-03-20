from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def resolve_torch_dtype(dtype: str | None):
    import torch

    if dtype is None:
        return torch.float32
    value = str(dtype).lower()
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {dtype}")


def select_device(preferred: str | None = None):
    import torch

    pref = str(preferred or "auto").lower()
    if pref in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    if pref in {"auto", "mps"} and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class RuntimeContext:
    device: Any
    torch_dtype: Any


def build_runtime_context(model_cfg: dict[str, Any]) -> RuntimeContext:
    import torch

    device = select_device(model_cfg.get("device", None))
    torch_dtype = resolve_torch_dtype(model_cfg.get("torch_dtype", "bf16" if device.type == "cuda" else "fp32"))

    if bool(model_cfg.get("allow_tf32", False)) and device.type == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    return RuntimeContext(device=device, torch_dtype=torch_dtype)


def maybe_build_quantized_transformer(model_cfg: dict[str, Any], torch_dtype):
    """Optionally load a 4-bit SD3 transformer.

    The SD3.5 model card shows loading an NF4-quantized `SD3Transformer2DModel`
    and then injecting it into `StableDiffusion3Pipeline.from_pretrained(...)`.
    """
    quant_cfg = (model_cfg.get("quantization", {}) or {})
    if not bool(quant_cfg.get("enabled", False)):
        return None

    try:
        from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
    except Exception as e:  # pragma: no cover - depends on runtime env
        raise RuntimeError(
            "4-bit transformer quantization requires diffusers with BitsAndBytesConfig support."
        ) from e

    repo_id = str(model_cfg["repo_id"])
    subfolder = str(quant_cfg.get("subfolder", "transformer"))
    compute_dtype = resolve_torch_dtype(quant_cfg.get("compute_dtype", None) or str(torch_dtype).split(".")[-1])
    qconf = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=str(quant_cfg.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_compute_dtype=compute_dtype,
    )
    return SD3Transformer2DModel.from_pretrained(
        repo_id,
        subfolder=subfolder,
        torch_dtype=torch_dtype,
        quantization_config=qconf,
    )


def apply_pipeline_runtime_options(pipe, model_cfg: dict[str, Any]):
    """Move a diffusers pipeline to the requested runtime configuration."""
    device = select_device(model_cfg.get("device", None))

    try:
        pipe.set_progress_bar_config(disable=bool(model_cfg.get("disable_progress_bar", False)))
    except Exception:
        pass

    if bool(model_cfg.get("enable_cpu_offload", False)):
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
            return pipe

    if bool(model_cfg.get("enable_sequential_cpu_offload", False)):
        if hasattr(pipe, "enable_sequential_cpu_offload"):
            pipe.enable_sequential_cpu_offload()
            return pipe

    if bool(model_cfg.get("enable_attention_slicing", False)):
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass

    try:
        pipe.to(device)
    except Exception:
        # Some quantized setups prefer explicit offload instead of `.to(...)`.
        pass
    return pipe

from __future__ import annotations

from typing import Any

from src.sd35_task_aware_vae.sd3.runtime import (
    apply_pipeline_runtime_options,
    build_runtime_context,
    maybe_build_quantized_transformer,
)
from src.sd35_task_aware_vae.sd3.transformer_factory import build_sd3_transformer, maybe_load_sd3_lora
from src.sd35_task_aware_vae.sd3.vae_factory import build_sd3_vae


def _attach_components(pipe, *, vae=None, transformer=None, scheduler=None):
    if hasattr(pipe, "register_modules"):
        kwargs = {}
        if vae is not None:
            kwargs["vae"] = vae
        if transformer is not None:
            kwargs["transformer"] = transformer
        if scheduler is not None:
            kwargs["scheduler"] = scheduler
        if kwargs:
            pipe.register_modules(**kwargs)
    else:
        if vae is not None:
            pipe.vae = vae
        if transformer is not None:
            pipe.transformer = transformer
        if scheduler is not None:
            pipe.scheduler = scheduler
    return pipe


def build_sd3_text2img_pipeline(model_cfg: dict[str, Any], vae_cfg: dict[str, Any] | None = None, transformer_cfg: dict[str, Any] | None = None):
    try:
        from diffusers import StableDiffusion3Pipeline
    except Exception as e:  # pragma: no cover - env dependent
        raise RuntimeError("diffusers is required to build the SD3.5 pipeline") from e

    vae_cfg = vae_cfg or {}
    runtime = build_runtime_context(model_cfg)
    transformer = build_sd3_transformer(model_cfg, transformer_cfg, torch_dtype=runtime.torch_dtype)
    if transformer is None:
        transformer = maybe_build_quantized_transformer(model_cfg, runtime.torch_dtype)
    vae = build_sd3_vae(model_cfg, vae_cfg, torch_dtype=runtime.torch_dtype, device=runtime.device)

    pipe = StableDiffusion3Pipeline.from_pretrained(
        str(model_cfg["repo_id"]),
        transformer=transformer,
        vae=vae,
        torch_dtype=runtime.torch_dtype,
    )
    pipe = _attach_components(pipe, vae=vae, transformer=transformer)
    pipe = maybe_load_sd3_lora(pipe, transformer_cfg=transformer_cfg, model_cfg=model_cfg)
    return apply_pipeline_runtime_options(pipe, model_cfg)


def build_sd3_img2img_pipeline(model_cfg: dict[str, Any], vae_cfg: dict[str, Any] | None = None, transformer_cfg: dict[str, Any] | None = None):
    vae_cfg = vae_cfg or {}
    runtime = build_runtime_context(model_cfg)
    transformer = build_sd3_transformer(model_cfg, transformer_cfg, torch_dtype=runtime.torch_dtype)
    if transformer is None:
        transformer = maybe_build_quantized_transformer(model_cfg, runtime.torch_dtype)
    vae = build_sd3_vae(model_cfg, vae_cfg, torch_dtype=runtime.torch_dtype, device=runtime.device)

    pipe = None
    try:
        from diffusers import StableDiffusion3Img2ImgPipeline

        pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
            str(model_cfg["repo_id"]),
            transformer=transformer,
            vae=vae,
            torch_dtype=runtime.torch_dtype,
        )
    except Exception:
        # Fallback to the task-first loader if the direct class name is not
        # exported in the installed diffusers version.
        try:
            from diffusers import AutoPipelineForImage2Image
        except Exception as e:  # pragma: no cover - env dependent
            raise RuntimeError("Unable to construct an SD3 image-to-image pipeline.") from e

        pipe = AutoPipelineForImage2Image.from_pretrained(
            str(model_cfg["repo_id"]),
            torch_dtype=runtime.torch_dtype,
        )
        pipe = _attach_components(pipe, vae=vae, transformer=transformer)

    pipe = _attach_components(pipe, vae=vae, transformer=transformer)
    pipe = maybe_load_sd3_lora(pipe, transformer_cfg=transformer_cfg, model_cfg=model_cfg)
    return apply_pipeline_runtime_options(pipe, model_cfg)

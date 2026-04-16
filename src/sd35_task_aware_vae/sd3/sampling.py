from __future__ import annotations

from typing import Any, Sequence

from PIL import Image

from src.sd35_task_aware_vae.sd3.prompts import resolve_prompts



def _default_size(model_cfg: dict[str, Any]) -> tuple[int, int]:
    image_cfg = model_cfg.get("image", {}) or {}
    height = int(image_cfg.get("height", image_cfg.get("image_size", 1024)))
    width = int(image_cfg.get("width", image_cfg.get("image_size", 1024)))
    return height, width



def _normalize_negative_prompts(negative_prompts: Sequence[str] | None) -> list[str] | None:
    if negative_prompts is None:
        return None
    values = [str(x) for x in negative_prompts]
    if all(not v.strip() for v in values):
        return None
    return values



def sample_text2img(pipe, model_cfg: dict[str, Any], prompts: Sequence[str], negative_prompts: Sequence[str] | None = None):
    height, width = _default_size(model_cfg)
    output = pipe(
        prompt=list(prompts),
        negative_prompt=_normalize_negative_prompts(negative_prompts),
        num_inference_steps=int(model_cfg.get("num_inference_steps", 40)),
        guidance_scale=float(model_cfg.get("guidance_scale", 4.5)),
        height=height,
        width=width,
        output_type="pil",
    )
    return list(output.images)



def sample_img2img(
    pipe,
    model_cfg: dict[str, Any],
    images: Sequence[Image.Image],
    prompts: Sequence[str],
    negative_prompts: Sequence[str] | None = None,
):
    height, width = _default_size(model_cfg)
    output = pipe(
        prompt=list(prompts),
        negative_prompt=_normalize_negative_prompts(negative_prompts),
        image=list(images),
        num_inference_steps=int(model_cfg.get("num_inference_steps", 40)),
        guidance_scale=float(model_cfg.get("guidance_scale", 4.5)),
        strength=float(model_cfg.get("strength", 0.6)),
        height=height,
        width=width,
        output_type="pil",
    )
    return list(output.images)

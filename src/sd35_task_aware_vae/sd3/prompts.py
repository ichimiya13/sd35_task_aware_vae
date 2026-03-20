from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from src.sd35_task_aware_vae.utils.config import load_yaml


DEFAULT_NEUTRAL_PROMPT = "ultra-widefield fundus photograph, clinical retinal imaging, medically faithful"
DEFAULT_NEGATIVE_PROMPT = "cartoon, illustration, painting, text overlay, watermark, duplicated vessels"


def load_prompt_templates(prompt_cfg: dict[str, Any] | str | Path | None) -> dict[str, Any]:
    if prompt_cfg is None:
        return {}
    if isinstance(prompt_cfg, (str, Path)):
        return load_yaml(prompt_cfg)
    return dict(prompt_cfg)


def build_neutral_prompts(batch_size: int, prompt_cfg: dict[str, Any] | None = None) -> list[str]:
    prompt_cfg = load_prompt_templates(prompt_cfg)
    prompt = str(prompt_cfg.get("neutral_prompt", DEFAULT_NEUTRAL_PROMPT))
    return [prompt for _ in range(batch_size)]


def build_negative_prompts(batch_size: int, prompt_cfg: dict[str, Any] | None = None) -> list[str]:
    prompt_cfg = load_prompt_templates(prompt_cfg)
    prompt = str(prompt_cfg.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT))
    return [prompt for _ in range(batch_size)]


def build_label_conditioned_prompts(
    labels,
    class_names: Sequence[str],
    prompt_cfg: dict[str, Any] | None = None,
    threshold: float = 0.5,
) -> list[str]:
    prompt_cfg = load_prompt_templates(prompt_cfg)
    prefix = str(prompt_cfg.get("label_prompt_prefix", "ultra-widefield fundus photograph showing "))
    suffix = str(prompt_cfg.get("label_prompt_suffix", ""))
    none_text = str(prompt_cfg.get("label_prompt_none", "no clearly visible retinal abnormality"))
    separator = str(prompt_cfg.get("label_separator", ", "))

    prompts: list[str] = []
    for row in labels:
        active = [class_names[i] for i, v in enumerate(row) if float(v) >= threshold]
        desc = separator.join(active) if active else none_text
        prompts.append(f"{prefix}{desc}{suffix}".strip())
    return prompts


def resolve_prompts(
    batch_size: int,
    labels=None,
    class_names: Sequence[str] | None = None,
    prompt_cfg: dict[str, Any] | None = None,
) -> tuple[list[str], list[str]]:
    prompt_cfg = load_prompt_templates(prompt_cfg)
    mode = str(prompt_cfg.get("mode", "neutral")).lower()

    if mode == "explicit":
        prompt = prompt_cfg.get("prompt", DEFAULT_NEUTRAL_PROMPT)
        negative = prompt_cfg.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
        prompts = [str(prompt) for _ in range(batch_size)]
        negatives = [str(negative) for _ in range(batch_size)]
        return prompts, negatives

    if mode == "label_conditioned":
        if labels is None or class_names is None:
            raise ValueError("labels and class_names are required for label-conditioned prompts")
        prompts = build_label_conditioned_prompts(
            labels,
            class_names=class_names,
            prompt_cfg=prompt_cfg,
            threshold=float(prompt_cfg.get("label_threshold", 0.5)),
        )
        negatives = build_negative_prompts(batch_size, prompt_cfg)
        return prompts, negatives

    # default: neutral
    return build_neutral_prompts(batch_size, prompt_cfg), build_negative_prompts(batch_size, prompt_cfg)

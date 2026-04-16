from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.sd35_task_aware_vae.utils.config import load_yaml


DEFAULT_NEUTRAL_PROMPT = "ultra-widefield fundus photograph, clinical retinal imaging, medically faithful"
DEFAULT_NEGATIVE_PROMPT = "cartoon, illustration, painting, text overlay, watermark, duplicated vessels"



def load_prompt_templates(prompt_cfg: dict[str, Any] | str | Path | None) -> dict[str, Any]:
    if prompt_cfg is None:
        return {}
    if isinstance(prompt_cfg, (str, Path)):
        return load_yaml(prompt_cfg)

    cfg = dict(prompt_cfg)
    template_file = cfg.pop("template_file", None) or cfg.pop("templates", None)
    if template_file is not None:
        base = load_yaml(template_file)
        if base is None:
            base = {}
        if not isinstance(base, dict):
            raise ValueError("prompt.template_file must load a YAML mapping")
        merged = dict(base)
        merged.update(cfg)
        return merged
    return cfg



def _ensure_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    return [str(value)]



def _sanitize_prompt_piece(text: str) -> str:
    return " ".join(str(text).strip().split())



def _pick_value(value: Any) -> str:
    choices = _ensure_str_list(value)
    if not choices:
        return ""
    if len(choices) == 1:
        return choices[0]
    return random.choice(choices)



def _resolve_label_phrase(class_name: str, prompt_cfg: dict[str, Any]) -> str:
    class_prompts = dict(prompt_cfg.get("class_prompts", {}) or {})
    label_aliases = dict(prompt_cfg.get("label_aliases", {}) or {})

    if class_name in class_prompts:
        return _sanitize_prompt_piece(_pick_value(class_prompts[class_name]))
    if class_name in label_aliases:
        return _sanitize_prompt_piece(_pick_value(label_aliases[class_name]))
    return _sanitize_prompt_piece(class_name)



def _sort_selected_labels(selected: list[str], prompt_cfg: dict[str, Any]) -> list[str]:
    priority = prompt_cfg.get("label_priority", None)
    if isinstance(priority, (list, tuple)) and len(priority) > 0:
        order = {str(name): idx for idx, name in enumerate(priority)}
        selected = sorted(selected, key=lambda x: (order.get(x, 10**9), x))
    elif bool(prompt_cfg.get("sort_labels", False)):
        selected = sorted(selected)

    if bool(prompt_cfg.get("shuffle_labels", False)) and len(selected) > 1:
        random.shuffle(selected)
    return selected



def _select_active_labels(row, class_names: Sequence[str], prompt_cfg: dict[str, Any], threshold: float) -> list[str]:
    exclude = {str(x) for x in _ensure_str_list(prompt_cfg.get("exclude_labels_in_prompt", []))}
    active = [class_names[i] for i, v in enumerate(row) if float(v) >= threshold and class_names[i] not in exclude]
    active = _sort_selected_labels(active, prompt_cfg)

    dropout_p = float(prompt_cfg.get("label_dropout_p", 0.0))
    if dropout_p > 0 and active:
        kept: list[str] = []
        for item in active:
            if random.random() >= dropout_p:
                kept.append(item)
        if kept:
            active = kept

    strategy = str(prompt_cfg.get("multi_label_strategy", "join_all")).lower()
    max_labels = int(prompt_cfg.get("max_labels_in_prompt", max(1, len(active) if active else 1)))
    if max_labels > 0 and len(active) > max_labels:
        if strategy in {"random_one", "single", "single_positive"}:
            active = [random.choice(active)]
        elif strategy in {"random_k", "sample_k"}:
            k = max(1, min(len(active), max_labels))
            active = random.sample(active, k=k)
        else:
            active = active[:max_labels]
    elif strategy in {"random_one", "single", "single_positive"} and active:
        active = [random.choice(active)]

    active = _sort_selected_labels(active, prompt_cfg)
    return active



def _resolve_negative_prompt(prompt_cfg: dict[str, Any]) -> str:
    if not bool(prompt_cfg.get("use_negative_prompt", True)):
        return ""
    prompt = prompt_cfg.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
    if prompt is None:
        return ""
    return _sanitize_prompt_piece(str(prompt))



def build_neutral_prompts(batch_size: int, prompt_cfg: dict[str, Any] | None = None) -> list[str]:
    prompt_cfg = load_prompt_templates(prompt_cfg)
    prompt = str(prompt_cfg.get("neutral_prompt", DEFAULT_NEUTRAL_PROMPT))
    return [prompt for _ in range(batch_size)]



def build_negative_prompts(batch_size: int, prompt_cfg: dict[str, Any] | None = None) -> list[str]:
    prompt_cfg = load_prompt_templates(prompt_cfg)
    prompt = _resolve_negative_prompt(prompt_cfg)
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
    neutral_p = float(prompt_cfg.get("neutral_prompt_probability", 0.0))
    empty_p = float(prompt_cfg.get("empty_prompt_probability", 0.0))

    prompts: list[str] = []
    for row in labels:
        if neutral_p > 0 and random.random() < neutral_p:
            prompts.append(str(prompt_cfg.get("neutral_prompt", DEFAULT_NEUTRAL_PROMPT)))
            continue
        if empty_p > 0 and random.random() < empty_p:
            prompts.append("")
            continue

        selected = _select_active_labels(row, class_names, prompt_cfg, threshold)
        phrases = [_resolve_label_phrase(name, prompt_cfg) for name in selected]
        phrases = [_sanitize_prompt_piece(x) for x in phrases if _sanitize_prompt_piece(x)]
        desc = separator.join(phrases) if phrases else none_text
        prompts.append(_sanitize_prompt_piece(f"{prefix}{desc}{suffix}"))
    return prompts



def _normalize_class_target_spec(target: Any) -> list[str]:
    if isinstance(target, str):
        if "," in target:
            return [x.strip() for x in target.split(",") if x.strip()]
        if "+" in target:
            return [x.strip() for x in target.split("+") if x.strip()]
        return [target]
    if isinstance(target, (list, tuple)):
        return [str(x) for x in target if str(x)]
    return [str(target)]



def _normalize_target_count_mapping(value: Any) -> dict[str, int]:
    if isinstance(value, Mapping):
        out: dict[str, int] = {}
        for key, count in value.items():
            out[str(key)] = max(0, int(count))
        return out
    return {}



def _resolve_num_images_for_target(
    cleaned_target: Sequence[str],
    *,
    default_count: int,
    count_overrides: Mapping[str, int] | None = None,
) -> int:
    count_overrides = count_overrides or {}
    target_id = "__".join(str(x) for x in cleaned_target)
    candidate_keys = [target_id]
    if len(cleaned_target) == 1:
        candidate_keys.extend([str(cleaned_target[0])])
    else:
        candidate_keys.extend([
            "+".join(str(x) for x in cleaned_target),
            ",".join(str(x) for x in cleaned_target),
        ])

    for key in candidate_keys:
        if key in count_overrides:
            return max(0, int(count_overrides[key]))
    return max(0, int(default_count))



def _resolve_explicit_target_prompt(target_labels: Sequence[str], prompt_cfg: dict[str, Any]) -> str | None:
    target_prompts = dict(prompt_cfg.get("target_prompts", {}) or {})
    if not target_prompts:
        return None
    target_id = "__".join(str(x) for x in target_labels)
    candidate_keys = [target_id]
    if len(target_labels) == 1:
        candidate_keys.append(str(target_labels[0]))
    else:
        candidate_keys.extend([
            "+".join(str(x) for x in target_labels),
            ",".join(str(x) for x in target_labels),
        ])
    for key in candidate_keys:
        if key in target_prompts:
            return _sanitize_prompt_piece(_pick_value(target_prompts[key]))
    return None



def _resolve_explicit_target_negative_prompt(target_labels: Sequence[str], prompt_cfg: dict[str, Any]) -> str | None:
    target_negative_prompts = dict(prompt_cfg.get("target_negative_prompts", {}) or {})
    if not target_negative_prompts:
        return None
    target_id = "__".join(str(x) for x in target_labels)
    candidate_keys = [target_id]
    if len(target_labels) == 1:
        candidate_keys.append(str(target_labels[0]))
    else:
        candidate_keys.extend([
            "+".join(str(x) for x in target_labels),
            ",".join(str(x) for x in target_labels),
        ])
    for key in candidate_keys:
        if key in target_negative_prompts:
            return _sanitize_prompt_piece(_pick_value(target_negative_prompts[key]))
    return None



def build_neutral_prompt_entries(
    prompt_cfg: dict[str, Any] | None = None,
    *,
    num_images: int,
) -> list[dict[str, Any]]:
    prompt_cfg = load_prompt_templates(prompt_cfg)
    prompts_pool = _ensure_str_list(prompt_cfg.get("neutral_prompts", prompt_cfg.get("neutral_prompt", DEFAULT_NEUTRAL_PROMPT)))
    if not prompts_pool:
        prompts_pool = [DEFAULT_NEUTRAL_PROMPT]
    negatives_pool = _ensure_str_list(prompt_cfg.get("negative_prompts", _resolve_negative_prompt(prompt_cfg)))
    if not negatives_pool:
        negatives_pool = [""]

    strategy = str(prompt_cfg.get("neutral_prompt_strategy", "repeat")).lower()
    entries: list[dict[str, Any]] = []
    for idx in range(max(0, int(num_images))):
        if strategy in {"cycle", "round_robin"}:
            prompt = prompts_pool[idx % len(prompts_pool)]
            negative = negatives_pool[idx % len(negatives_pool)]
        elif strategy in {"random", "sample"}:
            prompt = random.choice(prompts_pool)
            negative = random.choice(negatives_pool)
        else:
            prompt = prompts_pool[0]
            negative = negatives_pool[0]
        entries.append(
            {
                "target_id": str(prompt_cfg.get("neutral_target_id", "neutral")),
                "repeat_index": idx,
                "sample_index": idx,
                "prompt": _sanitize_prompt_piece(prompt),
                "negative_prompt": _sanitize_prompt_piece(negative),
            }
        )
    return entries



def build_class_prompt_entries(
    class_names: Sequence[str],
    prompt_cfg: dict[str, Any] | None = None,
    *,
    num_images_per_target: int | Mapping[str, int] = 1,
) -> list[dict[str, Any]]:
    prompt_cfg = load_prompt_templates(prompt_cfg)
    prompt_cfg = dict(prompt_cfg)
    prompt_cfg["neutral_prompt_probability"] = 0.0
    prompt_cfg["empty_prompt_probability"] = 0.0
    prompt_cfg["label_dropout_p"] = 0.0
    targets_cfg = prompt_cfg.get("class_targets", "__all__")
    if targets_cfg == "__all__":
        targets = [[str(name)] for name in class_names]
    else:
        raw_targets = _ensure_str_list(targets_cfg) if isinstance(targets_cfg, str) else list(targets_cfg)
        targets = [_normalize_class_target_spec(t) for t in raw_targets]

    if isinstance(num_images_per_target, Mapping):
        default_count = int(prompt_cfg.get("default_num_images_per_class", 0))
        count_overrides = _normalize_target_count_mapping(num_images_per_target)
    else:
        default_count = max(0, int(num_images_per_target))
        count_overrides = _normalize_target_count_mapping(prompt_cfg.get("num_images_per_class", {}))

    label_index = {str(name): idx for idx, name in enumerate(class_names)}
    rows = []
    for target in targets:
        cleaned = [str(x) for x in target if str(x) in label_index]
        if not cleaned:
            continue
        label_vec = [0.0 for _ in class_names]
        for name in cleaned:
            label_vec[label_index[name]] = 1.0

        target_count = _resolve_num_images_for_target(cleaned, default_count=default_count, count_overrides=count_overrides)
        if target_count <= 0:
            continue

        explicit_prompt = _resolve_explicit_target_prompt(cleaned, prompt_cfg)
        if explicit_prompt is None:
            prompt = build_label_conditioned_prompts([label_vec], class_names, prompt_cfg=prompt_cfg, threshold=0.5)[0]
        else:
            prompt = explicit_prompt

        explicit_negative = _resolve_explicit_target_negative_prompt(cleaned, prompt_cfg)
        negative = explicit_negative if explicit_negative is not None else build_negative_prompts(1, prompt_cfg=prompt_cfg)[0]
        target_id = "__".join(cleaned)
        for rep in range(target_count):
            rows.append(
                {
                    "target_labels": cleaned,
                    "target_id": target_id,
                    "repeat_index": int(rep),
                    "sample_index": int(rep),
                    "label_vector": list(label_vec),
                    "prompt": prompt,
                    "negative_prompt": negative,
                }
            )
    return rows



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
        negative = _resolve_negative_prompt(prompt_cfg)
        prompts = [str(prompt) for _ in range(batch_size)]
        negatives = [str(negative) for _ in range(batch_size)]
        return prompts, negatives

    if mode in {"label_conditioned", "class_conditioned", "multilabel_conditioned"}:
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

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.sd35_task_aware_vae.evaluation.generation_filter import filter_generated_probabilities
from src.sd35_task_aware_vae.labels.schema import load_label_schema
from src.sd35_task_aware_vae.sd3.pipeline_factory import build_sd3_img2img_pipeline, build_sd3_text2img_pipeline
from src.sd35_task_aware_vae.sd3.prompts import build_class_prompt_entries, resolve_prompts
from src.sd35_task_aware_vae.sd3.sampling import sample_img2img, sample_text2img
from src.sd35_task_aware_vae.teacher_classifier import build_convnext_large, build_teacher_transforms
from src.sd35_task_aware_vae.utils.config import dump_yaml
from src.sd35_task_aware_vae.utils.device import get_gpu_ids, set_visible_gpus
from src.sd35_task_aware_vae.utils.files import ensure_dir, write_csv, write_json
from src.sd35_task_aware_vae.utils.seed import seed_everything



def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    return obj or {}



def build_pil_transform(center_crop_size: int, image_size: int):
    from torchvision import transforms as T

    return T.Compose([
        T.CenterCrop(center_crop_size),
        T.Resize((image_size, image_size)),
    ])



def save_image_with_label(
    image,
    *,
    out_dir: Path,
    source_path: str,
    generation_index: int,
    save_label: bool,
    filename_pattern: str,
    data_root: Path,
) -> tuple[str, str | None]:
    src = Path(source_path)
    try:
        rel = src.relative_to(data_root)
    except Exception:
        rel = Path(src.name)
    stem = rel.stem
    suffix = rel.suffix or ".png"
    out_name = filename_pattern.format(stem=stem, idx=generation_index, suffix=suffix)
    out_rel = rel.with_name(out_name)
    out_path = out_dir / out_rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)

    label_out = None
    if save_label:
        src_label = src.with_suffix(".yaml")
        if src_label.is_file():
            label_path = out_path.with_suffix(".yaml")
            label_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_label, label_path)
            label_out = label_path.as_posix()
    return out_path.as_posix(), label_out



def _write_label_yaml(path: Path, class_names: list[str], label_vector: np.ndarray) -> str:
    payload = {str(name): int(float(label_vector[idx]) >= 0.5) for idx, name in enumerate(class_names)}
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
    return path.as_posix()



def _save_class_text2img_output(
    image,
    *,
    out_dir: Path,
    target_id: str,
    sample_index: int,
    filename_pattern: str,
    save_label: bool,
    class_names: list[str],
    label_vector: np.ndarray,
) -> tuple[str, str | None]:
    class_dir = out_dir / target_id
    class_dir.mkdir(parents=True, exist_ok=True)
    out_name = filename_pattern.format(target_id=target_id, idx=sample_index, suffix=".png")
    out_path = class_dir / out_name
    image.save(out_path)

    label_path = None
    if save_label:
        label_path = _write_label_yaml(out_path.with_suffix(".yaml"), class_names, label_vector)
    return out_path.as_posix(), label_path



def _build_teacher_if_needed(cfg: dict[str, Any], class_names: list[str], device):
    import torch

    teacher_ckpt = (cfg.get("teacher", {}) or {}).get("checkpoint", None)
    if not teacher_ckpt:
        return None

    teacher = build_convnext_large(
        num_classes=len(class_names),
        pretrained=bool((cfg.get("teacher", {}) or {}).get("imagenet_pretrained", True)),
    )
    sd = torch.load(teacher_ckpt, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    teacher.load_state_dict(sd, strict=False)
    teacher.to(device)
    teacher.eval()
    return teacher



def main() -> None:
    parser = argparse.ArgumentParser(description="Run SD3.5 augmentation generation from YAML config.")
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    gpu_ids = get_gpu_ids(cfg)
    set_visible_gpus(gpu_ids)
    seed_everything(int(cfg.get("seed", 42)), deterministic=bool(cfg.get("deterministic", False)))

    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    exp_name = str(cfg.get("experiment_name", "sd35_generate_aug"))
    model_cfg = cfg.get("model", {}) or {}
    vae_cfg = cfg.get("vae", {}) or {}
    prompt_cfg = cfg.get("prompt", cfg.get("prompts", {})) or {}
    transformer_cfg = cfg.get("transformer", {}) or {}
    gen_cfg = cfg.get("generation", {}) or {}
    filter_cfg = cfg.get("filter", {}) or {}
    out_cfg = cfg.get("output", {}) or {}

    out_root = ensure_dir(out_cfg.get("root_dir", "outputs/aug/sd3"))
    out_dir = ensure_dir(out_root / exp_name)
    dump_yaml(cfg, out_dir / "config_used.yaml")

    data_cfg = cfg.get("data", {}) or {}
    schema_path = data_cfg.get("label_schema_file", None)
    if not schema_path:
        raise KeyError("data.label_schema_file is required")
    class_names, label_groups, group_reduce, mask_cfg = load_label_schema(schema_path)
    class_names = list(class_names)

    image_cfg = cfg.get("image", {}) or {}
    center_crop_size = int(image_cfg.get("center_crop_size", 3072))
    image_size = int(image_cfg.get("image_size", 1024))
    teacher_mean = tuple(float(x) for x in data_cfg.get("mean", [0.485, 0.456, 0.406]))
    teacher_std = tuple(float(x) for x in data_cfg.get("std", [0.229, 0.224, 0.225]))
    pil_tf = build_pil_transform(center_crop_size, image_size)
    teacher_tf = build_teacher_transforms(
        center_crop_size=center_crop_size,
        image_size=image_size,
        mean=teacher_mean,
        std=teacher_std,
        train=False,
    )

    device = torch.device(
        model_cfg.get("device", "cuda") if torch.cuda.is_available() and str(model_cfg.get("device", "cuda")) != "cpu" else "cpu"
    )
    filter_enabled = bool(filter_cfg.get("enabled", True))
    teacher = _build_teacher_if_needed(cfg, class_names, device) if filter_enabled else None
    if filter_enabled and teacher is None:
        raise KeyError("teacher.checkpoint is required when filter.enabled=true")

    mode = str(gen_cfg.get("mode", "img2img")).lower()
    if mode == "text2img":
        pipe = build_sd3_text2img_pipeline(model_cfg, vae_cfg, transformer_cfg)
    elif mode == "img2img":
        pipe = build_sd3_img2img_pipeline(model_cfg, vae_cfg, transformer_cfg)
    else:
        raise ValueError(f"Unsupported generation.mode: {mode}")

    prompt_mode = str(prompt_cfg.get("mode", "neutral")).lower()
    rows: list[dict[str, Any]] = []
    saved_relpaths: list[str] = []

    if prompt_mode in {"class_text2img", "per_class", "class_targets"}:
        if mode != "text2img":
            raise ValueError("prompt.mode=class_text2img requires generation.mode=text2img")

        filename_pattern = str(gen_cfg.get("filename_pattern", "{target_id}__{idx:04d}{suffix}"))
        save_label = bool(out_cfg.get("copy_labels", True))
        num_images_per_target = int(gen_cfg.get("num_images_per_class", gen_cfg.get("num_images_per_input", 1)))
        batch_size = int(gen_cfg.get("batch_size", 1))
        entries = build_class_prompt_entries(class_names, prompt_cfg=prompt_cfg, num_images_per_target=num_images_per_target)
        keep_expected = bool(filter_cfg.get("use_target_labels", True))

        for start in range(0, len(entries), batch_size):
            chunk = entries[start:start + batch_size]
            prompts = [item["prompt"] for item in chunk]
            negative_prompts = [item["negative_prompt"] for item in chunk]
            generated = sample_text2img(pipe, model_cfg, prompts, negative_prompts)

            prob = None
            score = None
            keep_mask = np.ones((len(generated),), dtype=bool)
            expected = None
            if filter_enabled and teacher is not None:
                x_teacher = torch.stack([teacher_tf(img) for img in generated]).to(device)
                with torch.no_grad():
                    logits = teacher(x_teacher)
                    prob = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
                expected = np.stack([np.asarray(item["label_vector"], dtype=np.float32) for item in chunk], axis=0) if keep_expected else None
                result = filter_generated_probabilities(
                    prob,
                    expected_labels=expected,
                    min_match_score=float(filter_cfg.get("min_match_score", 0.5)),
                    min_max_probability=float(filter_cfg.get("min_max_probability", 0.0)),
                )
                keep_mask = result.keep_mask
                score = result.match_score

            for local_idx, (item, image) in enumerate(zip(chunk, generated)):
                kept = bool(keep_mask[local_idx])
                save_path = None
                label_path = None
                if kept or bool(out_cfg.get("save_rejected", False)):
                    save_path, label_path = _save_class_text2img_output(
                        image,
                        out_dir=out_dir,
                        target_id=str(item["target_id"]),
                        sample_index=start + local_idx,
                        filename_pattern=filename_pattern,
                        save_label=(save_label and kept),
                        class_names=class_names,
                        label_vector=np.asarray(item["label_vector"], dtype=np.float32),
                    )
                    if kept and save_path is not None:
                        try:
                            rel = Path(save_path).relative_to(out_dir)
                            saved_relpaths.append(rel.as_posix())
                        except Exception:
                            saved_relpaths.append(Path(save_path).name)
                rows.append(
                    {
                        "source_path": None,
                        "saved_path": save_path,
                        "saved_label_path": label_path,
                        "kept": kept,
                        "target_id": item["target_id"],
                        "target_labels": ",".join(item["target_labels"]),
                        "repeat_index": int(item["repeat_index"]),
                        "prompt": item["prompt"],
                        "negative_prompt": item["negative_prompt"],
                        "match_score": None if score is None else float(score[local_idx]),
                        "max_probability": None if prob is None else float(prob[local_idx].max()),
                    }
                )
    else:
        from src.sd35_task_aware_vae.datasets.image_dataset import MultiLabelMedicalDataset

        base_ds = MultiLabelMedicalDataset(
            root=data_cfg["root"],
            split=str(data_cfg.get("split", data_cfg.get("train_split", "train"))),
            classes=class_names,
            transform=lambda x: x,
            center_crop_size=center_crop_size,
            image_size=image_size,
            split_filename=str(data_cfg.get("split_filename", "default_split.yaml")),
            label_groups=label_groups,
            group_reduce=group_reduce,
            mask=mask_cfg,
        )

        class PathLabelDataset(Dataset):
            def __init__(self, base):
                self.image_paths = base.image_paths
                self.labels = base.labels

            def __len__(self):
                return len(self.image_paths)

            def __getitem__(self, idx):
                return str(self.image_paths[idx]), self.labels[idx]

        ds = PathLabelDataset(base_ds)
        loader = DataLoader(
            ds,
            batch_size=int(gen_cfg.get("batch_size", 2)),
            shuffle=False,
            num_workers=int(gen_cfg.get("num_workers", 0)),
        )

        num_images_per_input = int(gen_cfg.get("num_images_per_input", 1))
        filename_pattern = str(gen_cfg.get("filename_pattern", "{stem}__aug{idx:02d}.png"))
        save_label = bool(out_cfg.get("copy_labels", True))
        data_root = Path(data_cfg["root"])
        use_source_labels = bool(filter_cfg.get("use_source_labels", prompt_mode in {"label_conditioned", "class_conditioned", "multilabel_conditioned"}))

        for batch_paths, batch_labels in loader:
            label_np = batch_labels.numpy().astype(np.float32)
            prompts, negative_prompts = resolve_prompts(
                batch_size=len(batch_paths),
                labels=label_np,
                class_names=class_names,
                prompt_cfg=prompt_cfg,
            )

            pil_images = None
            if mode == "img2img":
                pil_images = []
                for p in batch_paths:
                    with Image.open(p) as img:
                        img = img.convert("RGB")
                        pil_images.append(pil_tf(img))

            for aug_idx in range(num_images_per_input):
                if mode == "text2img":
                    generated = sample_text2img(pipe, model_cfg, prompts, negative_prompts)
                else:
                    assert pil_images is not None
                    generated = sample_img2img(pipe, model_cfg, pil_images, prompts, negative_prompts)

                keep_mask = np.ones((len(generated),), dtype=bool)
                prob = None
                score = None
                if filter_enabled and teacher is not None:
                    x_teacher = torch.stack([teacher_tf(img) for img in generated]).to(device)
                    with torch.no_grad():
                        logits = teacher(x_teacher)
                        prob = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
                    result = filter_generated_probabilities(
                        prob,
                        expected_labels=(label_np if use_source_labels else None),
                        min_match_score=float(filter_cfg.get("min_match_score", 0.5)),
                        min_max_probability=float(filter_cfg.get("min_max_probability", 0.0)),
                    )
                    keep_mask = result.keep_mask
                    score = result.match_score

                for i, img in enumerate(generated):
                    kept = bool(keep_mask[i])
                    save_path = None
                    label_path = None
                    if kept or bool(out_cfg.get("save_rejected", False)):
                        save_path, label_path = save_image_with_label(
                            img,
                            out_dir=out_dir,
                            source_path=batch_paths[i],
                            generation_index=aug_idx,
                            save_label=(save_label and kept),
                            filename_pattern=filename_pattern,
                            data_root=data_root,
                        )
                        if kept and save_path is not None:
                            try:
                                rel = Path(save_path).relative_to(out_dir)
                                saved_relpaths.append(rel.as_posix())
                            except Exception:
                                saved_relpaths.append(Path(save_path).name)
                    rows.append(
                        {
                            "source_path": batch_paths[i],
                            "saved_path": save_path,
                            "saved_label_path": label_path,
                            "kept": kept,
                            "generation_index": aug_idx,
                            "prompt": prompts[i],
                            "negative_prompt": negative_prompts[i],
                            "match_score": None if score is None else float(score[i]),
                            "max_probability": None if prob is None else float(prob[i].max()),
                        }
                    )

    write_csv(rows, out_dir / "metadata.csv")
    write_json(
        {
            "experiment_name": exp_name,
            "num_generated": len(rows),
            "num_kept": int(sum(1 for r in rows if r["kept"])),
            "mode": mode,
            "prompt_mode": prompt_mode,
        },
        out_dir / "summary.json",
    )
    if bool(out_cfg.get("write_split_yaml", True)):
        split_name = str(out_cfg.get("split_name", f"{exp_name}_generated"))
        split_filename = str(out_cfg.get("split_filename", "generated_split.yaml"))
        write_json({split_name: saved_relpaths}, out_dir / split_filename.replace(".yaml", ".json"))
        with (out_dir / split_filename).open("w", encoding="utf-8") as f:
            yaml.safe_dump({split_name: saved_relpaths}, f, allow_unicode=True, sort_keys=False)

    print(f"[done] generated augmentation written to {out_dir}")


if __name__ == "__main__":
    main()

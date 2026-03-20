from __future__ import annotations

import argparse
import heapq
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.sd35_task_aware_vae.datasets.image_dataset import MultiLabelMedicalDataset
from src.sd35_task_aware_vae.evaluation.restore_eval import sanitize_id, summarize_restore_results
from src.sd35_task_aware_vae.labels.schema import load_label_schema
from src.sd35_task_aware_vae.sd3.pipeline_factory import build_sd3_img2img_pipeline
from src.sd35_task_aware_vae.sd3.restore import reverse_restore_batch
from src.sd35_task_aware_vae.teacher_classifier import build_convnext_large, build_teacher_transforms
from src.sd35_task_aware_vae.utils.config import dump_yaml
from src.sd35_task_aware_vae.utils.files import ensure_dir, write_csv, write_json
from src.sd35_task_aware_vae.utils.seed import seed_everything



def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    return obj or {}



def build_vae_transform(center_crop_size: int, image_size: int):
    from torchvision import transforms as T

    return T.Compose(
        [
            T.CenterCrop(center_crop_size),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Lambda(lambda x: x * 2.0 - 1.0),
        ]
    )



def build_vis_transform(center_crop_size: int, image_size: int):
    from torchvision import transforms as T

    return T.Compose(
        [
            T.CenterCrop(center_crop_size),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ]
    )



def normalize_for_teacher(x_minus1_1, mean: tuple[float, float, float], std: tuple[float, float, float]):
    import torch

    x01 = (x_minus1_1.clamp(-1.0, 1.0) + 1.0) / 2.0
    mean_t = torch.tensor(mean, device=x_minus1_1.device, dtype=x_minus1_1.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=x_minus1_1.device, dtype=x_minus1_1.dtype).view(1, 3, 1, 1)
    return (x01 - mean_t) / std_t



def forward_teacher(teacher, x, use_embedding_cosine: bool):
    import torch

    with torch.no_grad():
        logits = teacher(x)
        emb = None
        if use_embedding_cosine and hasattr(teacher, "forward_embedding"):
            emb = teacher.forward_embedding(x)
    return emb, logits



def save_topk_samples(topk_by_key: dict[str, list[tuple[float, int, dict[str, Any]]]], out_dir: Path) -> None:
    from torchvision.utils import save_image

    for key, heap in topk_by_key.items():
        target_dir = ensure_dir(out_dir / "worst_samples" / key)
        for rank, (_score, _idx, sample) in enumerate(sorted(heap, key=lambda x: x[0], reverse=True), start=1):
            base = target_dir / f"{rank:03d}_{sample['sample_id']}"
            save_image(sample["orig"], base.with_name(base.name + "_orig.png"))
            save_image(sample["restored"], base.with_name(base.name + "_restored.png"))
            save_image(sample["diff"], base.with_name(base.name + "_diff.png"))
            panel = np.concatenate(
                [
                    np.transpose(sample["orig"].numpy(), (1, 2, 0)),
                    np.transpose(sample["restored"].numpy(), (1, 2, 0)),
                    np.transpose(sample["diff"].numpy(), (1, 2, 0)),
                ],
                axis=1,
            )
            # Convert back to CHW for save_image.
            import torch

            panel_tensor = torch.from_numpy(np.transpose(panel, (2, 0, 1))).clamp(0.0, 1.0)
            save_image(panel_tensor, base.with_name(base.name + "_panel.png"))



def main() -> None:
    parser = argparse.ArgumentParser(description="Run SD3.5 reverse-restoration evaluation from YAML config.")
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(int(cfg.get("seed", 42)), deterministic=bool(cfg.get("deterministic", False)))

    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    exp_name = str(cfg.get("experiment_name", "sd35_restore_eval"))
    model_cfg = cfg.get("model", {}) or {}
    vae_cfg = cfg.get("vae", {}) or {}
    eval_cfg = cfg.get("eval", {}) or {}
    prompt_cfg = cfg.get("prompt", cfg.get("prompts", {})) or {}
    out_cfg = cfg.get("output", {}) or {}

    out_root = ensure_dir(out_cfg.get("root_dir", "outputs/eval/sd3_restore"))
    out_dir = ensure_dir(out_root / exp_name)
    dump_yaml(cfg, out_dir / "config_used.yaml")

    data_cfg = cfg.get("data", {}) or {}
    schema_path = data_cfg.get("label_schema_file", None)
    if not schema_path:
        raise KeyError("data.label_schema_file is required")
    class_names, label_groups, group_reduce, mask_cfg = load_label_schema(schema_path)

    image_cfg = cfg.get("image", {}) or {}
    center_crop_size = int(image_cfg.get("center_crop_size", 3072))
    image_size = int(image_cfg.get("image_size", 1024))
    mean = tuple(float(x) for x in data_cfg.get("mean", [0.485, 0.456, 0.406]))
    std = tuple(float(x) for x in data_cfg.get("std", [0.229, 0.224, 0.225]))

    teacher_tf = build_teacher_transforms(
        center_crop_size=center_crop_size,
        image_size=image_size,
        mean=mean,
        std=std,
        train=False,
    )
    vae_tf = build_vae_transform(center_crop_size, image_size)
    vis_tf = build_vis_transform(center_crop_size, image_size)

    base_ds = MultiLabelMedicalDataset(
        root=data_cfg["root"],
        split=str(data_cfg.get("split", data_cfg.get("val_split", "val"))),
        classes=class_names,
        transform=lambda x: x,
        center_crop_size=center_crop_size,
        image_size=image_size,
        split_filename=str(data_cfg.get("split_filename", "default_split.yaml")),
        label_groups=label_groups,
        group_reduce=group_reduce,
        mask=mask_cfg,
    )

    class RestoreDataset(Dataset):
        def __init__(self, base):
            self.base = base
            self.image_paths = base.image_paths
            self.labels = base.labels

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            path = self.image_paths[idx]
            label = self.labels[idx]
            with Image.open(path) as img:
                img = img.convert("RGB")
                x_teacher = teacher_tf(img)
                x_vae = vae_tf(img)
                x_vis = vis_tf(img)
            return x_teacher, x_vae, x_vis, label, str(path)

    ds = RestoreDataset(base_ds)
    batch_size = int(eval_cfg.get("batch_size", 2))
    num_workers = int(eval_cfg.get("num_workers", 2))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    teacher_cfg = cfg.get("teacher", {}) or {}
    teacher_ckpt = teacher_cfg.get("checkpoint", None)
    if not teacher_ckpt:
        raise KeyError("teacher.checkpoint is required")

    teacher = build_convnext_large(
        num_classes=len(class_names),
        pretrained=bool(teacher_cfg.get("imagenet_pretrained", True)),
    )
    sd = torch.load(teacher_ckpt, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    teacher.load_state_dict(sd, strict=False)

    device = torch.device(
        model_cfg.get("device", "cuda") if torch.cuda.is_available() and str(model_cfg.get("device", "cuda")) != "cpu" else "cpu"
    )
    teacher.to(device)
    teacher.eval()

    use_embedding_cos = bool((eval_cfg.get("agreement", {}) or {}).get("embedding_cosine", False))

    print("[info] building SD3.5 img2img pipeline...")
    pipe = build_sd3_img2img_pipeline(model_cfg, vae_cfg)
    if hasattr(pipe, "scheduler") and eval_cfg.get("scheduler", None):
        # Future extension point; currently we use the pipeline's default scheduler.
        pass

    y_true_chunks: list[np.ndarray] = []
    p_real_chunks: list[np.ndarray] = []
    emb_real_chunks: list[np.ndarray] = []

    print("[info] collecting real-image teacher predictions...")
    for x_teacher, _x_vae, _x_vis, labels, _paths in loader:
        x_teacher = x_teacher.to(device)
        emb, logits = forward_teacher(teacher, x_teacher, use_embedding_cos)
        y_true_chunks.append(labels.numpy().astype(np.float32))
        p_real_chunks.append(torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32))
        if emb is not None:
            emb_real_chunks.append(emb.detach().cpu().numpy().astype(np.float32))

    y_true = np.concatenate(y_true_chunks, axis=0)
    p_real = np.concatenate(p_real_chunks, axis=0)
    emb_real = np.concatenate(emb_real_chunks, axis=0) if emb_real_chunks else None

    timesteps = eval_cfg.get("timesteps", eval_cfg.get("start_timesteps", [50, 100, 200, 400])) or []
    timesteps = [int(t) for t in timesteps]
    reverse_steps = int(eval_cfg.get("reverse_steps", model_cfg.get("num_inference_steps", 40)))
    posterior = str(vae_cfg.get("posterior", "mode"))
    guidance_scale = float(eval_cfg.get("guidance_scale", model_cfg.get("guidance_scale", 4.5)))
    max_sequence_length = int(eval_cfg.get("max_sequence_length", model_cfg.get("max_sequence_length", 256)))
    save_topk = int(out_cfg.get("save_topk", eval_cfg.get("save_topk", 16)))

    p_by_key: dict[str, np.ndarray] = {}
    emb_by_key: dict[str, np.ndarray | None] = {}
    topk_by_key: dict[str, list[tuple[float, int, dict[str, Any]]]] = {f"t{t}": [] for t in timesteps}
    global_index = 0

    for timestep in timesteps:
        key = f"t{timestep}"
        print(f"[info] running reverse restoration for {key}...")
        restored_prob_chunks: list[np.ndarray] = []
        restored_emb_chunks: list[np.ndarray] = []
        sample_counter = 0
        offset = 0

        for x_teacher, x_vae, x_vis, labels, paths in loader:
            x_vae = x_vae.to(device=device, dtype=torch.float32)
            result = reverse_restore_batch(
                pipe,
                x_vae,
                labels=labels,
                class_names=class_names,
                prompt_cfg=prompt_cfg,
                posterior=posterior,
                start_timestep=timestep,
                reverse_steps=reverse_steps,
                guidance_scale=guidance_scale,
                max_sequence_length=max_sequence_length,
            )
            restored = result["restored"]
            restored_teacher = normalize_for_teacher(restored, mean, std).to(device)
            emb_rest, logits_rest = forward_teacher(teacher, restored_teacher, use_embedding_cos)
            prob_rest = torch.sigmoid(logits_rest).detach().cpu().numpy().astype(np.float32)
            restored_prob_chunks.append(prob_rest)
            if emb_rest is not None:
                restored_emb_chunks.append(emb_rest.detach().cpu().numpy().astype(np.float32))

            batch_real_prob = p_real[offset: offset + len(paths)]
            batch_scores = np.abs(prob_rest - batch_real_prob).mean(axis=1)
            if save_topk > 0:
                restored_vis = (restored.detach().cpu().clamp(-1.0, 1.0) + 1.0) / 2.0
                diff_vis = (restored_vis - x_vis).abs().clamp(0.0, 1.0)
                for i, score in enumerate(batch_scores.tolist()):
                    sample_id = sanitize_id(paths[i])
                    item = {
                        "sample_id": sample_id,
                        "path": paths[i],
                        "score": float(score),
                        "orig": x_vis[i].detach().cpu(),
                        "restored": restored_vis[i].detach().cpu(),
                        "diff": diff_vis[i].detach().cpu(),
                    }
                    heap = topk_by_key[key]
                    entry = (float(score), sample_counter, item)
                    if len(heap) < save_topk:
                        heapq.heappush(heap, entry)
                    elif float(score) > heap[0][0]:
                        heapq.heapreplace(heap, entry)
                    sample_counter += 1
            offset += len(paths)

        p_by_key[key] = np.concatenate(restored_prob_chunks, axis=0)
        emb_by_key[key] = np.concatenate(restored_emb_chunks, axis=0) if restored_emb_chunks else None

    summary, per_label_rows, curve_rows = summarize_restore_results(
        class_names=class_names,
        y_true=y_true,
        p_real=p_real,
        p_by_key=p_by_key,
        embeddings_real=emb_real,
        embeddings_by_key=emb_by_key,
        threshold_cfg=cfg.get("threshold", eval_cfg.get("threshold", {})) or {},
    )
    summary["timesteps"] = timesteps
    summary["reverse_steps"] = reverse_steps
    summary["prompt"] = prompt_cfg
    write_json(summary, out_dir / "summary.json")
    write_csv(per_label_rows, out_dir / "per_label.csv")
    write_csv(curve_rows, out_dir / "t_curve.csv")

    if save_topk > 0:
        save_topk_samples(topk_by_key, out_dir)

    print(f"[done] results written to {out_dir}")


if __name__ == "__main__":
    main()

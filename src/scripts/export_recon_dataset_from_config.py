from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import yaml

from src.sd35_task_aware_vae.datasets.image_dataset import MultiLabelMedicalDataset
from src.sd35_task_aware_vae.labels.schema import load_label_schema
from src.sd35_task_aware_vae.sd3.latent_codec import decode_from_latents, encode_to_latents
from src.sd35_task_aware_vae.sd3.vae_factory import build_sd3_vae
from src.sd35_task_aware_vae.utils.config import dump_yaml
from src.sd35_task_aware_vae.utils.files import ensure_dir
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



def _noisify_latents(latents, timesteps, noise_cfg: dict[str, Any], generator=None):
    import torch

    if not timesteps or timesteps == [0]:
        return {0: latents}

    mode = str(noise_cfg.get("mode", noise_cfg.get("scheduler", "gaussian"))).lower()
    out: dict[int, torch.Tensor] = {0: latents}
    if generator is None:
        eps = torch.randn_like(latents)
    else:
        eps = torch.randn(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)

    if mode in {"gaussian", "custom"}:
        denom = float(max(int(noise_cfg.get("num_train_timesteps", 1000)) - 1, 1))
        for t in timesteps:
            if int(t) == 0:
                continue
            beta = float(t) / denom
            beta = min(max(beta, 0.0), 0.999)
            out[int(t)] = (1.0 - beta) ** 0.5 * latents + beta**0.5 * eps
        return out

    if mode == "ddpm":
        try:
            from diffusers import DDPMScheduler
        except Exception as e:  # pragma: no cover
            raise RuntimeError("diffusers is required for noise mode='ddpm'") from e
        scheduler = DDPMScheduler(
            num_train_timesteps=int(noise_cfg.get("num_train_timesteps", 1000)),
            beta_schedule=str(noise_cfg.get("beta_schedule", "scaled_linear")),
        )
        for t in timesteps:
            if int(t) == 0:
                continue
            t_tensor = torch.full((latents.shape[0],), int(t), device=latents.device, dtype=torch.long)
            out[int(t)] = scheduler.add_noise(latents, eps, t_tensor)
        return out

    raise ValueError(f"Unsupported noise mode: {mode}")



def reconstruct_sd35_from_config(cfg: dict[str, Any], config_path: str | Path) -> Path:
    import torch
    from torch.utils.data import DataLoader
    from torchvision.utils import save_image

    seed_everything(int(cfg.get("seed", 42)), deterministic=bool(cfg.get("deterministic", False)))

    exp_name = str(cfg.get("experiment_name", "sd35_reconstruct"))
    model_cfg = cfg.get("model", {}) or {}
    vae_cfg = cfg.get("vae", {}) or {}
    data_cfg = cfg.get("data", {}) or {}
    image_cfg = cfg.get("image", {}) or {}
    infer_cfg = cfg.get("inference", {}) or {}
    noise_cfg = cfg.get("noise", {}) or {}
    out_cfg = cfg.get("output", {}) or {}

    out_root = ensure_dir(out_cfg.get("root_dir", "outputs/reconstructions/sd35"))
    out_dir = ensure_dir(out_root / exp_name)
    dump_yaml(cfg, out_dir / "config_used.yaml")

    schema_path = data_cfg.get("label_schema_file", None)
    if schema_path:
        class_names, label_groups, group_reduce, mask_cfg = load_label_schema(schema_path)
    else:
        class_names = list(data_cfg.get("classes", []))
        if not class_names:
            raise KeyError("data.label_schema_file or data.classes is required")
        label_groups = data_cfg.get("label_groups", {}) or {}
        group_reduce = data_cfg.get("group_reduce", "any")
        mask_cfg = data_cfg.get("mask", {}) or {}

    center_crop_size = int(image_cfg.get("center_crop_size", 3072))
    image_size = int(image_cfg.get("image_size", 1024))
    transform = build_vae_transform(center_crop_size, image_size)

    ds = MultiLabelMedicalDataset(
        root=data_cfg["root"],
        split=str(data_cfg.get("split", data_cfg.get("val_split", "val"))),
        classes=class_names,
        transform=transform,
        center_crop_size=center_crop_size,
        image_size=image_size,
        split_filename=str(data_cfg.get("split_filename", "default_split.yaml")),
        label_groups=label_groups,
        group_reduce=group_reduce,
        mask=mask_cfg,
    )
    loader = DataLoader(
        ds,
        batch_size=int(infer_cfg.get("batch_size", 4)),
        shuffle=False,
        num_workers=int(infer_cfg.get("num_workers", 2)),
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(
        model_cfg.get("device", "cuda") if torch.cuda.is_available() and str(model_cfg.get("device", "cuda")) != "cpu" else "cpu"
    )
    vae = build_sd3_vae(model_cfg, vae_cfg, torch_dtype=torch.float32, device=device)
    vae.eval()
    posterior_mode = str(vae_cfg.get("posterior", "mode"))
    timesteps = [int(t) for t in (noise_cfg.get("timesteps", [0]) or [0])]
    if 0 not in timesteps:
        timesteps = [0] + timesteps
    timesteps = sorted(set(timesteps))

    copy_labels = bool(out_cfg.get("copy_labels", True))
    preserve_relative_paths = bool(out_cfg.get("preserve_relative_paths", True))
    keep_original_name_for_t0 = bool(out_cfg.get("keep_original_name_for_t0", True))
    suffix_fmt = str(out_cfg.get("timestep_suffix_format", "__t{t:04d}"))
    split_filename = str(out_cfg.get("split_filename", "default_split.yaml"))
    split_name = str(out_cfg.get("split_name", data_cfg.get("split", data_cfg.get("val_split", "val"))))

    saved_relpaths: list[str] = []
    generator = None
    if noise_cfg.get("seed", None) is not None:
        generator = torch.Generator(device=device.type if device.type != "cpu" else None)
        generator.manual_seed(int(noise_cfg.get("seed")))

    for images, _labels, paths in loader:
        images = images.to(device=device, dtype=torch.float32)
        latents = encode_to_latents(vae, images, sample_mode=posterior_mode, generator=generator)
        latents_by_t = _noisify_latents(latents, timesteps, noise_cfg, generator=generator)
        for t, latent_t in latents_by_t.items():
            recon = decode_from_latents(vae, latent_t)
            recon01 = (recon.detach().cpu().clamp(-1.0, 1.0) + 1.0) / 2.0
            for i, src_path in enumerate(paths):
                src = Path(src_path)
                try:
                    rel = src.relative_to(Path(data_cfg["root"])) if preserve_relative_paths else Path(src.name)
                except Exception:
                    rel = Path(src.name)
                if int(t) == 0 and keep_original_name_for_t0:
                    rel_out = rel
                else:
                    rel_out = rel.with_name(rel.stem + suffix_fmt.format(t=int(t)) + rel.suffix)
                save_path = out_dir / rel_out
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_image(recon01[i], save_path)
                saved_relpaths.append(rel_out.as_posix())
                if copy_labels:
                    src_label = src.with_suffix(".yaml")
                    if src_label.is_file():
                        dst_label = save_path.with_suffix(".yaml")
                        dst_label.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src_label, dst_label)

    with (out_dir / split_filename).open("w", encoding="utf-8") as f:
        yaml.safe_dump({split_name: saved_relpaths}, f, allow_unicode=True, sort_keys=False)

    return out_dir



def main() -> None:
    parser = argparse.ArgumentParser(description="Export SD3.5 VAE reconstructions from YAML config.")
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    out_dir = reconstruct_sd35_from_config(cfg, args.config)
    print(f"[done] reconstructions written to {out_dir}")


if __name__ == "__main__":
    main()

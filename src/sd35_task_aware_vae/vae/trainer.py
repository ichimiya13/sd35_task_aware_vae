from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.sd35_task_aware_vae.datasets.image_dataset import MultiLabelMedicalDataset
from src.sd35_task_aware_vae.labels.schema import load_label_schema
from src.sd35_task_aware_vae.sd3.latent_codec import decode_from_latents, encode_to_latents
from src.sd35_task_aware_vae.sd3.vae_factory import apply_freeze_patterns, build_sd3_vae
from src.sd35_task_aware_vae.teacher_classifier import build_convnext_large
from src.sd35_task_aware_vae.utils.config import dump_yaml
from src.sd35_task_aware_vae.utils.files import ensure_dir, write_csv, write_json
from src.sd35_task_aware_vae.utils.seed import seed_everything
from src.sd35_task_aware_vae.utils.wandb import init_wandb_session, maybe_build_wandb_image
from src.sd35_task_aware_vae.vae.losses import feature_distance, posterior_kl_loss, reconstruction_loss


@dataclass
class TeacherViews:
    features: Any | None
    logits: Any | None


@dataclass
class EpochSummary:
    epoch: int
    split: str
    total_loss: float
    recon_loss: float
    kl_loss: float
    feature_loss: float
    logit_loss: float
    noise_feature_loss: float

    def as_row(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "split": self.split,
            "total_loss": self.total_loss,
            "recon_loss": self.recon_loss,
            "kl_loss": self.kl_loss,
            "feature_loss": self.feature_loss,
            "logit_loss": self.logit_loss,
            "noise_feature_loss": self.noise_feature_loss,
        }



def build_vae_transform(center_crop_size: int, image_size: int, augment: dict[str, Any] | None = None):
    from torchvision import transforms as T

    augment = augment or {}
    hflip_p = float(augment.get("hflip_p", 0.0))
    tfms = [
        T.CenterCrop(center_crop_size),
        T.Resize((image_size, image_size)),
    ]
    if hflip_p > 0:
        tfms.append(T.RandomHorizontalFlip(p=hflip_p))
    tfms += [
        T.ToTensor(),
        T.Lambda(lambda x: x * 2.0 - 1.0),
    ]
    return T.Compose(tfms)



def _teacher_stats(cfg: dict[str, Any]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    data_cfg = cfg.get("data", {}) or {}
    mean = tuple(float(x) for x in data_cfg.get("mean", [0.485, 0.456, 0.406]))
    std = tuple(float(x) for x in data_cfg.get("std", [0.229, 0.224, 0.225]))
    return mean, std



def normalize_for_teacher(x_minus1_1, mean: tuple[float, float, float], std: tuple[float, float, float]):
    import torch

    x01 = (x_minus1_1.clamp(-1.0, 1.0) + 1.0) / 2.0
    mean_t = torch.tensor(mean, device=x_minus1_1.device, dtype=x_minus1_1.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=x_minus1_1.device, dtype=x_minus1_1.dtype).view(1, 3, 1, 1)
    return (x01 - mean_t) / std_t



def build_datasets(cfg: dict[str, Any]):
    data_cfg = cfg.get("data", {}) or {}
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

    image_cfg = cfg.get("image", {}) or {}
    center_crop_size = int(image_cfg.get("center_crop_size", 3072))
    image_size = int(image_cfg.get("image_size", 1024))
    aug_cfg = cfg.get("augment", {}) or {}

    train_tf = build_vae_transform(center_crop_size, image_size, augment=aug_cfg)
    val_tf = build_vae_transform(center_crop_size, image_size, augment=None)

    common = dict(
        root=data_cfg["root"],
        classes=class_names,
        center_crop_size=center_crop_size,
        image_size=image_size,
        split_filename=str(data_cfg.get("split_filename", "default_split.yaml")),
        label_groups=label_groups,
        group_reduce=group_reduce,
        mask=mask_cfg,
    )
    train_ds = MultiLabelMedicalDataset(
        split=str(data_cfg.get("train_split", "train")),
        transform=train_tf,
        **common,
    )
    val_ds = MultiLabelMedicalDataset(
        split=str(data_cfg.get("val_split", "val")),
        transform=val_tf,
        **common,
    )
    return train_ds, val_ds, class_names



def build_teacher_if_needed(cfg: dict[str, Any], num_classes: int, device):
    import torch

    loss_cfg = cfg.get("loss", {}) or {}
    need_teacher = any(
        float(loss_cfg.get(k, 0.0)) > 0
        for k in ["feature_weight", "logit_weight", "noise_feature_weight"]
    )
    if not need_teacher:
        return None

    teacher_cfg = cfg.get("teacher", {}) or {}
    ckpt = teacher_cfg.get("checkpoint", None)
    if not ckpt:
        raise KeyError("teacher.checkpoint is required when teacher-based VAE losses are enabled")

    teacher = build_convnext_large(
        num_classes=num_classes,
        pretrained=bool(teacher_cfg.get("imagenet_pretrained", True)),
    )
    sd = torch.load(ckpt, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    teacher.load_state_dict(sd, strict=False)
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher



def _extract_teacher_views(
    teacher,
    x_teacher,
    *,
    feature_stage: str | int | None = "embedding",
):
    if teacher is None:
        return TeacherViews(features=None, logits=None)

    logits = teacher(x_teacher)
    features = None
    if hasattr(teacher, "forward_embedding") and feature_stage in {None, "embedding", "pooled", "final"}:
        features = teacher.forward_embedding(x_teacher)
    elif hasattr(teacher, "forward_features"):
        stage = None if feature_stage in {None, "final_map"} else feature_stage
        features = teacher.forward_features(x_teacher, stage=stage)
    return TeacherViews(features=features, logits=logits)



def _sample_noisy_latents(latents, noise_cfg: dict[str, Any], generator=None):
    import torch

    mode = str(noise_cfg.get("mode", "gaussian")).lower()
    if generator is None:
        eps = torch.randn_like(latents)
    else:
        eps = torch.randn(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)

    if mode == "gaussian":
        min_beta = float(noise_cfg.get("min_beta", 0.02))
        max_beta = float(noise_cfg.get("max_beta", 0.25))
        beta = torch.empty((latents.shape[0], 1, 1, 1), device=latents.device, dtype=latents.dtype).uniform_(min_beta, max_beta)
        return torch.sqrt(1.0 - beta) * latents + torch.sqrt(beta) * eps

    if mode == "ddpm":
        try:
            from diffusers import DDPMScheduler
        except Exception as e:  # pragma: no cover - optional dependency
            raise RuntimeError("diffusers is required for mode='ddpm'") from e
        scheduler = DDPMScheduler(
            num_train_timesteps=int(noise_cfg.get("num_train_timesteps", 1000)),
            beta_schedule=str(noise_cfg.get("beta_schedule", "scaled_linear")),
        )
        timesteps = torch.randint(
            low=int(noise_cfg.get("min_timestep", 0)),
            high=int(noise_cfg.get("max_timestep", scheduler.config.num_train_timesteps - 1)) + 1,
            size=(latents.shape[0],),
            device=latents.device,
        ).long()
        return scheduler.add_noise(latents, eps, timesteps)

    raise ValueError(f"Unsupported noise_conditioning.mode: {mode}")



def _compute_loss_terms(
    *,
    vae,
    batch,
    teacher,
    mean,
    std,
    loss_cfg: dict[str, Any],
    teacher_cfg: dict[str, Any],
    posterior_mode: str,
    noise_cfg: dict[str, Any],
    generator=None,
):
    import torch

    feature_kind = str(loss_cfg.get("feature_loss_type", "mse"))
    recon_kind = str(loss_cfg.get("recon_type", "l1"))
    feature_stage = teacher_cfg.get("feature_stage", "embedding")

    latents, posterior = encode_to_latents(
        vae,
        batch,
        sample_mode=posterior_mode,
        generator=generator,
        return_posterior=True,
    )
    recon = decode_from_latents(vae, latents)

    recon_loss_value = reconstruction_loss(recon, batch, kind=recon_kind)
    kl_loss_value = posterior_kl_loss(posterior)
    feature_loss_value = batch.new_tensor(0.0)
    logit_loss_value = batch.new_tensor(0.0)
    noise_feature_loss_value = batch.new_tensor(0.0)

    if teacher is not None:
        with torch.no_grad():
            target_teacher = normalize_for_teacher(batch, mean, std)
            target_views = _extract_teacher_views(teacher, target_teacher, feature_stage=feature_stage)

        recon_teacher = normalize_for_teacher(recon, mean, std)
        recon_views = _extract_teacher_views(teacher, recon_teacher, feature_stage=feature_stage)

        if float(loss_cfg.get("feature_weight", 0.0)) > 0 and target_views.features is not None and recon_views.features is not None:
            feature_loss_value = feature_distance(recon_views.features, target_views.features, kind=feature_kind)

        if float(loss_cfg.get("logit_weight", 0.0)) > 0 and target_views.logits is not None and recon_views.logits is not None:
            logit_loss_value = feature_distance(recon_views.logits, target_views.logits, kind=str(loss_cfg.get("logit_loss_type", "mse")))

        if float(loss_cfg.get("noise_feature_weight", 0.0)) > 0:
            noisy_latents = _sample_noisy_latents(latents, noise_cfg, generator=generator)
            recon_noise = decode_from_latents(vae, noisy_latents)
            recon_noise_teacher = normalize_for_teacher(recon_noise, mean, std)
            recon_noise_views = _extract_teacher_views(teacher, recon_noise_teacher, feature_stage=feature_stage)
            if target_views.features is not None and recon_noise_views.features is not None:
                noise_feature_loss_value = feature_distance(
                    recon_noise_views.features,
                    target_views.features,
                    kind=str(loss_cfg.get("noise_feature_loss_type", feature_kind)),
                )

    weights = {
        "recon": float(loss_cfg.get("recon_weight", 1.0)),
        "kl": float(loss_cfg.get("kl_weight", 1.0e-6)),
        "feature": float(loss_cfg.get("feature_weight", 0.0)),
        "logit": float(loss_cfg.get("logit_weight", 0.0)),
        "noise_feature": float(loss_cfg.get("noise_feature_weight", 0.0)),
    }
    total = (
        weights["recon"] * recon_loss_value
        + weights["kl"] * kl_loss_value
        + weights["feature"] * feature_loss_value
        + weights["logit"] * logit_loss_value
        + weights["noise_feature"] * noise_feature_loss_value
    )
    return {
        "total": total,
        "recon": recon_loss_value,
        "kl": kl_loss_value,
        "feature": feature_loss_value,
        "logit": logit_loss_value,
        "noise_feature": noise_feature_loss_value,
        "recon_image": recon,
    }



def _run_epoch(
    *,
    vae,
    loader,
    optimizer,
    scaler,
    device,
    amp_dtype,
    train: bool,
    teacher,
    mean,
    std,
    loss_cfg,
    teacher_cfg,
    posterior_mode,
    noise_cfg,
    grad_accum_steps: int,
    epoch: int,
    save_preview_dir: Path | None = None,
    global_step_start: int = 0,
    wandb_session=None,
    step_log_interval: int = 0,
):
    import torch
    from torchvision.utils import save_image

    vae.train(train)
    total = recon = kl = feat = logit = noise_feat = 0.0
    num_batches = 0
    optimizer_step = int(global_step_start)

    autocast_enabled = amp_dtype is not None and device.type == "cuda"
    preview_saved = False

    if train:
        optimizer.zero_grad(set_to_none=True)

    def _maybe_log_step(terms, step_idx: int):
        if (not train) or wandb_session is None or (not getattr(wandb_session, "enabled", False)):
            return
        if step_log_interval <= 0 or step_idx % max(1, step_log_interval) != 0:
            return
        lr_now = float(optimizer.param_groups[0]["lr"])
        payload = {
            "train/global_step": step_idx,
            "train/epoch": epoch,
            "train/loss_step": float(terms["total"].detach().cpu()),
            "train/recon_step": float(terms["recon"].detach().cpu()),
            "train/kl_step": float(terms["kl"].detach().cpu()),
            "train/feature_step": float(terms["feature"].detach().cpu()),
            "train/logit_step": float(terms["logit"].detach().cpu()),
            "train/noise_feature_step": float(terms["noise_feature"].detach().cpu()),
            "train/loss_running": float(total / max(1, num_batches)),
            "train/recon_running": float(recon / max(1, num_batches)),
            "train/kl_running": float(kl / max(1, num_batches)),
            "train/feature_running": float(feat / max(1, num_batches)),
            "train/logit_running": float(logit / max(1, num_batches)),
            "train/noise_feature_running": float(noise_feat / max(1, num_batches)),
            "train/lr": lr_now,
        }
        try:
            if scaler is not None:
                payload["train/grad_scale"] = float(scaler.get_scale())
        except Exception:
            pass
        wandb_session.log(payload, step=step_idx)

    for step, (images, _labels, _paths) in enumerate(loader):
        images = images.to(device=device, dtype=torch.float32)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                terms = _compute_loss_terms(
                    vae=vae,
                    batch=images,
                    teacher=teacher,
                    mean=mean,
                    std=std,
                    loss_cfg=loss_cfg,
                    teacher_cfg=teacher_cfg,
                    posterior_mode=posterior_mode,
                    noise_cfg=noise_cfg,
                )
                loss = terms["total"] / max(1, grad_accum_steps)

            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (step + 1) % max(1, grad_accum_steps) == 0:
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_step += 1

        total += float(terms["total"].detach().cpu())
        recon += float(terms["recon"].detach().cpu())
        kl += float(terms["kl"].detach().cpu())
        feat += float(terms["feature"].detach().cpu())
        logit += float(terms["logit"].detach().cpu())
        noise_feat += float(terms["noise_feature"].detach().cpu())
        num_batches += 1

        if train and (step + 1) % max(1, grad_accum_steps) == 0:
            _maybe_log_step(terms, optimizer_step)

        if save_preview_dir is not None and not preview_saved:
            preview_saved = True
            save_preview_dir.mkdir(parents=True, exist_ok=True)
            orig = (images[:4].detach().cpu().clamp(-1.0, 1.0) + 1.0) / 2.0
            rec = (terms["recon_image"][:4].detach().cpu().clamp(-1.0, 1.0) + 1.0) / 2.0
            grid = torch.cat([orig, rec], dim=3)
            save_image(grid, save_preview_dir / f"epoch_{epoch:03d}.png")

    if train and num_batches > 0 and (num_batches % max(1, grad_accum_steps) != 0):
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step += 1
        _maybe_log_step(terms, optimizer_step)

    denom = float(max(1, num_batches))
    summary = EpochSummary(
        epoch=epoch,
        split="train" if train else "val",
        total_loss=total / denom,
        recon_loss=recon / denom,
        kl_loss=kl / denom,
        feature_loss=feat / denom,
        logit_loss=logit / denom,
        noise_feature_loss=noise_feat / denom,
    )
    return summary, optimizer_step



def train_sd35_vae_from_config(cfg: dict[str, Any], config_path: str | Path) -> Path:
    import torch
    from torch.utils.data import DataLoader

    seed_everything(int(cfg.get("seed", 42)), deterministic=bool(cfg.get("deterministic", False)))

    exp_name = str(cfg.get("experiment_name", "sd35_vae_train"))
    out_cfg = cfg.get("output", {}) or {}
    out_root = ensure_dir(out_cfg.get("root_dir", "outputs/checkpoints/vae"))
    out_dir = ensure_dir(out_root / exp_name)
    ensure_dir(out_dir / "last")
    ensure_dir(out_dir / "best")
    ensure_dir(out_dir / "previews")

    dump_yaml(cfg, out_dir / "config_used.yaml")

    model_cfg = cfg.get("model", {}) or {}
    vae_cfg = cfg.get("vae", {}) or {}
    train_cfg = cfg.get("train", {}) or {}
    loss_cfg = cfg.get("loss", {}) or {}
    teacher_cfg = cfg.get("teacher", {}) or {}
    noise_cfg = cfg.get("noise_conditioning", {}) or {}

    if str(vae_cfg.get("backend", model_cfg.get("family", "sd35"))).lower() not in {"sd35", "sd3", "stable_diffusion_3", "stable-diffusion-3"}:
        raise ValueError("train_sd35_vae_from_config only supports vae.backend=sd35")

    device = torch.device(
        model_cfg.get("device", "cuda") if torch.cuda.is_available() and str(model_cfg.get("device", "cuda")) != "cpu" else "cpu"
    )
    dtype_str = str(model_cfg.get("torch_dtype", vae_cfg.get("dtype", "fp32"))).lower()
    amp_dtype = None
    if device.type == "cuda":
        if dtype_str in {"fp16", "float16", "half"}:
            amp_dtype = torch.float16
        elif dtype_str in {"bf16", "bfloat16"}:
            amp_dtype = torch.bfloat16

    train_ds, val_ds, class_names = build_datasets(cfg)
    num_classes = len(class_names)
    batch_size = int(train_cfg.get("batch_size", 4))
    num_workers = int(train_cfg.get("num_workers", 4))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=bool(train_cfg.get("drop_last", True)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    vae = build_sd3_vae(model_cfg, vae_cfg, torch_dtype=torch.float32, device=device)
    apply_freeze_patterns(
        vae,
        freeze_patterns=[str(x) for x in (vae_cfg.get("freeze_patterns", []) or [])],
        unfreeze_patterns=[str(x) for x in (vae_cfg.get("unfreeze_patterns", []) or [])],
    )

    trainable = [p for p in vae.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable VAE parameters remain after applying freeze/unfreeze patterns")

    lr = float(train_cfg.get("lr", 1.0e-5))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    optimizer_name = str(train_cfg.get("optimizer", "adamw")).lower()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)

    scheduler = None
    sched_name = str(train_cfg.get("lr_scheduler", "none")).lower()
    if sched_name in {"cosine", "cosineannealinglr"}:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(train_cfg.get("epochs", 10))))
    elif sched_name in {"step", "steplr"}:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(train_cfg.get("step_size", 1)),
            gamma=float(train_cfg.get("gamma", 0.5)),
        )

    teacher = build_teacher_if_needed(cfg, num_classes=num_classes, device=device)
    mean, std = _teacher_stats(cfg)
    posterior_mode = str(vae_cfg.get("posterior", "mode"))

    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16 and device.type == "cuda"))
    if not (amp_dtype == torch.float16 and device.type == "cuda"):
        scaler = None

    wandb_session = init_wandb_session(
        cfg,
        out_dir=out_dir,
        experiment_name=exp_name,
        default_project="sd35-vae-train",
        enabled=True,
    )
    wandb_session.set_summary("num_classes", num_classes)
    wandb_session.set_summary("num_train_samples", len(train_ds))
    wandb_session.set_summary("num_val_samples", len(val_ds))
    wandb_session.set_summary("trainable_params", int(sum(p.numel() for p in trainable)))

    history_rows: list[dict[str, Any]] = []
    best_val = math.inf
    best_epoch = -1
    epochs = int(train_cfg.get("epochs", 10))
    global_step = 0
    step_log_interval = int((cfg.get("wandb", {}) or {}).get("log_interval_steps", 0))

    try:
        for epoch in range(1, epochs + 1):
            train_summary, global_step = _run_epoch(
                vae=vae,
                loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                amp_dtype=amp_dtype,
                train=True,
                teacher=teacher,
                mean=mean,
                std=std,
                loss_cfg=loss_cfg,
                teacher_cfg=teacher_cfg,
                posterior_mode=posterior_mode,
                noise_cfg=noise_cfg,
                grad_accum_steps=grad_accum_steps,
                epoch=epoch,
                save_preview_dir=(out_dir / "previews") if bool(train_cfg.get("save_previews", True)) else None,
                global_step_start=global_step,
                wandb_session=wandb_session,
                step_log_interval=step_log_interval,
            )
            with torch.no_grad():
                val_summary, _ = _run_epoch(
                    vae=vae,
                    loader=val_loader,
                    optimizer=optimizer,
                    scaler=scaler,
                    device=device,
                    amp_dtype=amp_dtype,
                    train=False,
                    teacher=teacher,
                    mean=mean,
                    std=std,
                    loss_cfg=loss_cfg,
                    teacher_cfg=teacher_cfg,
                    posterior_mode=posterior_mode,
                    noise_cfg=noise_cfg,
                    grad_accum_steps=grad_accum_steps,
                    epoch=epoch,
                    global_step_start=global_step,
                )

            history_rows.extend([train_summary.as_row(), val_summary.as_row()])
            write_csv(history_rows, out_dir / "history.csv")

            vae.save_pretrained(out_dir / "last" / "vae")
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "history": history_rows,
                },
                out_dir / "last" / "train_state.pt",
            )

            preview_path = out_dir / "previews" / f"epoch_{epoch:03d}.png"
            epoch_payload = {
                "epoch": epoch,
                "global_step": global_step,
                "train/loss_epoch": train_summary.total_loss,
                "train/recon_epoch": train_summary.recon_loss,
                "train/kl_epoch": train_summary.kl_loss,
                "train/feature_epoch": train_summary.feature_loss,
                "train/logit_epoch": train_summary.logit_loss,
                "train/noise_feature_epoch": train_summary.noise_feature_loss,
                "val/loss": val_summary.total_loss,
                "val/recon": val_summary.recon_loss,
                "val/kl": val_summary.kl_loss,
                "val/feature": val_summary.feature_loss,
                "val/logit": val_summary.logit_loss,
                "val/noise_feature": val_summary.noise_feature_loss,
                "train/lr": float(optimizer.param_groups[0]["lr"]),
            }
            if preview_path.is_file():
                wb_img = maybe_build_wandb_image(wandb_session, str(preview_path), caption=f"epoch {epoch}")
                if wb_img is not None:
                    epoch_payload["preview/recon"] = wb_img
            wandb_session.log(epoch_payload, step=global_step)

            if val_summary.total_loss < best_val:
                best_val = val_summary.total_loss
                best_epoch = epoch
                vae.save_pretrained(out_dir / "best" / "vae")
                write_json(
                    {
                        "best_epoch": best_epoch,
                        "best_val_total_loss": best_val,
                        "train_summary": train_summary.as_row(),
                        "val_summary": val_summary.as_row(),
                    },
                    out_dir / "best" / "metrics.json",
                )
                wandb_session.set_summary("best_epoch", best_epoch)
                wandb_session.set_summary("best_val_total_loss", best_val)

            if scheduler is not None:
                scheduler.step()

        summary = {
            "experiment_name": exp_name,
            "num_classes": num_classes,
            "num_train_samples": len(train_ds),
            "num_val_samples": len(val_ds),
            "best_epoch": best_epoch,
            "best_val_total_loss": best_val,
            "trainable_params": int(sum(p.numel() for p in trainable)),
            "total_params": int(sum(p.numel() for p in vae.parameters())),
            "global_step": global_step,
        }
        write_json(summary, out_dir / "summary.json")
        for key, value in summary.items():
            wandb_session.set_summary(key, value)
        return out_dir
    finally:
        wandb_session.finish()

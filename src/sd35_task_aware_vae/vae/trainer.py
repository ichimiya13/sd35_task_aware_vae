from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

from src.sd35_task_aware_vae.datasets.image_dataset import MultiLabelMedicalDataset
from src.sd35_task_aware_vae.labels.schema import load_label_schema
from src.sd35_task_aware_vae.sd3.latent_codec import (
    apply_latent_stats_to_vae_config,
    decode_from_latents,
    encode_to_latents,
    estimate_latent_moments_from_loader,
)
from src.sd35_task_aware_vae.sd3.vae_factory import apply_freeze_patterns, build_sd3_vae
from src.sd35_task_aware_vae.teacher_classifier import build_convnext_large
from src.sd35_task_aware_vae.utils.config import dump_yaml
from src.sd35_task_aware_vae.utils.files import ensure_dir, write_csv, write_json
from src.sd35_task_aware_vae.utils.seed import seed_everything
from src.sd35_task_aware_vae.utils.wandb import init_wandb_session, maybe_build_wandb_image
from src.sd35_task_aware_vae.vae.losses import (
    LPIPSLoss,
    build_spatial_weight_map,
    feature_distance,
    gradient_loss,
    patch_reconstruction_loss,
    posterior_kl_loss,
    reconstruction_loss,
    weighted_reconstruction_loss,
)


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
    edge_loss: float
    weighted_recon_loss: float
    patch_recon_loss: float
    feature_loss: float
    logit_loss: float
    lpips_loss: float
    noise_feature_loss: float
    recon_term: float
    kl_term: float
    edge_term: float
    weighted_recon_term: float
    patch_recon_term: float
    feature_term: float
    logit_term: float
    lpips_term: float
    noise_feature_term: float

    def as_row(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "split": self.split,
            "total_loss": self.total_loss,
            "recon_loss": self.recon_loss,
            "kl_loss": self.kl_loss,
            "edge_loss": self.edge_loss,
            "weighted_recon_loss": self.weighted_recon_loss,
            "patch_recon_loss": self.patch_recon_loss,
            "feature_loss": self.feature_loss,
            "logit_loss": self.logit_loss,
            "lpips_loss": self.lpips_loss,
            "noise_feature_loss": self.noise_feature_loss,
            "recon_term": self.recon_term,
            "kl_term": self.kl_term,
            "edge_term": self.edge_term,
            "weighted_recon_term": self.weighted_recon_term,
            "patch_recon_term": self.patch_recon_term,
            "feature_term": self.feature_term,
            "logit_term": self.logit_term,
            "lpips_term": self.lpips_term,
            "noise_feature_term": self.noise_feature_term,
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



def _get_nested(cfg: dict[str, Any], *keys, default=None):
    obj: Any = cfg
    for key in keys:
        if not isinstance(obj, dict) or key not in obj:
            return default
        obj = obj[key]
    return obj



def _section(loss_cfg: dict[str, Any], name: str, *, legacy_prefix: str | None = None, default_weight: float = 0.0) -> dict[str, Any]:
    section = dict(loss_cfg.get(name, {}) or {})

    def _legacy_get(key: str, fallback=None):
        if legacy_prefix is None:
            return fallback
        return loss_cfg.get(f"{legacy_prefix}_{key}", fallback)

    if "weight" not in section:
        section["weight"] = _legacy_get("weight", default_weight)
    if "type" not in section:
        legacy_type = _legacy_get("type", None)
        if legacy_type is None and legacy_prefix not in {None, "kl"}:
            legacy_type = _legacy_get("loss_type", None)
        if legacy_type is not None:
            section["type"] = legacy_type
    return section



def _resolve_loss_config(loss_cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cfg = dict(loss_cfg or {})
    recon = _section(cfg, "recon", legacy_prefix="recon", default_weight=float(cfg.get("recon_weight", 1.0)))
    if "type" not in recon:
        recon["type"] = cfg.get("recon_type", "l1")
    recon.setdefault("epsilon", float(cfg.get("recon_epsilon", 1.0e-3)))

    kl = _section(cfg, "kl", legacy_prefix="kl", default_weight=float(cfg.get("kl_weight", 1.0e-6)))

    edge = _section(cfg, "edge", legacy_prefix="edge", default_weight=float(cfg.get("edge_weight", 0.0)))
    edge.setdefault("type", cfg.get("edge_type", "sobel_l1"))
    edge.setdefault("use_weight_map", bool(cfg.get("edge_use_weight_map", False)))

    weighted_recon = _section(
        cfg,
        "weighted_recon",
        legacy_prefix="weighted_recon",
        default_weight=float(cfg.get("weighted_recon_weight", 0.0)),
    )
    weighted_recon.setdefault("type", cfg.get("weighted_recon_type", recon.get("type", "l1")))
    weighted_recon.setdefault("epsilon", float(cfg.get("weighted_recon_epsilon", recon.get("epsilon", 1.0e-3))))

    patch_recon = _section(
        cfg,
        "patch_recon",
        legacy_prefix="patch_recon",
        default_weight=float(cfg.get("patch_recon_weight", 0.0)),
    )
    patch_recon.setdefault("type", cfg.get("patch_recon_type", recon.get("type", "l1")))
    patch_recon.setdefault("epsilon", float(cfg.get("patch_recon_epsilon", recon.get("epsilon", 1.0e-3))))
    patch_recon.setdefault("crop_ratio", cfg.get("patch_recon_crop_ratio", 0.5))
    patch_recon.setdefault("crop_size", cfg.get("patch_recon_crop_size", None))
    patch_recon.setdefault("center_x", float(cfg.get("patch_recon_center_x", 0.5)))
    patch_recon.setdefault("center_y", float(cfg.get("patch_recon_center_y", 0.5)))

    feature = _section(cfg, "feature", legacy_prefix="feature", default_weight=float(cfg.get("feature_weight", 0.0)))
    feature.setdefault("type", cfg.get("feature_loss_type", "mse"))

    logit = _section(cfg, "logit", legacy_prefix="logit", default_weight=float(cfg.get("logit_weight", 0.0)))
    logit.setdefault("type", cfg.get("logit_loss_type", "mse"))

    lpips = _section(cfg, "lpips", legacy_prefix="lpips", default_weight=float(cfg.get("lpips_weight", 0.0)))
    lpips.setdefault("net", cfg.get("lpips_net", "alex"))
    lpips.setdefault("spatial", bool(cfg.get("lpips_spatial", False)))

    noise_feature = _section(
        cfg,
        "noise_feature",
        legacy_prefix="noise_feature",
        default_weight=float(cfg.get("noise_feature_weight", 0.0)),
    )
    noise_feature.setdefault("type", cfg.get("noise_feature_loss_type", feature.get("type", "mse")))

    weight_map = dict(cfg.get("weight_map", {}) or {})
    weight_map.setdefault("mode", str(cfg.get("weight_map_mode", "none")))
    weight_map.setdefault("min_weight", float(cfg.get("weight_map_min_weight", 1.0)))
    weight_map.setdefault("max_weight", float(cfg.get("weight_map_max_weight", 1.0)))
    weight_map.setdefault("gamma", float(cfg.get("weight_map_gamma", 1.0)))
    weight_map.setdefault("inner_radius", float(cfg.get("weight_map_inner_radius", 0.0)))
    weight_map.setdefault("retina_threshold", float(cfg.get("weight_map_retina_threshold", 0.03)))
    weight_map.setdefault("apply_retina_mask", bool(cfg.get("weight_map_apply_retina_mask", True)))
    weight_map.setdefault("center_x", float(cfg.get("weight_map_center_x", 0.5)))
    weight_map.setdefault("center_y", float(cfg.get("weight_map_center_y", 0.5)))
    weight_map.setdefault("sigma", float(cfg.get("weight_map_sigma", 0.22)))
    weight_map.setdefault("sigma_x", cfg.get("weight_map_sigma_x", None))
    weight_map.setdefault("sigma_y", cfg.get("weight_map_sigma_y", None))

    return {
        "recon": recon,
        "kl": kl,
        "edge": edge,
        "weighted_recon": weighted_recon,
        "patch_recon": patch_recon,
        "feature": feature,
        "logit": logit,
        "lpips": lpips,
        "noise_feature": noise_feature,
        "weight_map": weight_map,
    }



def _get_weight(loss_cfg: dict[str, Any], name: str) -> float:
    resolved = _resolve_loss_config(loss_cfg)
    return float(resolved[name].get("weight", 0.0))



def build_teacher_if_needed(cfg: dict[str, Any], num_classes: int, device):
    import torch

    loss_cfg = cfg.get("loss", {}) or {}
    need_teacher = any(
        _get_weight(loss_cfg, key) > 0 for key in ["feature", "logit", "noise_feature"]
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




def _resolve_posterior_modes(vae_cfg: dict[str, Any]) -> tuple[str, str]:
    posterior_cfg = vae_cfg.get("posterior", None)
    if isinstance(posterior_cfg, dict):
        train_mode = str(posterior_cfg.get("train", posterior_cfg.get("mode", "mode")))
        eval_mode = str(posterior_cfg.get("eval", posterior_cfg.get("mode", train_mode)))
        return train_mode, eval_mode

    shared = str(vae_cfg.get("posterior", "mode"))
    train_mode = str(vae_cfg.get("posterior_train", shared))
    eval_mode = str(vae_cfg.get("posterior_eval", shared))
    return train_mode, eval_mode



def _build_lpips_if_needed(loss_cfg: dict[str, Any], device):
    lpips_cfg = _resolve_loss_config(loss_cfg)["lpips"]
    if float(lpips_cfg.get("weight", 0.0)) <= 0:
        return None
    module = LPIPSLoss(net=str(lpips_cfg.get("net", "alex")), spatial=bool(lpips_cfg.get("spatial", False)))
    module.to(device)
    return module



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
    lpips_module=None,
):
    import torch

    resolved = _resolve_loss_config(loss_cfg)
    feature_cfg = resolved["feature"]
    logit_cfg = resolved["logit"]
    recon_cfg = resolved["recon"]
    edge_cfg = resolved["edge"]
    weighted_recon_cfg = resolved["weighted_recon"]
    patch_recon_cfg = resolved["patch_recon"]
    lpips_cfg = resolved["lpips"]
    noise_feature_cfg = resolved["noise_feature"]
    weight_map_cfg = resolved["weight_map"]
    feature_stage = teacher_cfg.get("feature_stage", "embedding")

    latents, posterior = encode_to_latents(
        vae,
        batch,
        sample_mode=posterior_mode,
        generator=generator,
        return_posterior=True,
    )
    recon = decode_from_latents(vae, latents)

    weight_map = None
    if float(weighted_recon_cfg.get("weight", 0.0)) > 0 or (
        float(edge_cfg.get("weight", 0.0)) > 0 and bool(edge_cfg.get("use_weight_map", False))
    ):
        weight_map = build_spatial_weight_map(batch, weight_map_cfg)

    recon_loss_value = reconstruction_loss(
        recon,
        batch,
        kind=str(recon_cfg.get("type", "l1")),
        reduction="mean",
        epsilon=float(recon_cfg.get("epsilon", 1.0e-3)),
    )
    kl_loss_value = posterior_kl_loss(posterior)
    edge_loss_value = batch.new_tensor(0.0)
    weighted_recon_loss_value = batch.new_tensor(0.0)
    patch_recon_loss_value = batch.new_tensor(0.0)
    feature_loss_value = batch.new_tensor(0.0)
    logit_loss_value = batch.new_tensor(0.0)
    lpips_loss_value = batch.new_tensor(0.0)
    noise_feature_loss_value = batch.new_tensor(0.0)

    if float(edge_cfg.get("weight", 0.0)) > 0:
        edge_loss_value = gradient_loss(
            recon,
            batch,
            kind=str(edge_cfg.get("type", "sobel_l1")),
            weight_map=(weight_map if bool(edge_cfg.get("use_weight_map", False)) else None),
        )

    if float(weighted_recon_cfg.get("weight", 0.0)) > 0:
        if weight_map is None:
            raise ValueError("weighted_recon.weight > 0 requires loss.weight_map.mode != 'none'")
        weighted_recon_loss_value = weighted_reconstruction_loss(
            recon,
            batch,
            weight_map=weight_map,
            kind=str(weighted_recon_cfg.get("type", recon_cfg.get("type", "l1"))),
            epsilon=float(weighted_recon_cfg.get("epsilon", recon_cfg.get("epsilon", 1.0e-3))),
        )

    if float(patch_recon_cfg.get("weight", 0.0)) > 0:
        patch_recon_loss_value = patch_reconstruction_loss(
            recon,
            batch,
            kind=str(patch_recon_cfg.get("type", recon_cfg.get("type", "l1"))),
            epsilon=float(patch_recon_cfg.get("epsilon", recon_cfg.get("epsilon", 1.0e-3))),
            crop_size=(int(patch_recon_cfg["crop_size"]) if patch_recon_cfg.get("crop_size", None) not in {None, "", 0, "0"} else None),
            crop_ratio=(float(patch_recon_cfg["crop_ratio"]) if patch_recon_cfg.get("crop_ratio", None) not in {None, ""} else None),
            center_x=float(patch_recon_cfg.get("center_x", 0.5)),
            center_y=float(patch_recon_cfg.get("center_y", 0.5)),
        )

    if float(lpips_cfg.get("weight", 0.0)) > 0:
        if lpips_module is None:
            raise RuntimeError("LPIPS loss is enabled but lpips_module was not initialized")
        lpips_loss_value = lpips_module(recon, batch)

    if teacher is not None:
        with torch.no_grad():
            target_teacher = normalize_for_teacher(batch, mean, std)
            target_views = _extract_teacher_views(teacher, target_teacher, feature_stage=feature_stage)

        recon_teacher = normalize_for_teacher(recon, mean, std)
        recon_views = _extract_teacher_views(teacher, recon_teacher, feature_stage=feature_stage)

        if float(feature_cfg.get("weight", 0.0)) > 0 and target_views.features is not None and recon_views.features is not None:
            feature_loss_value = feature_distance(recon_views.features, target_views.features, kind=str(feature_cfg.get("type", "mse")))

        if float(logit_cfg.get("weight", 0.0)) > 0 and target_views.logits is not None and recon_views.logits is not None:
            logit_loss_value = feature_distance(recon_views.logits, target_views.logits, kind=str(logit_cfg.get("type", "mse")))

        if float(noise_feature_cfg.get("weight", 0.0)) > 0:
            noisy_latents = _sample_noisy_latents(latents, noise_cfg, generator=generator)
            recon_noise = decode_from_latents(vae, noisy_latents)
            recon_noise_teacher = normalize_for_teacher(recon_noise, mean, std)
            recon_noise_views = _extract_teacher_views(teacher, recon_noise_teacher, feature_stage=feature_stage)
            if target_views.features is not None and recon_noise_views.features is not None:
                noise_feature_loss_value = feature_distance(
                    recon_noise_views.features,
                    target_views.features,
                    kind=str(noise_feature_cfg.get("type", feature_cfg.get("type", "mse"))),
                )

    weights = {
        "recon": float(recon_cfg.get("weight", 1.0)),
        "kl": float(resolved["kl"].get("weight", 1.0e-6)),
        "edge": float(edge_cfg.get("weight", 0.0)),
        "weighted_recon": float(weighted_recon_cfg.get("weight", 0.0)),
        "patch_recon": float(patch_recon_cfg.get("weight", 0.0)),
        "feature": float(feature_cfg.get("weight", 0.0)),
        "logit": float(logit_cfg.get("weight", 0.0)),
        "lpips": float(lpips_cfg.get("weight", 0.0)),
        "noise_feature": float(noise_feature_cfg.get("weight", 0.0)),
    }
    total = (
        weights["recon"] * recon_loss_value
        + weights["kl"] * kl_loss_value
        + weights["edge"] * edge_loss_value
        + weights["weighted_recon"] * weighted_recon_loss_value
        + weights["patch_recon"] * patch_recon_loss_value
        + weights["feature"] * feature_loss_value
        + weights["logit"] * logit_loss_value
        + weights["lpips"] * lpips_loss_value
        + weights["noise_feature"] * noise_feature_loss_value
    )
    weighted_terms = {
        "recon": weights["recon"] * recon_loss_value,
        "kl": weights["kl"] * kl_loss_value,
        "edge": weights["edge"] * edge_loss_value,
        "weighted_recon": weights["weighted_recon"] * weighted_recon_loss_value,
        "patch_recon": weights["patch_recon"] * patch_recon_loss_value,
        "feature": weights["feature"] * feature_loss_value,
        "logit": weights["logit"] * logit_loss_value,
        "lpips": weights["lpips"] * lpips_loss_value,
        "noise_feature": weights["noise_feature"] * noise_feature_loss_value,
    }
    return {
        "total": total,
        "recon": recon_loss_value,
        "kl": kl_loss_value,
        "edge": edge_loss_value,
        "weighted_recon": weighted_recon_loss_value,
        "patch_recon": patch_recon_loss_value,
        "feature": feature_loss_value,
        "logit": logit_loss_value,
        "lpips": lpips_loss_value,
        "noise_feature": noise_feature_loss_value,
        "weighted_terms": weighted_terms,
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
    loss_cfg: dict[str, Any],
    teacher_cfg: dict[str, Any],
    posterior_mode: str,
    noise_cfg: dict[str, Any],
    grad_accum_steps: int,
    epoch: int,
    save_preview_dir: Path | None = None,
    global_step_start: int = 0,
    wandb_session=None,
    step_log_interval: int = 0,
    use_progress_bar: bool = True,
    progress_desc: str | None = None,
    progress_update_interval: int = 10,
    progress_mininterval: float = 0.5,
    progress_leave: bool = False,
    lpips_module=None,
):
    import torch
    from torchvision.utils import save_image

    autocast_enabled = amp_dtype is not None and device.type == "cuda"
    vae.train(train)
    if train:
        optimizer.zero_grad(set_to_none=True)

    total = recon = kl = edge = weighted_recon = patch_recon = feat = logit = lpips_val = noise_feat = 0.0
    recon_term = kl_term = edge_term = weighted_recon_term = patch_recon_term = feat_term = logit_term = lpips_term = noise_feat_term = 0.0
    num_batches = 0
    num_items = 0.0
    optimizer_step = int(global_step_start)
    preview_saved = False

    total_batches = len(loader)
    progress_update_interval = max(1, int(progress_update_interval))
    progress = None
    iterator = loader
    if tqdm is not None and use_progress_bar:
        progress = tqdm(
            loader,
            total=total_batches,
            desc=progress_desc or ("train" if train else "val"),
            dynamic_ncols=True,
            leave=progress_leave,
            mininterval=progress_mininterval,
        )
        iterator = progress

    def _maybe_log_step(terms: dict[str, Any], step_idx: int) -> None:
        if not train or wandb_session is None or not getattr(wandb_session, "enabled", False):
            return
        if step_log_interval <= 0 or step_idx <= global_step_start or (step_idx % max(1, step_log_interval)) != 0:
            return
        payload = {
            "train/global_step": step_idx,
            "train/epoch": epoch,
            "train/loss_step": float(terms["total"].detach().cpu()),
            "train/loss_running": float(total / max(1.0, num_items)),
            "train/lr": float(optimizer.param_groups[0]["lr"]),
            "train/recon_step": float(terms["recon"].detach().cpu()),
            "train/kl_step": float(terms["kl"].detach().cpu()),
            "train/edge_step": float(terms["edge"].detach().cpu()),
            "train/weighted_recon_step": float(terms["weighted_recon"].detach().cpu()),
            "train/patch_recon_step": float(terms["patch_recon"].detach().cpu()),
            "train/feature_step": float(terms["feature"].detach().cpu()),
            "train/logit_step": float(terms["logit"].detach().cpu()),
            "train/lpips_step": float(terms["lpips"].detach().cpu()),
            "train/noise_feature_step": float(terms["noise_feature"].detach().cpu()),
            "train/recon_term_step": float(terms["weighted_terms"]["recon"].detach().cpu()),
            "train/kl_term_step": float(terms["weighted_terms"]["kl"].detach().cpu()),
            "train/edge_term_step": float(terms["weighted_terms"]["edge"].detach().cpu()),
            "train/weighted_recon_term_step": float(terms["weighted_terms"]["weighted_recon"].detach().cpu()),
            "train/patch_recon_term_step": float(terms["weighted_terms"]["patch_recon"].detach().cpu()),
            "train/feature_term_step": float(terms["weighted_terms"]["feature"].detach().cpu()),
            "train/logit_term_step": float(terms["weighted_terms"]["logit"].detach().cpu()),
            "train/lpips_term_step": float(terms["weighted_terms"]["lpips"].detach().cpu()),
            "train/noise_feature_term_step": float(terms["weighted_terms"]["noise_feature"].detach().cpu()),
        }
        try:
            if scaler is not None:
                payload["train/grad_scale"] = float(scaler.get_scale())
        except Exception:
            pass
        wandb_session.log(payload, step=step_idx)

    for step, (images, _labels, _paths) in enumerate(iterator, start=1):
        images = images.to(device=device, dtype=torch.float32, non_blocking=(device.type == "cuda"))
        batch_items = float(images.shape[0])
        should_step = train and ((step % max(1, grad_accum_steps) == 0) or (step == total_batches))

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
                    lpips_module=lpips_module,
                )
                loss = terms["total"] / max(1, grad_accum_steps)

            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if should_step:
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_step += 1

        total += float(terms["total"].detach().cpu()) * batch_items
        recon += float(terms["recon"].detach().cpu()) * batch_items
        kl += float(terms["kl"].detach().cpu()) * batch_items
        edge += float(terms["edge"].detach().cpu()) * batch_items
        weighted_recon += float(terms["weighted_recon"].detach().cpu()) * batch_items
        patch_recon += float(terms["patch_recon"].detach().cpu()) * batch_items
        feat += float(terms["feature"].detach().cpu()) * batch_items
        logit += float(terms["logit"].detach().cpu()) * batch_items
        lpips_val += float(terms["lpips"].detach().cpu()) * batch_items
        noise_feat += float(terms["noise_feature"].detach().cpu()) * batch_items
        recon_term += float(terms["weighted_terms"]["recon"].detach().cpu()) * batch_items
        kl_term += float(terms["weighted_terms"]["kl"].detach().cpu()) * batch_items
        edge_term += float(terms["weighted_terms"]["edge"].detach().cpu()) * batch_items
        weighted_recon_term += float(terms["weighted_terms"]["weighted_recon"].detach().cpu()) * batch_items
        patch_recon_term += float(terms["weighted_terms"]["patch_recon"].detach().cpu()) * batch_items
        feat_term += float(terms["weighted_terms"]["feature"].detach().cpu()) * batch_items
        logit_term += float(terms["weighted_terms"]["logit"].detach().cpu()) * batch_items
        lpips_term += float(terms["weighted_terms"]["lpips"].detach().cpu()) * batch_items
        noise_feat_term += float(terms["weighted_terms"]["noise_feature"].detach().cpu()) * batch_items
        num_batches += 1
        num_items += batch_items

        if train and should_step:
            _maybe_log_step(terms, optimizer_step)

        if progress is not None and (step % progress_update_interval == 0 or step == total_batches):
            denom = float(max(1.0, num_items))
            postfix = {
                "loss": f"{(total / denom):.4f}",
                "recon": f"{(recon / denom):.4f}",
                "recon*": f"{(recon_term / denom):.4f}",
            }
            if train:
                postfix["lr"] = f"{optimizer.param_groups[0]['lr']:.2e}"
            progress.set_postfix(postfix)

        if save_preview_dir is not None and not preview_saved:
            preview_saved = True
            save_preview_dir.mkdir(parents=True, exist_ok=True)
            orig = (images[:4].detach().cpu().clamp(-1.0, 1.0) + 1.0) / 2.0
            rec = (terms["recon_image"][:4].detach().cpu().clamp(-1.0, 1.0) + 1.0) / 2.0
            grid = torch.cat([orig, rec], dim=3)
            save_image(grid, save_preview_dir / f"epoch_{epoch:03d}.png")

    if progress is not None:
        progress.close()

    denom = float(max(1.0, num_items))
    summary = EpochSummary(
        epoch=epoch,
        split="train" if train else "val",
        total_loss=total / denom,
        recon_loss=recon / denom,
        kl_loss=kl / denom,
        edge_loss=edge / denom,
        weighted_recon_loss=weighted_recon / denom,
        patch_recon_loss=patch_recon / denom,
        feature_loss=feat / denom,
        logit_loss=logit / denom,
        lpips_loss=lpips_val / denom,
        noise_feature_loss=noise_feat / denom,
        recon_term=recon_term / denom,
        kl_term=kl_term / denom,
        edge_term=edge_term / denom,
        weighted_recon_term=weighted_recon_term / denom,
        patch_recon_term=patch_recon_term / denom,
        feature_term=feat_term / denom,
        logit_term=logit_term / denom,
        lpips_term=lpips_term / denom,
        noise_feature_term=noise_feat_term / denom,
    )
    return summary, optimizer_step



def _save_vae_checkpoint(vae, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    vae.save_pretrained(path)



def _estimate_and_save_latent_stats(
    *,
    cfg: dict[str, Any],
    source_dir: Path,
    out_dir: Path,
    split_name: str,
    loader,
    model_cfg: dict[str, Any],
    vae_cfg: dict[str, Any],
    device,
):
    import torch

    latent_cfg = cfg.get("latent_stats", {}) or {}
    if not bool(latent_cfg.get("enabled", False)):
        return None

    sample_mode = str(latent_cfg.get("posterior", latent_cfg.get("sample_mode", vae_cfg.get("posterior_eval", vae_cfg.get("posterior", "mode")))))
    max_batches = latent_cfg.get("max_batches", None)
    max_batches = None if max_batches in {None, "", 0, "0"} else int(max_batches)

    local_vae_cfg = dict(vae_cfg)
    local_vae_cfg["checkpoint"] = str(source_dir)
    vae = build_sd3_vae(model_cfg, local_vae_cfg, torch_dtype=torch.float32, device=device)
    stats = estimate_latent_moments_from_loader(
        vae,
        loader,
        device=device,
        sample_mode=sample_mode,
        max_batches=max_batches,
    )

    apply_update = bool(latent_cfg.get("update_config", True))
    overwrite_source = bool(latent_cfg.get("overwrite_saved_dir", False))
    if apply_update:
        apply_latent_stats_to_vae_config(
            vae,
            shift=float(stats["recommended_shift_factor"]),
            scaling=float(stats["recommended_scaling_factor"]),
        )

    stats_path = out_dir / f"latent_stats_{split_name}.json"
    write_json(stats, stats_path)

    target_dir = source_dir if overwrite_source else (out_dir / f"{source_dir.name}_latent_calibrated")
    target_dir.mkdir(parents=True, exist_ok=True)
    if apply_update:
        vae.save_pretrained(target_dir)
    write_json(stats, target_dir / "latent_stats.json")
    return {
        "stats": stats,
        "stats_path": str(stats_path),
        "vae_dir": str(target_dir),
        "source": str(source_dir),
    }



def train_sd35_vae_from_config(cfg: dict[str, Any], config_path: str | Path) -> Path:
    import torch
    from torch.utils.data import DataLoader
    from src.sd35_task_aware_vae.utils.device import describe_visible_gpus

    seed_everything(int(cfg.get("seed", 42)), deterministic=bool(cfg.get("deterministic", False)))

    exp_name = str(cfg.get("experiment_name", "sd35_vae_train"))
    out_cfg = cfg.get("output", {}) or {}
    out_root = ensure_dir(out_cfg.get("root_dir", "outputs/checkpoints/vae"))
    out_dir = ensure_dir(out_root / exp_name)
    ensure_dir(out_dir / "last")
    ensure_dir(out_dir / "best")
    ensure_dir(out_dir / "previews")
    ensure_dir(out_dir / "epoch_summaries")

    dump_yaml(cfg, out_dir / "config_used.yaml")

    model_cfg = cfg.get("model", {}) or {}
    vae_cfg = cfg.get("vae", {}) or {}
    train_cfg = cfg.get("train", {}) or {}
    loss_cfg = cfg.get("loss", {}) or {}
    teacher_cfg = cfg.get("teacher", {}) or {}
    noise_cfg = cfg.get("noise_conditioning", {}) or {}

    if str(vae_cfg.get("backend", model_cfg.get("family", "sd35"))).lower() not in {"sd35", "sd3", "stable_diffusion_3", "stable-diffusion-3"}:
        raise ValueError("train_sd35_vae_from_config only supports vae.backend=sd35")

    requested_device = str(model_cfg.get("device", "cuda")).lower()
    cuda_requested = requested_device != "cpu"
    cuda_available = torch.cuda.is_available()
    if cuda_requested and not cuda_available:
        visible = describe_visible_gpus()
        raise RuntimeError(
            "CUDA device was requested but is not available. "
            f"Check runtime.gpu_ids / CUDA_VISIBLE_DEVICES (current='{visible}')."
        )

    device = torch.device(requested_device if cuda_available and cuda_requested else "cpu")
    if bool(model_cfg.get("allow_tf32", False)) and device.type == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
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
    posterior_train_mode, posterior_eval_mode = _resolve_posterior_modes(vae_cfg)

    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16 and device.type == "cuda"))
    if not (amp_dtype == torch.float16 and device.type == "cuda"):
        scaler = None

    lpips_module = _build_lpips_if_needed(loss_cfg, device)

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
    wandb_session.set_summary("posterior_train_mode", posterior_train_mode)
    wandb_session.set_summary("posterior_eval_mode", posterior_eval_mode)

    history_rows: list[dict[str, Any]] = []
    best_val = math.inf
    best_epoch = -1
    epochs = int(train_cfg.get("epochs", 10))
    global_step = 0
    step_log_interval = int((cfg.get("wandb", {}) or {}).get("log_interval_steps", 0))
    use_pbar = bool(train_cfg.get("progress_bar", True)) and (tqdm is not None)
    pbar_mininterval = float(train_cfg.get("tqdm_mininterval", 0.5))
    pbar_update_interval = int(train_cfg.get("tqdm_update_interval", 10))
    pbar_leave = bool(train_cfg.get("tqdm_leave", False))

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
                posterior_mode=posterior_train_mode,
                noise_cfg=noise_cfg,
                grad_accum_steps=grad_accum_steps,
                epoch=epoch,
                save_preview_dir=(out_dir / "previews") if bool(train_cfg.get("save_previews", True)) else None,
                global_step_start=global_step,
                wandb_session=wandb_session,
                step_log_interval=step_log_interval,
                use_progress_bar=use_pbar,
                progress_desc=f"Train {epoch}/{epochs}",
                progress_update_interval=pbar_update_interval,
                progress_mininterval=pbar_mininterval,
                progress_leave=pbar_leave,
                lpips_module=lpips_module,
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
                    posterior_mode=posterior_eval_mode,
                    noise_cfg=noise_cfg,
                    grad_accum_steps=grad_accum_steps,
                    epoch=epoch,
                    global_step_start=global_step,
                    use_progress_bar=use_pbar,
                    progress_desc=f"Val   {epoch}/{epochs}",
                    progress_update_interval=pbar_update_interval,
                    progress_mininterval=pbar_mininterval,
                    progress_leave=pbar_leave,
                    lpips_module=lpips_module,
                )

            history_rows.extend([train_summary.as_row(), val_summary.as_row()])
            write_csv(history_rows, out_dir / "history.csv")
            write_json(
                {"train": train_summary.as_row(), "val": val_summary.as_row()},
                out_dir / "epoch_summaries" / f"epoch_{epoch:03d}.json",
            )

            _save_vae_checkpoint(vae, out_dir / "last" / "vae")
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
                "train/edge_epoch": train_summary.edge_loss,
                "train/weighted_recon_epoch": train_summary.weighted_recon_loss,
                "train/patch_recon_epoch": train_summary.patch_recon_loss,
                "train/feature_epoch": train_summary.feature_loss,
                "train/logit_epoch": train_summary.logit_loss,
                "train/lpips_epoch": train_summary.lpips_loss,
                "train/noise_feature_epoch": train_summary.noise_feature_loss,
                "train/recon_term_epoch": train_summary.recon_term,
                "train/kl_term_epoch": train_summary.kl_term,
                "train/edge_term_epoch": train_summary.edge_term,
                "train/weighted_recon_term_epoch": train_summary.weighted_recon_term,
                "train/patch_recon_term_epoch": train_summary.patch_recon_term,
                "train/feature_term_epoch": train_summary.feature_term,
                "train/logit_term_epoch": train_summary.logit_term,
                "train/lpips_term_epoch": train_summary.lpips_term,
                "train/noise_feature_term_epoch": train_summary.noise_feature_term,
                "val/loss": val_summary.total_loss,
                "val/recon": val_summary.recon_loss,
                "val/kl": val_summary.kl_loss,
                "val/edge": val_summary.edge_loss,
                "val/weighted_recon": val_summary.weighted_recon_loss,
                "val/patch_recon": val_summary.patch_recon_loss,
                "val/feature": val_summary.feature_loss,
                "val/logit": val_summary.logit_loss,
                "val/lpips": val_summary.lpips_loss,
                "val/noise_feature": val_summary.noise_feature_loss,
                "val/recon_term": val_summary.recon_term,
                "val/kl_term": val_summary.kl_term,
                "val/edge_term": val_summary.edge_term,
                "val/weighted_recon_term": val_summary.weighted_recon_term,
                "val/patch_recon_term": val_summary.patch_recon_term,
                "val/feature_term": val_summary.feature_term,
                "val/logit_term": val_summary.logit_term,
                "val/lpips_term": val_summary.lpips_term,
                "val/noise_feature_term": val_summary.noise_feature_term,
                "train/lr": float(optimizer.param_groups[0]["lr"]),
            }
            if preview_path.is_file():
                wb_img = maybe_build_wandb_image(wandb_session, str(preview_path), caption=f"epoch {epoch}")
                if wb_img is not None:
                    epoch_payload["preview/recon"] = wb_img
            wandb_session.log(epoch_payload, step=global_step)
            print(
                f"[epoch {epoch}/{epochs}] "
                f"train: total={train_summary.total_loss:.6f}, recon={train_summary.recon_loss:.6f}, "
                f"recon_term={train_summary.recon_term:.6f}, kl={train_summary.kl_loss:.6f}, kl_term={train_summary.kl_term:.6f}, "
                f"patch={train_summary.patch_recon_loss:.6f}, wpatch={train_summary.patch_recon_term:.6f} | "
                f"val: total={val_summary.total_loss:.6f}, recon={val_summary.recon_loss:.6f}, "
                f"recon_term={val_summary.recon_term:.6f}, kl={val_summary.kl_loss:.6f}, kl_term={val_summary.kl_term:.6f}, "
                f"patch={val_summary.patch_recon_loss:.6f}, wpatch={val_summary.patch_recon_term:.6f}"
            )

            if val_summary.total_loss < best_val:
                best_val = val_summary.total_loss
                best_epoch = epoch
                _save_vae_checkpoint(vae, out_dir / "best" / "vae")
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
            "posterior_train_mode": posterior_train_mode,
            "posterior_eval_mode": posterior_eval_mode,
        }

        latent_results = {}
        latent_cfg = cfg.get("latent_stats", {}) or {}
        if bool(latent_cfg.get("enabled", False)):
            split = str(latent_cfg.get("split", "train")).lower()
            loader = train_loader if split == "train" else val_loader
            if (out_dir / "best" / "vae").is_dir():
                latent_results["best"] = _estimate_and_save_latent_stats(
                    cfg=cfg,
                    source_dir=out_dir / "best" / "vae",
                    out_dir=out_dir / "best",
                    split_name=split,
                    loader=loader,
                    model_cfg=model_cfg,
                    vae_cfg=vae_cfg,
                    device=device,
                )
            if bool(latent_cfg.get("also_calibrate_last", True)) and (out_dir / "last" / "vae").is_dir():
                latent_results["last"] = _estimate_and_save_latent_stats(
                    cfg=cfg,
                    source_dir=out_dir / "last" / "vae",
                    out_dir=out_dir / "last",
                    split_name=split,
                    loader=loader,
                    model_cfg=model_cfg,
                    vae_cfg=vae_cfg,
                    device=device,
                )
            summary["latent_stats"] = latent_results

        write_json(summary, out_dir / "summary.json")
        for key, value in summary.items():
            wandb_session.set_summary(key, value)
        return out_dir
    finally:
        wandb_session.finish()

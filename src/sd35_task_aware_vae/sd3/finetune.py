from __future__ import annotations

import copy
import math
import os
import re
import shutil
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

from src.sd35_task_aware_vae.sd3.prompts import resolve_prompts
from src.sd35_task_aware_vae.sd3.runtime import resolve_torch_dtype
from src.sd35_task_aware_vae.sd3.vae_factory import apply_freeze_patterns, build_sd3_vae
from src.sd35_task_aware_vae.utils.config import dump_yaml
from src.sd35_task_aware_vae.utils.files import ensure_dir, write_csv, write_json
from src.sd35_task_aware_vae.utils.seed import seed_everything
from src.sd35_task_aware_vae.utils.wandb import init_wandb_session, maybe_build_wandb_image
from src.sd35_task_aware_vae.vae.trainer import (
    _compute_loss_terms,
    _teacher_stats,
    build_datasets,
    build_teacher_if_needed,
)


DEFAULT_LORA_TARGET_MODULES = [
    "attn.add_k_proj",
    "attn.add_q_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "attn.to_k",
    "attn.to_out.0",
    "attn.to_q",
    "attn.to_v",
]


@dataclass
class PromptBatch:
    prompts: list[str]
    negative_prompts: list[str]
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor


@dataclass
class EpochMetrics:
    epoch: int
    split: str
    total_loss: float
    diffusion_loss: float
    vae_loss: float
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
            "diffusion_loss": self.diffusion_loss,
            "vae_loss": self.vae_loss,
            "recon_loss": self.recon_loss,
            "kl_loss": self.kl_loss,
            "feature_loss": self.feature_loss,
            "logit_loss": self.logit_loss,
            "noise_feature_loss": self.noise_feature_loss,
        }


@dataclass
class DistributedContext:
    use_ddp: bool
    rank: int
    local_rank: int
    world_size: int
    is_main_process: bool
    device: torch.device


class PromptManager:
    def __init__(
        self,
        pipe,
        *,
        prompt_cfg: dict[str, Any],
        class_names: Sequence[str],
        device: torch.device,
        max_sequence_length: int,
        offload_static_text_encoders: bool = True,
    ) -> None:
        self.pipe = pipe
        self.prompt_cfg = prompt_cfg or {}
        self.class_names = list(class_names)
        self.device = device
        self.max_sequence_length = int(max_sequence_length)
        self.mode = str(self.prompt_cfg.get("mode", "neutral")).lower()
        self.static = self.mode in {"neutral", "explicit"}
        self._cached_prompt: list[str] | None = None
        self._cached_negative: list[str] | None = None
        self._cached_embeds: torch.Tensor | None = None
        self._cached_pooled: torch.Tensor | None = None

        if self.static:
            self._prime_static_cache()
            if offload_static_text_encoders:
                self._offload_text_encoders_to_cpu()

    def _prime_static_cache(self) -> None:
        prompt, negative = resolve_prompts(
            batch_size=1,
            labels=None,
            class_names=self.class_names,
            prompt_cfg=self.prompt_cfg,
        )
        prompt_embeds, pooled_prompt_embeds = self._encode_prompts(prompt)
        self._cached_prompt = prompt
        self._cached_negative = negative
        self._cached_embeds = prompt_embeds.detach().cpu()
        self._cached_pooled = pooled_prompt_embeds.detach().cpu()

    def _offload_text_encoders_to_cpu(self) -> None:
        for attr in ["text_encoder", "text_encoder_2", "text_encoder_3"]:
            module = getattr(self.pipe, attr, None)
            if module is None:
                continue
            try:
                module.to("cpu")
            except Exception:
                pass

    def _encode_prompts(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            prompt_embeds, _neg, pooled_prompt_embeds, _neg_pooled = self.pipe.encode_prompt(
                prompt=list(prompts),
                prompt_2=None,
                prompt_3=None,
                negative_prompt=None,
                negative_prompt_2=None,
                negative_prompt_3=None,
                do_classifier_free_guidance=False,
                device=self.device,
                num_images_per_prompt=1,
                max_sequence_length=self.max_sequence_length,
            )
        return prompt_embeds, pooled_prompt_embeds

    def encode_batch(self, labels) -> PromptBatch:
        if self.static:
            if self._cached_embeds is None or self._cached_pooled is None or self._cached_prompt is None or self._cached_negative is None:
                raise RuntimeError("Static prompt cache has not been initialized.")
            batch_size = int(labels.shape[0]) if labels is not None else 1
            prompt_embeds = self._cached_embeds.to(device=self.device).repeat(batch_size, 1, 1)
            pooled_prompt_embeds = self._cached_pooled.to(device=self.device).repeat(batch_size, 1)
            return PromptBatch(
                prompts=self._cached_prompt * batch_size,
                negative_prompts=self._cached_negative * batch_size,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
            )

        batch_size = int(labels.shape[0])
        prompts, negative_prompts = resolve_prompts(
            batch_size=batch_size,
            labels=labels.detach().cpu() if hasattr(labels, "detach") else labels,
            class_names=self.class_names,
            prompt_cfg=self.prompt_cfg,
        )
        prompt_embeds, pooled_prompt_embeds = self._encode_prompts(prompts)
        return PromptBatch(
            prompts=prompts,
            negative_prompts=negative_prompts,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
        )



def _resolve_train_targets(cfg: dict[str, Any]) -> dict[str, Any]:
    train_targets = cfg.get("train_targets", {}) or {}
    if train_targets:
        return train_targets
    # backward-compatible defaults: train transformer if explicitly requested in model_finetune,
    # otherwise treat this script as transformer full fine-tuning only.
    return {
        "transformer": {"mode": str((cfg.get("transformer", {}) or {}).get("mode", "full"))},
        "vae": {"enabled": bool((cfg.get("vae", {}) or {}).get("train", False))},
        "text_encoders": {"enabled": False},
    }



def _load_state_dict_file(path: Path):
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except Exception as e:  # pragma: no cover - optional dependency
            raise RuntimeError("safetensors is required to load .safetensors checkpoints") from e
        return load_file(str(path))

    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        for key in ["state_dict", "model", "transformer", "module"]:
            if key in state and isinstance(state[key], dict):
                return state[key]
    return state

def _resolve_transformer_source(model_cfg: dict[str, Any], transformer_cfg: dict[str, Any]) -> tuple[str, str]:
    checkpoint = transformer_cfg.get("checkpoint", None)
    if checkpoint:
        return str(checkpoint), str(transformer_cfg.get("subfolder", ""))

    repo_id = str(transformer_cfg.get("model_repo_id", model_cfg.get("repo_id")))
    subfolder = str(transformer_cfg.get("subfolder", "transformer"))
    return repo_id, subfolder

def _build_sd3_transformer(model_cfg: dict[str, Any], transformer_cfg: dict[str, Any], torch_dtype):
    from pathlib import Path
    from diffusers import SD3Transformer2DModel

    source, subfolder = _resolve_transformer_source(model_cfg, transformer_cfg)
    source_path = Path(source)

    if source_path.is_file() and source_path.suffix.lower() in {".pt", ".pth", ".bin", ".safetensors"}:
        base_repo = str(transformer_cfg.get("model_repo_id", model_cfg.get("repo_id")))
        base_subfolder = str(transformer_cfg.get("base_subfolder", "transformer"))
        transformer = SD3Transformer2DModel.from_pretrained(
            base_repo,
            subfolder=base_subfolder,
            torch_dtype=torch_dtype,
        )
        state_dict = _load_state_dict_file(source_path)
        missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[warn] transformer state_dict load: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        return transformer

    load_kwargs = {"torch_dtype": torch_dtype}
    if subfolder:
        load_kwargs["subfolder"] = subfolder
    return SD3Transformer2DModel.from_pretrained(source, **load_kwargs)



def _resolve_lora_target_modules(lora_cfg: dict[str, Any]) -> list[str]:
    target_modules = lora_cfg.get("target_modules", None)
    if isinstance(target_modules, str):
        target_modules = [x.strip() for x in target_modules.split(",") if x.strip()]
    elif isinstance(target_modules, (list, tuple)):
        target_modules = [str(x) for x in target_modules]
    else:
        target_modules = list(DEFAULT_LORA_TARGET_MODULES)

    target_blocks = lora_cfg.get("target_blocks", None)
    if isinstance(target_blocks, str):
        target_blocks = [int(x.strip()) for x in target_blocks.split(",") if x.strip()]
    elif isinstance(target_blocks, (list, tuple)):
        target_blocks = [int(x) for x in target_blocks]

    if target_blocks:
        expanded: list[str] = []
        for block in target_blocks:
            for module in target_modules:
                expanded.append(f"transformer_blocks.{block}.{module}")
        return expanded
    return target_modules



def _apply_transformer_lora(transformer, lora_cfg: dict[str, Any]) -> list[str]:
    try:
        from peft import LoraConfig
    except Exception as e:  # pragma: no cover - optional dependency
        raise RuntimeError("peft is required for transformer mode='lora'") from e

    target_modules = _resolve_lora_target_modules(lora_cfg)
    config = LoraConfig(
        r=int(lora_cfg.get("rank", 8)),
        lora_alpha=int(lora_cfg.get("alpha", lora_cfg.get("rank", 8))),
        lora_dropout=float(lora_cfg.get("dropout", 0.0)),
        init_lora_weights=str(lora_cfg.get("init_lora_weights", "gaussian")),
        target_modules=target_modules,
    )
    transformer.add_adapter(config)
    if bool(lora_cfg.get("upcast_trainable_params", True)):
        try:
            from diffusers.training_utils import cast_training_params

            cast_training_params(transformer, dtype=torch.float32)
        except Exception:
            for p in transformer.parameters():
                if p.requires_grad:
                    p.data = p.data.float()
    return target_modules



def _build_optimizer(param_groups: list[dict[str, Any]], optimizer_cfg: dict[str, Any]):
    name = str(optimizer_cfg.get("name", optimizer_cfg.get("optimizer", "adamw"))).lower()
    use_8bit_adam = bool(optimizer_cfg.get("use_8bit_adam", False))
    betas = optimizer_cfg.get("betas", [0.9, 0.999])
    betas = (float(betas[0]), float(betas[1]))
    eps = float(optimizer_cfg.get("eps", optimizer_cfg.get("adam_epsilon", 1e-8)))

    if use_8bit_adam:
        try:
            import bitsandbytes as bnb  # type: ignore
        except Exception as e:  # pragma: no cover - optional dependency
            raise RuntimeError("bitsandbytes is required for use_8bit_adam=true") from e

        if name == "adam":
            return bnb.optim.Adam8bit(param_groups, betas=betas, eps=eps)
        return bnb.optim.AdamW8bit(param_groups, betas=betas, eps=eps)

    if name == "adam":
        return torch.optim.Adam(param_groups, betas=betas, eps=eps)
    return torch.optim.AdamW(param_groups, betas=betas, eps=eps)



def _build_lr_scheduler(optimizer, train_cfg: dict[str, Any], total_steps: int):
    name = str(train_cfg.get("lr_scheduler", "constant")).lower()
    warmup_steps = int(train_cfg.get("lr_warmup_steps", 0))
    num_cycles = int(train_cfg.get("lr_num_cycles", 1))
    power = float(train_cfg.get("lr_power", 1.0))

    try:
        from diffusers.optimization import get_scheduler

        return get_scheduler(
            name,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max(1, total_steps),
            num_cycles=num_cycles,
            power=power,
        )
    except Exception:
        pass

    if name in {"constant", "none"}:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)

    if name in {"constant_with_warmup", "linear"}:
        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            if name == "constant_with_warmup":
                return 1.0
            remain = max(1, total_steps - warmup_steps)
            return max(0.0, float(total_steps - step) / float(remain))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))

    raise ValueError(f"Unsupported lr_scheduler: {name}")



def _compute_density_for_timestep_sampling_fallback(
    *,
    weighting_scheme: str,
    batch_size: int,
    device: torch.device,
    logit_mean: float,
    logit_std: float,
    mode_scale: float,
) -> torch.Tensor:
    scheme = str(weighting_scheme).lower()
    if scheme == "logit_normal":
        return torch.sigmoid(torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device=device))
    if scheme == "mode":
        # Conservative approximation when diffusers.training_utils is unavailable.
        u = torch.rand(batch_size, device=device)
        return 1.0 - torch.pow(1.0 - u, max(1.0, mode_scale))
    if scheme == "cosmap":
        u = torch.rand(batch_size, device=device)
        return 1.0 - torch.cos(0.5 * math.pi * u)
    return torch.rand(batch_size, device=device)



def _compute_loss_weighting_fallback(weighting_scheme: str, sigmas: torch.Tensor) -> torch.Tensor:
    scheme = str(weighting_scheme).lower()
    if scheme == "sigma_sqrt":
        return torch.sqrt(torch.clamp(sigmas, min=1.0e-5))
    return torch.ones_like(sigmas)



"""
def _ensure_scheduler_state(noise_scheduler, device: torch.device) -> None:
    if getattr(noise_scheduler, "timesteps", None) is not None and getattr(noise_scheduler, "sigmas", None) is not None:
        return
    num_train_timesteps = int(getattr(noise_scheduler.config, "num_train_timesteps", 1000))
    try:
        noise_scheduler.set_timesteps(num_train_timesteps, device=device)
    except TypeError:
        noise_scheduler.set_timesteps(num_train_timesteps)
"""

def _ensure_scheduler_state(noise_scheduler, device: torch.device) -> None:
    has_t = getattr(noise_scheduler, "timesteps", None) is not None
    has_s = getattr(noise_scheduler, "sigmas", None) is not None

    if has_t and has_s:
        try:
            noise_scheduler.timesteps = noise_scheduler.timesteps.to(device)
            noise_scheduler.sigmas = noise_scheduler.sigmas.to(device)
        except Exception:
            pass
        return

    num_train_timesteps = int(getattr(noise_scheduler.config, "num_train_timesteps", 1000))
    try:
        noise_scheduler.set_timesteps(num_train_timesteps, device=device)
    except TypeError:
        noise_scheduler.set_timesteps(num_train_timesteps)
        if getattr(noise_scheduler, "timesteps", None) is not None:
            noise_scheduler.timesteps = noise_scheduler.timesteps.to(device)
        if getattr(noise_scheduler, "sigmas", None) is not None:
            noise_scheduler.sigmas = noise_scheduler.sigmas.to(device)


def _get_sigmas(noise_scheduler, timesteps: torch.Tensor, n_dim: int = 4, dtype=torch.float32, device: torch.device | None = None):
    if device is None:
        device = timesteps.device
    _ensure_scheduler_state(noise_scheduler, device)
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma



def _sample_training_timesteps(noise_scheduler, batch_size: int, device: torch.device, diffusion_cfg: dict[str, Any]) -> torch.Tensor:
    weighting_scheme = str(diffusion_cfg.get("weighting_scheme", "logit_normal"))
    logit_mean = float(diffusion_cfg.get("logit_mean", 0.0))
    logit_std = float(diffusion_cfg.get("logit_std", 1.0))
    mode_scale = float(diffusion_cfg.get("mode_scale", 1.29))
    try:
        from diffusers.training_utils import compute_density_for_timestep_sampling

        u = compute_density_for_timestep_sampling(
            weighting_scheme=weighting_scheme,
            batch_size=batch_size,
            logit_mean=logit_mean,
            logit_std=logit_std,
            mode_scale=mode_scale,
        ).to(device=device)
    except Exception:
        u = _compute_density_for_timestep_sampling_fallback(
            weighting_scheme=weighting_scheme,
            batch_size=batch_size,
            device=device,
            logit_mean=logit_mean,
            logit_std=logit_std,
            mode_scale=mode_scale,
        )

    _ensure_scheduler_state(noise_scheduler, device)
    num_train_timesteps = int(getattr(noise_scheduler.config, "num_train_timesteps", len(noise_scheduler.timesteps)))
    indices = (u * num_train_timesteps).long().clamp_(0, len(noise_scheduler.timesteps) - 1)
    return noise_scheduler.timesteps[indices].to(device=device)



def _compute_sd3_loss_weighting(diffusion_cfg: dict[str, Any], sigmas: torch.Tensor) -> torch.Tensor:
    weighting_scheme = str(diffusion_cfg.get("weighting_scheme", "logit_normal"))
    try:
        from diffusers.training_utils import compute_loss_weighting_for_sd3

        return compute_loss_weighting_for_sd3(weighting_scheme=weighting_scheme, sigmas=sigmas)
    except Exception:
        return _compute_loss_weighting_fallback(weighting_scheme, sigmas)



def _save_lora_weights(pipe, transformer, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from peft.utils import get_peft_model_state_dict
    except Exception as e:  # pragma: no cover - optional dependency
        raise RuntimeError("peft is required to save LoRA weights") from e

    state_dict = get_peft_model_state_dict(transformer)
    if hasattr(pipe.__class__, "save_lora_weights"):
        pipe.__class__.save_lora_weights(str(out_dir), transformer_lora_layers=state_dict)
    try:
        transformer.save_pretrained(out_dir / "transformer_peft")
    except Exception:
        pass



def _prune_old_checkpoints(checkpoint_root: Path, limit: int) -> None:
    if limit <= 0:
        return
    checkpoints = sorted([p for p in checkpoint_root.glob("step_*") if p.is_dir()], key=lambda p: p.name)
    while len(checkpoints) > limit:
        oldest = checkpoints.pop(0)
        shutil.rmtree(oldest, ignore_errors=True)



def _build_preview_recon(vae, images: torch.Tensor, posterior_mode: str) -> torch.Tensor:
    from src.sd35_task_aware_vae.sd3.latent_codec import decode_from_latents, encode_to_latents

    with torch.no_grad():
        latents = encode_to_latents(vae, images, sample_mode=posterior_mode)
        recon = decode_from_latents(vae, latents)
    return recon



def _save_recon_preview(path: Path, images: torch.Tensor, recon: torch.Tensor) -> Path:
    from torchvision.utils import save_image

    path.parent.mkdir(parents=True, exist_ok=True)
    orig = (images[:4].detach().cpu().clamp(-1.0, 1.0) + 1.0) / 2.0
    rec = (recon[:4].detach().cpu().clamp(-1.0, 1.0) + 1.0) / 2.0
    grid = torch.cat([orig, rec], dim=3)
    save_image(grid, path)
    return path



def _maybe_generate_text2img_preview(pipe, preview_cfg: dict[str, Any], out_path: Path):
    if not bool(preview_cfg.get("enabled", False)):
        return None
    if str(preview_cfg.get("kind", "")).lower() != "text2img":
        return None

    prompt = str(preview_cfg.get("prompt", "ultra-widefield fundus photograph"))
    negative_prompt = str(preview_cfg.get("negative_prompt", "")) or None
    num_inference_steps = int(preview_cfg.get("num_inference_steps", 20))
    guidance_scale = float(preview_cfg.get("guidance_scale", 4.5))
    seed = preview_cfg.get("seed", None)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe._execution_device if hasattr(pipe, "_execution_device") else None).manual_seed(int(seed))

    try:
        with torch.no_grad():
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path)
        return out_path
    except Exception as e:
        print(f"[warn] preview generation failed: {e}", flush=True)
        return None



def _collect_trainable_param_groups(
    *,
    transformer,
    vae,
    transformer_mode: str,
    train_transformer: bool,
    train_vae: bool,
    optimizer_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    groups: list[dict[str, Any]] = []
    counts = {"transformer": 0, "vae": 0}

    base_lr = float(optimizer_cfg.get("lr", 1.0e-5))
    transformer_lr = float(optimizer_cfg.get("transformer_lr", base_lr))
    vae_lr = float(optimizer_cfg.get("vae_lr", base_lr))
    weight_decay = float(optimizer_cfg.get("weight_decay", 0.0))

    if train_transformer:
        params = [p for p in transformer.parameters() if p.requires_grad]
        if params:
            groups.append({"params": params, "lr": transformer_lr, "weight_decay": weight_decay})
            counts["transformer"] = int(sum(p.numel() for p in params))

    if train_vae:
        params = [p for p in vae.parameters() if p.requires_grad]
        if params:
            groups.append({"params": params, "lr": vae_lr, "weight_decay": weight_decay})
            counts["vae"] = int(sum(p.numel() for p in params))

    if not groups:
        raise RuntimeError("No trainable parameters were selected. Check train_targets and freeze/unfreeze patterns.")
    return groups, counts



def _prepare_resume_sources(cfg: dict[str, Any]) -> tuple[dict[str, Any], Path | None]:
    cfg = copy.deepcopy(cfg)
    train_cfg = cfg.get("train", {}) or {}
    resume_from = train_cfg.get("resume_from", None)
    if not resume_from:
        return cfg, None

    resume_dir = Path(str(resume_from))
    if not resume_dir.exists():
        raise FileNotFoundError(f"Resume directory not found: {resume_dir}")

    transformer_dir = resume_dir / "transformer"
    vae_dir = resume_dir / "vae"
    if transformer_dir.is_dir():
        cfg.setdefault("train_targets", {}).setdefault("transformer", {})["checkpoint"] = str(transformer_dir)
    lora_dir = resume_dir / "lora"
    if lora_dir.is_dir():
        cfg.setdefault("train_targets", {}).setdefault("transformer", {})["resume_lora"] = str(lora_dir)
    if vae_dir.is_dir():
        cfg.setdefault("vae", {})["checkpoint"] = str(vae_dir)
    return cfg, resume_dir



def _load_resume_state(resume_dir: Path | None):
    if resume_dir is None:
        return None
    state_path = resume_dir / "train_state.pt"
    if not state_path.is_file():
        return None
    return torch.load(state_path, map_location="cpu")



def _load_lora_checkpoint_if_needed(transformer, transformer_cfg: dict[str, Any]):
    resume_lora = transformer_cfg.get("resume_lora", None)
    if not resume_lora:
        return transformer

    if hasattr(transformer, "load_adapter"):
        try:
            transformer.load_adapter(str(resume_lora), adapter_name="default")
            return transformer
        except Exception:
            pass

    try:
        from peft import PeftModel
    except Exception as e:  # pragma: no cover - optional dependency
        raise RuntimeError("peft is required to resume a LoRA transformer") from e

    try:
        return PeftModel.from_pretrained(transformer, str(resume_lora), is_trainable=True)
    except Exception as e:
        raise RuntimeError(f"Failed to resume LoRA checkpoint from {resume_lora}") from e



def _compute_diffusion_terms(
    *,
    transformer,
    prompt_batch: PromptBatch,
    latents: torch.Tensor,
    noise_scheduler,
    diffusion_cfg: dict[str, Any],
):
    noise = torch.randn_like(latents)
    timesteps = _sample_training_timesteps(
        noise_scheduler,
        batch_size=latents.shape[0],
        device=latents.device,
        diffusion_cfg=diffusion_cfg,
    )
    sigmas = _get_sigmas(noise_scheduler, timesteps, n_dim=latents.ndim, dtype=latents.dtype, device=latents.device)
    noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise

    model_pred = transformer(
        hidden_states=noisy_model_input,
        timestep=timesteps,
        encoder_hidden_states=prompt_batch.prompt_embeds,
        pooled_projections=prompt_batch.pooled_prompt_embeds,
        return_dict=False,
    )[0]

    precondition_outputs = bool(diffusion_cfg.get("precondition_outputs", True))
    if precondition_outputs:
        model_pred = model_pred * (-sigmas) + noisy_model_input
        target = latents
    else:
        target = noise - latents

    weighting = _compute_sd3_loss_weighting(diffusion_cfg, sigmas)
    diff_loss = torch.mean(
        (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
        dim=1,
    ).mean()
    return {
        "loss": diff_loss,
        "mean_timestep": float(timesteps.float().mean().detach().cpu()),
        "mean_sigma": float(sigmas.float().mean().detach().cpu()),
    }



def _build_training_latents(
    *,
    vae,
    images: torch.Tensor,
    posterior_mode: str,
    requires_grad: bool,
    target_dtype,
):
    from src.sd35_task_aware_vae.sd3.latent_codec import encode_to_latents

    with torch.set_grad_enabled(requires_grad):
        latents = encode_to_latents(vae, images, sample_mode=posterior_mode)
    return latents.to(dtype=target_dtype)



def _setup_distributed_context(cfg: dict[str, Any], model_cfg: dict[str, Any]) -> DistributedContext:
    distributed_cfg = cfg.get("distributed", {}) or {}
    requested_ddp = bool(distributed_cfg.get("enabled", False))
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    env_rank = int(os.environ.get("RANK", "0"))
    env_local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    use_ddp = requested_ddp and env_world_size > 1 and torch.cuda.is_available()
    if requested_ddp and env_world_size <= 1:
        print(
            "[warn] distributed.enabled=true but WORLD_SIZE=1. Launch with torchrun to enable DDP, or disable distributed.",
            flush=True,
        )

    if use_ddp:
        import datetime
        import torch.distributed as dist

        torch.cuda.set_device(env_local_rank)
        if not dist.is_initialized():
            timeout_seconds = int(distributed_cfg.get("timeout_seconds", 7200))
            dist.init_process_group(
                backend=str(distributed_cfg.get("backend", "nccl")),
                init_method="env://",
                timeout=datetime.timedelta(seconds=timeout_seconds),
            )
        device = torch.device(f"cuda:{env_local_rank}")
    else:
        if torch.cuda.is_available() and str(model_cfg.get("device", "cuda")) != "cpu":
            try:
                torch.cuda.set_device(0)
            except Exception:
                pass
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    return DistributedContext(
        use_ddp=use_ddp,
        rank=env_rank if use_ddp else 0,
        local_rank=env_local_rank if use_ddp else 0,
        world_size=env_world_size if use_ddp else 1,
        is_main_process=(env_rank == 0) if use_ddp else True,
        device=device,
    )



def _barrier_if_needed(context: DistributedContext) -> None:
    if not context.use_ddp:
        return
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:
        pass



def _cleanup_distributed(context: DistributedContext | None) -> None:
    if context is None or not context.use_ddp:
        return
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass



def _run_epoch(
    *,
    pipe,
    transformer,
    vae,
    teacher,
    prompt_manager: PromptManager,
    loader,
    optimizer,
    lr_scheduler,
    scaler,
    device: torch.device,
    amp_dtype,
    epoch: int,
    train: bool,
    class_names: Sequence[str],
    posterior_mode: str,
    train_transformer: bool,
    train_vae: bool,
    diffusion_cfg: dict[str, Any],
    loss_cfg: dict[str, Any],
    objective_cfg: dict[str, Any],
    teacher_cfg: dict[str, Any],
    noise_cfg: dict[str, Any],
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    grad_accum_steps: int,
    global_step_start: int,
    wandb_session,
    step_log_interval: int,
    max_batches: int | None = None,
    vae_diffusion_gradients: bool = False,
    max_grad_norm: float = 0.0,
    use_progress_bar: bool = True,
    progress_desc: str | None = None,
    progress_update_interval: int = 10,
    progress_mininterval: float = 0.5,
    progress_leave: bool = False,
    is_main_process: bool = True,
    use_distributed: bool = False,
    ddp_no_sync_module=None,
):
    autocast_enabled = amp_dtype is not None and device.type == "cuda"
    total_loss = diffusion_loss = vae_loss = 0.0
    recon_loss = kl_loss = feature_loss = logit_loss = noise_feature_loss = 0.0
    num_batches = 0
    num_items = 0.0
    global_step = int(global_step_start)

    transformer.train(train and train_transformer)
    vae.train(train and train_vae)

    if train:
        optimizer.zero_grad(set_to_none=True)

    objective_diffusion_weight = float(objective_cfg.get("diffusion_weight", 1.0))
    objective_vae_weight = float(objective_cfg.get("vae_weight", 1.0))

    total_batches = len(loader)
    if max_batches is not None:
        total_batches = min(int(max_batches), total_batches)
    progress_update_interval = max(1, int(progress_update_interval))

    progress = None
    iterator = loader
    if tqdm is not None and use_progress_bar and is_main_process:
        progress = tqdm(
            loader,
            total=total_batches,
            desc=progress_desc or ("train" if train else "val"),
            dynamic_ncols=True,
            leave=progress_leave,
            mininterval=progress_mininterval,
        )
        iterator = progress

    for batch_idx, (images, labels, _paths) in enumerate(iterator, start=1):
        if batch_idx > total_batches:
            break

        images = images.to(device=device, dtype=torch.float32, non_blocking=(device.type == "cuda"))
        labels = labels.to(device=device, dtype=torch.float32, non_blocking=(device.type == "cuda"))
        batch_items = float(images.shape[0])
        should_step = train and ((batch_idx % max(1, grad_accum_steps) == 0) or (batch_idx == total_batches))

        sync_context = nullcontext()
        if train and ddp_no_sync_module is not None and not should_step:
            sync_context = ddp_no_sync_module.no_sync()

        with sync_context:
            with torch.set_grad_enabled(train):
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                    prompt_batch = prompt_manager.encode_batch(labels)
                    terms_total = images.new_tensor(0.0)
                    diffusion_terms = None
                    vae_terms = None

                    if train_transformer:
                        latents = _build_training_latents(
                            vae=vae,
                            images=images,
                            posterior_mode=posterior_mode,
                            requires_grad=(train and train_vae and vae_diffusion_gradients),
                            target_dtype=(amp_dtype if amp_dtype is not None else torch.float32),
                        )
                        diffusion_terms = _compute_diffusion_terms(
                            transformer=transformer,
                            prompt_batch=prompt_batch,
                            latents=latents,
                            noise_scheduler=pipe.scheduler,
                            diffusion_cfg=diffusion_cfg,
                        )
                        terms_total = terms_total + objective_diffusion_weight * diffusion_terms["loss"]

                    if train_vae:
                        vae_terms = _compute_loss_terms(
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
                        terms_total = terms_total + objective_vae_weight * vae_terms["total"]

                    loss = terms_total / float(max(1, grad_accum_steps))

                if train:
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    if should_step:
                        if scaler is not None:
                            scaler.unscale_(optimizer)
                        if max_grad_norm > 0:
                            params_to_clip = [p for group in optimizer.param_groups for p in group["params"] if p.requires_grad]
                            torch.nn.utils.clip_grad_norm_(params_to_clip, max_grad_norm)
                        if scaler is not None:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        if lr_scheduler is not None:
                            lr_scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1

        num_batches += 1
        num_items += batch_items
        total_loss += float(terms_total.detach().cpu()) * batch_items
        if diffusion_terms is not None:
            diffusion_loss += float(diffusion_terms["loss"].detach().cpu()) * batch_items
        if vae_terms is not None:
            vae_loss += float(vae_terms["total"].detach().cpu()) * batch_items
            recon_loss += float(vae_terms["recon"].detach().cpu()) * batch_items
            kl_loss += float(vae_terms["kl"].detach().cpu()) * batch_items
            feature_loss += float(vae_terms["feature"].detach().cpu()) * batch_items
            logit_loss += float(vae_terms["logit"].detach().cpu()) * batch_items
            noise_feature_loss += float(vae_terms["noise_feature"].detach().cpu()) * batch_items

        if progress is not None and (batch_idx % progress_update_interval == 0 or batch_idx == total_batches):
            denom = float(max(1.0, num_items))
            postfix: dict[str, str] = {"loss": f"{(total_loss / denom):.4f}"}
            if diffusion_terms is not None:
                postfix["diff"] = f"{(diffusion_loss / denom):.4f}"
            if vae_terms is not None:
                postfix["vae"] = f"{(vae_loss / denom):.4f}"
            if train:
                postfix["lr"] = f"{optimizer.param_groups[0]['lr']:.2e}"
            progress.set_postfix(postfix)

        if train and wandb_session is not None and getattr(wandb_session, "enabled", False):
            if step_log_interval > 0 and should_step and global_step > global_step_start and global_step % max(1, step_log_interval) == 0:
                running_denom = float(max(1.0, num_items))
                payload = {
                    "train/global_step": global_step,
                    "train/epoch": epoch,
                    "train/loss_step": float(terms_total.detach().cpu()),
                    "train/loss_running": float(total_loss / running_denom),
                    "train/lr": float(optimizer.param_groups[0]["lr"]),
                }
                if diffusion_terms is not None:
                    payload["train/diffusion_step"] = float(diffusion_terms["loss"].detach().cpu())
                    payload["train/mean_timestep"] = float(diffusion_terms["mean_timestep"])
                    payload["train/mean_sigma"] = float(diffusion_terms["mean_sigma"])
                if vae_terms is not None:
                    payload.update(
                        {
                            "train/vae_step": float(vae_terms["total"].detach().cpu()),
                            "train/recon_step": float(vae_terms["recon"].detach().cpu()),
                            "train/kl_step": float(vae_terms["kl"].detach().cpu()),
                            "train/feature_step": float(vae_terms["feature"].detach().cpu()),
                            "train/logit_step": float(vae_terms["logit"].detach().cpu()),
                            "train/noise_feature_step": float(vae_terms["noise_feature"].detach().cpu()),
                        }
                    )
                try:
                    if scaler is not None:
                        payload["train/grad_scale"] = float(scaler.get_scale())
                except Exception:
                    pass
                wandb_session.log(payload, step=global_step)

    if progress is not None:
        progress.close()

    if use_distributed:
        try:
            import torch.distributed as dist

            packed = torch.tensor(
                [
                    total_loss,
                    diffusion_loss,
                    vae_loss,
                    recon_loss,
                    kl_loss,
                    feature_loss,
                    logit_loss,
                    noise_feature_loss,
                    num_items,
                ],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            (
                total_loss,
                diffusion_loss,
                vae_loss,
                recon_loss,
                kl_loss,
                feature_loss,
                logit_loss,
                noise_feature_loss,
                num_items,
            ) = [float(x) for x in packed.tolist()]
        except Exception:
            pass

    denom = float(max(1.0, num_items))
    metrics = EpochMetrics(
        epoch=epoch,
        split="train" if train else "val",
        total_loss=total_loss / denom,
        diffusion_loss=diffusion_loss / denom,
        vae_loss=vae_loss / denom,
        recon_loss=recon_loss / denom,
        kl_loss=kl_loss / denom,
        feature_loss=feature_loss / denom,
        logit_loss=logit_loss / denom,
        noise_feature_loss=noise_feature_loss / denom,
    )
    return metrics, global_step



def train_sd35_system_from_config(cfg: dict[str, Any], config_path: str | Path) -> Path:
    try:
        from diffusers import StableDiffusion3Pipeline
    except Exception as e:  # pragma: no cover - optional dependency
        raise RuntimeError("diffusers is required for SD3.5 finetuning") from e

    cfg, resume_dir = _prepare_resume_sources(cfg)
    resume_state = _load_resume_state(resume_dir)

    model_cfg = cfg.get("model", {}) or {}
    dist_context = _setup_distributed_context(cfg, model_cfg)
    wandb_session = None
    try:
        seed_everything(int(cfg.get("seed", 42)) + int(dist_context.rank), deterministic=bool(cfg.get("deterministic", False)))

        exp_name = str(cfg.get("experiment_name", "sd35_system_finetune"))
        out_root = ensure_dir((cfg.get("output", {}) or {}).get("root_dir", "outputs/checkpoints/sd3"))
        out_dir = ensure_dir(out_root / exp_name)
        checkpoint_root = ensure_dir(out_dir / "checkpoints")
        if dist_context.is_main_process:
            ensure_dir(out_dir / "last")
            ensure_dir(out_dir / "best")
            ensure_dir(out_dir / "previews")
            dump_yaml(cfg, out_dir / "config_used.yaml")
        _barrier_if_needed(dist_context)

        vae_cfg = cfg.get("vae", {}) or {}
        train_cfg = cfg.get("train", {}) or {}
        teacher_cfg = cfg.get("teacher", {}) or {}
        loss_cfg = cfg.get("loss", {}) or {}
        diffusion_cfg = cfg.get("diffusion_loss", {}) or {}
        objective_cfg = cfg.get("objective", {}) or {}
        prompt_cfg = cfg.get("prompt", cfg.get("prompts", {})) or {}
        preview_cfg = cfg.get("preview", {}) or {}
        train_targets = _resolve_train_targets(cfg)
        transformer_cfg = train_targets.get("transformer", {}) or {}
        target_vae_cfg = train_targets.get("vae", {}) or {}
        text_cfg = train_targets.get("text_encoders", {}) or {}
        lora_cfg = cfg.get("lora", {}) or {}
        noise_cfg = cfg.get("noise_conditioning", {}) or {}
        distributed_cfg = cfg.get("distributed", {}) or {}

        if bool(text_cfg.get("enabled", False)):
            raise NotImplementedError("Text-encoder fine-tuning is not implemented in this repo yet. Keep train_targets.text_encoders.enabled=false.")

        transformer_mode = str(transformer_cfg.get("mode", "full")).lower()
        if transformer_mode not in {"full", "lora", "frozen", "off", "none"}:
            raise ValueError(f"Unsupported transformer mode: {transformer_mode}")

        train_transformer = transformer_mode not in {"frozen", "off", "none"}
        train_vae = bool(target_vae_cfg.get("enabled", False))
        vae_diffusion_gradients = bool(target_vae_cfg.get("diffusion_gradients", False))
        posterior_mode = str(target_vae_cfg.get("posterior", vae_cfg.get("posterior", "mode")))

        if not train_transformer and not train_vae:
            raise ValueError("Nothing to train. Enable train_targets.transformer or train_targets.vae.")
        if dist_context.use_ddp and train_vae:
            raise NotImplementedError(
                "DDP is currently supported for DiT-only / LoRA training. Keep train_targets.vae.enabled=false when distributed.enabled=true."
            )

        device = dist_context.device
        load_dtype = resolve_torch_dtype(model_cfg.get("load_dtype", model_cfg.get("torch_dtype", "bf16" if device.type == "cuda" else "fp32")))
        mixed_precision_str = str(train_cfg.get("mixed_precision", model_cfg.get("torch_dtype", "bf16" if device.type == "cuda" else "fp32"))).lower()
        amp_dtype = None
        if device.type == "cuda":
            if mixed_precision_str in {"fp16", "float16", "half"}:
                amp_dtype = torch.float16
            elif mixed_precision_str in {"bf16", "bfloat16"}:
                amp_dtype = torch.bfloat16

        if bool(model_cfg.get("allow_tf32", False)) and device.type == "cuda":
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass

        train_ds, val_ds, class_names = build_datasets(cfg)
        batch_size = int(train_cfg.get("batch_size", 1))
        num_workers = int(train_cfg.get("num_workers", 4))
        grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))
        drop_last = bool(train_cfg.get("drop_last", True))

        train_sampler = None
        val_sampler = None
        if dist_context.use_ddp:
            from torch.utils.data import DistributedSampler

            train_sampler = DistributedSampler(
                train_ds,
                num_replicas=dist_context.world_size,
                rank=dist_context.rank,
                shuffle=True,
                drop_last=drop_last,
            )
            val_sampler = DistributedSampler(
                val_ds,
                num_replicas=dist_context.world_size,
                rank=dist_context.rank,
                shuffle=False,
                drop_last=False,
            )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=drop_last,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

        custom_transformer = _build_sd3_transformer(model_cfg, transformer_cfg, torch_dtype=load_dtype)
        custom_vae = build_sd3_vae(model_cfg, vae_cfg, torch_dtype=load_dtype, device=None)
        if custom_vae is None:
            raise RuntimeError("build_sd3_vae returned None.")
        if custom_transformer is None:
            raise RuntimeError("_build_sd3_transformer returned None.")

        pipe = StableDiffusion3Pipeline.from_pretrained(
            str(model_cfg["repo_id"]),
            transformer=custom_transformer,
            vae=custom_vae,
            torch_dtype=load_dtype,
        )
        pipe.to(device)

        transformer = pipe.transformer
        vae = pipe.vae
        if transformer is None or vae is None:
            raise RuntimeError("Failed to initialize SD3 pipeline components.")

        pipe.scheduler = copy.deepcopy(pipe.scheduler)
        _ensure_scheduler_state(pipe.scheduler, device)

        if bool(train_cfg.get("gradient_checkpointing", False)) and hasattr(transformer, "enable_gradient_checkpointing"):
            transformer.enable_gradient_checkpointing()

        for p in transformer.parameters():
            p.requires_grad_(False)
        for p in vae.parameters():
            p.requires_grad_(False)

        if train_transformer:
            if transformer_mode == "full":
                for p in transformer.parameters():
                    p.requires_grad_(True)
                apply_freeze_patterns(
                    transformer,
                    freeze_patterns=[str(x) for x in (transformer_cfg.get("freeze_patterns", []) or [])],
                    unfreeze_patterns=[str(x) for x in (transformer_cfg.get("unfreeze_patterns", []) or [])],
                )
                if bool(transformer_cfg.get("upcast_trainable_params", False)):
                    for p in transformer.parameters():
                        if p.requires_grad:
                            p.data = p.data.float()
            elif transformer_mode == "lora":
                target_modules = _apply_transformer_lora(transformer, lora_cfg)
                transformer_cfg["resolved_lora_target_modules"] = target_modules
                transformer = _load_lora_checkpoint_if_needed(transformer, transformer_cfg)
                if hasattr(pipe, "register_modules"):
                    pipe.register_modules(transformer=transformer)
                else:
                    pipe.transformer = transformer

        if train_vae:
            for p in vae.parameters():
                p.requires_grad_(True)
            apply_freeze_patterns(
                vae,
                freeze_patterns=[str(x) for x in (target_vae_cfg.get("freeze_patterns", vae_cfg.get("freeze_patterns", [])) or [])],
                unfreeze_patterns=[str(x) for x in (target_vae_cfg.get("unfreeze_patterns", vae_cfg.get("unfreeze_patterns", [])) or [])],
            )
            if bool(target_vae_cfg.get("upcast_trainable_params", False)):
                for p in vae.parameters():
                    if p.requires_grad:
                        p.data = p.data.float()

        prompt_manager = PromptManager(
            pipe,
            prompt_cfg=prompt_cfg,
            class_names=class_names,
            device=device,
            max_sequence_length=int(model_cfg.get("max_sequence_length", prompt_cfg.get("max_sequence_length", 256))),
            offload_static_text_encoders=bool(prompt_cfg.get("offload_static_text_encoders", False)),
        )

        teacher = build_teacher_if_needed(
            {**cfg, "loss": loss_cfg, "teacher": teacher_cfg},
            num_classes=len(class_names),
            device=device,
        ) if train_vae else None
        mean, std = _teacher_stats(cfg)

        optimizer_cfg = cfg.get("optimizer", {}) or {}
        param_groups, trainable_counts = _collect_trainable_param_groups(
            transformer=transformer,
            vae=vae,
            transformer_mode=transformer_mode,
            train_transformer=train_transformer,
            train_vae=train_vae,
            optimizer_cfg=optimizer_cfg,
        )
        optimizer = _build_optimizer(param_groups, optimizer_cfg)

        updates_per_epoch = max(1, math.ceil(len(train_loader) / max(1, grad_accum_steps)))
        max_train_steps = train_cfg.get("max_train_steps", None)
        epochs = int(train_cfg.get("epochs", 1))
        if max_train_steps is None:
            max_train_steps = updates_per_epoch * epochs
        else:
            max_train_steps = int(max_train_steps)
            epochs = int(math.ceil(max_train_steps / float(updates_per_epoch)))

        lr_scheduler = _build_lr_scheduler(optimizer, train_cfg, total_steps=max_train_steps)
        scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16 and device.type == "cuda"))
        if not (amp_dtype == torch.float16 and device.type == "cuda"):
            scaler = None

        transformer_train_module = transformer
        if dist_context.use_ddp and train_transformer:
            from torch.nn.parallel import DistributedDataParallel as DDP

            ddp_kwargs = dict(
                find_unused_parameters=bool(distributed_cfg.get("find_unused_parameters", False)),
                broadcast_buffers=bool(distributed_cfg.get("broadcast_buffers", False)),
            )
            if device.type == "cuda":
                ddp_kwargs.update(device_ids=[dist_context.local_rank], output_device=dist_context.local_rank)
            transformer_train_module = DDP(transformer, **ddp_kwargs)

        start_epoch = 0
        global_step = 0
        best_val = math.inf
        history_rows: list[dict[str, Any]] = []
        if resume_state is not None:
            try:
                optimizer.load_state_dict(resume_state.get("optimizer", {}))
            except Exception as e:
                if dist_context.is_main_process:
                    print(f"[warn] failed to load optimizer state: {e}", flush=True)
            if lr_scheduler is not None and resume_state.get("scheduler", None) is not None:
                try:
                    lr_scheduler.load_state_dict(resume_state["scheduler"])
                except Exception as e:
                    if dist_context.is_main_process:
                        print(f"[warn] failed to load lr scheduler state: {e}", flush=True)
            if scaler is not None and resume_state.get("scaler", None) is not None:
                try:
                    scaler.load_state_dict(resume_state["scaler"])
                except Exception as e:
                    if dist_context.is_main_process:
                        print(f"[warn] failed to load scaler state: {e}", flush=True)
            start_epoch = int(resume_state.get("epoch", 0))
            global_step = int(resume_state.get("global_step", 0))
            best_val = float(resume_state.get("best_val_total_loss", math.inf))
            history_rows = list(resume_state.get("history", [])) if isinstance(resume_state.get("history", []), list) else []

        wandb_session = init_wandb_session(
            cfg,
            out_dir=out_dir,
            experiment_name=exp_name,
            default_project="sd35-system-finetune",
            enabled=dist_context.is_main_process,
        )
        wandb_session.set_summary("transformer_mode", transformer_mode)
        wandb_session.set_summary("train_transformer_params", trainable_counts["transformer"])
        wandb_session.set_summary("train_vae_params", trainable_counts["vae"])
        wandb_session.set_summary("num_train_samples", len(train_ds))
        wandb_session.set_summary("num_val_samples", len(val_ds))
        wandb_session.set_summary("world_size", dist_context.world_size)
        wandb_session.set_summary("use_ddp", dist_context.use_ddp)
        wandb_session.set_summary("per_gpu_batch_size", batch_size)
        wandb_session.set_summary("effective_batch_size", batch_size * grad_accum_steps * dist_context.world_size)

        step_log_interval = int((cfg.get("wandb", {}) or {}).get("log_interval_steps", train_cfg.get("log_interval_steps", 0)))
        save_every_n_steps = int(train_cfg.get("checkpoint_interval_steps", 0))
        save_total_limit = int(train_cfg.get("checkpoint_total_limit", train_cfg.get("save_total_limit", 3)))
        num_val_batches = train_cfg.get("num_val_batches", None)
        if num_val_batches is not None:
            num_val_batches = int(num_val_batches)
        local_num_val_batches = None
        if num_val_batches is not None:
            local_num_val_batches = int(math.ceil(num_val_batches / float(dist_context.world_size))) if dist_context.use_ddp else num_val_batches

        use_pbar = bool(train_cfg.get("progress_bar", True)) and (tqdm is not None)
        pbar_mininterval = float(train_cfg.get("tqdm_mininterval", 0.5))
        pbar_update_interval = int(train_cfg.get("tqdm_update_interval", 10))
        pbar_leave = bool(train_cfg.get("tqdm_leave", False))

        best_epoch = int(resume_state.get("best_epoch", -1)) if resume_state is not None else -1

        def _save_checkpoint(target_dir: Path, epoch_idx: int, step_idx: int, val_metrics: EpochMetrics | None) -> None:
            if not dist_context.is_main_process:
                return
            target_dir.mkdir(parents=True, exist_ok=True)
            if transformer_mode == "lora":
                _save_lora_weights(pipe, transformer, target_dir / "lora")
            else:
                transformer.save_pretrained(target_dir / "transformer")
            if train_vae or bool(target_vae_cfg.get("save_even_if_frozen", False)):
                vae.save_pretrained(target_dir / "vae")
            torch.save(
                {
                    "epoch": epoch_idx,
                    "global_step": step_idx,
                    "best_val_total_loss": best_val,
                    "best_epoch": best_epoch,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "history": history_rows,
                    "trainable_counts": trainable_counts,
                    "transformer_mode": transformer_mode,
                    "val_metrics": val_metrics.as_row() if val_metrics is not None else None,
                },
                target_dir / "train_state.pt",
            )

        for epoch in range(start_epoch + 1, epochs + 1):
            if global_step >= max_train_steps:
                break

            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            remaining_updates = max(0, max_train_steps - global_step)
            if remaining_updates == 0:
                break
            train_max_batches = remaining_updates * max(1, grad_accum_steps)

            train_metrics, global_step = _run_epoch(
                pipe=pipe,
                transformer=transformer_train_module,
                vae=vae,
                teacher=teacher,
                prompt_manager=prompt_manager,
                loader=train_loader,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                scaler=scaler,
                device=device,
                amp_dtype=amp_dtype,
                epoch=epoch,
                train=True,
                class_names=class_names,
                posterior_mode=posterior_mode,
                train_transformer=train_transformer,
                train_vae=train_vae,
                diffusion_cfg=diffusion_cfg,
                loss_cfg=loss_cfg,
                objective_cfg=objective_cfg,
                teacher_cfg=teacher_cfg,
                noise_cfg=noise_cfg,
                mean=mean,
                std=std,
                grad_accum_steps=grad_accum_steps,
                global_step_start=global_step,
                wandb_session=wandb_session,
                step_log_interval=step_log_interval,
                max_batches=train_max_batches,
                vae_diffusion_gradients=vae_diffusion_gradients,
                max_grad_norm=float(train_cfg.get("gradient_clip_norm", diffusion_cfg.get("max_grad_norm", objective_cfg.get("max_grad_norm", 0.0) or 0.0))),
                use_progress_bar=use_pbar,
                progress_desc=f"Train {epoch}/{epochs}",
                progress_update_interval=pbar_update_interval,
                progress_mininterval=pbar_mininterval,
                progress_leave=pbar_leave,
                is_main_process=dist_context.is_main_process,
                use_distributed=dist_context.use_ddp,
                ddp_no_sync_module=(transformer_train_module if dist_context.use_ddp and train_transformer else None),
            )

            with torch.no_grad():
                val_metrics, _ = _run_epoch(
                    pipe=pipe,
                    transformer=transformer_train_module,
                    vae=vae,
                    teacher=teacher,
                    prompt_manager=prompt_manager,
                    loader=val_loader,
                    optimizer=optimizer,
                    lr_scheduler=None,
                    scaler=scaler,
                    device=device,
                    amp_dtype=amp_dtype,
                    epoch=epoch,
                    train=False,
                    class_names=class_names,
                    posterior_mode=posterior_mode,
                    train_transformer=train_transformer,
                    train_vae=train_vae,
                    diffusion_cfg=diffusion_cfg,
                    loss_cfg=loss_cfg,
                    objective_cfg=objective_cfg,
                    teacher_cfg=teacher_cfg,
                    noise_cfg=noise_cfg,
                    mean=mean,
                    std=std,
                    grad_accum_steps=grad_accum_steps,
                    global_step_start=global_step,
                    wandb_session=wandb_session,
                    step_log_interval=0,
                    max_batches=local_num_val_batches,
                    vae_diffusion_gradients=False,
                    max_grad_norm=0.0,
                    use_progress_bar=use_pbar,
                    progress_desc=f"Val   {epoch}/{epochs}",
                    progress_update_interval=pbar_update_interval,
                    progress_mininterval=pbar_mininterval,
                    progress_leave=pbar_leave,
                    is_main_process=dist_context.is_main_process,
                    use_distributed=dist_context.use_ddp,
                    ddp_no_sync_module=None,
                )

            if dist_context.is_main_process:
                history_rows.extend([train_metrics.as_row(), val_metrics.as_row()])
                write_csv(history_rows, out_dir / "history.csv")

                preview_payload: dict[str, Any] = {}
                if train_vae and bool(preview_cfg.get("enabled", True)) and str(preview_cfg.get("kind", "vae_reconstruction")).lower() == "vae_reconstruction":
                    preview_batch = next(iter(val_loader))
                    preview_images = preview_batch[0].to(device=device, dtype=torch.float32)
                    recon = _build_preview_recon(vae, preview_images, posterior_mode=posterior_mode)
                    preview_path = _save_recon_preview(out_dir / "previews" / f"epoch_{epoch:03d}_recon.png", preview_images, recon)
                    wb_img = maybe_build_wandb_image(wandb_session, str(preview_path), caption=f"epoch {epoch}")
                    if wb_img is not None:
                        preview_payload["preview/reconstruction"] = wb_img
                else:
                    preview_path = _maybe_generate_text2img_preview(pipe, preview_cfg, out_dir / "previews" / f"epoch_{epoch:03d}_text2img.png")
                    if preview_path is not None:
                        wb_img = maybe_build_wandb_image(wandb_session, str(preview_path), caption=f"epoch {epoch}")
                        if wb_img is not None:
                            preview_payload["preview/text2img"] = wb_img

                epoch_payload = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "train/loss_epoch": train_metrics.total_loss,
                    "train/diffusion_epoch": train_metrics.diffusion_loss,
                    "train/vae_epoch": train_metrics.vae_loss,
                    "train/recon_epoch": train_metrics.recon_loss,
                    "train/kl_epoch": train_metrics.kl_loss,
                    "train/feature_epoch": train_metrics.feature_loss,
                    "train/logit_epoch": train_metrics.logit_loss,
                    "train/noise_feature_epoch": train_metrics.noise_feature_loss,
                    "val/loss": val_metrics.total_loss,
                    "val/diffusion": val_metrics.diffusion_loss,
                    "val/vae": val_metrics.vae_loss,
                    "val/recon": val_metrics.recon_loss,
                    "val/kl": val_metrics.kl_loss,
                    "val/feature": val_metrics.feature_loss,
                    "val/logit": val_metrics.logit_loss,
                    "val/noise_feature": val_metrics.noise_feature_loss,
                    "train/lr": float(optimizer.param_groups[0]["lr"]),
                }
                epoch_payload.update(preview_payload)
                wandb_session.log(epoch_payload, step=global_step)

                _save_checkpoint(out_dir / "last", epoch, global_step, val_metrics)
                if save_every_n_steps > 0 and global_step > 0 and global_step % save_every_n_steps == 0:
                    step_dir = checkpoint_root / f"step_{global_step:08d}"
                    _save_checkpoint(step_dir, epoch, global_step, val_metrics)
                    _prune_old_checkpoints(checkpoint_root, save_total_limit)

                if val_metrics.total_loss < best_val:
                    best_val = val_metrics.total_loss
                    best_epoch = epoch
                    _save_checkpoint(out_dir / "best", epoch, global_step, val_metrics)
                    write_json(
                        {
                            "best_epoch": best_epoch,
                            "best_val_total_loss": best_val,
                            "train_metrics": train_metrics.as_row(),
                            "val_metrics": val_metrics.as_row(),
                        },
                        out_dir / "best" / "metrics.json",
                    )
                    wandb_session.set_summary("best_epoch", best_epoch)
                    wandb_session.set_summary("best_val_total_loss", best_val)

            _barrier_if_needed(dist_context)

        if dist_context.is_main_process:
            summary = {
                "experiment_name": exp_name,
                "transformer_mode": transformer_mode,
                "train_transformer": train_transformer,
                "train_vae": train_vae,
                "vae_diffusion_gradients": vae_diffusion_gradients,
                "train_transformer_params": trainable_counts["transformer"],
                "train_vae_params": trainable_counts["vae"],
                "num_train_samples": len(train_ds),
                "num_val_samples": len(val_ds),
                "best_epoch": best_epoch,
                "best_val_total_loss": best_val,
                "global_step": global_step,
                "world_size": dist_context.world_size,
                "effective_batch_size": batch_size * grad_accum_steps * dist_context.world_size,
            }
            write_json(summary, out_dir / "summary.json")
            for key, value in summary.items():
                wandb_session.set_summary(key, value)
        return out_dir
    finally:
        if wandb_session is not None:
            wandb_session.finish()
        _cleanup_distributed(dist_context)

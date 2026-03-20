from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from src.sd35_task_aware_vae.sd3.latent_codec import decode_from_latents, encode_to_latents
from src.sd35_task_aware_vae.sd3.prompts import resolve_prompts


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    m = (max_shift - base_shift) / float(max_seq_len - base_seq_len)
    b = base_shift - m * float(base_seq_len)
    return float(image_seq_len * m + b)



def _image_seq_len(pipe, height: int, width: int) -> int:
    vae_scale_factor = int(getattr(pipe, "vae_scale_factor", 8))
    transformer = getattr(pipe, "transformer", None)
    patch_size = 2
    if transformer is not None:
        tcfg = getattr(transformer, "config", None)
        if tcfg is not None and hasattr(tcfg, "patch_size"):
            patch_size = int(getattr(tcfg, "patch_size"))
        elif hasattr(transformer, "patch_size"):
            patch_size = int(getattr(transformer, "patch_size"))
    return (height // vae_scale_factor // patch_size) * (width // vae_scale_factor // patch_size)



def build_reverse_timesteps(start_timestep: int, reverse_steps: int) -> list[float]:
    if reverse_steps <= 0:
        raise ValueError("reverse_steps must be >= 1")
    start = float(start_timestep)
    if reverse_steps == 1:
        return [start]
    return [float(x) for x in np.linspace(start, 1.0, reverse_steps, dtype=np.float32)]



def prepare_restore_prompt_embeds(
    pipe,
    prompts: Sequence[str],
    negative_prompts: Sequence[str],
    guidance_scale: float,
    max_sequence_length: int = 256,
):
    import torch

    do_cfg = float(guidance_scale) > 1.0
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
        prompt=list(prompts),
        prompt_2=None,
        prompt_3=None,
        negative_prompt=list(negative_prompts),
        negative_prompt_2=None,
        negative_prompt_3=None,
        do_classifier_free_guidance=do_cfg,
        device=pipe._execution_device if hasattr(pipe, "_execution_device") else None,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
    )
    if do_cfg:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
    return prompt_embeds, pooled_prompt_embeds



def reverse_restore_batch(
    pipe,
    images,
    *,
    labels=None,
    class_names: Sequence[str] | None = None,
    prompt_cfg: dict[str, Any] | None = None,
    posterior: str = "mode",
    start_timestep: int,
    reverse_steps: int,
    guidance_scale: float,
    max_sequence_length: int = 256,
    generator=None,
):
    import torch

    device = pipe._execution_device if hasattr(pipe, "_execution_device") else images.device
    vae_dtype = getattr(pipe.vae, "dtype", images.dtype)
    batch_size = images.shape[0]
    height, width = int(images.shape[-2]), int(images.shape[-1])

    prompts, negative_prompts = resolve_prompts(
        batch_size=batch_size,
        labels=labels,
        class_names=class_names,
        prompt_cfg=prompt_cfg,
    )

    prompt_embeds, pooled_prompt_embeds = prepare_restore_prompt_embeds(
        pipe,
        prompts=prompts,
        negative_prompts=negative_prompts,
        guidance_scale=guidance_scale,
        max_sequence_length=max_sequence_length,
    )

    latents = encode_to_latents(
        pipe.vae,
        images.to(device=device, dtype=vae_dtype),
        sample_mode=posterior,
        generator=generator,
    )

    scheduler = pipe.scheduler
    custom_timesteps = build_reverse_timesteps(start_timestep=start_timestep, reverse_steps=reverse_steps)
    scheduler_kwargs: dict[str, Any] = {}
    if getattr(scheduler, "config", None) is not None and getattr(scheduler.config, "use_dynamic_shifting", False):
        scheduler_kwargs["mu"] = calculate_shift(
            _image_seq_len(pipe, height=height, width=width),
            base_seq_len=int(getattr(scheduler.config, "base_image_seq_len", 256)),
            max_seq_len=int(getattr(scheduler.config, "max_image_seq_len", 4096)),
            base_shift=float(getattr(scheduler.config, "base_shift", 0.5)),
            max_shift=float(getattr(scheduler.config, "max_shift", 1.15)),
        )
    scheduler.set_timesteps(timesteps=custom_timesteps, device=device, **scheduler_kwargs)

    if generator is None:
        noise = torch.randn(latents.shape, device=device, dtype=latents.dtype)
    else:
        noise = torch.randn(latents.shape, generator=generator, device=device, dtype=latents.dtype)

    if hasattr(scheduler, "scale_noise"):
        latent_timestep = scheduler.timesteps[:1].repeat(batch_size)
        latents = scheduler.scale_noise(latents, latent_timestep, noise)
    elif hasattr(scheduler, "add_noise"):
        latent_timestep = scheduler.timesteps[:1].repeat(batch_size).to(dtype=torch.long)
        latents = scheduler.add_noise(latents, noise, latent_timestep)
    else:
        raise RuntimeError("Scheduler does not support forward noising via scale_noise/add_noise.")

    do_cfg = float(guidance_scale) > 1.0
    for timestep in scheduler.timesteps:
        latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
        timestep_batch = timestep.repeat(latent_model_input.shape[0])
        noise_pred = pipe.transformer(
            hidden_states=latent_model_input,
            timestep=timestep_batch,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=getattr(pipe, "joint_attention_kwargs", None),
            return_dict=False,
        )[0]
        if do_cfg:
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + float(guidance_scale) * (noise_text - noise_uncond)
        latents = scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]

    restored = decode_from_latents(pipe.vae, latents)
    timesteps_list = (
        [float(t) for t in scheduler.timesteps.detach().cpu().tolist()]
        if hasattr(scheduler.timesteps, "detach")
        else [float(t) for t in scheduler.timesteps]
    )
    return {
        "prompts": prompts,
        "negative_prompts": negative_prompts,
        "restored": restored,
        "latents": latents,
        "timesteps": timesteps_list,
    }

from __future__ import annotations

import math
from typing import Any



def get_latent_scaling_config(vae) -> tuple[float, float]:
    scaling = float(getattr(vae.config, "scaling_factor", 1.0))
    shift = float(getattr(vae.config, "shift_factor", 0.0))
    return scaling, shift



def retrieve_raw_latents(vae, images, sample_mode: str = "mode", generator=None, return_posterior: bool = False):
    posterior = vae.encode(images)
    dist = posterior.latent_dist if hasattr(posterior, "latent_dist") else posterior
    mode = str(sample_mode).lower()
    if mode in {"mode", "mean", "argmax", "deterministic"}:
        raw_latents = dist.mode()
    elif mode == "sample":
        if generator is None:
            raw_latents = dist.sample()
        else:
            raw_latents = dist.sample(generator)
    else:
        raise ValueError(f"Unsupported sample_mode: {sample_mode}")

    if return_posterior:
        return raw_latents, posterior
    return raw_latents



def encode_to_latents(vae, images, sample_mode: str = "mode", generator=None, return_posterior: bool = False):
    """Encode pixel-space images to SD3 latent space.

    Diffusers' SD3 img2img path uses:
      encoded = retrieve_latents(vae.encode(image))
      latents = (encoded - vae.config.shift_factor) * vae.config.scaling_factor
    This helper centralizes the same convention so custom VAE evaluation and the
    pipeline codepath use consistent latent preprocessing.
    """
    raw_latents, posterior = retrieve_raw_latents(
        vae,
        images,
        sample_mode=sample_mode,
        generator=generator,
        return_posterior=True,
    )
    scaling, shift = get_latent_scaling_config(vae)
    latents = (raw_latents - shift) * scaling
    if return_posterior:
        return latents, posterior
    return latents



def decode_from_latents(vae, latents, *, return_dict: bool = False):
    scaling, shift = get_latent_scaling_config(vae)
    decoder_input = (latents / scaling) + shift
    out = vae.decode(decoder_input, return_dict=return_dict)
    if return_dict:
        return out
    if isinstance(out, tuple):
        return out[0]
    if hasattr(out, "sample"):
        return out.sample
    return out



def get_vae_scale_factor(vae) -> int:
    block_out = getattr(vae.config, "block_out_channels", None)
    if block_out is None:
        return 8
    return 2 ** (len(block_out) - 1)



def estimate_latent_moments_from_loader(
    vae,
    loader,
    *,
    device=None,
    sample_mode: str = "mode",
    max_batches: int | None = None,
) -> dict[str, Any]:
    import torch

    vae.eval()
    if device is None:
        try:
            device = next(vae.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    total_elems = 0.0
    total_sum = 0.0
    total_sq = 0.0
    channel_sum = None
    channel_sq = None
    channel_elems = 0.0
    num_batches = 0
    num_samples = 0

    with torch.no_grad():
        for batch_idx, (images, _labels, _paths) in enumerate(loader):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            images = images.to(device=device, dtype=torch.float32, non_blocking=(device.type == "cuda"))
            raw = retrieve_raw_latents(vae, images, sample_mode=sample_mode)
            raw_f = raw.float()

            total_elems += float(raw_f.numel())
            total_sum += float(raw_f.sum().item())
            total_sq += float(raw_f.square().sum().item())

            csum = raw_f.sum(dim=(0, 2, 3))
            csq = raw_f.square().sum(dim=(0, 2, 3))
            if channel_sum is None:
                channel_sum = csum.detach().cpu()
                channel_sq = csq.detach().cpu()
            else:
                channel_sum += csum.detach().cpu()
                channel_sq += csq.detach().cpu()
            channel_elems += float(raw_f.shape[0] * raw_f.shape[2] * raw_f.shape[3])
            num_batches += 1
            num_samples += int(raw_f.shape[0])

    if total_elems <= 0 or channel_sum is None or channel_sq is None or channel_elems <= 0:
        raise RuntimeError("No latent statistics were accumulated; check the loader or max_batches setting.")

    mean = total_sum / total_elems
    var = max(total_sq / total_elems - mean * mean, 0.0)
    std = math.sqrt(var + 1.0e-12)

    cmean = (channel_sum / channel_elems).numpy().astype(float).tolist()
    cvar_tensor = torch.clamp(channel_sq / channel_elems - (channel_sum / channel_elems).pow(2), min=0.0)
    cstd = cvar_tensor.sqrt().numpy().astype(float).tolist()

    current_scaling, current_shift = get_latent_scaling_config(vae)
    recommended_shift = float(mean)
    recommended_scaling = float(1.0 / max(std, 1.0e-6))
    normalized_mean = (mean - recommended_shift) * recommended_scaling
    normalized_std = std * recommended_scaling

    return {
        "num_batches": int(num_batches),
        "num_samples": int(num_samples),
        "sample_mode": str(sample_mode),
        "global_mean": float(mean),
        "global_std": float(std),
        "per_channel_mean": cmean,
        "per_channel_std": cstd,
        "current_shift_factor": float(current_shift),
        "current_scaling_factor": float(current_scaling),
        "recommended_shift_factor": float(recommended_shift),
        "recommended_scaling_factor": float(recommended_scaling),
        "normalized_global_mean_after_recommendation": float(normalized_mean),
        "normalized_global_std_after_recommendation": float(normalized_std),
    }



def apply_latent_stats_to_vae_config(vae, *, shift: float, scaling: float):
    setattr(vae.config, "shift_factor", float(shift))
    setattr(vae.config, "scaling_factor", float(scaling))
    return vae

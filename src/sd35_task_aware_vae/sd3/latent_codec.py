from __future__ import annotations

from typing import Any


def get_latent_scaling_config(vae) -> tuple[float, float]:
    scaling = float(getattr(vae.config, "scaling_factor", 1.0))
    shift = float(getattr(vae.config, "shift_factor", 0.0))
    return scaling, shift


def encode_to_latents(vae, images, sample_mode: str = "mode", generator=None, return_posterior: bool = False):
    """Encode pixel-space images to SD3 latent space.

    Diffusers' SD3 img2img path uses:
      encoded = retrieve_latents(vae.encode(image))
      latents = (encoded - vae.config.shift_factor) * vae.config.scaling_factor
    This helper centralizes the same convention so custom VAE evaluation and the
    pipeline codepath use consistent latent preprocessing.
    """
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

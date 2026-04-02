from __future__ import annotations

__all__ = ["train_sd35_vae_from_config"]



def __getattr__(name: str):
    if name == "train_sd35_vae_from_config":
        from .trainer import train_sd35_vae_from_config

        return train_sd35_vae_from_config
    raise AttributeError(name)

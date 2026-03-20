from __future__ import annotations

from typing import Literal


def reconstruction_loss(pred, target, kind: str = "l1"):
    import torch.nn.functional as F

    mode = str(kind).lower()
    if mode in {"l1", "mae"}:
        return F.l1_loss(pred, target)
    if mode in {"l2", "mse"}:
        return F.mse_loss(pred, target)
    if mode in {"smooth_l1", "huber"}:
        return F.smooth_l1_loss(pred, target)
    raise ValueError(f"Unsupported reconstruction loss: {kind}")



def posterior_kl_loss(posterior):
    """Compute KL(q(z|x) || N(0, I)) for diffusers' posterior objects."""
    import torch

    obj = posterior.latent_dist if hasattr(posterior, "latent_dist") else posterior
    if hasattr(obj, "kl"):
        kl = obj.kl()
        if isinstance(kl, torch.Tensor):
            return kl.mean() if kl.ndim > 0 else kl

    if not (hasattr(obj, "mean") and hasattr(obj, "logvar")):
        raise RuntimeError("posterior does not expose mean/logvar; cannot compute KL")
    mean = obj.mean
    logvar = obj.logvar
    kl = 0.5 * (mean.pow(2) + logvar.exp() - 1.0 - logvar)
    dims = tuple(range(1, kl.ndim))
    if dims:
        kl = kl.sum(dim=dims)
    return kl.mean()



def feature_distance(x, y, kind: str = "mse"):
    import torch.nn.functional as F

    mode = str(kind).lower()
    if mode in {"mse", "l2"}:
        return F.mse_loss(x, y)
    if mode in {"l1", "mae"}:
        return F.l1_loss(x, y)
    if mode in {"cos", "cosine"}:
        x_n = F.normalize(x.flatten(1), dim=1)
        y_n = F.normalize(y.flatten(1), dim=1)
        return 1.0 - (x_n * y_n).sum(dim=1).mean()
    raise ValueError(f"Unsupported feature distance: {kind}")

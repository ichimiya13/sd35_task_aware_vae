from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal



def _reduce_loss(loss, reduction: str = "mean"):
    import torch

    mode = str(reduction).lower()
    if mode == "none":
        return loss
    if mode == "sum":
        return loss.sum()
    if mode == "mean":
        return loss.mean()
    raise ValueError(f"Unsupported reduction: {reduction}")



def charbonnier_loss(pred, target, epsilon: float = 1.0e-3, reduction: str = "mean"):
    import torch

    loss = torch.sqrt((pred - target).pow(2) + float(epsilon) ** 2)
    return _reduce_loss(loss, reduction=reduction)



def reconstruction_loss(pred, target, kind: str = "l1", reduction: str = "mean", epsilon: float = 1.0e-3):
    import torch.nn.functional as F

    mode = str(kind).lower()
    if mode in {"l1", "mae"}:
        return F.l1_loss(pred, target, reduction=reduction)
    if mode in {"l2", "mse"}:
        return F.mse_loss(pred, target, reduction=reduction)
    if mode in {"smooth_l1", "huber"}:
        return F.smooth_l1_loss(pred, target, reduction=reduction)
    if mode in {"charbonnier", "charb"}:
        return charbonnier_loss(pred, target, epsilon=epsilon, reduction=reduction)
    raise ValueError(f"Unsupported reconstruction loss: {kind}")



def weighted_reconstruction_loss(
    pred,
    target,
    *,
    weight_map,
    kind: str = "l1",
    epsilon: float = 1.0e-3,
    normalize: bool = True,
):
    import torch

    if weight_map is None:
        return reconstruction_loss(pred, target, kind=kind, reduction="mean", epsilon=epsilon)

    base = reconstruction_loss(pred, target, kind=kind, reduction="none", epsilon=epsilon)
    if weight_map.ndim == 3:
        weight_map = weight_map.unsqueeze(1)
    weight_map = weight_map.to(device=base.device, dtype=base.dtype)
    while weight_map.ndim < base.ndim:
        weight_map = weight_map.unsqueeze(1)

    weighted = base * weight_map
    if normalize:
        denom = torch.clamp(weight_map.sum() * max(1, base.shape[1]), min=1.0)
        return weighted.sum() / denom
    return weighted.mean()



def _sobel_gradients(x):
    import torch
    import torch.nn.functional as F

    c = x.shape[1]
    kernel_x = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], device=x.device, dtype=x.dtype)
    kernel_y = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]], device=x.device, dtype=x.dtype)
    kernel_x = kernel_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    kernel_y = kernel_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    grad_x = F.conv2d(x, kernel_x, padding=1, groups=c)
    grad_y = F.conv2d(x, kernel_y, padding=1, groups=c)
    return grad_x, grad_y



def _laplacian_response(x):
    import torch
    import torch.nn.functional as F

    c = x.shape[1]
    kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=x.device, dtype=x.dtype)
    kernel = kernel.view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    return F.conv2d(x, kernel, padding=1, groups=c)



def gradient_loss(
    pred,
    target,
    *,
    kind: str = "sobel_l1",
    weight_map=None,
    normalize: bool = True,
):
    import torch

    mode = str(kind).lower()
    if mode in {"sobel_l1", "sobel"}:
        px, py = _sobel_gradients(pred)
        tx, ty = _sobel_gradients(target)
        loss_map = (px - tx).abs() + (py - ty).abs()
    elif mode in {"sobel_l2", "sobel_mse"}:
        px, py = _sobel_gradients(pred)
        tx, ty = _sobel_gradients(target)
        loss_map = (px - tx).pow(2) + (py - ty).pow(2)
    elif mode in {"laplacian_l1", "lap_l1"}:
        loss_map = (_laplacian_response(pred) - _laplacian_response(target)).abs()
    elif mode in {"laplacian_l2", "lap_l2", "laplacian_mse"}:
        loss_map = (_laplacian_response(pred) - _laplacian_response(target)).pow(2)
    else:
        raise ValueError(f"Unsupported gradient loss: {kind}")

    if weight_map is None:
        return loss_map.mean()

    if weight_map.ndim == 3:
        weight_map = weight_map.unsqueeze(1)
    weight_map = weight_map.to(device=loss_map.device, dtype=loss_map.dtype)
    while weight_map.ndim < loss_map.ndim:
        weight_map = weight_map.unsqueeze(1)

    weighted = loss_map * weight_map
    if normalize:
        denom = torch.clamp(weight_map.sum() * max(1, loss_map.shape[1]), min=1.0)
        return weighted.sum() / denom
    return weighted.mean()



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
    if mode in {"smooth_l1", "huber"}:
        return F.smooth_l1_loss(x, y)
    if mode in {"cos", "cosine"}:
        x_n = F.normalize(x.flatten(1), dim=1)
        y_n = F.normalize(y.flatten(1), dim=1)
        return 1.0 - (x_n * y_n).sum(dim=1).mean()
    raise ValueError(f"Unsupported feature distance: {kind}")


@dataclass
class LPIPSLoss:
    net: str = "alex"
    lpips: bool = True
    spatial: bool = False

    def __post_init__(self) -> None:
        try:
            import lpips  # type: ignore
        except Exception as e:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "lpips is required when loss.lpips.weight > 0. Install it with `pip install lpips`."
            ) from e

        self.module = lpips.LPIPS(net=self.net, lpips=self.lpips, spatial=self.spatial)
        self.module.eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    def to(self, device):
        self.module.to(device)
        return self

    def __call__(self, pred, target):
        out = self.module(pred, target)
        if hasattr(out, "mean"):
            return out.mean()
        return out

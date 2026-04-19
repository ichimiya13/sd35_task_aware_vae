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




def _latent_tokens_for_distribution(latents, max_tokens: int | None = None):
    """Flatten latent maps to [N, C] tokens and optionally subsample tokens.

    SD3.5 latents have a small channel dimension but a large spatial grid.
    Matching covariance / Gram matrices over a random token subset keeps the
    loss cheap while still constraining the aggregated latent distribution.
    """
    import torch

    if latents.ndim != 4:
        raise ValueError(f"Expected latents with shape [B, C, H, W], got {tuple(latents.shape)}")
    tokens = latents.permute(0, 2, 3, 1).reshape(-1, latents.shape[1]).float()
    if max_tokens is not None and int(max_tokens) > 0 and tokens.shape[0] > int(max_tokens):
        idx = torch.randperm(tokens.shape[0], device=tokens.device)[: int(max_tokens)]
        tokens = tokens.index_select(0, idx)
    return tokens



def latent_covariance_gram_loss(
    latents,
    reference_latents,
    *,
    kind: str = "mean_covariance",
    max_tokens: int | None = 4096,
    include_mean: bool | None = None,
    mean_weight: float = 1.0,
    matrix_weight: float = 1.0,
    eps: float = 1.0e-6,
):
    """Match current latent distribution to a reference latent distribution.

    ``latents`` and ``reference_latents`` are expected to be the scaled latents
    passed to the DiT.  Supported ``kind`` values are:

    - ``covariance`` / ``mean_covariance``: centered channel covariance.
    - ``gram`` / ``mean_gram``: uncentered second-moment Gram matrix.
    - ``correlation`` / ``mean_correlation``: normalized covariance.
    - ``mean`` / ``mean_only``: channel mean only.

    The loss compares [C, C] channel statistics of sampled latent tokens.
    """
    import torch
    import torch.nn.functional as F

    mode = str(kind).lower()
    if include_mean is None:
        include_mean = mode.startswith("mean_") or mode in {"mean", "mean_only"}
    mode = mode.replace("mean_", "")

    x = _latent_tokens_for_distribution(latents, max_tokens=max_tokens)
    y = _latent_tokens_for_distribution(reference_latents.detach(), max_tokens=max_tokens)
    n = min(x.shape[0], y.shape[0])
    if n < 2:
        return latents.new_tensor(0.0)
    x = x[:n]
    y = y[:n]

    mu_x = x.mean(dim=0)
    mu_y = y.mean(dim=0)
    loss = torch.zeros((), device=latents.device, dtype=torch.float32)
    if include_mean or mode in {"mean", "mean_only"}:
        loss = loss + float(mean_weight) * F.mse_loss(mu_x, mu_y)
    if mode in {"mean", "mean_only"}:
        return loss.to(dtype=latents.dtype)

    if mode in {"cov", "covariance"}:
        xc = x - mu_x
        yc = y - mu_y
        mat_x = xc.transpose(0, 1).matmul(xc) / float(max(1, n - 1))
        mat_y = yc.transpose(0, 1).matmul(yc) / float(max(1, n - 1))
    elif mode in {"gram", "second_moment", "second-moment"}:
        mat_x = x.transpose(0, 1).matmul(x) / float(n)
        mat_y = y.transpose(0, 1).matmul(y) / float(n)
    elif mode in {"corr", "correlation"}:
        xc = x - mu_x
        yc = y - mu_y
        cov_x = xc.transpose(0, 1).matmul(xc) / float(max(1, n - 1))
        cov_y = yc.transpose(0, 1).matmul(yc) / float(max(1, n - 1))
        std_x = torch.sqrt(torch.diag(cov_x).clamp_min(float(eps)))
        std_y = torch.sqrt(torch.diag(cov_y).clamp_min(float(eps)))
        mat_x = cov_x / (std_x[:, None] * std_x[None, :] + float(eps))
        mat_y = cov_y / (std_y[:, None] * std_y[None, :] + float(eps))
    else:
        raise ValueError(f"Unsupported latent distribution matching kind: {kind}")

    loss = loss + float(matrix_weight) * F.mse_loss(mat_x, mat_y)
    return loss.to(dtype=latents.dtype)


def patch_reconstruction_loss(
    pred,
    target,
    *,
    kind: str = "l1",
    epsilon: float = 1.0e-3,
    crop_size: int | None = None,
    crop_ratio: float | None = None,
    center_x: float = 0.5,
    center_y: float = 0.5,
):
    """Compute reconstruction loss on a single spatial crop shared by pred and target.

    The crop is defined on the resized training image. `center_x` and `center_y` are
    normalized coordinates in [0, 1] where 0.5 corresponds to the image center.
    """
    import math

    h = int(pred.shape[-2])
    w = int(pred.shape[-1])
    if crop_size is None:
        ratio = float(0.5 if crop_ratio is None else crop_ratio)
        patch = max(1, int(round(min(h, w) * ratio)))
    else:
        patch = max(1, int(crop_size))
    patch_h = min(h, patch)
    patch_w = min(w, patch)

    cx = float(center_x) * max(0, w - 1)
    cy = float(center_y) * max(0, h - 1)
    left = int(round(cx - patch_w / 2.0))
    top = int(round(cy - patch_h / 2.0))
    left = min(max(left, 0), max(0, w - patch_w))
    top = min(max(top, 0), max(0, h - patch_h))

    pred_patch = pred[..., top : top + patch_h, left : left + patch_w]
    target_patch = target[..., top : top + patch_h, left : left + patch_w]
    return reconstruction_loss(pred_patch, target_patch, kind=kind, reduction="mean", epsilon=epsilon)




def build_spatial_weight_map(batch, cfg: dict | None):
    import torch

    cfg = cfg or {}
    mode = str(cfg.get("mode", "none")).lower()
    if mode in {"none", "off", "disabled"}:
        return None

    supported_peripheral = {"peripheral", "radial", "periphery"}
    supported_center = {"center_gaussian", "gaussian_center", "center", "gaussian"}
    if mode not in supported_peripheral | supported_center:
        raise ValueError(f"Unsupported weight_map.mode: {mode}")

    b, _c, h, w = batch.shape
    dtype = batch.dtype
    device = batch.device
    yy = torch.linspace(0.0, 1.0, steps=h, device=device, dtype=dtype).view(h, 1).expand(h, w)
    xx = torch.linspace(0.0, 1.0, steps=w, device=device, dtype=dtype).view(1, w).expand(h, w)

    gamma = float(cfg.get("gamma", 1.0))
    min_w = float(cfg.get("min_weight", 1.0))
    max_w = float(cfg.get("max_weight", 1.0))

    if mode in supported_peripheral:
        rr = torch.sqrt((xx - 0.5).pow(2) + (yy - 0.5).pow(2)) / math.sqrt(0.5)
        rr = rr.clamp(0.0, 1.0)
        inner = float(cfg.get("inner_radius", 0.0))
        if inner > 0:
            rr = ((rr - inner) / max(1.0e-6, 1.0 - inner)).clamp(0.0, 1.0)
        if gamma != 1.0:
            rr = rr.pow(gamma)
        base = min_w + (max_w - min_w) * rr
    else:
        center_x = float(cfg.get("center_x", 0.5))
        center_y = float(cfg.get("center_y", 0.5))
        sigma_default = float(cfg.get("sigma", 0.22))
        sigma_x = cfg.get("sigma_x", None)
        sigma_y = cfg.get("sigma_y", None)
        sigma_x = float(sigma_default if sigma_x in {None, '', 0, '0'} else sigma_x)
        sigma_y = float(sigma_default if sigma_y in {None, '', 0, '0'} else sigma_y)
        sigma_x = max(1.0e-4, sigma_x)
        sigma_y = max(1.0e-4, sigma_y)

        dx = (xx - center_x) / sigma_x
        dy = (yy - center_y) / sigma_y
        base_score = torch.exp(-0.5 * (dx.pow(2) + dy.pow(2)))
        if gamma != 1.0:
            base_score = base_score.pow(gamma)
        base = min_w + (max_w - min_w) * base_score

    weight_map = base.unsqueeze(0).unsqueeze(0).expand(b, 1, h, w).clone()

    if bool(cfg.get("apply_retina_mask", True)):
        x01 = (batch.clamp(-1.0, 1.0) + 1.0) / 2.0
        retina = (x01.mean(dim=1, keepdim=True) > float(cfg.get("retina_threshold", 0.03))).to(dtype=dtype)
        weight_map = 1.0 + retina * (weight_map - 1.0)

    return weight_map


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

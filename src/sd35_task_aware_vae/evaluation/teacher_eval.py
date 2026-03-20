from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from src.sd35_task_aware_vae.teacher_classifier.metrics import compute_multilabel_metrics


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


def choose_global_threshold_macro_f1(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    grid_start: float = 0.01,
    grid_end: float = 0.99,
    grid_num: int = 199,
) -> dict[str, Any]:
    y_true = y_true.astype(np.int32)
    thresholds = np.linspace(grid_start, grid_end, grid_num)

    best_t = 0.5
    best_score = -1.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(np.int32)
        tp = ((y_pred == 1) & (y_true == 1)).sum(axis=0).astype(np.float64)
        fp = ((y_pred == 1) & (y_true == 0)).sum(axis=0).astype(np.float64)
        fn = ((y_pred == 0) & (y_true == 1)).sum(axis=0).astype(np.float64)
        f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
        score = float(f1.mean())
        if (score > best_score) or (abs(score - best_score) < 1e-12 and t > best_t):
            best_t = float(t)
            best_score = score

    return {
        "best_threshold": float(best_t),
        "best_macro_f1": float(best_score),
        "grid": {"start": grid_start, "end": grid_end, "num": int(grid_num)},
    }


def bernoulli_kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    q = np.clip(q, eps, 1.0 - eps)
    return p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))


def compute_agreement_metrics(
    p_ref: np.ndarray,
    p_cmp: np.ndarray,
    *,
    threshold: float,
    embeddings_ref: np.ndarray | None = None,
    embeddings_cmp: np.ndarray | None = None,
) -> dict[str, Any]:
    dp = np.abs(p_cmp - p_ref)
    kl = bernoulli_kl(p_ref, p_cmp)
    yhat_ref = (p_ref >= threshold).astype(np.int32)
    yhat_cmp = (p_cmp >= threshold).astype(np.int32)
    flip = (yhat_ref != yhat_cmp).astype(np.float32)

    out: dict[str, Any] = {
        "mean_abs_dp": float(dp.mean()),
        "mean_kl": float(kl.mean()),
        "flip_rate": float(flip.mean()),
        "mean_abs_dp_per_class": dp.mean(axis=0).astype(np.float64),
        "mean_kl_per_class": kl.mean(axis=0).astype(np.float64),
        "flip_rate_per_class": flip.mean(axis=0).astype(np.float64),
    }

    if embeddings_ref is not None and embeddings_cmp is not None:
        num = (embeddings_ref * embeddings_cmp).sum(axis=1)
        den = np.linalg.norm(embeddings_ref, axis=1) * np.linalg.norm(embeddings_cmp, axis=1) + 1e-12
        out["mean_cosine"] = float(np.mean(num / den))
    return out


def build_per_label_rows(
    *,
    class_names: Sequence[str],
    metrics_real: dict[str, Any],
    metrics_by_key: dict[str, dict[str, Any]],
    agreement_by_key: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    num_classes = len(class_names)
    real_auroc = metrics_real.get("per_class_auroc", [None] * num_classes)
    real_auprc = metrics_real.get("per_class_auprc", [None] * num_classes)

    for idx, class_name in enumerate(class_names):
        row: dict[str, Any] = {
            "class": class_name,
            "auroc_real": real_auroc[idx],
            "auprc_real": real_auprc[idx],
        }
        for key, metric in metrics_by_key.items():
            row[f"auroc_{key}"] = metric.get("per_class_auroc", [None] * num_classes)[idx]
            row[f"auprc_{key}"] = metric.get("per_class_auprc", [None] * num_classes)[idx]
            agree = agreement_by_key[key]
            row[f"mean_abs_dp_{key}"] = float(agree["mean_abs_dp_per_class"][idx])
            row[f"mean_kl_{key}"] = float(agree["mean_kl_per_class"][idx])
            row[f"flip_rate_{key}"] = float(agree["flip_rate_per_class"][idx])
        rows.append(row)
    return rows


def compute_gt_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    return compute_multilabel_metrics(y_true, y_prob)

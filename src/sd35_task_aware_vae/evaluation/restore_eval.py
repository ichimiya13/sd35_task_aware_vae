from __future__ import annotations

import re
from typing import Any, Sequence

import numpy as np

from src.sd35_task_aware_vae.evaluation.teacher_eval import (
    build_per_label_rows,
    choose_global_threshold_macro_f1,
    compute_agreement_metrics,
    compute_gt_metrics,
)


def sanitize_id(path: str, max_len: int = 120) -> str:
    s = path.replace("\\", "/")
    s = re.sub(r"[^0-9A-Za-z._/-]+", "_", s)
    s = s.strip("/_")
    if len(s) > max_len:
        s = s[-max_len:]
    return s.replace("/", "__")



def summarize_restore_results(
    *,
    class_names: Sequence[str],
    y_true: np.ndarray,
    p_real: np.ndarray,
    p_by_key: dict[str, np.ndarray],
    embeddings_real: np.ndarray | None = None,
    embeddings_by_key: dict[str, np.ndarray | None] | None = None,
    threshold_cfg: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    threshold_cfg = threshold_cfg or {}
    thr_mode = str(threshold_cfg.get("mode", "search_on_real_val")).lower()

    if thr_mode in {"fixed", "value"}:
        threshold = float(threshold_cfg.get("value", 0.5))
        thr_info = {"mode": "fixed", "best_threshold": threshold}
    else:
        grid = threshold_cfg.get("grid", {}) or {}
        thr_info = choose_global_threshold_macro_f1(
            y_true,
            p_real,
            grid_start=float(grid.get("start", 0.01)),
            grid_end=float(grid.get("end", 0.99)),
            grid_num=int(grid.get("num", 199)),
        )
        threshold = float(thr_info["best_threshold"])

    metrics_real = compute_gt_metrics(y_true, p_real)
    metrics_by_key: dict[str, dict[str, Any]] = {}
    agreement_by_key: dict[str, dict[str, Any]] = {}
    t_rows: list[dict[str, Any]] = []

    embeddings_by_key = embeddings_by_key or {}
    for key, prob in p_by_key.items():
        metrics_by_key[key] = compute_gt_metrics(y_true, prob)
        agreement_by_key[key] = compute_agreement_metrics(
            p_real,
            prob,
            threshold=threshold,
            embeddings_ref=embeddings_real,
            embeddings_cmp=embeddings_by_key.get(key),
        )
        row = {
            "key": key,
            "macro_auroc": metrics_by_key[key].get("macro_auroc", None),
            "macro_auprc": metrics_by_key[key].get("macro_auprc", None),
            "mean_abs_dp": agreement_by_key[key].get("mean_abs_dp", None),
            "mean_kl": agreement_by_key[key].get("mean_kl", None),
            "flip_rate": agreement_by_key[key].get("flip_rate", None),
        }
        if "mean_cosine" in agreement_by_key[key]:
            row["mean_cosine"] = agreement_by_key[key]["mean_cosine"]
        t_rows.append(row)

    per_label_rows = build_per_label_rows(
        class_names=class_names,
        metrics_real=metrics_real,
        metrics_by_key=metrics_by_key,
        agreement_by_key=agreement_by_key,
    )

    summary = {
        "num_samples": int(y_true.shape[0]),
        "num_classes": int(len(class_names)),
        "threshold": {"value": threshold, **thr_info},
        "teacher_metrics_real": metrics_real,
        "variants": {
            key: {
                "teacher_metrics": metrics_by_key[key],
                "agreement": {
                    k: (float(v) if isinstance(v, (float, int, np.floating)) else v)
                    for k, v in agreement_by_key[key].items()
                    if not k.endswith("_per_class")
                },
            }
            for key in p_by_key
        },
        "curve": t_rows,
    }
    return summary, per_label_rows, t_rows

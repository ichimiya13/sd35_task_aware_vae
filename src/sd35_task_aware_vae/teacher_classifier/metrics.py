from __future__ import annotations

import numpy as np



def compute_multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    c = y_true.shape[1]
    aurocs: list[float | None] = []
    aps: list[float | None] = []

    for idx in range(c):
        yt = y_true[:, idx]
        yp = y_prob[:, idx]
        if yt.max() == 0 or yt.min() == 1:
            aurocs.append(None)
            aps.append(None)
            continue
        aurocs.append(float(roc_auc_score(yt, yp)))
        aps.append(float(average_precision_score(yt, yp)))

    def safe_mean(xs):
        vals = [float(x) for x in xs if x is not None]
        return float(np.mean(vals)) if vals else None

    macro_auroc = safe_mean(aurocs)
    macro_ap = safe_mean(aps)
    return {
        "macro_auroc": macro_auroc,
        "macro_auprc": macro_ap,
        "macro_map": macro_ap,
        "per_class_auroc": aurocs,
        "per_class_auprc": aps,
        "per_class_ap": aps,
    }

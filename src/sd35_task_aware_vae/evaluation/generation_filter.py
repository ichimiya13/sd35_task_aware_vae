from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass
class GenerationFilterResult:
    keep_mask: np.ndarray
    teacher_prob: np.ndarray
    match_score: np.ndarray


def compute_expected_label_match(prob: np.ndarray, expected_labels: np.ndarray | None) -> np.ndarray:
    if expected_labels is None:
        return prob.max(axis=1)

    expected = expected_labels.astype(np.float32)
    denom = np.clip(expected.sum(axis=1), 1.0, None)
    return ((prob * expected).sum(axis=1) / denom).astype(np.float32)


def filter_generated_probabilities(
    prob: np.ndarray,
    *,
    expected_labels: np.ndarray | None = None,
    min_match_score: float = 0.5,
    min_max_probability: float = 0.0,
) -> GenerationFilterResult:
    score = compute_expected_label_match(prob, expected_labels)
    keep = (score >= float(min_match_score)) & (prob.max(axis=1) >= float(min_max_probability))
    return GenerationFilterResult(keep_mask=keep.astype(bool), teacher_prob=prob, match_score=score)


def run_teacher_filter(
    teacher,
    x_teacher,
    *,
    expected_labels: np.ndarray | None = None,
    min_match_score: float = 0.5,
    min_max_probability: float = 0.0,
):
    import torch

    with torch.no_grad():
        logits = teacher(x_teacher)
        prob = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
    return filter_generated_probabilities(
        prob,
        expected_labels=expected_labels,
        min_match_score=min_match_score,
        min_max_probability=min_max_probability,
    )

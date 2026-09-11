"""Evaluation utilities reproducing the paper's Section 4 analysis:
accuracy/precision/recall/F1 (Table 1), normalized confusion matrix
(Fig. 25), per-class ROC-AUC (Fig. 27).
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from spectronet.config import CLASSES


@dataclasses.dataclass
class EvaluationResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: np.ndarray  # normalized, rows=truth, cols=pred
    per_class_auc: Dict[str, float]


def evaluate_predictions(
    y_true_onehot: np.ndarray, y_pred_proba: np.ndarray, classes: List[str] = CLASSES
) -> EvaluationResult:
    y_true = np.argmax(y_true_onehot, axis=1)
    y_pred = np.argmax(y_pred_proba, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    cm_normalized = cm.astype(np.float64) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    per_class_auc = {}
    for i, cls in enumerate(classes):
        try:
            per_class_auc[cls] = roc_auc_score(y_true_onehot[:, i], y_pred_proba[:, i])
        except ValueError:
            per_class_auc[cls] = float("nan")

    return EvaluationResult(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        confusion=cm_normalized,
        per_class_auc=per_class_auc,
    )


def results_table(results: Dict[str, EvaluationResult]) -> pd.DataFrame:
    """Builds the paper's Table 1 style comparison across models."""
    rows = []
    for model_name, r in results.items():
        rows.append(
            {
                "Model": model_name,
                "Accuracy": round(r.accuracy, 2),
                "Precision": round(r.precision, 2),
                "Recall": round(r.recall, 2),
                "F1 Score": round(r.f1, 2),
            }
        )
    return pd.DataFrame(rows).sort_values("F1 Score", ascending=False).reset_index(drop=True)


def roc_points(y_true_onehot: np.ndarray, y_pred_proba: np.ndarray, class_index: int):
    fpr, tpr, _ = roc_curve(y_true_onehot[:, class_index], y_pred_proba[:, class_index])
    return fpr, tpr

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)


def expected_calibration_error(y_true: np.ndarray, prob_harmful: np.ndarray, bins: int = 10) -> float:
    pred = (prob_harmful >= 0.5).astype(int)
    conf = np.maximum(prob_harmful, 1.0 - prob_harmful)
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs((pred[mask] == y_true[mask]).mean() - conf[mask].mean())
    return float(ece)


def compute_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict:
    pred = prob.argmax(axis=1)
    p, r, f, _ = precision_recall_fscore_support(y_true, pred, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    harmful_fn = int(cm[1, 0])
    harmful_tp = int(cm[1, 1])
    out = {
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, pred)) if len(set(y_true)) > 1 else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "beneficial_precision": float(p[0]),
        "beneficial_recall": float(r[0]),
        "beneficial_f1": float(f[0]),
        "harmful_precision": float(p[1]),
        "harmful_recall": float(r[1]),
        "harmful_f1": float(f[1]),
        "harmful_false_negative_rate": float(harmful_fn / max(harmful_fn + harmful_tp, 1)),
        "brier_score": float(brier_score_loss(y_true, prob[:, 1])),
        "ece": expected_calibration_error(y_true, prob[:, 1]),
    }
    try:
        out["auroc"] = float(roc_auc_score(y_true, prob[:, 1]))
    except Exception:
        out["auroc"] = float("nan")
    try:
        out["auprc"] = float(average_precision_score(y_true, prob[:, 1]))
    except Exception:
        out["auprc"] = float("nan")
    out["confusion_matrix"] = cm.tolist()
    return out


def confusion_df(metrics: dict) -> pd.DataFrame:
    cm = metrics["confusion_matrix"]
    return pd.DataFrame(cm, index=["true_beneficial", "true_harmful"], columns=["pred_beneficial", "pred_harmful"])


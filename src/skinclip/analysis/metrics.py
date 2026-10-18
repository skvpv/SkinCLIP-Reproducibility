from __future__ import annotations

import numpy as np


def confusion(y: np.ndarray, pred: np.ndarray, K: int, w: np.ndarray | None = None) -> np.ndarray:
    w = np.ones(len(y)) if w is None else w
    return np.bincount(y * K + pred, weights=w, minlength=K * K).reshape(K, K)


def from_confusion(C: np.ndarray) -> dict:
    tp = np.diag(C)
    support = C.sum(1)
    predicted = C.sum(0)
    present = support > 0
    recall = np.divide(tp, support, out=np.zeros_like(tp, dtype=float), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp, dtype=float), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp, dtype=float),
                   where=(precision + recall) > 0)
    total = C.sum()
    spec = np.array([(total - support[k] - predicted[k] + tp[k]) / (total - support[k]) if total > support[k] else np.nan
                     for k in range(len(tp))])
    return {"bma": float(recall[present].mean()), "macro_f1": float(f1[present].mean()), "accuracy": float(tp.sum() / total),
            "recall": recall.tolist(), "precision": precision.tolist(), "f1": f1.tolist(), "specificity": spec.tolist(),
            "support": support.tolist()}


def classification_metrics(y: np.ndarray, probs: np.ndarray, w: np.ndarray | None = None, auroc: bool = True) -> dict:
    K = probs.shape[1]
    pred = probs.argmax(1)
    out = from_confusion(confusion(y, pred, K, w))
    if auroc:
        from sklearn.metrics import roc_auc_score

        aucs = []
        for k in range(K):
            pos = y == k
            if pos.any() and (~pos).any():
                aucs.append(roc_auc_score(pos, probs[:, k], sample_weight=w))
        out["macro_auroc"] = float(np.mean(aucs)) if aucs else float("nan")
    return out

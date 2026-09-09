"""Métricas e escolha de threshold orientadas ao problema (classe positiva = risco)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)


def resumo(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "recall_risco": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision_risco": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1_risco": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float((tp + tn) / (tp + tn + fp + fn)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def escolher_threshold(y_true, y_prob, recall_minimo: float = 0.85) -> float:
    """
    Menor custo para o gestor é deixar passar um município em risco (FN).
    Escolhe o maior threshold cujo recall da classe risco ainda é >= recall_minimo
    (ou seja, a melhor precisão possível dado o recall exigido).
    """
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve devolve len(thr) = len(rec) - 1
    ok = np.where(rec[:-1] >= recall_minimo)[0]
    if len(ok) == 0:
        return 0.5
    return float(thr[ok].max())


def tabela_cv(resultados: dict) -> pd.DataFrame:
    linhas = []
    for nome, r in resultados.items():
        linhas.append({
            "modelo": nome,
            "roc_auc": f"{r['roc_auc_mean']:.3f} ± {r['roc_auc_std']:.3f}",
            "pr_auc": f"{r['pr_auc_mean']:.3f} ± {r['pr_auc_std']:.3f}",
            "recall_risco@0.5": f"{r['recall_mean']:.3f} ± {r['recall_std']:.3f}",
            "f1_risco@0.5": f"{r['f1_mean']:.3f} ± {r['f1_std']:.3f}",
            "fit_s": f"{r['fit_time']:.1f}",
        })
    return pd.DataFrame(linhas)

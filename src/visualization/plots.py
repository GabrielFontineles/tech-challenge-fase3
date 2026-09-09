"""Gráficos padronizados do projeto (matplotlib/seaborn, sem dependência de dados)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,  # noqa: E402
                             confusion_matrix)
from sklearn.calibration import calibration_curve  # noqa: E402

sns.set_theme(style="whitegrid", context="talk", font_scale=0.8)
PALETA = {"risco": "#c0392b", "ok": "#2e86c1", "neutro": "#7f8c8d"}


def _salvar(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


def plot_resultados_modelo(cv_baselines: dict, y_te, prob_te, thr, nome, path: Path):
    fig, ax = plt.subplots(2, 2, figsize=(14, 11))
    nomes = list(cv_baselines)
    roc = [cv_baselines[n]["roc_auc_mean"] for n in nomes]
    pr = [cv_baselines[n]["pr_auc_mean"] for n in nomes]
    x = np.arange(len(nomes))
    ax[0, 0].bar(x - 0.2, roc, 0.4, label="ROC-AUC", color=PALETA["ok"])
    ax[0, 0].bar(x + 0.2, pr, 0.4, label="PR-AUC", color=PALETA["risco"])
    ax[0, 0].set_xticks(x, nomes, rotation=20, ha="right")
    ax[0, 0].set_ylim(0, 1)
    ax[0, 0].set_title("Comparação de modelos (CV 5-fold, treino)")
    ax[0, 0].legend()
    RocCurveDisplay.from_predictions(y_te, prob_te, ax=ax[0, 1], name=nome)
    ax[0, 1].plot([0, 1], [0, 1], "k--", lw=1)
    ax[0, 1].set_title("Curva ROC — teste")
    PrecisionRecallDisplay.from_predictions(y_te, prob_te, ax=ax[1, 0], name=nome)
    ax[1, 0].axhline(np.mean(y_te), ls="--", color=PALETA["neutro"], label="prevalência")
    ax[1, 0].set_title("Precision-Recall — teste")
    ax[1, 0].legend()
    cm = confusion_matrix(y_te, (np.asarray(prob_te) >= thr).astype(int))
    ConfusionMatrixDisplay(cm, display_labels=["Não risco", "Risco"]).plot(ax=ax[1, 1], colorbar=False, cmap="Blues")
    ax[1, 1].set_title(f"Matriz de confusão — teste (threshold={thr:.2f})")
    _salvar(fig, path)


def plot_calibracao(y_te, prob_bruta, prob_cal, path: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for p, lab, cor in ((prob_bruta, "sem calibração", PALETA["neutro"]), (prob_cal, "isotônica", PALETA["risco"])):
        fo, mp = calibration_curve(y_te, p, n_bins=10, strategy="quantile")
        ax.plot(mp, fo, marker="o", label=lab, color=cor)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("probabilidade prevista")
    ax.set_ylabel("fração observada em risco")
    ax.set_title("Calibração — teste")
    ax.legend()
    _salvar(fig, path)


def plot_matriz_transicao(df: pd.DataFrame, col_t: str, col_t1: str, corte: float, path: Path):
    """Quantos municípios mudaram de estado entre t e t+1 (hipótese H1)."""
    est_t = np.where(df[col_t] < corte, "risco (t)", "ok (t)")
    est_t1 = np.where(df[col_t1] < corte, "risco (t+1)", "ok (t+1)")
    tab = pd.crosstab(est_t, est_t1)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(tab, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
    ax.set_title(f"Transição de estado {col_t} → {col_t1} (corte {corte:.0f}%)")
    _salvar(fig, path)
    return tab


def plot_importancias(imp: pd.Series, path: Path, titulo="Importância por permutação (teste)", top=20):
    imp = imp.sort_values(ascending=True).tail(top)
    fig, ax = plt.subplots(figsize=(9, 0.4 * len(imp) + 1.5))
    ax.barh(imp.index, imp.values, color=PALETA["ok"])
    ax.set_title(titulo)
    _salvar(fig, path)

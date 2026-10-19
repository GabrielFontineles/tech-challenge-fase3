"""
Funcoes de visualizacao compartilhadas
Fase 3 — Tech Challenge FIAP

Graficos usados em mais de um ponto do projeto (treino, interpretabilidade
e notebooks). Cada funcao desenha em um eixo (ax) recebido como argumento,
o que permite combinar os graficos em figuras com varios paineis.
"""

import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)

ROTULOS_CLASSES = ["Nao risco", "Em risco"]


def plot_roc(ax, y_true, prob):
    """Curva ROC com o valor de ROC-AUC na legenda."""
    fpr, tpr, _ = roc_curve(y_true, prob)
    ax.plot(fpr, tpr, linewidth=2, label=f"ROC-AUC = {roc_auc_score(y_true, prob):.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set(title="Curva ROC", xlabel="Taxa de falsos positivos",
           ylabel="Taxa de verdadeiros positivos")
    ax.legend()
    return ax


def plot_precision_recall(ax, y_true, prob, recall_alvo=None):
    """Curva precision-recall, com linha opcional no recall alvo."""
    precisao, recall, _ = precision_recall_curve(y_true, prob)
    ax.plot(recall, precisao, linewidth=2, color="orange",
            label=f"PR-AUC = {average_precision_score(y_true, prob):.3f}")
    if recall_alvo is not None:
        ax.axvline(recall_alvo, color="red", linestyle="--", label=f"Recall alvo ({recall_alvo})")
    ax.set(title="Curva Precision-Recall", xlabel="Recall", ylabel="Precision")
    ax.legend()
    return ax


def plot_calibracao(ax, y_true, prob, n_bins=10):
    """Curva de calibracao: probabilidade prevista x fracao observada."""
    fracao, media = calibration_curve(y_true, prob, n_bins=n_bins)
    ax.plot(media, fracao, "o-", label="Modelo")
    ax.plot([0, 1], [0, 1], "k--", label="Calibracao perfeita")
    ax.set(title="Curva de calibracao", xlabel="Probabilidade prevista",
           ylabel="Fracao observada em risco")
    ax.legend()
    return ax


def plot_matriz_confusao(ax, y_true, pred, threshold=None):
    """Matriz de confusao anotada."""
    sns.heatmap(confusion_matrix(y_true, pred), annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=ROTULOS_CLASSES, yticklabels=ROTULOS_CLASSES)
    titulo = "Matriz de confusao"
    if threshold is not None:
        titulo += f" (threshold = {threshold:.3f})"
    ax.set(title=titulo, xlabel="Previsto", ylabel="Real")
    return ax


def plot_importancia(ax, rotulos, valores, titulo, xlabel, erros=None, cor="steelblue"):
    """Barras horizontais; recebe os itens em ordem decrescente e mostra o maior no topo."""
    rotulos, valores = list(rotulos)[::-1], list(valores)[::-1]
    erros = list(erros)[::-1] if erros is not None else None
    ax.barh(rotulos, valores, xerr=erros, color=cor, alpha=0.85)
    ax.set(title=titulo, xlabel=xlabel)
    return ax

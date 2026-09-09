"""
Interpretabilidade: importância por permutação + SHAP no modelo salvo.

    python -m src.evaluation.shap_analysis
    python -m src.evaluation.shap_analysis --sem-historico

Carrega models/modelo_final*.joblib (NÃO retreina) e usa o mesmo split do treino
(random_state em src/config.py), garantindo que o SHAP é calculado no conjunto de teste.
"""
from __future__ import annotations

import argparse

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from src import config as C  # noqa: E402
from src.preprocessing.features import separar_xy  # noqa: E402
from src.preprocessing.pipeline import nomes_features  # noqa: E402
from src.visualization.plots import plot_importancias  # noqa: E402


def carregar(sufixo: str):
    df = pd.read_parquet(C.PROCESSED_DIR / "dataset_modelagem.parquet")
    X, y = separar_xy(df, incluir_historico=(sufixo == ""))
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.RANDOM_STATE)
    modelo = joblib.load(C.MODELS_DIR / f"modelo_final{sufixo}.joblib")
    return df, modelo, X_tr, X_te, y_tr, y_te


def explicador(modelo, X_bg_proc: pd.DataFrame):
    clf = modelo.named_steps["clf"]
    nome = clf.__class__.__name__
    if nome in ("HistGradientBoostingClassifier", "RandomForestClassifier", "GradientBoostingClassifier"):
        return shap.TreeExplainer(clf, feature_perturbation="tree_path_dependent")
    if nome == "LogisticRegression":
        return shap.LinearExplainer(clf, X_bg_proc)
    return shap.Explainer(clf.predict_proba, shap.sample(X_bg_proc, 200, random_state=C.RANDOM_STATE))


def _valores_classe_risco(sv):
    """Normaliza a saída do SHAP (varia entre versões/modelos) para a classe positiva."""
    vals = sv.values if hasattr(sv, "values") else sv
    if isinstance(vals, list):
        return np.asarray(vals[1])
    vals = np.asarray(vals)
    if vals.ndim == 3:
        return vals[:, :, 1] if vals.shape[2] == 2 else vals[:, :, 0]
    return vals


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sem-historico", action="store_true")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args(argv)
    sufixo = "_sem_historico" if args.sem_historico else ""

    df, modelo, X_tr, X_te, y_tr, y_te = carregar(sufixo)
    pre = modelo.named_steps["pre"]
    nomes = nomes_features(modelo)
    X_te_proc = pd.DataFrame(np.asarray(pre.transform(X_te)), columns=nomes, index=X_te.index)
    X_tr_proc = pd.DataFrame(np.asarray(pre.transform(X_tr)), columns=nomes, index=X_tr.index)

    print("[1] importância por permutação (PR-AUC, teste)")
    pi = permutation_importance(modelo, X_te, y_te, scoring="average_precision", n_repeats=10,
                                random_state=C.RANDOM_STATE, n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=X_te.columns).sort_values(ascending=False)
    imp.to_csv(C.REPORTS_DIR / f"importancia_permutacao{sufixo}.csv", header=["queda_pr_auc"])
    print(imp.head(args.top).to_string())
    plot_importancias(imp, C.IMAGES_DIR / f"10_importancia_permutacao{sufixo}.png", top=args.top)

    print("\n[2] SHAP (teste)")
    exp = explicador(modelo, X_tr_proc)
    sv = exp(X_te_proc) if not isinstance(exp, shap.TreeExplainer) else exp(X_te_proc, check_additivity=False)
    vals = _valores_classe_risco(sv)
    shap_imp = pd.Series(np.abs(vals).mean(axis=0), index=nomes).sort_values(ascending=False)
    shap_imp.to_csv(C.REPORTS_DIR / f"importancia_shap{sufixo}.csv", header=["mean_abs_shap"])
    print(shap_imp.head(args.top).to_string())

    plt.figure()
    shap.summary_plot(vals, X_te_proc, feature_names=nomes, max_display=args.top, show=False)
    plt.title("SHAP — impacto na probabilidade de risco (teste)")
    plt.savefig(C.IMAGES_DIR / f"11_shap_summary{sufixo}.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(vals, X_te_proc, feature_names=nomes, plot_type="bar", max_display=args.top, show=False)
    plt.savefig(C.IMAGES_DIR / f"12_shap_bar{sufixo}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # dependence plots das 4 principais features numéricas
    top4 = [f for f in shap_imp.index if "missingindicator" not in f and not f.startswith("cat__")][:4]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, f in zip(axes.ravel(), top4):
        j = nomes.index(f)
        ax.scatter(X_te_proc[f], vals[:, j], s=8, alpha=0.5, color="#c0392b")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel(f + " (padronizado)")
        ax.set_ylabel("SHAP (→ risco)")
    fig.suptitle("SHAP dependence — 4 principais features")
    fig.tight_layout()
    fig.savefig(C.IMAGES_DIR / f"13_shap_dependence{sufixo}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # waterfall de 3 municípios: maior risco, limítrofe, e "falso negativo" mais grave
    prob = modelo.predict_proba(X_te)[:, 1]
    idx_alto = int(np.argmax(prob))
    idx_lim = int(np.argmin(np.abs(prob - 0.5)))
    fn_mask = (y_te.values == 1)
    idx_fn = int(np.argmin(np.where(fn_mask, prob, np.inf)))
    base_value = float(np.ravel(sv.base_values[0])[-1]) if hasattr(sv, "base_values") else 0.0
    for tag, i in (("maior_risco", idx_alto), ("limitrofe", idx_lim), ("risco_nao_capturado", idx_fn)):
        e = shap.Explanation(values=vals[i], base_values=base_value, data=X_te_proc.iloc[i].values, feature_names=nomes)
        plt.figure()
        shap.plots.waterfall(e, max_display=12, show=False)
        mid = int(df.loc[X_te.index[i], C.COL_ID])
        plt.title(f"Município {mid} — {tag} (p={prob[i]:.2f}, real={'risco' if y_te.iloc[i] else 'ok'})")
        plt.savefig(C.IMAGES_DIR / f"14_waterfall_{tag}{sufixo}.png", dpi=150, bbox_inches="tight")
        plt.close()
    print("✓ gráficos 10–14 gerados em images/")


if __name__ == "__main__":
    main()

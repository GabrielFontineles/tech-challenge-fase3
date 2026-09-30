"""
Interpretabilidade do modelo — v2.1
Fase 3 — Tech Challenge FIAP

1. SHAP Values do modelo base (sem calibracao), por feature e agregados por grupo
2. Importancia por permutacao em grupo, no conjunto de teste, usando o
   modelo calibrado (queda de PR-AUC ao embaralhar juntas as colunas do grupo)

Grupos de features:
- Historico educacional 2023 (Bloco A)
- Estado e regiao (sigla_uf, regiao)
- Porte e densidade (IBGE Censo 2022)
- Socioeconomico (Atlas 2010 e PIB 2021)

O script apenas calcula e exporta os resultados; a interpretacao
e feita a partir das tabelas geradas em reports/.

Uso:
    python src/evaluation/shap_analysis_v2.py
    python src/evaluation/shap_analysis_v2.py --sem-historico
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TARGET = "em_risco_2024"
N_REPETICOES = 10
DATASET = Path("data/processed/dataset_enriquecido_v2.parquet")
MODELS_DIR = Path("models")
IMAGES_DIR = Path("images")
REPORTS_DIR = Path("reports")


def carregar(sufixo):
    """Carrega modelos, metadata e reconstroi o mesmo split do treino."""
    with open(MODELS_DIR / f"metadata{sufixo}.json", encoding="utf-8") as f:
        metadata = json.load(f)
    modelo_base = joblib.load(MODELS_DIR / f"modelo_base{sufixo}.joblib")
    modelo_final = joblib.load(MODELS_DIR / f"modelo_final{sufixo}.joblib")

    numericas = metadata["features_numericas"]
    categoricas = metadata["features_categoricas"]
    df = pd.read_parquet(DATASET)
    X = df[numericas + categoricas]
    y = df[TARGET].astype(int)
    X_train, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    print(f"Modelo: {metadata['modelo']} | variante: {metadata['variante']}")
    print(f"Teste: {len(X_test)} municipios")
    return metadata, modelo_base, modelo_final, X_train, X_test, y_test


def grupo_da_feature(original, metadata):
    if original in ("sigla_uf", "regiao"):
        return "Estado e regiao"
    fontes = metadata["fontes_externas"]
    censo = next(v for k, v in fontes.items() if "Censo 2022" in k)
    if original in censo:
        return "Porte e densidade"
    socio = [c for k, v in fontes.items() if "Censo 2022" not in k for c in v]
    if original in socio:
        return "Socioeconomico"
    return "Historico educacional 2023"


def mapear_features(modelo_base, metadata):
    """Relaciona cada coluna transformada a sua feature original e grupo."""
    categoricas = metadata["features_categoricas"]
    linhas = []
    for nome in modelo_base.named_steps["pre"].get_feature_names_out():
        transformador, resto = nome.split("__", 1)
        if transformador == "num":
            original = resto.replace("missingindicator_", "", 1)
        else:
            original = next(c for c in categoricas if resto.startswith(c + "_"))
        linhas.append({"feature": resto, "feature_original": original,
                       "grupo": grupo_da_feature(original, metadata)})
    return pd.DataFrame(linhas)


def calcular_shap(modelo_base, X_train, X_test):
    pre = modelo_base.named_steps["pre"]
    clf = modelo_base.named_steps["clf"]
    X_train_proc = pre.transform(X_train)
    X_test_proc = pre.transform(X_test)

    if isinstance(clf, LogisticRegression):
        explainer = shap.LinearExplainer(clf, X_train_proc)
    else:
        explainer = shap.TreeExplainer(clf)
    valores = explainer.shap_values(X_test_proc)
    if isinstance(valores, list):
        valores = valores[1]
    valores = np.asarray(valores)
    if valores.ndim == 3:
        valores = valores[:, :, 1]
    return valores, X_test_proc


def permutacao_por_grupo(modelo_final, X_test, y_test, metadata):
    """Queda de PR-AUC ao embaralhar juntas as colunas originais de cada grupo."""
    colunas = metadata["features_numericas"] + metadata["features_categoricas"]
    grupos = {}
    for c in colunas:
        grupos.setdefault(grupo_da_feature(c, metadata), []).append(c)

    referencia = average_precision_score(y_test, modelo_final.predict_proba(X_test)[:, 1])
    rng = np.random.default_rng(RANDOM_STATE)
    linhas = []
    for grupo, cols in grupos.items():
        quedas = []
        for _ in range(N_REPETICOES):
            X_perm = X_test.copy()
            ordem = rng.permutation(len(X_perm))
            X_perm[cols] = X_test[cols].values[ordem]
            quedas.append(referencia - average_precision_score(
                y_test, modelo_final.predict_proba(X_perm)[:, 1]))
        linhas.append({"grupo": grupo, "n_features": len(cols),
                       "queda_pr_auc_media": float(np.mean(quedas)),
                       "queda_pr_auc_std": float(np.std(quedas))})
    tabela = pd.DataFrame(linhas).sort_values("queda_pr_auc_media", ascending=False)
    return tabela, referencia


def graficos(shap_values, X_test_proc, mapa, por_grupo, permutacao, sufixo, titulo):
    importancia = np.abs(shap_values).mean(axis=0)
    top = np.argsort(importancia)[::-1][:15]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top)), importancia[top][::-1], color="steelblue", alpha=0.85)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(mapa["feature"].values[top][::-1], fontsize=9)
    ax.set_xlabel("Media |SHAP| (log-odds)")
    ax.set_title(f"Top 15 features — SHAP\n{titulo}")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / f"16_shap_importancia_v21{sufixo}.png", dpi=150)
    plt.close()

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_proc, feature_names=list(mapa["feature"]),
                      max_display=15, show=False)
    plt.title(f"SHAP summary — {titulo}")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / f"17_shap_summary_v21{sufixo}.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].barh(por_grupo["grupo"][::-1], por_grupo["participacao_pct"][::-1],
                 color="steelblue", alpha=0.85)
    axes[0].set(title="Participacao no SHAP total (%)", xlabel="%")
    axes[1].barh(permutacao["grupo"][::-1], permutacao["queda_pr_auc_media"][::-1],
                 xerr=permutacao["queda_pr_auc_std"][::-1], color="darkorange", alpha=0.85)
    axes[1].set(title="Queda de PR-AUC ao embaralhar o grupo", xlabel="Queda de PR-AUC")
    fig.suptitle(f"Importancia por grupo de features — {titulo}")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / f"21_importancia_grupos_v21{sufixo}.png", dpi=150)
    plt.close()
    print(f"Graficos salvos com sufixo '_v21{sufixo}'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sem-historico", action="store_true")
    args = parser.parse_args()
    sufixo = "_sem_historico" if args.sem_historico else ""
    titulo = "modelo sem historico" if args.sem_historico else "modelo completo"

    print("=" * 60)
    print(f"INTERPRETABILIDADE v2.1 — {titulo.upper()}")
    print("=" * 60)
    metadata, modelo_base, modelo_final, X_train, X_test, y_test = carregar(sufixo)

    mapa = mapear_features(modelo_base, metadata)
    shap_values, X_test_proc = calcular_shap(modelo_base, X_train, X_test)
    mapa["media_abs_shap"] = np.abs(shap_values).mean(axis=0)

    por_grupo = (mapa.groupby("grupo")["media_abs_shap"].sum()
                 .sort_values(ascending=False).reset_index())
    por_grupo["participacao_pct"] = por_grupo["media_abs_shap"] / por_grupo["media_abs_shap"].sum() * 100

    permutacao, referencia = permutacao_por_grupo(modelo_final, X_test, y_test, metadata)

    REPORTS_DIR.mkdir(exist_ok=True)
    mapa.sort_values("media_abs_shap", ascending=False).to_csv(
        REPORTS_DIR / f"shap_importancia{sufixo}.csv", index=False)
    por_grupo.merge(permutacao, on="grupo").to_csv(
        REPORTS_DIR / f"importancia_grupos{sufixo}.csv", index=False)

    print("\nTop 15 features (media |SHAP|):")
    print(mapa.sort_values("media_abs_shap", ascending=False).head(15)
          [["feature", "grupo", "media_abs_shap"]].to_string(index=False))
    print("\nParticipacao por grupo no SHAP total:")
    print(por_grupo[["grupo", "participacao_pct"]].round(1).to_string(index=False))
    print(f"\nPermutacao por grupo (PR-AUC de referencia no teste: {referencia:.4f}):")
    print(permutacao.round(4).to_string(index=False))

    graficos(shap_values, X_test_proc, mapa, por_grupo, permutacao, sufixo, titulo)


if __name__ == "__main__":
    main()

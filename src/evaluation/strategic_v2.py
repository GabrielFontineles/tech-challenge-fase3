"""
Analise estrategica — v2.1
Fase 3 — Tech Challenge FIAP

Perguntas de negocio respondidas com o modelo calibrado (variante completa):
1. Quais municipios apresentam maior risco?
   Probabilidades out-of-fold (cada municipio avaliado por um modelo que
   nao o viu no treino), faixas de risco e validacao da calibracao por faixa.
2. Como o risco se distribui entre os estados?
3. Quais municipios possuem perfis estruturais semelhantes?
   K-Means sobre variaveis socioeconomicas e territoriais (sem o target);
   k entre 3 e 6, escolhido pela silhueta. O risco observado e usado apenas
   para descrever os grupos, depois do agrupamento.
4. Quais municipios podem nao atingir a meta de 2030?
   Cenario de tendencia: variacao media nacional observada 2023->2024,
   aplicada a partir da taxa de 2024, comparada a meta municipal de 2030.
5. Lista de priorizacao: alto risco previsto e ritmo necessario acima
   de 2x a tendencia nacional.

O script apenas calcula e exporta os resultados em reports/ e images/.
"""

import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TARGET = "em_risco_2024"
ANO_BASE, ANO_META = 2024, 2030
K_CANDIDATOS = range(3, 7)
DATASET = Path("data/processed/dataset_enriquecido_v2.parquet")
MODELS_DIR = Path("models")
IMAGES_DIR = Path("images")
REPORTS_DIR = Path("reports")

FEATURES_CLUSTER = [
    "log_populacao_2022", "densidade_2022", "log_pib_per_capita",
    "idhm_educacao", "idhm_renda", "idhm_longevidade", "gini",
    "pct_extremamente_pobres", "analfabetismo_15mais",
    "freq_escolar_4a5", "pct_rural",
]


def carregar():
    with open(MODELS_DIR / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    modelo = joblib.load(MODELS_DIR / "modelo_final.joblib")
    df = pd.read_parquet(DATASET)
    print(f"Modelo: {metadata['modelo']} (calibrado) | threshold: {metadata['threshold']:.3f}")
    print(f"Municipios: {len(df)}")
    return metadata, modelo, df


def probabilidades_out_of_fold(modelo, df, metadata):
    """Cada municipio recebe a probabilidade de um modelo treinado sem ele."""
    print("\nCalculando probabilidades out-of-fold (5 folds)...")
    colunas = metadata["features_numericas"] + metadata["features_categoricas"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_predict(clone(modelo), df[colunas], df[TARGET].astype(int),
                             cv=cv, method="predict_proba", n_jobs=-1)[:, 1]


def faixas_de_risco(df, threshold):
    """Faixas ancoradas no threshold e taxa de risco observada em cada uma."""
    if not 0.2 < threshold < 0.7:
        raise ValueError(f"Threshold fora do intervalo esperado para as faixas: {threshold}")
    df["faixa_risco"] = pd.cut(df["prob_risco"], bins=[0, 0.2, threshold, 0.7, 1.0001],
                               labels=["Baixo", "Moderado", "Alto", "Critico"],
                               include_lowest=True)
    tabela = df.groupby("faixa_risco", observed=True).agg(
        municipios=("id_municipio", "count"),
        prob_media=("prob_risco", "mean"),
        taxa_risco_observada=(TARGET, "mean"),
    ).reset_index()
    tabela["pct_municipios"] = tabela["municipios"] / len(df) * 100
    return tabela


def risco_por_uf(df, threshold):
    tabela = df.groupby("sigla_uf").agg(
        regiao=("regiao", "first"),
        municipios=("id_municipio", "count"),
        prob_media=("prob_risco", "mean"),
        acima_threshold=("prob_risco", lambda s: int((s >= threshold).sum())),
        taxa_risco_observada=(TARGET, "mean"),
    ).reset_index()
    tabela["pct_acima_threshold"] = tabela["acima_threshold"] / tabela["municipios"] * 100
    return tabela.sort_values("prob_media", ascending=False)


def clusters_estruturais(df):
    """K-Means sobre variaveis estruturais; k escolhido pela silhueta (3 a 6)."""
    preparo = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    Z = preparo.fit_transform(df[FEATURES_CLUSTER])

    silhuetas = {}
    for k in K_CANDIDATOS:
        rotulos = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(Z)
        silhuetas[k] = float(silhouette_score(Z, rotulos))
    k = max(silhuetas, key=silhuetas.get)
    df["cluster"] = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(Z)

    agregacoes = {"municipios": ("id_municipio", "count")}
    for c in FEATURES_CLUSTER:
        agregacoes[c] = (c, "median")
    agregacoes["taxa_risco_observada"] = (TARGET, "mean")
    agregacoes["prob_media"] = ("prob_risco", "mean")
    agregacoes["regiao_predominante"] = ("regiao", lambda s: s.value_counts().index[0])
    agregacoes["pct_regiao_predominante"] = (
        "regiao", lambda s: s.value_counts(normalize=True).iloc[0] * 100)
    perfil = df.groupby("cluster").agg(**agregacoes).reset_index()

    coordenadas = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(Z)
    return perfil, silhuetas, k, coordenadas


def projecao_2030(df):
    """Cenario de tendencia nacional e ritmo necessario para a meta municipal."""
    variacao = float((df["taxa_alf_2024"] - df["taxa_alf_2023"]).mean())
    anos = ANO_META - ANO_BASE
    df["taxa_projetada_2030"] = (df["taxa_alf_2024"] + variacao * anos).clip(upper=100)
    df["ritmo_necessario"] = (df["meta_2030"] - df["taxa_alf_2024"]) / anos

    def situacao(ritmo):
        if pd.isna(ritmo):
            return "Sem meta definida"
        if ritmo <= 0:
            return "Meta ja atingida"
        if ritmo <= variacao:
            return "Atinge no ritmo atual"
        if ritmo <= 2 * variacao:
            return "Precisa de ate 2x o ritmo"
        return "Precisa de mais de 2x o ritmo"

    df["situacao_2030"] = df["ritmo_necessario"].apply(situacao)
    return variacao


def priorizacao(df, threshold):
    candidatos = df[(df["prob_risco"] >= threshold) &
                    (df["situacao_2030"] == "Precisa de mais de 2x o ritmo")]
    colunas = ["nome_municipio", "id_municipio", "sigla_uf", "regiao",
               "taxa_alf_2023", "taxa_alf_2024", "prob_risco", "faixa_risco",
               "meta_2030", "ritmo_necessario", "cluster"]
    return candidatos.sort_values(["prob_risco", "ritmo_necessario"],
                                  ascending=False)[colunas]


def graficos(df, faixas, por_uf, perfil, coordenadas, k, threshold):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].hist(df["prob_risco"], bins=30, color="steelblue", edgecolor="white")
    axes[0].axvline(threshold, color="red", linestyle="--", label=f"Threshold ({threshold:.2f})")
    axes[0].set(title="Probabilidade de risco (out-of-fold, calibrada)",
                xlabel="Probabilidade", ylabel="Municipios")
    axes[0].legend()
    x = np.arange(len(faixas))
    axes[1].bar(x - 0.2, faixas["prob_media"], 0.4, label="Probabilidade media prevista")
    axes[1].bar(x + 0.2, faixas["taxa_risco_observada"], 0.4, label="Taxa de risco observada")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(faixas["faixa_risco"].astype(str))
    axes[1].set(title="Previsto vs. observado por faixa", ylabel="Proporcao")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "22_faixas_risco_v21.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 9))
    ordem = por_uf.sort_values("prob_media")
    ax.barh(ordem["sigla_uf"], ordem["prob_media"], color="indianred", alpha=0.85)
    ax.axvline(threshold, color="black", linestyle="--", label=f"Threshold ({threshold:.2f})")
    ax.set(title="Probabilidade media de risco por UF", xlabel="Probabilidade media")
    ax.legend()
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "23_risco_por_uf_v21.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    dispersao = axes[0].scatter(coordenadas[:, 0], coordenadas[:, 1], c=df["cluster"],
                                cmap="tab10", s=8, alpha=0.6)
    axes[0].set(title=f"Clusters estruturais (k={k}) — projecao PCA",
                xlabel="Componente 1", ylabel="Componente 2")
    axes[0].legend(*dispersao.legend_elements(), title="Cluster")
    axes[1].bar(perfil["cluster"].astype(str), perfil["taxa_risco_observada"],
                color="darkorange", alpha=0.85)
    axes[1].set(title="Taxa de risco observada em 2024 por cluster",
                xlabel="Cluster", ylabel="Proporcao em risco")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "24_clusters_estruturais_v21.png", dpi=150)
    plt.close()

    ordem_situacao = ["Meta ja atingida", "Atinge no ritmo atual", "Precisa de ate 2x o ritmo",
                      "Precisa de mais de 2x o ritmo", "Sem meta definida"]
    tabela = (pd.crosstab(df["regiao"], df["situacao_2030"], normalize="index") * 100)
    tabela = tabela[[c for c in ordem_situacao if c in tabela.columns]]
    tabela.plot(kind="barh", stacked=True, figsize=(12, 6), colormap="RdYlGn_r")
    plt.title("Situacao em relacao a meta de 2030 por regiao (%)")
    plt.xlabel("% dos municipios")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "25_projecao_2030_v21.png", dpi=150)
    plt.close()
    print("\nGraficos salvos: 22 a 25 (_v21)")


def main():
    print("=" * 60)
    print("ANALISE ESTRATEGICA v2.1")
    print("=" * 60)
    metadata, modelo, df = carregar()
    threshold = metadata["threshold"]

    df["prob_risco"] = probabilidades_out_of_fold(modelo, df, metadata)
    faixas = faixas_de_risco(df, threshold)
    por_uf = risco_por_uf(df, threshold)
    perfil, silhuetas, k, coordenadas = clusters_estruturais(df)
    variacao = projecao_2030(df)
    prioritarios = priorizacao(df, threshold)

    REPORTS_DIR.mkdir(exist_ok=True)
    df.sort_values("prob_risco", ascending=False)[
        ["nome_municipio", "id_municipio", "sigla_uf", "regiao", "prob_risco", "faixa_risco",
         "taxa_alf_2023", "taxa_alf_2024", TARGET, "meta_2030", "ritmo_necessario",
         "situacao_2030", "cluster"]
    ].to_csv(REPORTS_DIR / "ranking_risco_v21.csv", index=False)
    faixas.to_csv(REPORTS_DIR / "faixas_risco_v21.csv", index=False)
    por_uf.to_csv(REPORTS_DIR / "risco_por_uf_v21.csv", index=False)
    perfil.to_csv(REPORTS_DIR / "perfil_clusters_v21.csv", index=False)
    prioritarios.to_csv(REPORTS_DIR / "priorizacao_municipios_v21.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n[1] Faixas de risco (previsto vs. observado):")
    print(faixas.round(3).to_string(index=False))
    print("\n[2] Risco por UF (10 maiores probabilidades medias):")
    print(por_uf.head(10).round(3).to_string(index=False))
    print("\n[3] Silhueta por k:", {kk: round(v, 3) for kk, v in silhuetas.items()}, f"-> k = {k}")
    print("Perfil dos clusters (medianas):")
    print(perfil.round(2).T.to_string())
    print(f"\n[4] Variacao media nacional observada 2023->2024: {variacao:.2f} pontos/ano")
    print(df["situacao_2030"].value_counts().to_string())
    print(f"\n[5] Municipios prioritarios: {len(prioritarios)}")
    print(prioritarios.head(15)[["nome_municipio", "prob_risco", "taxa_alf_2024",
                                 "meta_2030", "ritmo_necessario", "cluster"]]
          .round(2).to_string(index=False))

    graficos(df, faixas, por_uf, perfil, coordenadas, k, threshold)


if __name__ == "__main__":
    main()

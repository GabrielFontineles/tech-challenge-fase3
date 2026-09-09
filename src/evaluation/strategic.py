"""
Aplicação estratégica: ranking de risco, clusters de similaridade, projeção de metas.

    python -m src.evaluation.strategic

Responde às perguntas de negócio do enunciado:
  - Quais municípios apresentam maior risco?           → reports/ranking_risco_municipios.csv
  - Quais regiões possuem padrões semelhantes?         → clusters (blocos B/C, sem a taxa) + perfil
  - Como prever municípios que não atingirão metas?    → probabilidade de risco + projeção linear até 2030
  - Lista de priorização (alto risco + baixo IDHM + alto gap)
Usa o modelo CALIBRADO salvo (probabilidades interpretáveis como frequência).
"""
from __future__ import annotations

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src import config as C  # noqa: E402
from src.preprocessing.features import separar_xy  # noqa: E402

ANO_META = 2030


def ranking_risco(df, modelo, thr):
    X, _ = separar_xy(df)
    prob = modelo.predict_proba(X)[:, 1]
    r = df[[C.COL_ID, "sigla_uf", "regiao", f"{C.COL_TAXA}_t", "taxa_alfabetizacao_t1", C.TARGET]].copy()
    r["prob_risco"] = prob
    r["classificado_risco"] = (prob >= thr).astype(int)
    r["faixa_risco"] = pd.cut(prob, [0, 0.25, 0.5, 0.75, 1.0], labels=["baixo", "moderado", "alto", "muito alto"], include_lowest=True)
    r = r.sort_values("prob_risco", ascending=False)
    r.to_csv(C.REPORTS_DIR / "ranking_risco_municipios.csv", index=False)
    print(f"  municípios classificados em risco (thr={thr:.2f}): {r['classificado_risco'].sum()} "
          f"| distribuição de faixas: {r['faixa_risco'].value_counts().to_dict()}")
    return r


def clusters_similaridade(df, k_range=range(3, 8)):
    """Clusters sobre território + socioeconômico (blocos B/C) — nunca sobre a taxa."""
    X, _ = separar_xy(df, incluir_historico=False)
    num = X.select_dtypes("number")
    if num.shape[1] < 2:
        print("  (poucas variáveis estruturais disponíveis — clusters pulados; rode download_external)")
        return None
    Z = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit_transform(num)
    melhor = None
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=C.RANDOM_STATE).fit(Z)
        s = silhouette_score(Z, km.labels_, sample_size=min(3000, len(Z)), random_state=C.RANDOM_STATE)
        print(f"  k={k}: silhueta={s:.3f}")
        if melhor is None or s > melhor[1]:
            melhor = (k, s, km.labels_)
    k, s, labels = melhor
    df = df.assign(cluster=labels)
    perfil = df.groupby("cluster").agg(
        n=(C.COL_ID, "size"), pct_risco=(C.TARGET, "mean"), taxa_t=(f"{C.COL_TAXA}_t", "mean"),
        **{c: (c, "mean") for c in num.columns[:8]},
        regiao_predominante=("regiao", lambda s_: s_.mode().iat[0]),
    ).round(3)
    perfil.to_csv(C.REPORTS_DIR / "perfil_clusters.csv")
    print(f"  k escolhido={k} (silhueta {s:.3f})\n{perfil[['n', 'pct_risco', 'taxa_t', 'regiao_predominante']].to_string()}")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=perfil.index, y=perfil["pct_risco"], ax=ax, color="#c0392b")
    ax.set_ylabel("fração em risco (t+1)")
    ax.set_title(f"Risco por cluster estrutural (k={k})")
    fig.tight_layout()
    fig.savefig(C.IMAGES_DIR / "15_clusters_risco.png", dpi=150)
    plt.close(fig)
    return df[[C.COL_ID, "cluster"]]


def projecao_meta(df, r):
    """Tendência linear t→t+1 extrapolada até 2030 vs meta municipal (heurística transparente)."""
    p = df[[C.COL_ID, f"{C.COL_TAXA}_t", "taxa_alfabetizacao_t1", f"{C.COL_META_2030}_t"]].copy()
    p["variacao_anual"] = p["taxa_alfabetizacao_t1"] - p[f"{C.COL_TAXA}_t"]
    anos = ANO_META - C.ANO_TARGET
    p["taxa_projetada_2030"] = (p["taxa_alfabetizacao_t1"] + anos * p["variacao_anual"]).clip(0, 100)
    p["atinge_meta_projecao"] = p["taxa_projetada_2030"] >= p[f"{C.COL_META_2030}_t"]
    p["variacao_necessaria_ano"] = (p[f"{C.COL_META_2030}_t"] - p["taxa_alfabetizacao_t1"]) / anos
    p = p.merge(r[[C.COL_ID, "prob_risco"]], on=C.COL_ID)
    p["status_meta"] = np.select(
        [p["atinge_meta_projecao"] & (p["prob_risco"] < 0.25),
         p["atinge_meta_projecao"] | (p["prob_risco"] < 0.5)],
        ["trajetória adequada", "atenção"], default="fora da trajetória")
    p.to_csv(C.REPORTS_DIR / "projecao_meta_2030.csv", index=False)
    print(f"  status meta 2030: {p['status_meta'].value_counts().to_dict()}")
    return p


def lista_priorizacao(df, r, p, n=200):
    cols_vuln = [c for c in ("atlas_idhm_educacao", "atlas_idhm", "pib_per_capita_2021") if c in df.columns]
    base = r.merge(p[[C.COL_ID, "variacao_necessaria_ano", "status_meta"]], on=C.COL_ID)
    if cols_vuln:
        base = base.merge(df[[C.COL_ID] + cols_vuln], on=C.COL_ID)
        vuln = base[cols_vuln[0]].rank(pct=True)
        base["score_priorizacao"] = 0.6 * base["prob_risco"] + 0.4 * (1 - vuln)
    else:
        base["score_priorizacao"] = base["prob_risco"]
    base = base.sort_values("score_priorizacao", ascending=False).head(n)
    base.to_csv(C.REPORTS_DIR / "priorizacao_municipios.csv", index=False)
    print(f"  lista de priorização: {len(base)} municípios | UFs mais frequentes: {base['sigla_uf'].value_counts().head(5).to_dict()}")
    return base


def main():
    df = pd.read_parquet(C.PROCESSED_DIR / "dataset_modelagem.parquet")
    meta = json.loads((C.MODELS_DIR / "metadata.json").read_text())
    modelo = joblib.load(C.MODELS_DIR / "modelo_final_calibrado.joblib")
    thr = meta["threshold"]
    print("[1] ranking de risco")
    r = ranking_risco(df, modelo, thr)
    print("[2] clusters de similaridade estrutural")
    clusters_similaridade(df)
    print("[3] projeção de metas 2030")
    p = projecao_meta(df, r)
    print("[4] lista de priorização")
    lista_priorizacao(df, r, p)
    print("✓ reports/ranking_risco_municipios.csv, perfil_clusters.csv, projecao_meta_2030.csv, priorizacao_municipios.csv")


if __name__ == "__main__":
    main()

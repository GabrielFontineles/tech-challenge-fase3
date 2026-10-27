"""
Previsao de risco para 2025 - uso prospectivo do modelo final
Fase 3 - Tech Challenge FIAP

O modelo foi treinado com features do ano anterior (2023) para prever o risco
do ano seguinte (2024). Aqui ele e aplicado um ano a frente: os resultados de
2024 entram nas colunas *_2023 ("ano anterior") e a saida e a probabilidade de
cada municipio ter menos de 60% das criancas alfabetizadas em 2025.

O que muda em relacao ao treino:
- Bloco A (historico educacional): passa a vir de 2024.
- Blocos B e C (IBGE 2021/2022 e Atlas 2010): os mesmos do treino.
- Universo: todos os municipios avaliados em 2024, inclusive os que nao tinham
  resultado em 2023, exceto as UFs que o modelo nunca viu (AC e DF; RR nao
  aparece na Silver).

Ressalva: o indicador de 2024 vem de avaliacoes estaduais equalizadas com a
escala do Saeb. Se a distribuicao de 2024 diferir da de 2023, as probabilidades
podem ficar deslocadas. Nao existe 2025 observado, entao esta lista nao tem
metrica: a validacao do modelo continua sendo a do conjunto de teste de 2024.

As colunas prop_nivel_*_2023 valem 0 em todos os municipios na Silver de 2023,
entao o modelo nunca as usou (uma arvore nao divide uma coluna constante).
Os valores reais de 2024 nessas colunas nao alteram a previsao.

Saidas:
- reports/risco_2025.csv: probabilidade e faixa por municipio
- reports/priorizacao_2025.csv: alto risco em 2025 e ritmo insuficiente para 2030
- reports/risco_2025_por_uf.csv: resumo por UF
- reports/previsao_2025_resumo.json: numeros gerais e diagnostico de mudanca de distribuicao
"""

import json
import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.build_dataset import UFS_FORA_DO_MODELO, construir_features_2023
from src.data.download_external import juntar_fontes_externas

warnings.filterwarnings("ignore")

ANO_BASE, ANO_PREVISTO, ANO_META = 2024, 2025, 2030
SILVER = Path("data/raw/municipio_silver.parquet")
EXTERNAL_DIR = Path("data/external")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
FAIXAS = ["Baixo", "Moderado", "Alto", "Critico"]


def carregar_silver():
    df = pd.read_parquet(SILVER)
    return df[df["rede"] == "Total"].copy()


def montar_features_ano_base(silver, ano=ANO_BASE):
    """Bloco A a partir do ano-base, sem as UFs que o modelo nunca viu."""
    df = construir_features_2023(silver, ano=ano)
    df["id_municipio"] = df["id_municipio"].astype(str)
    fora = df["sigla_uf"].isin(UFS_FORA_DO_MODELO)
    if fora.any():
        ufs = ", ".join(sorted(df.loc[fora, "sigla_uf"].unique()))
        print(f"  Removidos {fora.sum()} municipios de UFs fora do modelo: {ufs}")
    return df[~fora].reset_index(drop=True)


def carregar_fontes_externas():
    arquivos = {
        "censo": EXTERNAL_DIR / "ibge_censo2022.parquet",
        "pib": EXTERNAL_DIR / "ibge_pib2021.parquet",
        "atlas": EXTERNAL_DIR / "atlas_2010.parquet",
    }
    faltando = [str(p) for p in arquivos.values() if not p.exists()]
    if faltando:
        sys.exit(f"ERRO: fontes externas ausentes: {faltando}\n"
                 "Rode antes: python src/data/download_external.py")
    return {nome: pd.read_parquet(p) for nome, p in arquivos.items()}


def faixa(prob, threshold):
    return pd.cut(prob, bins=[0, 0.2, threshold, 0.7, 1.0001], labels=FAIXAS,
                  include_lowest=True)


def variacao_nacional(silver):
    """Variacao media 2023 -> 2024 dos municipios pareados (a mesma da analise estrategica)."""
    wide = silver.pivot_table(index="id_municipio", columns="ano",
                              values="taxa_alfabetizacao").dropna()
    return float((wide[2024] - wide[2023]).mean())


def situacao_2030(ritmo, variacao):
    if pd.isna(ritmo):
        return "Sem meta definida"
    if ritmo <= 0:
        return "Meta ja atingida"
    if ritmo <= variacao:
        return "Atinge no ritmo atual"
    if ritmo <= 2 * variacao:
        return "Precisa de ate 2x o ritmo"
    return "Precisa de mais de 2x o ritmo"


def diagnostico_distribuicao(silver):
    """Compara o Bloco A de 2023 e de 2024 nos municipios pareados."""
    f23 = montar_features_ano_base(silver, ano=2023).set_index("id_municipio")
    f24 = montar_features_ano_base(silver, ano=2024).set_index("id_municipio")
    comuns = f23.index.intersection(f24.index)
    linhas = {}
    for col in ["taxa_alf_2023", "media_pt_2023", "particip_2023", "prop_nivel_0_2023"]:
        a, b = f23.loc[comuns, col], f24.loc[comuns, col]
        linhas[col] = {"media_2023": round(float(a.mean()), 3),
                       "media_2024": round(float(b.mean()), 3),
                       "desvio_2023": round(float(a.std()), 3),
                       "desvio_2024": round(float(b.std()), 3)}
    return {"municipios_pareados": int(len(comuns)), "colunas": linhas}


def main():
    print("=" * 60)
    print(f"PREVISAO DE RISCO {ANO_PREVISTO} (features de {ANO_BASE})")
    print("=" * 60)

    with open(MODELS_DIR / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    modelo = joblib.load(MODELS_DIR / "modelo_final.joblib")
    threshold = metadata["threshold"]
    colunas = metadata["features_numericas"] + metadata["features_categoricas"]
    print(f"Modelo: {metadata['modelo']} (calibrado) | threshold: {threshold:.3f}")

    silver = carregar_silver()
    base = montar_features_ano_base(silver)
    ext = carregar_fontes_externas()
    df = juntar_fontes_externas(base, ext["censo"], ext["pib"], ext["atlas"])

    faltando = [c for c in colunas if c not in df.columns]
    if faltando:
        sys.exit(f"ERRO: colunas do modelo ausentes na base de {ANO_BASE}: {faltando}")
    print(f"Municipios pontuados: {len(df)} de {df['sigla_uf'].nunique()} UFs")

    df["prob_risco_2025"] = modelo.predict_proba(df[colunas])[:, 1]
    df["acima_threshold"] = (df["prob_risco_2025"] >= threshold).astype(int)
    df["faixa_risco_2025"] = faixa(df["prob_risco_2025"], threshold)

    # Contexto: resultado observado no ano-base e situacao em relacao a meta de 2030
    df = df.rename(columns={"taxa_alf_2023": "taxa_alf_2024"})
    df["em_risco_2024"] = (df["taxa_alf_2024"] < metadata["corte_risco"]).astype(int)
    variacao = variacao_nacional(silver)
    df["ritmo_necessario"] = (df["meta_2030"] - df["taxa_alf_2024"]) / (ANO_META - ANO_BASE)
    df["situacao_2030"] = df["ritmo_necessario"].apply(situacao_2030, variacao=variacao)

    REPORTS_DIR.mkdir(exist_ok=True)
    saida = ["nome_municipio", "id_municipio", "sigla_uf", "regiao", "porte",
             "taxa_alf_2024", "em_risco_2024", "prob_risco_2025", "acima_threshold",
             "faixa_risco_2025", "meta_2030", "ritmo_necessario", "situacao_2030"]
    risco = df[saida].sort_values("prob_risco_2025", ascending=False)
    risco.to_csv(REPORTS_DIR / "risco_2025.csv", index=False)

    prioridade = risco[(risco["acima_threshold"] == 1) &
                       (risco["situacao_2030"] == "Precisa de mais de 2x o ritmo")]
    prioridade.to_csv(REPORTS_DIR / "priorizacao_2025.csv", index=False)

    por_uf = (df.groupby(["sigla_uf", "regiao"])
              .agg(municipios=("id_municipio", "count"),
                   prob_media_2025=("prob_risco_2025", "mean"),
                   acima_threshold=("acima_threshold", "sum"),
                   taxa_risco_2024=("em_risco_2024", "mean"))
              .reset_index().sort_values("prob_media_2025", ascending=False))
    por_uf.to_csv(REPORTS_DIR / "risco_2025_por_uf.csv", index=False)

    faixas = (df.groupby("faixa_risco_2025", observed=True)
              .agg(municipios=("id_municipio", "count"),
                   prob_media=("prob_risco_2025", "mean"),
                   em_risco_2024=("em_risco_2024", "mean"))
              .reset_index())

    resumo = {
        "ano_base": ANO_BASE,
        "ano_previsto": ANO_PREVISTO,
        "modelo": metadata["modelo"],
        "threshold": threshold,
        "municipios_pontuados": int(len(df)),
        "ufs": int(df["sigla_uf"].nunique()),
        "ufs_fora_do_modelo": sorted(UFS_FORA_DO_MODELO),
        "prob_media_2025": round(float(df["prob_risco_2025"].mean()), 4),
        "acima_threshold": int(df["acima_threshold"].sum()),
        "pct_acima_threshold": round(float(df["acima_threshold"].mean() * 100), 2),
        "em_risco_observado_2024_pct": round(float(df["em_risco_2024"].mean() * 100), 2),
        "priorizados_2025": int(len(prioridade)),
        "variacao_nacional_pts_ano": round(variacao, 3),
        "faixas": faixas.assign(faixa_risco_2025=faixas["faixa_risco_2025"].astype(str))
                        .round(4).to_dict(orient="records"),
        "diagnostico_distribuicao_bloco_a": diagnostico_distribuicao(silver),
        "ressalva": ("Indicador de 2024 baseado em avaliacoes estaduais equalizadas com o Saeb; "
                     "pode haver mudanca de distribuicao em relacao a 2023. Sem 2025 observado, "
                     "a lista nao tem metrica de desempenho."),
    }
    with open(REPORTS_DIR / "previsao_2025_resumo.json", "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2)

    print(f"\nProbabilidade media de risco em 2025: {resumo['prob_media_2025']:.3f}")
    print(f"Acima do threshold: {resumo['acima_threshold']} ({resumo['pct_acima_threshold']:.1f}%)"
          f" | em risco observado em 2024: {resumo['em_risco_observado_2024_pct']:.1f}%")
    print(f"Priorizados (alto risco e mais de 2x o ritmo): {resumo['priorizados_2025']}")
    print("\nFaixas:")
    print(faixas.round(3).to_string(index=False))
    print("\nDiagnostico do Bloco A (2023 vs 2024, municipios pareados):")
    for col, v in resumo["diagnostico_distribuicao_bloco_a"]["colunas"].items():
        print(f"  {col}: media {v['media_2023']} -> {v['media_2024']} | "
              f"desvio {v['desvio_2023']} -> {v['desvio_2024']}")
    print(f"\nSalvo em {REPORTS_DIR}/: risco_2025.csv, priorizacao_2025.csv, "
          "risco_2025_por_uf.csv, previsao_2025_resumo.json")


if __name__ == "__main__":
    main()

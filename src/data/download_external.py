"""
Download de dados externos — fontes reais
Fase 3 — Tech Challenge FIAP

Fontes:
- IBGE, API SIDRA (sem credencial):
    Tabela 4714: Censo 2022 — populacao residente, area e densidade
    Tabela 5938: PIB dos Municipios 2021 (mil reais)
- Atlas do Desenvolvimento Humano no Brasil (Censo 2010):
    arquivo censo_total_1991_2010.xlsx, aba "MUN 91-00-10",
    baixado manualmente em atlasbrasil.org.br e salvo em data/external/raw/

Nao ha geracao de dados simulados: se uma fonte faltar ou falhar,
o script interrompe a execucao com erro explicito.
Todas as fontes sao anteriores a 2024 (ano do target).
"""

import sys
import requests
import numpy as np
import pandas as pd
from pathlib import Path

EXTERNAL_DIR = Path("data/external")
RAW_EXTERNAL_DIR = EXTERNAL_DIR / "raw"
PROCESSED_DIR = Path("data/processed")
EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

SIDRA_URL = "https://apisidra.ibge.gov.br/values"
ATLAS_ARQUIVO = RAW_EXTERNAL_DIR / "censo_total_1991_2010.xlsx"
ATLAS_ABA = "MUN 91-00-10"

# Colunas do Atlas utilizadas -> nome no projeto
ATLAS_COLUNAS = {
    "IDHM": "idhm",
    "IDHM_E": "idhm_educacao",
    "IDHM_R": "idhm_renda",
    "IDHM_L": "idhm_longevidade",
    "RDPC": "renda_per_capita_2010",
    "GINI": "gini",
    "PMPOB": "pct_pobres",
    "PIND": "pct_extremamente_pobres",
    "T_ANALF15M": "analfabetismo_15mais",
    "E_ANOSESTUDO": "expectativa_anos_estudo",
    "T_FREQ4A5": "freq_escolar_4a5",
}


def consultar_sidra(consulta):
    """Consulta a API SIDRA e retorna DataFrame com id_municipio e valor."""
    url = f"{SIDRA_URL}/{consulta}"
    print(f"  Consultando: {url}")
    resposta = requests.get(url, timeout=300)
    resposta.raise_for_status()
    dados = resposta.json()
    df = pd.DataFrame(dados[1:])  # primeira linha e o cabecalho
    df["id_municipio"] = df["D1C"].astype(str)
    df["valor"] = pd.to_numeric(df["V"], errors="coerce")  # '-' e '...' viram NaN
    print(f"  Linhas recebidas: {len(df)}")
    return df


def baixar_populacao_censo2022():
    """Tabela 4714 — populacao, area e densidade (Censo 2022)."""
    print("\n[1/3] IBGE — Censo 2022 (tabela 4714)")
    df = consultar_sidra("t/4714/n6/all/v/all/p/2022")

    wide = df.pivot_table(index="id_municipio", columns="D2N",
                          values="valor", aggfunc="first").reset_index()

    mapa = {}
    for col in wide.columns:
        nome = str(col).lower()
        if nome.startswith("população residente"):
            mapa[col] = "populacao_2022"
        elif nome.startswith("área"):
            mapa[col] = "area_km2"
        elif nome.startswith("densidade"):
            mapa[col] = "densidade_2022"

    esperadas = {"populacao_2022", "area_km2", "densidade_2022"}
    if set(mapa.values()) != esperadas:
        sys.exit(f"ERRO: variaveis inesperadas na tabela 4714: {list(wide.columns)}")

    wide = wide.rename(columns=mapa)[["id_municipio"] + sorted(esperadas)]
    wide["log_populacao_2022"] = np.log(wide["populacao_2022"])
    wide["porte"] = pd.cut(
        wide["populacao_2022"],
        bins=[0, 5_000, 20_000, 50_000, 100_000, 500_000, np.inf],
        labels=["ate_5mil", "5_20mil", "20_50mil",
                "50_100mil", "100_500mil", "acima_500mil"],
    ).astype(str)

    caminho = EXTERNAL_DIR / "ibge_censo2022.parquet"
    wide.to_parquet(caminho, index=False)
    print(f"  Salvo: {caminho} ({len(wide)} municipios)")
    return wide


def baixar_pib_2021():
    """Tabela 5938 — PIB a precos correntes 2021 (mil reais)."""
    print("\n[2/3] IBGE — PIB dos Municipios 2021 (tabela 5938)")
    df = consultar_sidra("t/5938/n6/all/v/37/p/2021")
    pib = df[["id_municipio", "valor"]].rename(columns={"valor": "pib_2021_mil_reais"})

    caminho = EXTERNAL_DIR / "ibge_pib2021.parquet"
    pib.to_parquet(caminho, index=False)
    print(f"  Salvo: {caminho} ({len(pib)} municipios)")
    return pib


def carregar_atlas_2010():
    """Atlas do Desenvolvimento Humano — recorte municipal de 2010."""
    print("\n[3/3] Atlas do Desenvolvimento Humano — Censo 2010")
    cache = EXTERNAL_DIR / "atlas_2010.parquet"
    if cache.exists():
        print(f"  Usando cache: {cache}")
        return pd.read_parquet(cache)

    if not ATLAS_ARQUIVO.exists():
        sys.exit(
            f"ERRO: arquivo do Atlas nao encontrado em {ATLAS_ARQUIVO}\n"
            "Baixe a base municipal completa em atlasbrasil.org.br "
            "(censo_total_1991_2010.xlsx) e salve nessa pasta."
        )

    print(f"  Lendo {ATLAS_ARQUIVO} (pode levar cerca de 1 minuto)...")
    df = pd.read_excel(ATLAS_ARQUIVO, sheet_name=ATLAS_ABA)

    faltando = [c for c in list(ATLAS_COLUNAS) + ["ANO", "Codmun7", "pesoRUR", "pesotot"]
                if c not in df.columns]
    if faltando:
        sys.exit(f"ERRO: colunas ausentes no Atlas: {faltando}")

    df = df[df["ANO"] == 2010].copy()
    atlas = df[["Codmun7"] + list(ATLAS_COLUNAS)].rename(columns=ATLAS_COLUNAS)
    atlas = atlas.rename(columns={"Codmun7": "id_municipio"})
    atlas["id_municipio"] = atlas["id_municipio"].astype(str)
    atlas["pct_rural"] = df["pesoRUR"].values / df["pesotot"].values * 100

    atlas.to_parquet(cache, index=False)
    print(f"  Salvo: {cache} ({len(atlas)} municipios)")
    return atlas


def enriquecer_dataset(df_censo, df_pib, df_atlas):
    """Junta as fontes externas ao dataset temporal."""
    print("\nEnriquecendo dataset de modelagem...")
    base = pd.read_parquet(PROCESSED_DIR / "dataset_modelagem_v2.parquet")
    base["id_municipio"] = base["id_municipio"].astype(str)

    if not base["id_municipio"].str.len().eq(7).all():
        sys.exit("ERRO: dataset base contem id_municipio fora do padrao de 7 digitos")

    df = base.merge(df_censo, on="id_municipio", how="left")
    df = df.merge(df_pib, on="id_municipio", how="left")
    df = df.merge(df_atlas, on="id_municipio", how="left")

    # PIB per capita: PIB 2021 (mil reais -> reais) / populacao Censo 2022
    df["pib_per_capita"] = df["pib_2021_mil_reais"] * 1000 / df["populacao_2022"]
    df["log_pib_per_capita"] = np.log(df["pib_per_capita"])

    print("\nCobertura do merge (municipios do dataset com dado externo):")
    for col in ["populacao_2022", "pib_per_capita", "idhm", "analfabetismo_15mais"]:
        cobertos = df[col].notna().sum()
        print(f"  {col}: {cobertos}/{len(df)} ({cobertos / len(df) * 100:.1f}%)")

    caminho = PROCESSED_DIR / "dataset_enriquecido_v2.parquet"
    df.to_parquet(caminho, index=False)
    print(f"\nSalvo: {caminho} — shape {df.shape}")
    return df


def main():
    print("=" * 60)
    print("DOWNLOAD DE DADOS EXTERNOS — FONTES REAIS")
    print("=" * 60)
    df_censo = baixar_populacao_censo2022()
    df_pib = baixar_pib_2021()
    df_atlas = carregar_atlas_2010()
    enriquecer_dataset(df_censo, df_pib, df_atlas)
    print("\n" + "=" * 60)
    print("Concluido. Nenhum dado simulado foi gerado.")
    print("=" * 60)


if __name__ == "__main__":
    main()

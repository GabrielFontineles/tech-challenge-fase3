"""
Download de Dados Externos — Enriquecimento da Base
Fase 3 — Tech Challenge FIAP

Baixa dados externos via Base dos Dados (BigQuery):
- Atlas do Desenvolvimento Humano (IDHM, renda, educação)
- IBGE — Estimativas populacionais 2023
- IBGE — PIB dos municípios

Pré-requisito: projeto GCP configurado
export GCP_BILLING_PROJECT_ID="seu-projeto-gcp"
"""

import pandas as pd
import os
from pathlib import Path

EXTERNAL_DIR = Path("data/external")
EXTERNAL_DIR.mkdir(exist_ok=True)

# Projeto GCP para billing do BigQuery
GCP_PROJECT = os.environ.get("GCP_BILLING_PROJECT_ID", "")


def baixar_atlas_idhm():
    """
    Atlas do Desenvolvimento Humano — Censo 2010.
    IDHM, IDHM-Educação, IDHM-Renda, % pobreza, Gini, renda per capita.
    """
    print("\n[1/3] Baixando Atlas do Desenvolvimento Humano...")

    if not GCP_PROJECT:
        print("  ⚠️  GCP_BILLING_PROJECT_ID não configurado.")
        print("  Gerando dados simulados para desenvolvimento...")
        return _simular_atlas()

    try:
        import basedosdados as bd

        query = """
        SELECT
            id_municipio,
            ano,
            idhm,
            idhm_educacao,
            idhm_longevidade,
            idhm_renda,
            renda_per_capita,
            gini,
            pct_extremamente_pobres,
            pct_pobres,
            populacao
        FROM `basedosdados.mundo_onu_adh.municipio`
        WHERE ano = 2010
        """

        df = bd.read_sql(query, billing_project_id=GCP_PROJECT)
        df['id_municipio'] = df['id_municipio'].astype(str).str[:6]
        caminho = EXTERNAL_DIR / "atlas_idhm.parquet"
        df.to_parquet(caminho, index=False)
        print(f"  ✓ {len(df)} municípios baixados")
        print(f"  ✓ Salvo em: {caminho}")
        return df

    except Exception as e:
        print(f"  ✗ Erro ao baixar Atlas: {e}")
        print("  Gerando dados simulados...")
        return _simular_atlas()


def _simular_atlas():
    """Gera dados simulados do Atlas para desenvolvimento local."""
    import numpy as np
    np.random.seed(42)

    # Carrega ids de municípios do dataset principal
    df_base = pd.read_parquet("data/processed/dataset_modelagem_v2.parquet")
    ids = df_base['id_municipio'].unique()

    df = pd.DataFrame({
        'id_municipio': ids,
        'idhm': np.random.uniform(0.4, 0.85, len(ids)).round(3),
        'idhm_educacao': np.random.uniform(0.3, 0.85, len(ids)).round(3),
        'idhm_longevidade': np.random.uniform(0.6, 0.9, len(ids)).round(3),
        'idhm_renda': np.random.uniform(0.4, 0.85, len(ids)).round(3),
        'renda_per_capita': np.random.uniform(300, 2000, len(ids)).round(2),
        'gini': np.random.uniform(0.35, 0.65, len(ids)).round(3),
        'pct_pobres': np.random.uniform(5, 60, len(ids)).round(2),
        'pct_extremamente_pobres': np.random.uniform(2, 30, len(ids)).round(2),
    })

    caminho = EXTERNAL_DIR / "atlas_idhm_simulado.parquet"
    df.to_parquet(caminho, index=False)
    print(f"  ✓ {len(df)} municípios simulados")
    print(f"  ✓ Salvo em: {caminho}")
    print("  ⚠️  DADOS SIMULADOS — substituir por dados reais antes da entrega")
    return df


def baixar_populacao_ibge():
    """
    IBGE — Estimativas populacionais 2023.
    """
    print("\n[2/3] Baixando estimativas populacionais IBGE 2023...")

    if not GCP_PROJECT:
        print("  ⚠️  GCP_BILLING_PROJECT_ID não configurado.")
        print("  Gerando dados simulados...")
        return _simular_populacao()

    try:
        import basedosdados as bd

        query = """
        SELECT
            id_municipio,
            ano,
            populacao
        FROM `basedosdados.br_ibge_populacao.municipio`
        WHERE ano = 2023
        """

        df = bd.read_sql(query, billing_project_id=GCP_PROJECT)
        df['id_municipio'] = df['id_municipio'].astype(str)
        caminho = EXTERNAL_DIR / "populacao_ibge.parquet"
        df.to_parquet(caminho, index=False)
        print(f"  ✓ {len(df)} municípios baixados")
        print(f"  ✓ Salvo em: {caminho}")
        return df

    except Exception as e:
        print(f"  ✗ Erro ao baixar IBGE: {e}")
        print("  Gerando dados simulados...")
        return _simular_populacao()


def _simular_populacao():
    """Gera dados simulados de população."""
    import numpy as np
    np.random.seed(42)

    df_base = pd.read_parquet("data/processed/dataset_modelagem_v2.parquet")
    ids = df_base['id_municipio'].unique()

    populacao = np.random.lognormal(mean=9.5, sigma=1.2, size=len(ids)).astype(int)

    df = pd.DataFrame({
        'id_municipio': ids,
        'populacao_2023': populacao,
        'log_populacao': np.log(populacao),
        'porte': pd.cut(
            populacao,
            bins=[0, 5000, 20000, 100000, 500000, float('inf')],
            labels=['Muito pequeno', 'Pequeno', 'Medio', 'Grande', 'Muito grande']
        ).astype(str)
    })

    caminho = EXTERNAL_DIR / "populacao_ibge_simulado.parquet"
    df.to_parquet(caminho, index=False)
    print(f"  ✓ {len(df)} municípios simulados")
    print(f"  ✓ Salvo em: {caminho}")
    print("  ⚠️  DADOS SIMULADOS — substituir por dados reais antes da entrega")
    return df


def baixar_pib_ibge():
    """
    IBGE — PIB dos Municípios (último disponível: 2021).
    """
    print("\n[3/3] Baixando PIB municipal IBGE...")

    if not GCP_PROJECT:
        print("  ⚠️  GCP_BILLING_PROJECT_ID não configurado.")
        print("  Gerando dados simulados...")
        return _simular_pib()

    try:
        import basedosdados as bd

        query = """
        SELECT
            id_municipio,
            ano,
            pib,
            pib_per_capita,
            va_agropecuaria,
            va_industria,
            va_servicos
        FROM `basedosdados.br_ibge_pib.municipio`
        WHERE ano = 2021
        """

        df = bd.read_sql(query, billing_project_id=GCP_PROJECT)
        df['id_municipio'] = df['id_municipio'].astype(str)
        caminho = EXTERNAL_DIR / "pib_ibge.parquet"
        df.to_parquet(caminho, index=False)
        print(f"  ✓ {len(df)} municípios baixados")
        print(f"  ✓ Salvo em: {caminho}")
        return df

    except Exception as e:
        print(f"  ✗ Erro ao baixar PIB: {e}")
        print("  Gerando dados simulados...")
        return _simular_pib()


def _simular_pib():
    """Gera dados simulados de PIB."""
    import numpy as np
    np.random.seed(42)

    df_base = pd.read_parquet("data/processed/dataset_modelagem_v2.parquet")
    ids = df_base['id_municipio'].unique()

    pib_per_capita = np.random.lognormal(mean=9.0, sigma=0.8, size=len(ids))

    df = pd.DataFrame({
        'id_municipio': ids,
        'pib_per_capita': pib_per_capita.round(2),
        'log_pib_per_capita': np.log(pib_per_capita).round(4),
    })

    caminho = EXTERNAL_DIR / "pib_ibge_simulado.parquet"
    df.to_parquet(caminho, index=False)
    print(f"  ✓ {len(df)} municípios simulados")
    print(f"  ✓ Salvo em: {caminho}")
    print("  ⚠️  DADOS SIMULADOS — substituir por dados reais antes da entrega")
    return df


def enriquecer_dataset(df_atlas, df_pop, df_pib):
    """
    Junta os dados externos ao dataset principal.
    """
    print("\nEnriquecendo dataset principal...")

    df_base = pd.read_parquet("data/processed/dataset_modelagem_v2.parquet")
    print(f"  Dataset base: {df_base.shape}")

    # Join com Atlas
    df = pd.merge(df_base, df_atlas, on='id_municipio', how='left')

    # Join com população
    pop_cols = ['id_municipio'] + [c for c in df_pop.columns if c != 'id_municipio']
    df = pd.merge(df, df_pop[pop_cols], on='id_municipio', how='left')

    # Join com PIB
    pib_cols = ['id_municipio'] + [c for c in df_pib.columns if c != 'id_municipio']
    df = pd.merge(df, df_pib[pib_cols], on='id_municipio', how='left')

    print(f"  Dataset enriquecido: {df.shape}")

    # Salva dataset enriquecido
    caminho = Path("data/processed") / "dataset_enriquecido_v2.parquet"
    df.to_parquet(caminho, index=False)
    print(f"  ✓ Salvo em: {caminho}")

    # Missing values após join
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(f"\n  Missing values após enriquecimento:")
        for col, count in missing.items():
            print(f"    {col}: {count} ({count/len(df)*100:.1f}%)")

    return df


def main():
    print("=" * 60)
    print("DOWNLOAD DE DADOS EXTERNOS")
    print("=" * 60)
    print(f"\nGCP Project: {GCP_PROJECT or 'NÃO CONFIGURADO (modo simulação)'}")

    df_atlas = baixar_atlas_idhm()
    df_pop = baixar_populacao_ibge()
    df_pib = baixar_pib_ibge()
    df_enriquecido = enriquecer_dataset(df_atlas, df_pop, df_pib)

    print("\n" + "=" * 60)
    print("✓ Download e enriquecimento concluídos!")
    print(f"  Dataset final: {df_enriquecido.shape}")
    print("\nPara usar dados reais, configure:")
    print("  export GCP_BILLING_PROJECT_ID='seu-projeto-gcp'")
    print("=" * 60)


if __name__ == "__main__":
    main()

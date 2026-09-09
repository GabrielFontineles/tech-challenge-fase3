"""
Build Dataset — Construção do Dataset Temporal
Fase 3 — Tech Challenge FIAP

Constrói o dataset de modelagem com design temporal correto:
- Features: dados de 2023 (histórico)
- Target: em_risco_2024 (taxa_alfabetizacao_2024 < 60%)

Elimina o data leakage estrutural da versão anterior.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)

# Mapeamento de UF
UF_MAP = {
    11: 'RO', 12: 'AC', 13: 'AM', 14: 'RR', 15: 'PA',
    16: 'AP', 17: 'TO', 21: 'MA', 22: 'PI', 23: 'CE',
    24: 'RN', 25: 'PB', 26: 'PE', 27: 'AL', 28: 'SE',
    29: 'BA', 31: 'MG', 32: 'ES', 33: 'RJ', 35: 'SP',
    41: 'PR', 42: 'SC', 43: 'RS', 50: 'MS', 51: 'MT',
    52: 'GO', 53: 'DF'
}

REGIOES = {
    'RO': 'Norte', 'AC': 'Norte', 'AM': 'Norte', 'RR': 'Norte',
    'PA': 'Norte', 'AP': 'Norte', 'TO': 'Norte',
    'MA': 'Nordeste', 'PI': 'Nordeste', 'CE': 'Nordeste',
    'RN': 'Nordeste', 'PB': 'Nordeste', 'PE': 'Nordeste',
    'AL': 'Nordeste', 'SE': 'Nordeste', 'BA': 'Nordeste',
    'MG': 'Sudeste', 'ES': 'Sudeste', 'RJ': 'Sudeste', 'SP': 'Sudeste',
    'PR': 'Sul', 'SC': 'Sul', 'RS': 'Sul',
    'MS': 'Centro-Oeste', 'MT': 'Centro-Oeste',
    'GO': 'Centro-Oeste', 'DF': 'Centro-Oeste'
}

def carregar_municipio_silver():
    """Carrega dados Silver de município."""
    print("Carregando municipio_silver...")
    df = pd.read_parquet(RAW_DIR / "municipio_silver.parquet")
    df_total = df[df['rede'] == 'Total'].copy()
    print(f"  Shape total: {df_total.shape}")
    return df_total

def construir_features_2023(df):
    """
    Bloco A — Histórico educacional 2023.
    Todas as features são de 2023 — seguro usar como preditores do target 2024.
    """
    print("\nConstruindo features 2023 (Bloco A)...")
    df_2023 = df[df['ano'] == 2023].copy()

    # Extrai cod_uf e adiciona sigla/região
    df_2023['cod_uf'] = df_2023['id_municipio'].astype(str).str[:2].astype(int)
    df_2023['sigla_uf'] = df_2023['cod_uf'].map(UF_MAP)
    df_2023['regiao'] = df_2023['sigla_uf'].map(REGIOES)

    # Calcula taxa média da UF em 2023 para feature relativa
    taxa_uf_2023 = df_2023.groupby('sigla_uf')['taxa_alfabetizacao'].mean()
    df_2023['taxa_uf_2023'] = df_2023['sigla_uf'].map(taxa_uf_2023)
    df_2023['taxa_vs_uf_2023'] = df_2023['taxa_alfabetizacao'] - df_2023['taxa_uf_2023']

    # Gap para a meta 2030
    df_2023['gap_meta_2030_2023'] = df_2023['meta_alfabetizacao_2030'] - df_2023['taxa_alfabetizacao']

    # Renomeia colunas com sufixo _2023 para clareza
    colunas_renomear = {
        'taxa_alfabetizacao': 'taxa_alf_2023',
        'media_portugues': 'media_pt_2023',
        'percentual_participacao': 'particip_2023',
        'nivel_alfabetizacao': 'nivel_alf_2023',
        'proporcao_aluno_nivel_0': 'prop_nivel_0_2023',
        'proporcao_aluno_nivel_1': 'prop_nivel_1_2023',
        'proporcao_aluno_nivel_2': 'prop_nivel_2_2023',
        'proporcao_aluno_nivel_3': 'prop_nivel_3_2023',
        'proporcao_aluno_nivel_4': 'prop_nivel_4_2023',
        'proporcao_aluno_nivel_5': 'prop_nivel_5_2023',
        'proporcao_aluno_nivel_6': 'prop_nivel_6_2023',
        'proporcao_aluno_nivel_7': 'prop_nivel_7_2023',
        'proporcao_aluno_nivel_8': 'prop_nivel_8_2023',
        'meta_alfabetizacao_2030': 'meta_2030',
        'distancia_meta_2030': 'dist_meta_2030_2023',
    }

    df_2023 = df_2023.rename(columns=colunas_renomear)

    # Seleciona colunas finais do Bloco A
    colunas_bloco_a = [
        'id_municipio', 'sigla_uf', 'regiao', 'cod_uf',
        'taxa_alf_2023', 'media_pt_2023', 'particip_2023',
        'nivel_alf_2023', 'meta_2030', 'gap_meta_2030_2023',
        'dist_meta_2030_2023', 'taxa_vs_uf_2023',
        'prop_nivel_0_2023', 'prop_nivel_1_2023', 'prop_nivel_2_2023',
        'prop_nivel_3_2023', 'prop_nivel_4_2023', 'prop_nivel_5_2023',
        'prop_nivel_6_2023', 'prop_nivel_7_2023', 'prop_nivel_8_2023',
    ]

    df_features = df_2023[colunas_bloco_a].copy()
    print(f"  Municípios com features 2023: {len(df_features)}")
    return df_features

def construir_target_2024(df):
    """
    Target: em_risco_2024
    Classe positiva = risco (taxa_2024 < 60%)
    NENHUMA coluna de 2024 além do target.
    """
    print("\nConstruindo target 2024...")
    df_2024 = df[df['ano'] == 2024][['id_municipio', 'taxa_alfabetizacao']].copy()
    df_2024 = df_2024.rename(columns={'taxa_alfabetizacao': 'taxa_alf_2024'})

    # Target binário — classe positiva = risco
    CORTE = 60.0
    df_2024['em_risco_2024'] = (df_2024['taxa_alf_2024'] < CORTE).astype(int)

    dist = df_2024['em_risco_2024'].value_counts()
    print(f"  Municípios 2024: {len(df_2024)}")
    print(f"  Em risco (1): {dist.get(1, 0)} ({dist.get(1, 0)/len(df_2024)*100:.1f}%)")
    print(f"  Não em risco (0): {dist.get(0, 0)} ({dist.get(0, 0)/len(df_2024)*100:.1f}%)")
    return df_2024

def parear_datasets(df_features, df_target):
    """
    Faz o join t/t+1 — só municípios presentes em ambos os anos.
    Essa é a base de modelagem limpa.
    """
    print("\nPareando 2023 → 2024...")
    df = pd.merge(df_features, df_target, on='id_municipio', how='inner')
    print(f"  Municípios pareados: {len(df)}")
    print(f"  Features: {len(df.columns) - 3} (excluindo id, taxa_2024 e target)")
    return df

def salvar_dataset(df):
    """Salva dataset final e exibe resumo."""
    caminho = PROCESSED_DIR / "dataset_modelagem_v2.parquet"
    df.to_parquet(caminho, index=False)
    print(f"\n✓ Dataset salvo em: {caminho}")
    print(f"  Shape: {df.shape}")

    print("\nResumo de missing values:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        for col, count in missing.items():
            print(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
    else:
        print("  Nenhum missing value!")

    print(f"\nDistribuição do target:")
    dist = df['em_risco_2024'].value_counts()
    print(f"  Em risco (1):     {dist[1]} ({dist[1]/len(df)*100:.1f}%)")
    print(f"  Não em risco (0): {dist[0]} ({dist[0]/len(df)*100:.1f}%)")

def main():
    print("=" * 60)
    print("BUILD DATASET — DESIGN TEMPORAL CORRETO")
    print("Features: 2023 | Target: 2024")
    print("=" * 60)

    df_total = carregar_municipio_silver()
    df_features = construir_features_2023(df_total)
    df_target = construir_target_2024(df_total)
    df_final = parear_datasets(df_features, df_target)
    salvar_dataset(df_final)

    print("\n" + "=" * 60)
    print("✓ Dataset temporal construído com sucesso!")
    print("  Próximo passo: download de dados externos")
    print("=" * 60)

if __name__ == "__main__":
    main()

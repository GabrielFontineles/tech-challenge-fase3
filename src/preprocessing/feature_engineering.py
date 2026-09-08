"""
Feature Engineering — Criação de Variáveis para o Modelo ML
Fase 3 — Tech Challenge FIAP
Cria a variável alvo e features derivadas para o modelo preditivo.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.style.use('seaborn-v0_8')
IMAGES_DIR = Path("images")
DATA_DIR = Path("data/processed")
DATA_DIR.mkdir(exist_ok=True)

REGIOES = {
    'AC': 'Norte', 'AM': 'Norte', 'AP': 'Norte', 'PA': 'Norte',
    'RO': 'Norte', 'RR': 'Norte', 'TO': 'Norte',
    'AL': 'Nordeste', 'BA': 'Nordeste', 'CE': 'Nordeste',
    'MA': 'Nordeste', 'PB': 'Nordeste', 'PE': 'Nordeste',
    'PI': 'Nordeste', 'RN': 'Nordeste', 'SE': 'Nordeste',
    'DF': 'Centro-Oeste', 'GO': 'Centro-Oeste',
    'MS': 'Centro-Oeste', 'MT': 'Centro-Oeste',
    'ES': 'Sudeste', 'MG': 'Sudeste', 'RJ': 'Sudeste', 'SP': 'Sudeste',
    'PR': 'Sul', 'RS': 'Sul', 'SC': 'Sul'
}

def carregar_dados():
    """Carrega dados Silver do município."""
    print("Carregando dados...")
    df = pd.read_parquet("data/raw/municipio_silver.parquet")
    print(f"  Shape original: {df.shape}")
    return df

def criar_dataset_modelagem(df):
    """
    Cria o dataset final para modelagem.
    Foco: municípios 2024, rede Total — base mais completa.
    """
    print("\n[1/6] Filtrando dados para modelagem...")

    df_model = df[
        (df['ano'] == 2024) &
        (df['rede'] == 'Total')
    ].copy()

    print(f"  Registros após filtro: {len(df_model)}")
    return df_model

def criar_variavel_alvo(df):
    """
    Cria variável alvo binária:
    1 = município no caminho ou já atingiu a meta 2030
    0 = município em risco ou crítico
    Ponto de corte: 60% (abaixo = em risco)
    """
    print("\n[2/6] Criando variável alvo...")

    df['alfabetizado'] = (df['taxa_alfabetizacao'] >= 60).astype(int)

    dist = df['alfabetizado'].value_counts()
    print(f"  Classe 1 (alfabetizado): {dist[1]} ({dist[1]/len(df)*100:.1f}%)")
    print(f"  Classe 0 (em risco): {dist[0]} ({dist[0]/len(df)*100:.1f}%)")

    return df

def criar_features_geograficas(df):
    """Cria features baseadas em localização geográfica."""
    print("\n[3/6] Criando features geográficas...")

    # Extrai código da UF do id_municipio
    df['cod_uf'] = df['id_municipio'].astype(str).str[:2].astype(int)

    # Mapeamento UF → sigla → região
    uf_map = {
        11: 'RO', 12: 'AC', 13: 'AM', 14: 'RR', 15: 'PA',
        16: 'AP', 17: 'TO', 21: 'MA', 22: 'PI', 23: 'CE',
        24: 'RN', 25: 'PB', 26: 'PE', 27: 'AL', 28: 'SE',
        29: 'BA', 31: 'MG', 32: 'ES', 33: 'RJ', 35: 'SP',
        41: 'PR', 42: 'SC', 43: 'RS', 50: 'MS', 51: 'MT',
        52: 'GO', 53: 'DF'
    }

    df['sigla_uf'] = df['cod_uf'].map(uf_map)
    df['regiao'] = df['sigla_uf'].map(REGIOES)

    # Encoding ordinal da região por desempenho médio histórico
    regiao_encoding = {
        'Norte': 1, 'Nordeste': 2, 'Centro-Oeste': 3,
        'Sudeste': 4, 'Sul': 5
    }
    df['regiao_encoded'] = df['regiao'].map(regiao_encoding)

    print(f"  Features criadas: cod_uf, sigla_uf, regiao, regiao_encoded")
    return df

def criar_features_desempenho(df):
    """Cria features baseadas em desempenho educacional."""
    print("\n[4/6] Criando features de desempenho...")

    # Distância normalizada da meta 2030
    df['distancia_meta_normalizada'] = (
        df['distancia_meta_2030'] / df['meta_alfabetizacao_2030']
    ).fillna(0)

    # Score de proficiência combinado (níveis altos = bom)
    niveis_altos = ['proporcao_aluno_nivel_5', 'proporcao_aluno_nivel_6',
                    'proporcao_aluno_nivel_7', 'proporcao_aluno_nivel_8']
    niveis_baixos = ['proporcao_aluno_nivel_0', 'proporcao_aluno_nivel_1',
                     'proporcao_aluno_nivel_2']

    niveis_altos_disp = [c for c in niveis_altos if c in df.columns]
    niveis_baixos_disp = [c for c in niveis_baixos if c in df.columns]

    if niveis_altos_disp:
        df['score_niveis_altos'] = df[niveis_altos_disp].sum(axis=1)
    if niveis_baixos_disp:
        df['score_niveis_baixos'] = df[niveis_baixos_disp].sum(axis=1)

    # Flag: município atingiu meta 2030
    df['atingiu_meta_2030'] = (df['distancia_meta_2030'] <= 0).astype(int)

    # Flag: participação alta (acima da mediana)
    mediana_part = df['percentual_participacao'].median()
    df['participacao_alta'] = (
        df['percentual_participacao'] >= mediana_part
    ).astype(int)

    print(f"  Features criadas: distancia_meta_normalizada, "
          f"score_niveis_altos, score_niveis_baixos, "
          f"atingiu_meta_2030, participacao_alta")
    return df

def selecionar_features_finais(df):
    """Seleciona e organiza as features finais para o modelo."""
    print("\n[5/6] Selecionando features finais...")

    features = [
        # Target
        'alfabetizado',
        # Identificadores
        'id_municipio', 'sigla_uf', 'regiao',
        # Features numéricas principais
        'taxa_alfabetizacao', 'media_portugues',
        'meta_alfabetizacao_2030', 'distancia_meta_2030',
        'percentual_participacao', 'nivel_alfabetizacao',
        # Features geográficas
        'cod_uf', 'regiao_encoded',
        # Features de desempenho
        'distancia_meta_normalizada', 'atingiu_meta_2030',
        'participacao_alta',
    ]

    # Adiciona scores de níveis se disponíveis
    if 'score_niveis_altos' in df.columns:
        features.append('score_niveis_altos')
    if 'score_niveis_baixos' in df.columns:
        features.append('score_niveis_baixos')

    # Adiciona proporções de níveis
    for i in range(9):
        col = f'proporcao_aluno_nivel_{i}'
        if col in df.columns:
            features.append(col)

    features_disponiveis = [f for f in features if f in df.columns]
    df_final = df[features_disponiveis].copy()

    print(f"  Total de features: {len(features_disponiveis) - 4}")
    print(f"  Shape final: {df_final.shape}")

    # Missing values
    missing = df_final.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(f"\n  Missing values a tratar:")
        for col, count in missing.items():
            print(f"    {col}: {count} ({count/len(df_final)*100:.1f}%)")

    return df_final

def salvar_dataset(df):
    """Salva dataset processado para uso na modelagem."""
    print("\n[6/6] Salvando dataset processado...")

    caminho = DATA_DIR / "dataset_modelagem.parquet"
    df.to_parquet(caminho, index=False)
    print(f"  ✓ Salvo em: {caminho}")
    print(f"  Shape: {df.shape}")

def visualizar_variavel_alvo(df):
    """Visualiza a distribuição da variável alvo."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distribuição da variável alvo
    dist = df['alfabetizado'].value_counts()
    cores = ['#d32f2f', '#4caf50']
    axes[0].bar(['Em risco (0)', 'Alfabetizado (1)'],
                [dist[0], dist[1]], color=cores, alpha=0.85)
    axes[0].set_title('Distribuição da Variável Alvo')
    axes[0].set_ylabel('Quantidade de Municípios')
    for i, v in enumerate([dist[0], dist[1]]):
        axes[0].text(i, v + 20, f'{v}\n({v/len(df)*100:.1f}%)',
                    ha='center', fontweight='bold')

    # Taxa por região
    taxa_regiao = df.groupby('regiao')['alfabetizado'].mean().sort_values()
    cores_reg = ['#d32f2f' if v < 0.5 else '#ff9800' if v < 0.7
                 else '#4caf50' for v in taxa_regiao.values]
    axes[1].barh(taxa_regiao.index, taxa_regiao.values * 100,
                 color=cores_reg, alpha=0.85)
    axes[1].axvline(x=50, color='black', linestyle='--', linewidth=1)
    axes[1].set_title('% Municípios Alfabetizados por Região')
    axes[1].set_xlabel('% de Municípios Classe 1')

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / '08_variavel_alvo.png', dpi=150)
    plt.close()
    print("\n✓ Gráfico salvo: images/08_variavel_alvo.png")

def main():
    print("=" * 60)
    print("FEATURE ENGINEERING — FASE 3 TECH CHALLENGE FIAP")
    print("=" * 60)

    df = carregar_dados()
    df = criar_dataset_modelagem(df)
    df = criar_variavel_alvo(df)
    df = criar_features_geograficas(df)
    df = criar_features_desempenho(df)
    df_final = selecionar_features_finais(df)
    visualizar_variavel_alvo(df)
    salvar_dataset(df_final)

    print("\n" + "=" * 60)
    print("✓ Feature Engineering concluído!")
    print("  Dataset pronto para modelagem em: data/processed/")
    print("=" * 60)

if __name__ == "__main__":
    main()

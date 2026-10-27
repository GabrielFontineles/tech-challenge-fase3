"""
EDA Inicial — Análise Exploratória dos Dados
Fase 3 — Tech Challenge FIAP
Analisa os datasets da camada Gold e Silver para entender
distribuições, padrões e formular hipóteses para o modelo ML.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configurações visuais
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
IMAGES_DIR = Path("images")
IMAGES_DIR.mkdir(exist_ok=True)

def carregar_dados():
    """Carrega todos os datasets disponíveis."""
    print("=" * 60)
    print("CARREGANDO DADOS")
    print("=" * 60)

    dados = {}

    # Gold Layer
    dados['ranking'] = pd.read_parquet("data/gold/ranking_estados.parquet")
    dados['evolucao'] = pd.read_parquet("data/gold/evolucao_temporal.parquet")
    dados['municipal'] = pd.read_parquet("data/gold/analise_municipal.parquet")
    dados['brasil'] = pd.read_parquet("data/gold/visao_brasil.parquet")

    # Silver Layer
    dados['municipio'] = pd.read_parquet("data/raw/municipio_silver.parquet")
    dados['uf'] = pd.read_parquet("data/raw/uf_silver.parquet")

    for nome, df in dados.items():
        print(f"\n{nome}:")
        print(f"  Shape: {df.shape}")
        print(f"  Colunas: {list(df.columns)}")

    return dados

def analisar_distribuicao_alfabetizacao(dados):
    """Analisa a distribuição da taxa de alfabetização."""
    print("\n" + "=" * 60)
    print("DISTRIBUIÇÃO DA TAXA DE ALFABETIZAÇÃO")
    print("=" * 60)

    df = dados['municipio']
    df_2024 = df[(df['ano'] == 2024) & (df['rede'] == 'Total')]

    print(f"\nEstatísticas descritivas (municípios 2024 - rede Total):")
    stats = df_2024['taxa_alfabetizacao'].describe()
    print(stats.to_string())

    # Histograma
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df_2024['taxa_alfabetizacao'].dropna(), bins=30,
                 color='steelblue', edgecolor='white', alpha=0.8)
    axes[0].axvline(x=80, color='red', linestyle='--', linewidth=2,
                    label='Meta 2030 (80%)')
    axes[0].axvline(x=df_2024['taxa_alfabetizacao'].mean(), color='orange',
                    linestyle='--', linewidth=2,
                    label=f'Média ({df_2024["taxa_alfabetizacao"].mean():.1f}%)')
    axes[0].set_title('Distribuição da Taxa de Alfabetização\nMunicípios 2024')
    axes[0].set_xlabel('Taxa de Alfabetização (%)')
    axes[0].set_ylabel('Quantidade de Municípios')
    axes[0].legend()

    # Boxplot por ano
    df_total = df[df['rede'] == 'Total']
    df_total.boxplot(column='taxa_alfabetizacao', by='ano', ax=axes[1])
    axes[1].set_title('Taxa de Alfabetização por Ano')
    axes[1].set_xlabel('Ano')
    axes[1].set_ylabel('Taxa de Alfabetização (%)')

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / '01_distribuicao_alfabetizacao.png', dpi=150)
    plt.close()
    print("\n✓ Gráfico salvo: images/01_distribuicao_alfabetizacao.png")

def analisar_por_estado(dados):
    """Analisa desempenho por estado."""
    print("\n" + "=" * 60)
    print("ANÁLISE POR ESTADO")
    print("=" * 60)

    df = dados['ranking']
    df_2024 = df[df['ano'] == 2024].sort_values('taxa_alfabetizacao',
                                                  ascending=True)

    print(f"\nTop 5 estados (2024):")
    print(df_2024.tail(5)[['sigla_uf', 'taxa_alfabetizacao',
                             'situacao_meta_2030']].to_string(index=False))

    print(f"\nBottom 5 estados (2024):")
    print(df_2024.head(5)[['sigla_uf', 'taxa_alfabetizacao',
                             'situacao_meta_2030']].to_string(index=False))

    # Gráfico de barras horizontal
    fig, ax = plt.subplots(figsize=(12, 10))

    cores = ['#d32f2f' if t < 50 else '#ff9800' if t < 70
             else '#4caf50' for t in df_2024['taxa_alfabetizacao']]

    bars = ax.barh(df_2024['sigla_uf'], df_2024['taxa_alfabetizacao'],
                   color=cores, alpha=0.85)
    ax.axvline(x=80, color='red', linestyle='--', linewidth=2,
               label='Meta 2030 (80%)')
    ax.set_title('Taxa de Alfabetização por Estado — 2024\n(Rede Total)',
                 fontsize=14)
    ax.set_xlabel('Taxa de Alfabetização (%)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / '02_ranking_estados.png', dpi=150)
    plt.close()
    print("\n✓ Gráfico salvo: images/02_ranking_estados.png")

def analisar_evolucao_temporal(dados):
    """Analisa evolução 2023 → 2024."""
    print("\n" + "=" * 60)
    print("EVOLUÇÃO TEMPORAL 2023 → 2024")
    print("=" * 60)

    df = dados['evolucao']

    avancaram = df[df['tendencia'] == 'Avançou']
    regrediram = df[df['tendencia'] == 'Regrediu']

    print(f"\nEstados que avançaram: {len(avancaram)}")
    print(f"Estados que regrediram: {len(regrediram)}")
    print(f"\nMaior avanço: {df.iloc[0]['sigla_uf']} "
          f"(+{df.iloc[0]['variacao_pontos']:.1f} pts)")
    print(f"Maior queda: {df.iloc[-1]['sigla_uf']} "
          f"({df.iloc[-1]['variacao_pontos']:.1f} pts)")

    # Gráfico de variação
    df_sorted = df.sort_values('variacao_pontos')
    cores = ['#d32f2f' if v < 0 else '#4caf50'
             for v in df_sorted['variacao_pontos']]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(df_sorted['sigla_uf'], df_sorted['variacao_pontos'],
            color=cores, alpha=0.85)
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_title('Variação da Taxa de Alfabetização\n2023 → 2024 por Estado',
                 fontsize=14)
    ax.set_xlabel('Variação em Pontos Percentuais')

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / '03_evolucao_temporal.png', dpi=150)
    plt.close()
    print("\n✓ Gráfico salvo: images/03_evolucao_temporal.png")

def analisar_missing_values(dados):
    """Analisa valores ausentes nos datasets."""
    print("\n" + "=" * 60)
    print("ANÁLISE DE VALORES AUSENTES")
    print("=" * 60)

    for nome, df in dados.items():
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) > 0:
            print(f"\n{nome}:")
            for col, count in missing.items():
                pct = count / len(df) * 100
                print(f"  {col}: {count} ({pct:.1f}%)")
        else:
            print(f"\n{nome}: ✓ Sem valores ausentes")

def formular_hipoteses(dados):
    """Formula hipóteses analíticas baseadas na EDA."""
    print("\n" + "=" * 60)
    print("HIPÓTESES ANALÍTICAS")
    print("=" * 60)

    hipoteses = [
        "H1: Municípios com taxa < 50% em 2023 têm maior probabilidade "
        "de não atingir a meta em 2030",
        "H2: Estados do Norte e Nordeste apresentam taxas "
        "sistematicamente menores que Sul e Sudeste",
        "H3: A variação positiva de 2023→2024 é preditor de atingimento "
        "da meta 2030",
        "H4: Municípios com alta participação percentual tendem a ter "
        "melhores taxas de alfabetização",
        "H5: A distância atual da meta 2030 é o fator mais importante "
        "para classificar municípios em risco"
    ]

    for h in hipoteses:
        print(f"\n  • {h}")

def main():
    print("=" * 60)
    print("EDA INICIAL — FASE 3 TECH CHALLENGE FIAP")
    print("Predição de Alfabetização no Brasil")
    print("=" * 60)

    dados = carregar_dados()
    analisar_distribuicao_alfabetizacao(dados)
    analisar_por_estado(dados)
    analisar_evolucao_temporal(dados)
    analisar_missing_values(dados)
    formular_hipoteses(dados)

    print("\n" + "=" * 60)
    print("✓ EDA Inicial concluída!")
    print(f"  Gráficos salvos em: images/")
    print("=" * 60)

if __name__ == "__main__":
    main()

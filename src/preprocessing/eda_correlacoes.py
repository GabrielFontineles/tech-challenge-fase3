"""
EDA Correlações — Análise de Correlações e Padrões
Fase 3 — Tech Challenge FIAP
Analisa correlações entre variáveis e identifica padrões
relevantes para a modelagem preditiva.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.style.use('seaborn-v0_8')
IMAGES_DIR = Path("images")
IMAGES_DIR.mkdir(exist_ok=True)

# Mapeamento de UF para região
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
    df_municipio = pd.read_parquet("data/raw/municipio_silver.parquet")
    df_uf = pd.read_parquet("data/raw/uf_silver.parquet")
    df_evolucao = pd.read_parquet("data/gold/evolucao_temporal.parquet")
    df_ranking = pd.read_parquet("data/gold/ranking_estados.parquet")
    return df_municipio, df_uf, df_evolucao, df_ranking

def analisar_correlacoes_municipio(df_municipio):
    """Matriz de correlação das variáveis numéricas do município."""
    print("=" * 60)
    print("MATRIZ DE CORRELAÇÕES — MUNICÍPIOS")
    print("=" * 60)

    df = df_municipio[
        (df_municipio['ano'] == 2024) &
        (df_municipio['rede'] == 'Total')
    ].copy()

    colunas_numericas = [
        'taxa_alfabetizacao', 'media_portugues',
        'meta_alfabetizacao_2030', 'distancia_meta_2030',
        'percentual_participacao', 'nivel_alfabetizacao',
        'proporcao_aluno_nivel_0', 'proporcao_aluno_nivel_1',
        'proporcao_aluno_nivel_2', 'proporcao_aluno_nivel_3'
    ]

    colunas_disponiveis = [c for c in colunas_numericas if c in df.columns]
    corr = df[colunas_disponiveis].corr()

    # Top correlações com taxa_alfabetizacao
    print("\nTop correlações com taxa_alfabetizacao:")
    corr_taxa = corr['taxa_alfabetizacao'].drop('taxa_alfabetizacao').sort_values(
        key=abs, ascending=False)
    for col, val in corr_taxa.head(8).items():
        print(f"  {col}: {val:.3f}")

    # Heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, ax=ax, square=True, linewidths=0.5)
    ax.set_title('Matriz de Correlações — Municípios 2024', fontsize=14)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / '04_correlacoes_municipio.png', dpi=150)
    plt.close()
    print("\n✓ Gráfico salvo: images/04_correlacoes_municipio.png")

def analisar_por_regiao(df_ranking):
    """Analisa desempenho por região geográfica."""
    print("\n" + "=" * 60)
    print("ANÁLISE POR REGIÃO GEOGRÁFICA")
    print("=" * 60)

    df = df_ranking[df_ranking['ano'] == 2024].copy()
    df['regiao'] = df['sigla_uf'].map(REGIOES)

    resumo = df.groupby('regiao')['taxa_alfabetizacao'].agg(
        ['mean', 'min', 'max', 'std']).round(2)
    resumo.columns = ['Média', 'Mínimo', 'Máximo', 'Desvio Padrão']
    print("\nDesempenho por região (2024):")
    print(resumo.to_string())

    # Boxplot por região
    fig, ax = plt.subplots(figsize=(12, 6))
    ordem = df.groupby('regiao')['taxa_alfabetizacao'].mean().sort_values(
        ascending=False).index

    df.boxplot(column='taxa_alfabetizacao', by='regiao', ax=ax,
               positions=range(len(ordem)))
    ax.axhline(y=80, color='red', linestyle='--', linewidth=2,
               label='Meta 2030 (80%)')
    ax.set_xticklabels(ordem)
    ax.set_title('Taxa de Alfabetização por Região — 2024')
    ax.set_xlabel('Região')
    ax.set_ylabel('Taxa de Alfabetização (%)')
    ax.legend()
    plt.suptitle('')
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / '05_analise_regional.png', dpi=150)
    plt.close()
    print("\n✓ Gráfico salvo: images/05_analise_regional.png")

    # Valida hipótese H2
    media_nordeste_norte = df[df['regiao'].isin(['Nordeste', 'Norte'])][
        'taxa_alfabetizacao'].mean()
    media_sul_sudeste = df[df['regiao'].isin(['Sul', 'Sudeste'])][
        'taxa_alfabetizacao'].mean()
    print(f"\nValidação H2:")
    print(f"  Média Norte/Nordeste: {media_nordeste_norte:.1f}%")
    print(f"  Média Sul/Sudeste: {media_sul_sudeste:.1f}%")
    print(f"  Gap: {media_sul_sudeste - media_nordeste_norte:.1f} pontos")
    print(f"  H2 {'CONFIRMADA' if media_sul_sudeste > media_nordeste_norte else 'REFUTADA'} ✓")

def analisar_municipios_risco(df_municipio):
    """Identifica municípios em situação crítica."""
    print("\n" + "=" * 60)
    print("MUNICÍPIOS EM SITUAÇÃO DE RISCO")
    print("=" * 60)

    df = df_municipio[
        (df_municipio['ano'] == 2024) &
        (df_municipio['rede'] == 'Total')
    ].copy()

    # Classifica municípios
    df['situacao'] = pd.cut(
        df['taxa_alfabetizacao'],
        bins=[0, 40, 60, 80, 100],
        labels=['Crítico (<40%)', 'Em risco (40-60%)',
                'No caminho (60-80%)', 'Meta atingida (>80%)']
    )

    distribuicao = df['situacao'].value_counts()
    print("\nDistribuição dos municípios por situação (2024):")
    for situacao, count in distribuicao.items():
        pct = count / len(df) * 100
        print(f"  {situacao}: {count} municípios ({pct:.1f}%)")

    # Gráfico de pizza
    fig, ax = plt.subplots(figsize=(10, 8))
    cores = ['#d32f2f', '#ff9800', '#2196f3', '#4caf50']
    ax.pie(distribuicao.values, labels=distribuicao.index,
           colors=cores, autopct='%1.1f%%', startangle=90)
    ax.set_title('Distribuição dos Municípios por Situação\nem Relação à Meta 2030', fontsize=14)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / '06_municipios_risco.png', dpi=150)
    plt.close()
    print("\n✓ Gráfico salvo: images/06_municipios_risco.png")

def analisar_participacao_vs_taxa(df_municipio):
    """Analisa relação entre participação e taxa de alfabetização."""
    print("\n" + "=" * 60)
    print("PARTICIPAÇÃO vs TAXA DE ALFABETIZAÇÃO")
    print("=" * 60)

    df = df_municipio[
        (df_municipio['ano'] == 2024) &
        (df_municipio['rede'] == 'Total') &
        (df_municipio['percentual_participacao'].notna())
    ].copy()

    correlacao = df['percentual_participacao'].corr(df['taxa_alfabetizacao'])
    print(f"\nCorrelação participação × taxa: {correlacao:.3f}")
    print(f"H4 {'CONFIRMADA' if abs(correlacao) > 0.2 else 'FRACA'}")

    # Scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df['percentual_participacao'], df['taxa_alfabetizacao'],
               alpha=0.3, color='steelblue', s=10)
    ax.axhline(y=80, color='red', linestyle='--', linewidth=1.5,
               label='Meta 2030')

    # Linha de tendência
    z = np.polyfit(df['percentual_participacao'].dropna(),
                   df.loc[df['percentual_participacao'].notna(),
                          'taxa_alfabetizacao'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['percentual_participacao'].min(),
                         df['percentual_participacao'].max(), 100)
    ax.plot(x_line, p(x_line), 'orange', linewidth=2, label='Tendência')

    ax.set_xlabel('Percentual de Participação (%)')
    ax.set_ylabel('Taxa de Alfabetização (%)')
    ax.set_title(f'Participação vs Taxa de Alfabetização\n'
                 f'(correlação: {correlacao:.3f})', fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / '07_participacao_vs_taxa.png', dpi=150)
    plt.close()
    print("\n✓ Gráfico salvo: images/07_participacao_vs_taxa.png")

def main():
    print("=" * 60)
    print("EDA CORRELAÇÕES — FASE 3 TECH CHALLENGE FIAP")
    print("=" * 60)

    df_municipio, df_uf, df_evolucao, df_ranking = carregar_dados()
    analisar_correlacoes_municipio(df_municipio)
    analisar_por_regiao(df_ranking)
    analisar_municipios_risco(df_municipio)
    analisar_participacao_vs_taxa(df_municipio)

    print("\n" + "=" * 60)
    print("✓ EDA Correlações concluída!")
    print(f"  4 novos gráficos salvos em: images/")
    print("=" * 60)

if __name__ == "__main__":
    main()

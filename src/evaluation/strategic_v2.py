"""
Análise Estratégica v2 — Usa modelo salvo
Fase 3 — Tech Challenge FIAP

Responde perguntas de negócio usando o modelo treinado:
- Ranking de municípios por risco
- Clustering por perfil socioeconômico
- Projeção até 2030
- Lista de priorização para política pública
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
IMAGES_DIR = Path("images")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

TARGET = 'em_risco_2024'

REGIOES = {
    11: 'Norte', 12: 'Norte', 13: 'Norte', 14: 'Norte', 15: 'Norte',
    16: 'Norte', 17: 'Norte', 21: 'Nordeste', 22: 'Nordeste',
    23: 'Nordeste', 24: 'Nordeste', 25: 'Nordeste', 26: 'Nordeste',
    27: 'Nordeste', 28: 'Nordeste', 29: 'Nordeste', 31: 'Sudeste',
    32: 'Sudeste', 33: 'Sudeste', 35: 'Sudeste', 41: 'Sul',
    42: 'Sul', 43: 'Sul', 50: 'Centro-Oeste', 51: 'Centro-Oeste',
    52: 'Centro-Oeste', 53: 'Centro-Oeste'
}

def carregar_tudo():
    """Carrega modelo, metadata e dataset."""
    print("Carregando modelo e dados...")
    pipeline = joblib.load("models/modelo_final.joblib")

    with open("models/metadata.json", encoding='utf-8') as f:
        metadata = json.load(f)

    threshold = metadata['threshold']
    features_num = metadata['features_numericas']
    features_cat = metadata['features_categoricas']

    caminho = Path("data/processed/dataset_enriquecido_v2.parquet")
    if not caminho.exists():
        caminho = Path("data/processed/dataset_modelagem_v2.parquet")
    df = pd.read_parquet(caminho)

    features_all = [f for f in features_num + features_cat if f in df.columns]
    X = df[features_all]
    y = df[TARGET]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"  Modelo: {metadata['modelo']} | Threshold: {threshold:.3f}")
    print(f"  Dataset: {df.shape}")
    return pipeline, df, X, y, threshold

def ranking_municipios_risco(pipeline, df, X, threshold):
    """Q1: Ranking de municípios por probabilidade de risco."""
    print("\n" + "=" * 60)
    print("Q1: RANKING DE MUNICÍPIOS POR RISCO")
    print("=" * 60)

    prob_risco = pipeline.predict_proba(X)[:, 1]
    df_rank = df[['id_municipio', 'taxa_alf_2023', 'em_risco_2024']].copy()
    df_rank['prob_risco'] = prob_risco
    df_rank['cod_uf'] = df_rank['id_municipio'].astype(str).str[:2].astype(int)
    df_rank['regiao'] = df_rank['cod_uf'].map(REGIOES)

    df_rank['nivel_risco'] = pd.cut(
        prob_risco,
        bins=[0, 0.3, 0.5, 0.7, 1.0],
        labels=['Baixo', 'Moderado', 'Alto', 'Crítico']
    )

    dist = df_rank['nivel_risco'].value_counts()
    print("\nDistribuição por nível de risco:")
    for nivel, count in dist.sort_index().items():
        pct = count / len(df_rank) * 100
        print(f"  {nivel}: {count} municípios ({pct:.1f}%)")

    # Top 20 municípios de maior risco
    top_risco = df_rank.nlargest(20, 'prob_risco')[
        ['id_municipio', 'regiao', 'taxa_alf_2023', 'prob_risco']
    ]
    print(f"\nTop 20 municípios de maior risco:")
    print(top_risco.to_string(index=False))

    # Gráfico distribuição de probabilidades
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(prob_risco, bins=30, color='steelblue',
                 edgecolor='white', alpha=0.85)
    axes[0].axvline(x=threshold, color='red', linestyle='--',
                    linewidth=2, label=f'Threshold ({threshold:.2f})')
    axes[0].set_xlabel('Probabilidade de Risco')
    axes[0].set_ylabel('Municípios')
    axes[0].set_title('Distribuição de Probabilidades\nModelo v2')
    axes[0].legend()

    risco_regiao = df_rank.groupby('regiao')['prob_risco'].mean().sort_values()
    cores = ['#d32f2f' if v > 0.5 else '#ff9800' if v > 0.4
             else '#4caf50' for v in risco_regiao.values]
    axes[1].barh(risco_regiao.index, risco_regiao.values,
                 color=cores, alpha=0.85)
    axes[1].axvline(x=0.5, color='red', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Probabilidade Média de Risco')
    axes[1].set_title('Risco Médio por Região')

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / '18_ranking_risco_v2.png', dpi=150)
    plt.close()
    print("\n✓ Gráfico salvo: images/18_ranking_risco_v2.png")

    return df_rank

def clustering_perfil(df):
    """Q2: Clustering por perfil socioeconômico e educacional."""
    print("\n" + "=" * 60)
    print("Q2: CLUSTERING POR PERFIL EDUCACIONAL")
    print("=" * 60)

    features_cluster = ['taxa_alf_2023', 'media_pt_2023', 'particip_2023']
    features_disponiveis = [f for f in features_cluster if f in df.columns]

    df_cluster = df[features_disponiveis].dropna()

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df_cluster)

    # Silhueta para escolher k
    from sklearn.metrics import silhouette_score
    scores = {}
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        scores[k] = silhouette_score(X_scaled, labels)

    melhor_k = max(scores, key=scores.get)
    print(f"\nMelhor k por silhueta: {melhor_k} (score: {scores[melhor_k]:.3f})")

    kmeans = KMeans(n_clusters=melhor_k, random_state=42, n_init=10)
    df.loc[df_cluster.index, 'cluster'] = kmeans.fit_predict(X_scaled)

    perfil = df.groupby('cluster')[features_disponiveis].mean().round(2)
    print("\nPerfil médio por cluster:")
    print(perfil.to_string())

    # Nomeia por taxa média
    ordem_taxa = perfil['taxa_alf_2023'].sort_values()
    nomes_cluster = {}
    labels_nomes = ['Crítico', 'Vulnerável', 'Em desenvolvimento', 'Avançado']
    for i, (idx, _) in enumerate(ordem_taxa.items()):
        nomes_cluster[idx] = labels_nomes[i] if i < len(labels_nomes) else f'Grupo {i}'

    print(f"\nNomenclatura dos clusters:")
    for c, nome in nomes_cluster.items():
        taxa = perfil.loc[c, 'taxa_alf_2023']
        print(f"  Cluster {int(c)}: {nome} (taxa média: {taxa}%)")

    # Gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        df.loc[df_cluster.index, 'taxa_alf_2023'],
        df.loc[df_cluster.index, 'media_pt_2023'],
        c=df.loc[df_cluster.index, 'cluster'],
        cmap='RdYlGn', alpha=0.4, s=15
    )
    ax.set_xlabel('Taxa de Alfabetização 2023 (%)')
    ax.set_ylabel('Média Português 2023')
    ax.set_title(f'Clustering de Municípios (K-Means, k={melhor_k})')
    plt.colorbar(scatter, ax=ax, label='Cluster')
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / '19_clustering_v2.png', dpi=150)
    plt.close()
    print("\n✓ Gráfico salvo: images/19_clustering_v2.png")

    return df

def projecao_2030(df):
    """Q3: Projeção de municípios que não atingirão a meta 2030."""
    print("\n" + "=" * 60)
    print("Q3: PROJEÇÃO ATÉ 2030")
    print("=" * 60)

    df_proj = df[['id_municipio', 'taxa_alf_2023',
                  'em_risco_2024', 'meta_2030']].copy()
    df_proj = df_proj.dropna(subset=['meta_2030'])

    # Variação anual observada
    variacao_anual = 2.36  # média nacional 2023→2024

    anos_restantes = 2030 - 2024
    df_proj['taxa_projetada_2030'] = (
        df_proj['taxa_alf_2023'] + variacao_anual * (anos_restantes + 1)
    )
    df_proj['atingira_meta'] = (
        df_proj['taxa_projetada_2030'] >= df_proj['meta_2030']
    )

    total = len(df_proj)
    atingira = df_proj['atingira_meta'].sum()
    nao_atingira = total - atingira

    print(f"\nProjeção com variação anual de {variacao_anual:.2f} pts:")
    print(f"  Municípios que atingirão a meta: {atingira} ({atingira/total*100:.1f}%)")
    print(f"  Municípios que NÃO atingirão:    {nao_atingira} ({nao_atingira/total*100:.1f}%)")

    # Gráfico
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df_proj['taxa_alf_2023'],
               df_proj['taxa_projetada_2030'],
               c=df_proj['atingira_meta'].map({True: '#4caf50', False: '#d32f2f'}),
               alpha=0.4, s=10)
    ax.axhline(y=80, color='red', linestyle='--', linewidth=2, label='Meta 2030 (80%)')
    ax.plot([0, 100], [0, 100], 'k--', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('Taxa de Alfabetização 2023 (%)')
    ax.set_ylabel('Taxa Projetada 2030 (%)')
    ax.set_title('Projeção de Municípios até 2030\n(verde = atingirá meta | vermelho = não atingirá)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / '20_projecao_2030_v2.png', dpi=150)
    plt.close()
    print("\n✓ Gráfico salvo: images/20_projecao_2030_v2.png")

    return df_proj

def gerar_lista_priorizacao(df_rank, df_proj):
    """Gera lista de municípios prioritários para intervenção."""
    print("\n" + "=" * 60)
    print("LISTA DE PRIORIZAÇÃO")
    print("=" * 60)

    df_prior = pd.merge(
        df_rank[['id_municipio', 'prob_risco', 'taxa_alf_2023',
                 'regiao', 'nivel_risco']],
        df_proj[['id_municipio', 'taxa_projetada_2030',
                 'meta_2030', 'atingira_meta']],
        on='id_municipio', how='inner'
    )

    # Prioridade: alto risco + não atingirá meta
    df_prior['prioridade'] = (
        df_prior['prob_risco'] * (1 - df_prior['atingira_meta'].astype(int))
    )

    top_prior = df_prior.nlargest(50, 'prob_risco')[
        ['id_municipio', 'regiao', 'taxa_alf_2023',
         'prob_risco', 'taxa_projetada_2030', 'meta_2030', 'atingira_meta']
    ].round(2)

    caminho = REPORTS_DIR / "priorizacao_municipios_v2.csv"
    top_prior.to_csv(caminho, index=False)
    print(f"\n✓ Lista salva em: {caminho}")
    print(f"\nTop 10 municípios prioritários:")
    print(top_prior.head(10).to_string(index=False))

    return top_prior

def main():
    print("=" * 60)
    print("ANÁLISE ESTRATÉGICA v2 — MODELO SALVO")
    print("=" * 60)

    pipeline, df, X, y, threshold = carregar_tudo()

    df_rank = ranking_municipios_risco(pipeline, df, X, threshold)
    df = clustering_perfil(df)
    df_proj = projecao_2030(df)
    gerar_lista_priorizacao(df_rank, df_proj)

    print("\n" + "=" * 60)
    print("✓ Análise Estratégica v2 concluída!")
    print("  3 gráficos + 1 CSV gerados")
    print("=" * 60)

if __name__ == "__main__":
    main()

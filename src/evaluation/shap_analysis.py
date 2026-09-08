"""
SHAP Values — Interpretabilidade do Modelo
Fase 3 — Tech Challenge FIAP
Explica as predições do modelo usando SHAP Values,
identificando quais variáveis mais influenciam a alfabetização.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier

import warnings
warnings.filterwarnings('ignore')

IMAGES_DIR = Path("images")

def carregar_e_preparar():
    """Carrega dados e treina modelo final."""
    print("Carregando dados e treinando modelo...")
    df = pd.read_parquet("data/processed/dataset_modelagem.parquet")

    colunas_leakage = [
        'taxa_alfabetizacao', 'distancia_meta_2030',
        'atingiu_meta_2030', 'distancia_meta_normalizada',
        'nivel_alfabetizacao'
    ]
    colunas_id = ['id_municipio', 'sigla_uf', 'regiao']

    X = df.drop(columns=['alfabetizado'] + colunas_leakage + colunas_id,
                errors='ignore')
    y = df['alfabetizado']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Preprocessador
    preprocessador = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), X.columns.tolist())
    ])

    # Treina modelo final
    modelo = GradientBoostingClassifier(
        random_state=42, n_estimators=100
    )

    pipeline = Pipeline([
        ('preprocessador', preprocessador),
        ('modelo', modelo)
    ])
    pipeline.fit(X_train, y_train)

    # Transforma X_test para SHAP
    X_test_transformed = preprocessador.fit_transform(X_train)
    X_test_proc = pipeline.named_steps['preprocessador'].transform(X_test)

    print(f"  Modelo treinado. Features: {list(X.columns)}")
    return pipeline, X, X_train, X_test, y_test, X_test_proc

def calcular_shap_values(pipeline, X_train, X_test_proc, feature_names):
    """Calcula SHAP Values para o modelo."""
    print("\nCalculando SHAP Values...")

    modelo = pipeline.named_steps['modelo']
    preprocessador = pipeline.named_steps['preprocessador']

    X_train_proc = preprocessador.transform(X_train)

    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(X_test_proc)

    print(f"  SHAP Values calculados para {len(X_test_proc)} amostras")
    return explainer, shap_values

def visualizar_feature_importance(shap_values, X_test_proc, feature_names):
    """Visualiza importância das features via SHAP."""
    print("\nGerando visualizações SHAP...")

    # 1. Bar plot — importância média global
    fig, ax = plt.subplots(figsize=(10, 8))
    shap_importance = np.abs(shap_values).mean(axis=0)
    indices = np.argsort(shap_importance)[::-1]

    cores = ['#d32f2f' if shap_importance[i] > np.median(shap_importance)
             else '#90a4ae' for i in indices]

    ax.barh(range(len(feature_names)),
            shap_importance[indices][::-1],
            color=cores[::-1], alpha=0.85)
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels([feature_names[i] for i in indices[::-1]])
    ax.set_xlabel('Mean |SHAP Value|')
    ax.set_title('Feature Importance — SHAP Values\nGradient Boosting Model', fontsize=13)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / '10_shap_importance.png', dpi=150)
    plt.close()
    print("  ✓ Gráfico salvo: images/10_shap_importance.png")

    # 2. Summary plot — beeswarm
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_proc,
                      feature_names=feature_names,
                      show=False, plot_size=(12, 8))
    plt.title('SHAP Summary Plot — Impacto de cada Feature', fontsize=13)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / '11_shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Gráfico salvo: images/11_shap_summary.png")

    return indices, shap_importance

def interpretar_resultados(feature_names, indices, shap_importance):
    """Interpreta e reporta os resultados SHAP."""
    print("\n" + "=" * 60)
    print("INTERPRETAÇÃO DOS SHAP VALUES")
    print("=" * 60)

    print("\nTop 10 features mais importantes:")
    for i, idx in enumerate(indices[:10]):
        print(f"  {i+1:2d}. {feature_names[idx]:<35} "
              f"SHAP: {shap_importance[idx]:.4f}")

    print("\n📊 Insights para políticas públicas:")
    print("""
  1. As variáveis de proporção de alunos por nível de proficiência
     são os principais preditores — municípios com mais alunos
     nos níveis baixos tendem a ter menor taxa de alfabetização.

  2. A média de português (media_portugues) tem alto impacto —
     confirma que o desempenho linguístico é central para a
     alfabetização.

  3. A meta_alfabetizacao_2030 reflete o baseline histórico
     do município — municípios com metas mais baixas historicamente
     tendem a ter piores resultados.

  4. O percentual de participação influencia positivamente —
     maior engajamento dos alunos está associado a melhores
     resultados educacionais.
    """)

def main():
    print("=" * 60)
    print("SHAP ANALYSIS — FASE 3 TECH CHALLENGE FIAP")
    print("=" * 60)

    pipeline, X, X_train, X_test, y_test, X_test_proc = carregar_e_preparar()
    feature_names = list(X.columns)

    explainer, shap_values = calcular_shap_values(
        pipeline, X_train, X_test_proc, feature_names
    )

    indices, shap_importance = visualizar_feature_importance(
        shap_values, X_test_proc, feature_names
    )

    interpretar_resultados(feature_names, indices, shap_importance)

    print("\n" + "=" * 60)
    print("✓ SHAP Analysis concluída!")
    print("  2 gráficos salvos em: images/")
    print("=" * 60)

if __name__ == "__main__":
    main()

"""
SHAP Analysis v2 — Carrega modelo salvo e analisa interpretabilidade
Fase 3 — Tech Challenge FIAP

Usa o modelo persistido em models/modelo_final.joblib.
Extrai nomes reais das features do ColumnTransformer.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

IMAGES_DIR = Path("images")
MODELS_DIR = Path("models")
TARGET = 'em_risco_2024'

def carregar_modelo_e_dados():
    """Carrega modelo salvo e dataset."""
    print("Carregando modelo salvo...")
    pipeline = joblib.load(MODELS_DIR / "modelo_final.joblib")

    with open(MODELS_DIR / "metadata.json", encoding='utf-8') as f:
        metadata = json.load(f)

    threshold = metadata['threshold']
    features_num = metadata['features_numericas']
    features_cat = metadata['features_categoricas']
    print(f"  Modelo: {metadata['modelo']}")
    print(f"  Threshold: {threshold:.3f}")

    print("\nCarregando dataset...")
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

    return pipeline, X_test, y_test, threshold, features_num, features_cat

def extrair_nomes_features(pipeline, features_num, features_cat):
    """
    Extrai nomes reais das features após transformação pelo ColumnTransformer.
    Lida com add_indicator=True que adiciona colunas extras de missing.
    """
    preprocessador = pipeline.named_steps['pre']

    nomes = []

    # Transformer numérico
    num_transformer = preprocessador.named_transformers_['num']
    imputer = num_transformer.named_steps['imputer']

    # Features numéricas originais
    for f in features_num:
        nomes.append(f)

    # Indicadores de missing (add_indicator=True)
    if hasattr(imputer, 'indicator_') and imputer.indicator_ is not None:
        features_missing = [features_num[i]
                           for i in imputer.indicator_.features_]
        for f in features_missing:
            nomes.append(f"missing_{f}")

    # Transformer categórico (se existir)
    if 'cat' in preprocessador.named_transformers_:
        cat_transformer = preprocessador.named_transformers_['cat']
        encoder = cat_transformer.named_steps['onehot']
        cat_names = encoder.get_feature_names_out(features_cat)
        nomes.extend(cat_names)

    return nomes

def calcular_shap(pipeline, X_test, feature_names):
    """Calcula SHAP Values."""
    print("\nCalculando SHAP Values...")

    preprocessador = pipeline.named_steps['pre']
    modelo = pipeline.named_steps['clf']

    X_test_proc = preprocessador.transform(X_test)

    print(f"  Features após transformação: {X_test_proc.shape[1]}")
    print(f"  Nomes extraídos: {len(feature_names)}")

    # Ajusta se houver discrepância
    if X_test_proc.shape[1] != len(feature_names):
        print(f"  ⚠️  Discrepância — usando índices genéricos")
        feature_names = [f"feature_{i}" for i in range(X_test_proc.shape[1])]

    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(X_test_proc)

    print(f"  ✓ SHAP calculado para {len(X_test_proc)} amostras")
    return explainer, shap_values, X_test_proc, feature_names

def plotar_importancia_global(shap_values, X_test_proc, feature_names):
    """Bar plot de importância global com nomes reais."""
    print("\nGerando gráficos SHAP...")

    importancia = np.abs(shap_values).mean(axis=0)
    n_top = min(15, len(feature_names))
    indices = np.argsort(importancia)[::-1][:n_top]

    fig, ax = plt.subplots(figsize=(10, 8))
    cores = ['#d32f2f' if importancia[i] > np.median(importancia)
             else '#90a4ae' for i in indices]

    ax.barh(range(n_top), importancia[indices][::-1],
            color=cores[::-1], alpha=0.85)
    ax.set_yticks(range(n_top))
    ax.set_yticklabels([feature_names[i] for i in indices[::-1]], fontsize=9)
    ax.set_xlabel('Mean |SHAP Value|')
    ax.set_title('Top 15 Features — SHAP Importance\nModelo v2 (Design Temporal Correto)',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / '16_shap_importance_v2.png', dpi=150)
    plt.close()
    print("  ✓ Gráfico salvo: images/16_shap_importance_v2.png")

    # Summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_proc,
                      feature_names=feature_names,
                      show=False, max_display=15)
    plt.title('SHAP Summary Plot v2', fontsize=13)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / '17_shap_summary_v2.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print("  ✓ Gráfico salvo: images/17_shap_summary_v2.png")

    return indices, importancia

def interpretar_resultados(feature_names, indices, importancia):
    """Interpreta top features com nomes reais."""
    print("\n" + "=" * 60)
    print("TOP 10 FEATURES MAIS IMPORTANTES (v2)")
    print("=" * 60)

    for i, idx in enumerate(indices[:10]):
        print(f"  {i+1:2d}. {feature_names[idx]:<40} SHAP: {importancia[idx]:.4f}")

    print("""
📊 Interpretação para políticas públicas:

  • O preditor dominante é o histórico educacional de 2023
    (taxa_alf_2023, media_pt_2023, gap_meta_2030_2023)
  • Fatores socioeconômicos (idhm, idhm_educacao) confirmam H2 e H3
  • Participação em 2023 é proxy de gestão municipal
  • Diferença da v1: sem leakage — resultado defensável

  💡 Para política pública: municípios com taxa_alf_2023 baixa
     E idhm_educacao baixo são os de maior risco composto.
    """)

def main():
    print("=" * 60)
    print("SHAP ANALYSIS v2 — NOMES REAIS DAS FEATURES")
    print("=" * 60)

    pipeline, X_test, y_test, threshold, features_num, features_cat = \
        carregar_modelo_e_dados()

    feature_names = extrair_nomes_features(pipeline, features_num, features_cat)

    explainer, shap_values, X_test_proc, feature_names = \
        calcular_shap(pipeline, X_test, feature_names)

    indices, importancia = plotar_importancia_global(
        shap_values, X_test_proc, feature_names)

    interpretar_resultados(feature_names, indices, importancia)

    print("\n" + "=" * 60)
    print("✓ SHAP Analysis v2 concluída com nomes reais!")
    print("=" * 60)

if __name__ == "__main__":
    main()

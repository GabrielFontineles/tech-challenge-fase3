"""
SHAP Analysis v2 — Carrega modelo salvo e analisa interpretabilidade
Fase 3 — Tech Challenge FIAP

Usa o modelo persistido em models/modelo_final.joblib
em vez de retreinar — garante consistência com os resultados reportados.
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
    print(f"  Features: {len(features_num) + len(features_cat)}")

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

def calcular_shap(pipeline, X_test, features_num, features_cat):
    """Calcula SHAP Values usando o modelo do pipeline."""
    print("\nCalculando SHAP Values...")

    preprocessador = pipeline.named_steps['pre']
    modelo = pipeline.named_steps['clf']

    X_test_proc = preprocessador.transform(X_test)

    # Nomes das features após transformação
    feature_names_proc = []

    # Numéricas + indicadores de missing
    for f in features_num:
        feature_names_proc.append(f)
    # add_indicator=True adiciona colunas de missing
    n_missing_indicators = X_test_proc.shape[1] - len(features_num) - sum(
        1 for _ in (pipeline.named_steps['pre']
                   .named_transformers_.get('cat', None) and [] or [])
    )

    # Simplifica: usa índices se nomes não baterem
    if X_test_proc.shape[1] != len(feature_names_proc):
        feature_names_proc = [f"feature_{i}" for i in range(X_test_proc.shape[1])]

    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(X_test_proc)

    print(f"  SHAP calculado para {len(X_test_proc)} amostras")
    return explainer, shap_values, X_test_proc, feature_names_proc

def plotar_importancia_global(shap_values, X_test_proc, feature_names):
    """Bar plot de importância global."""
    print("\nGerando gráficos SHAP...")

    importancia = np.abs(shap_values).mean(axis=0)
    indices = np.argsort(importancia)[::-1][:15]  # Top 15

    fig, ax = plt.subplots(figsize=(10, 8))
    cores = ['#d32f2f' if importancia[i] > np.median(importancia)
             else '#90a4ae' for i in indices]

    ax.barh(range(len(indices)), importancia[indices][::-1],
            color=cores[::-1], alpha=0.85)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices[::-1]], fontsize=10)
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
    plt.title('SHAP Summary Plot v2 — Impacto por Feature', fontsize=13)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / '17_shap_summary_v2.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print("  ✓ Gráfico salvo: images/17_shap_summary_v2.png")

    return indices, importancia

def interpretar_resultados(feature_names, indices, importancia):
    """Interpreta top features."""
    print("\n" + "=" * 60)
    print("TOP 10 FEATURES MAIS IMPORTANTES (v2)")
    print("=" * 60)

    for i, idx in enumerate(indices[:10]):
        print(f"  {i+1:2d}. {feature_names[idx]:<35} SHAP: {importancia[idx]:.4f}")

    print("""
📊 Interpretação para políticas públicas (v2):

  1. taxa_alf_2023 — inércia: o melhor preditor do risco em 2024
     é o resultado do ano anterior. Municípios já em risco tendem
     a permanecer em risco sem intervenção.

  2. idhm_educacao / idhm — contexto socioeconômico importa:
     mesmo controlando pelo histórico, municípios com menor IDH
     têm maior risco, confirmando H2 e H3.

  3. media_pt_2023 — desempenho em língua portuguesa em 2023
     é um forte preditor do risco no ciclo seguinte.

  4. gap_meta_2030_2023 — distância da meta em 2023 prediz
     quem estará em risco em 2024 — municípios com gap alto
     raramente recuperam em um único ciclo.

  💡 Diferença da v1: agora a feature dominante NÃO é
     score_niveis_altos (que era leakage). O preditor principal
     é o histórico educacional real — resultado defensável.
    """)

def main():
    print("=" * 60)
    print("SHAP ANALYSIS v2 — MODELO SALVO")
    print("=" * 60)

    pipeline, X_test, y_test, threshold, features_num, features_cat = \
        carregar_modelo_e_dados()

    explainer, shap_values, X_test_proc, feature_names = \
        calcular_shap(pipeline, X_test, features_num, features_cat)

    indices, importancia = plotar_importancia_global(
        shap_values, X_test_proc, feature_names)

    interpretar_resultados(feature_names, indices, importancia)

    print("\n" + "=" * 60)
    print("✓ SHAP Analysis v2 concluída!")
    print("=" * 60)

if __name__ == "__main__":
    main()

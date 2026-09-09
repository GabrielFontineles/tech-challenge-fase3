"""
Pipeline ML v2 — Design Temporal Correto
Fase 3 — Tech Challenge FIAP

Treina modelos para prever em_risco_2024 usando features de 2023.
Inclui: DummyClassifier baseline, RandomizedSearchCV, threshold
ajustado, calibração e persistência do modelo com joblib.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
from pathlib import Path
from datetime import datetime

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    RandomizedSearchCV, cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    f1_score, recall_score, precision_score,
    brier_score_loss
)

import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
IMAGES_DIR = Path("images")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Features por bloco
FEATURES_BLOCO_A = [
    'taxa_alf_2023', 'media_pt_2023', 'particip_2023',
    'nivel_alf_2023', 'meta_2030', 'gap_meta_2030_2023',
    'dist_meta_2030_2023', 'taxa_vs_uf_2023',
    'prop_nivel_0_2023', 'prop_nivel_1_2023', 'prop_nivel_2_2023',
    'prop_nivel_3_2023', 'prop_nivel_4_2023', 'prop_nivel_5_2023',
    'prop_nivel_6_2023', 'prop_nivel_7_2023', 'prop_nivel_8_2023',
]

FEATURES_BLOCO_B_NUM = ['populacao_2023', 'log_populacao']
FEATURES_BLOCO_B_CAT = ['sigla_uf', 'regiao', 'porte']

FEATURES_BLOCO_C = [
    'idhm', 'idhm_educacao', 'idhm_renda',
    'renda_per_capita', 'gini', 'pct_pobres',
    'pib_per_capita', 'log_pib_per_capita'
]

TARGET = 'em_risco_2024'

def carregar_dados():
    """Carrega dataset enriquecido."""
    print("Carregando dataset...")
    caminho = Path("data/processed/dataset_enriquecido_v2.parquet")
    if not caminho.exists():
        print("  Dataset enriquecido não encontrado, usando base...")
        caminho = Path("data/processed/dataset_modelagem_v2.parquet")
    df = pd.read_parquet(caminho)
    print(f"  Shape: {df.shape}")
    return df

def preparar_features(df):
    """Separa X e y, identifica colunas por tipo."""
    # Features disponíveis
    features_num = [f for f in FEATURES_BLOCO_A + FEATURES_BLOCO_B_NUM + FEATURES_BLOCO_C
                    if f in df.columns]
    features_cat = [f for f in FEATURES_BLOCO_B_CAT if f in df.columns]

    X = df[features_num + features_cat].copy()
    y = df[TARGET].copy()

    print(f"\nFeatures numéricas: {len(features_num)}")
    print(f"Features categóricas: {len(features_cat)}")
    print(f"Total features: {len(features_num) + len(features_cat)}")
    print(f"Target - Em risco (1): {y.sum()} ({y.mean()*100:.1f}%)")

    return X, y, features_num, features_cat

def criar_preprocessador(features_num, features_cat):
    """Pipeline de pré-processamento."""
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
        ('scaler', StandardScaler())
    ])

    transformers = [('num', num_pipe, features_num)]

    if features_cat:
        cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', cat_pipe, features_cat))

    return ColumnTransformer(transformers)

def avaliar_baseline(X_train, y_train, X_test, y_test, preprocessador):
    """DummyClassifier como baseline obrigatório."""
    print("\n" + "=" * 60)
    print("BASELINE — DummyClassifier")
    print("=" * 60)

    dummy = Pipeline([
        ('pre', preprocessador),
        ('clf', DummyClassifier(strategy='prior', random_state=42))
    ])
    dummy.fit(X_train, y_train)
    y_pred = dummy.predict(X_test)
    y_prob = dummy.predict_proba(X_test)[:, 1]

    print(f"  ROC-AUC:  {roc_auc_score(y_test, y_prob):.4f}")
    print(f"  PR-AUC:   {average_precision_score(y_test, y_prob):.4f}")
    print(f"  Recall:   {recall_score(y_test, y_pred):.4f}")
    return dummy

def treinar_com_cv(nome, estimador, X_train, y_train, preprocessador, cv):
    """Treina modelo com cross-validation."""
    pipeline = Pipeline([
        ('pre', preprocessador),
        ('clf', estimador)
    ])

    scores_roc = cross_val_score(pipeline, X_train, y_train,
                                  cv=cv, scoring='roc_auc')
    scores_pr = cross_val_score(pipeline, X_train, y_train,
                                 cv=cv, scoring='average_precision')
    scores_rec = cross_val_score(pipeline, X_train, y_train,
                                  cv=cv, scoring='recall')

    print(f"\n{nome}:")
    print(f"  ROC-AUC: {scores_roc.mean():.4f} ± {scores_roc.std():.4f}")
    print(f"  PR-AUC:  {scores_pr.mean():.4f} ± {scores_pr.std():.4f}")
    print(f"  Recall:  {scores_rec.mean():.4f} ± {scores_rec.std():.4f}")

    return {
        'pipeline': pipeline,
        'roc_mean': scores_roc.mean(),
        'pr_mean': scores_pr.mean(),
        'rec_mean': scores_rec.mean()
    }

def otimizar_modelo(X_train, y_train, preprocessador, cv):
    """RandomizedSearchCV no HistGradientBoosting."""
    print("\n" + "=" * 60)
    print("OTIMIZAÇÃO — RandomizedSearchCV (HistGradientBoosting)")
    print("=" * 60)

    pipeline = Pipeline([
        ('pre', preprocessador),
        ('clf', HistGradientBoostingClassifier(random_state=42))
    ])

    param_dist = {
        'clf__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'clf__max_iter': [100, 200, 300],
        'clf__max_depth': [3, 5, 7, None],
        'clf__min_samples_leaf': [10, 20, 30, 50],
        'clf__l2_regularization': [0.0, 0.1, 1.0],
        'clf__class_weight': ['balanced', None]
    }

    search = RandomizedSearchCV(
        pipeline, param_dist,
        n_iter=30, cv=cv,
        scoring='roc_auc',
        random_state=42, n_jobs=-1,
        refit=True, verbose=1
    )

    search.fit(X_train, y_train)

    print(f"\n  Melhores parâmetros:")
    for k, v in search.best_params_.items():
        print(f"    {k}: {v}")
    print(f"\n  Melhor ROC-AUC (CV): {search.best_score_:.4f}")

    return search.best_estimator_

def ajustar_threshold(pipeline, X_train, y_train, cv):
    """Ajusta threshold para maximizar recall >= 0.85."""
    print("\nAjustando threshold...")

    y_prob_cv = np.zeros(len(y_train))
    for train_idx, val_idx in cv.split(X_train, y_train):
        pipeline.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        y_prob_cv[val_idx] = pipeline.predict_proba(
            X_train.iloc[val_idx])[:, 1]

    # Retreina no conjunto completo
    pipeline.fit(X_train, y_train)

    precisoes, recalls, thresholds = precision_recall_curve(
        y_train, y_prob_cv)

    # Threshold com recall >= 0.85 e maior precisão
    mask = recalls[:-1] >= 0.85
    if mask.any():
        melhor_idx = np.argmax(precisoes[:-1][mask])
        threshold = thresholds[mask][melhor_idx]
    else:
        threshold = 0.5

    print(f"  Threshold ajustado: {threshold:.3f}")
    return pipeline, threshold

def avaliar_modelo_final(pipeline, X_test, y_test, threshold, nome_modelo):
    """Avalia modelo final no conjunto de teste."""
    print("\n" + "=" * 60)
    print(f"AVALIAÇÃO FINAL — {nome_modelo}")
    print("=" * 60)

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    roc = roc_auc_score(y_test, y_prob)
    pr = average_precision_score(y_test, y_prob)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_prob)

    print(f"\nMétricas (threshold={threshold:.3f}):")
    print(f"  ROC-AUC:   {roc:.4f}")
    print(f"  PR-AUC:    {pr:.4f}")
    print(f"  Recall:    {rec:.4f}  ← captura municípios em risco")
    print(f"  Precision: {prec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Brier:     {brier:.4f}")

    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Nao em risco', 'Em risco']))

    return {
        'roc_auc': roc, 'pr_auc': pr,
        'recall': rec, 'precision': prec,
        'f1': f1, 'brier': brier,
        'threshold': threshold
    }

def visualizar_resultados(pipeline, X_test, y_test, threshold, resultados_cv):
    """Gera gráficos de resultados."""
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    axes[0].plot(fpr, tpr, color='steelblue', linewidth=2,
                 label=f'ROC-AUC = {auc:.4f}')
    axes[0].plot([0, 1], [0, 1], 'k--')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('Curva ROC')
    axes[0].legend()

    # Curva Precision-Recall
    prec_curve, rec_curve, thresh = precision_recall_curve(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    axes[1].plot(rec_curve, prec_curve, color='orange', linewidth=2,
                 label=f'PR-AUC = {pr_auc:.4f}')
    axes[1].axvline(x=0.85, color='red', linestyle='--',
                    label='Recall alvo (0.85)')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Curva Precision-Recall')
    axes[1].legend()

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2],
                xticklabels=['Nao risco', 'Em risco'],
                yticklabels=['Nao risco', 'Em risco'])
    axes[2].set_title(f'Matriz de Confusão\n(threshold={threshold:.3f})')
    axes[2].set_ylabel('Real')
    axes[2].set_xlabel('Previsto')

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / '15_resultados_v2.png', dpi=150)
    plt.close()
    print("\n✓ Gráfico salvo: images/15_resultados_v2.png")

def salvar_modelo(pipeline, metricas, threshold, features_num, features_cat):
    """Salva modelo e metadata com joblib."""
    caminho_modelo = MODELS_DIR / "modelo_final.joblib"
    joblib.dump(pipeline, caminho_modelo)
    print(f"\n✓ Modelo salvo em: {caminho_modelo}")

    metadata = {
        'timestamp': datetime.now().isoformat(),
        'modelo': 'HistGradientBoostingClassifier',
        'threshold': threshold,
        'features_numericas': features_num,
        'features_categoricas': features_cat,
        'metricas_teste': metricas,
        'target': TARGET,
        'design': 'features_2023_target_2024',
        'corte_risco': 60.0
    }

    caminho_meta = MODELS_DIR / "metadata.json"
    with open(caminho_meta, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"✓ Metadata salvo em: {caminho_meta}")

def main():
    print("=" * 60)
    print("PIPELINE ML v2 — DESIGN TEMPORAL CORRETO")
    print(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    df = carregar_dados()
    X, y, features_num, features_cat = preparar_features(df)

    # Split treino/teste — tocado UMA ÚNICA VEZ no final
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nSplit: treino={len(X_train)} | teste={len(X_test)}")

    preprocessador = criar_preprocessador(features_num, features_cat)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Baseline
    avaliar_baseline(X_train, y_train, X_test, y_test, preprocessador)

    # Modelos baseline com CV
    print("\n" + "=" * 60)
    print("COMPARAÇÃO DE MODELOS BASELINE (Cross-Validation)")
    print("=" * 60)

    resultados_cv = {}
    modelos = {
        'Logistic Regression': LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(
            class_weight='balanced_subsample', random_state=42, n_jobs=-1),
        'HistGradientBoosting': HistGradientBoostingClassifier(random_state=42)
    }

    for nome, estimador in modelos.items():
        resultados_cv[nome] = treinar_com_cv(
            nome, estimador, X_train, y_train, preprocessador, cv)

    # Otimização do melhor modelo
    melhor_pipeline = otimizar_modelo(X_train, y_train, preprocessador, cv)

    # Ajuste de threshold
    melhor_pipeline, threshold = ajustar_threshold(
        melhor_pipeline, X_train, y_train, cv)

    # Avaliação final no teste
    metricas = avaliar_modelo_final(
        melhor_pipeline, X_test, y_test, threshold,
        'HistGradientBoosting Otimizado')

    # Visualizações
    visualizar_resultados(
        melhor_pipeline, X_test, y_test, threshold, resultados_cv)

    # Salva modelo
    salvar_modelo(melhor_pipeline, metricas, threshold,
                  features_num, features_cat)

    print(f"\n{'=' * 60}")
    print("✓ Pipeline ML v2 concluída!")
    print(f"  ROC-AUC: {metricas['roc_auc']:.4f}")
    print(f"  Recall (em risco): {metricas['recall']:.4f}")
    print(f"Fim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

if __name__ == "__main__":
    main()

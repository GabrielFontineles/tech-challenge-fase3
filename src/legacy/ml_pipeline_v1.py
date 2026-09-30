"""
Pipeline de Machine Learning — Predição de Alfabetização
Fase 3 — Tech Challenge FIAP
Treina e avalia modelos para prever municípios alfabetizados.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score
)

import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
IMAGES_DIR = Path("images")
MODELS_DIR = Path("src/modeling")

def carregar_dados():
    """Carrega dataset processado pelo feature engineering."""
    print("Carregando dataset processado...")
    df = pd.read_parquet("data/processed/dataset_modelagem.parquet")
    print(f"  Shape: {df.shape}")
    return df

def preparar_features(df):
    """Prepara X e y para modelagem."""
    print("\nPreparando features...")

    # Remove colunas que causariam data leakage
    # taxa_alfabetizacao é usada para criar o target — remover
    # distancia_meta_2030 é derivada da taxa — remover
    # atingiu_meta_2030 é derivada da taxa — remover
    colunas_leakage = [
        'taxa_alfabetizacao', 'distancia_meta_2030',
        'atingiu_meta_2030', 'distancia_meta_normalizada',
        'nivel_alfabetizacao'
    ]
    colunas_id = ['id_municipio', 'sigla_uf', 'regiao']

    X = df.drop(columns=['alfabetizado'] + colunas_leakage + colunas_id,
                errors='ignore')
    y = df['alfabetizado']

    print(f"  Features usadas: {list(X.columns)}")
    print(f"  Shape X: {X.shape} | Shape y: {y.shape}")
    print(f"  Distribuição alvo: {y.value_counts().to_dict()}")

    return X, y

def criar_preprocessador(X):
    """Cria preprocessador com imputação e encoding."""
    colunas_numericas = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    colunas_categoricas = X.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"\n  Colunas numéricas: {len(colunas_numericas)}")
    print(f"  Colunas categóricas: {len(colunas_categoricas)}")

    # Pipeline numérico: imputação + normalização
    pipeline_numerico = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline categórico: imputação + encoding
    pipeline_categorico = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessador = ColumnTransformer(
        transformers=[
            ('num', pipeline_numerico, colunas_numericas),
            ('cat', pipeline_categorico, colunas_categoricas)
        ] if colunas_categoricas else [
            ('num', pipeline_numerico, colunas_numericas)
        ]
    )

    return preprocessador

def treinar_modelos(X_train, y_train, preprocessador):
    """Treina múltiplos modelos baseline."""
    print("\n" + "=" * 60)
    print("TREINANDO MODELOS BASELINE")
    print("=" * 60)

    modelos = {
        'Logistic Regression': LogisticRegression(
            random_state=42, max_iter=1000
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=42, max_depth=10
        ),
        'Random Forest': RandomForestClassifier(
            random_state=42, n_estimators=100, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            random_state=42, n_estimators=100
        )
    }

    resultados = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for nome, modelo in modelos.items():
        pipeline = Pipeline([
            ('preprocessador', preprocessador),
            ('modelo', modelo)
        ])

        # Cross-validation
        scores_acc = cross_val_score(pipeline, X_train, y_train,
                                      cv=cv, scoring='accuracy')
        scores_f1 = cross_val_score(pipeline, X_train, y_train,
                                     cv=cv, scoring='f1')
        scores_roc = cross_val_score(pipeline, X_train, y_train,
                                      cv=cv, scoring='roc_auc')

        resultados[nome] = {
            'pipeline': pipeline,
            'acc_mean': scores_acc.mean(),
            'acc_std': scores_acc.std(),
            'f1_mean': scores_f1.mean(),
            'f1_std': scores_f1.std(),
            'roc_mean': scores_roc.mean(),
            'roc_std': scores_roc.std()
        }

        print(f"\n{nome}:")
        print(f"  Accuracy:  {scores_acc.mean():.4f} ± {scores_acc.std():.4f}")
        print(f"  F1-Score:  {scores_f1.mean():.4f} ± {scores_f1.std():.4f}")
        print(f"  ROC-AUC:   {scores_roc.mean():.4f} ± {scores_roc.std():.4f}")

    return resultados

def selecionar_melhor_modelo(resultados, X_train, y_train, X_test, y_test):
    """Seleciona o melhor modelo e avalia no conjunto de teste."""
    print("\n" + "=" * 60)
    print("SELEÇÃO DO MELHOR MODELO")
    print("=" * 60)

    # Seleciona por ROC-AUC
    melhor = max(resultados.items(),
                 key=lambda x: x[1]['roc_mean'])
    nome_melhor, info_melhor = melhor

    print(f"\nMelhor modelo: {nome_melhor}")
    print(f"  ROC-AUC CV: {info_melhor['roc_mean']:.4f}")

    # Treina no conjunto completo de treino
    pipeline = info_melhor['pipeline']
    pipeline.fit(X_train, y_train)

    # Avalia no conjunto de teste
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print(f"\nResultados no conjunto de teste:")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"  F1-Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")

    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Em risco', 'Alfabetizado']))

    return pipeline, nome_melhor, y_pred, y_prob

def visualizar_resultados(resultados, y_test, y_pred, y_prob, nome_melhor):
    """Gera visualizações dos resultados."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Comparação de modelos
    nomes = list(resultados.keys())
    roc_means = [resultados[n]['roc_mean'] for n in nomes]
    f1_means = [resultados[n]['f1_mean'] for n in nomes]

    x = np.arange(len(nomes))
    width = 0.35
    axes[0].bar(x - width/2, roc_means, width, label='ROC-AUC',
                color='steelblue', alpha=0.85)
    axes[0].bar(x + width/2, f1_means, width, label='F1-Score',
                color='orange', alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(nomes, rotation=15, ha='right')
    axes[0].set_title('Comparação de Modelos\n(Cross-Validation)')
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[0].axhline(y=0.8, color='red', linestyle='--', alpha=0.5)

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=['Em risco', 'Alfabetizado'],
                yticklabels=['Em risco', 'Alfabetizado'])
    axes[1].set_title(f'Matriz de Confusão\n{nome_melhor}')
    axes[1].set_ylabel('Real')
    axes[1].set_xlabel('Previsto')

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    axes[2].plot(fpr, tpr, color='steelblue', linewidth=2,
                 label=f'ROC Curve (AUC = {auc:.4f})')
    axes[2].plot([0, 1], [0, 1], 'k--', linewidth=1)
    axes[2].set_xlabel('False Positive Rate')
    axes[2].set_ylabel('True Positive Rate')
    axes[2].set_title(f'Curva ROC\n{nome_melhor}')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / '09_resultados_modelos.png', dpi=150)
    plt.close()
    print("\n✓ Gráfico salvo: images/09_resultados_modelos.png")

def main():
    print("=" * 60)
    print("PIPELINE ML — FASE 3 TECH CHALLENGE FIAP")
    print("=" * 60)

    # Carrega dados
    df = carregar_dados()

    # Prepara features
    X, y = preparar_features(df)

    # Split treino/teste — sem data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nSplit treino/teste:")
    print(f"  Treino: {X_train.shape[0]} registros")
    print(f"  Teste:  {X_test.shape[0]} registros")

    # Cria preprocessador
    preprocessador = criar_preprocessador(X_train)

    # Treina modelos
    resultados = treinar_modelos(X_train, y_train, preprocessador)

    # Seleciona melhor modelo
    pipeline, nome_melhor, y_pred, y_prob = selecionar_melhor_modelo(
        resultados, X_train, y_train, X_test, y_test
    )

    # Visualiza resultados
    visualizar_resultados(resultados, y_test, y_pred, y_prob, nome_melhor)

    print(f"\n{'=' * 60}")
    print("✓ Pipeline ML concluída!")
    print(f"  Melhor modelo: {nome_melhor}")
    print("=" * 60)

if __name__ == "__main__":
    main()

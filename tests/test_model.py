"""
Testes para src/modeling/train.py
Verifica integridade do modelo treinado e suas métricas.
"""

import pytest
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score


@pytest.fixture
def modelo_e_metadata():
    """Carrega modelo e metadata."""
    modelo_path = Path("models/modelo_final.joblib")
    meta_path = Path("models/metadata.json")
    if not modelo_path.exists() or not meta_path.exists():
        pytest.skip("Modelo não encontrado — rode train.py primeiro")
    pipeline = joblib.load(modelo_path)
    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)
    return pipeline, metadata


@pytest.fixture
def dados_teste(modelo_e_metadata):
    """Prepara conjunto de teste."""
    _, metadata = modelo_e_metadata
    caminho = Path("data/processed/dataset_enriquecido_v2.parquet")
    if not caminho.exists():
        caminho = Path("data/processed/dataset_modelagem_v2.parquet")
    if not caminho.exists():
        pytest.skip("Dataset não encontrado")

    df = pd.read_parquet(caminho)
    features_num = metadata["features_numericas"]
    features_cat = metadata["features_categoricas"]
    features_all = [f for f in features_num + features_cat if f in df.columns]

    X = df[features_all]
    y = df["em_risco_2024"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_test, y_test


def test_modelo_existe():
    """Arquivo do modelo deve existir."""
    assert Path("models/modelo_final.joblib").exists(), \
        "modelo_final.joblib não encontrado"


def test_metadata_existe():
    """Metadata do modelo deve existir."""
    assert Path("models/metadata.json").exists(), \
        "metadata.json não encontrado"


def test_metadata_campos_obrigatorios(modelo_e_metadata):
    """Metadata deve conter campos obrigatórios."""
    _, metadata = modelo_e_metadata
    campos = ["modelo", "threshold", "features_numericas",
              "features_categoricas", "metricas_teste", "design"]
    faltando = [c for c in campos if c not in metadata]
    assert len(faltando) == 0, f"Campos faltando no metadata: {faltando}"


def test_design_temporal(modelo_e_metadata):
    """Modelo deve usar design temporal correto."""
    _, metadata = modelo_e_metadata
    assert metadata.get("design") == "features_2023_target_2024", \
        "Design temporal incorreto no metadata"


def test_threshold_valido(modelo_e_metadata):
    """Threshold deve estar entre 0 e 1."""
    _, metadata = modelo_e_metadata
    threshold = metadata["threshold"]
    assert 0 < threshold < 1, f"Threshold inválido: {threshold}"


def test_roc_auc_minimo(modelo_e_metadata, dados_teste):
    """ROC-AUC no teste deve ser >= 0.80."""
    pipeline, metadata = modelo_e_metadata
    X_test, y_test = dados_teste
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    assert auc >= 0.80, f"ROC-AUC abaixo do mínimo: {auc:.4f} < 0.80"


def test_recall_minimo(modelo_e_metadata, dados_teste):
    """Recall da classe risco deve ser >= 0.80."""
    pipeline, metadata = modelo_e_metadata
    X_test, y_test = dados_teste
    threshold = metadata["threshold"]
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    rec = recall_score(y_test, y_pred)
    assert rec >= 0.80, f"Recall abaixo do mínimo: {rec:.4f} < 0.80"


def test_sem_leakage_features(modelo_e_metadata):
    """Features do modelo não devem conter variáveis de 2024."""
    _, metadata = modelo_e_metadata
    features = metadata["features_numericas"] + metadata["features_categoricas"]
    proibidas = ["taxa_alf_2024", "score_niveis_altos",
                 "proporcao_aluno_nivel_0", "proporcao_aluno_nivel_1"]
    encontradas = [f for f in proibidas if f in features]
    assert len(encontradas) == 0, \
        f"Features de 2024 encontradas (leakage!): {encontradas}"


def test_predict_proba_valido(modelo_e_metadata, dados_teste):
    """Probabilidades devem estar entre 0 e 1."""
    pipeline, _ = modelo_e_metadata
    X_test, _ = dados_teste
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    assert y_prob.min() >= 0.0, "Probabilidade negativa encontrada"
    assert y_prob.max() <= 1.0, "Probabilidade > 1 encontrada"

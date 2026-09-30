"""
Testes do modelo treinado: metadata, ausencia de leakage e reprodutibilidade.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

MODELS_DIR = Path("models")
DATASET = Path("data/processed/dataset_enriquecido_v2.parquet")
VARIANTES = {"": "completo", "_sem_historico": "sem_historico",
             "_socioeconomico": "socioeconomico"}


@pytest.fixture(scope="module")
def metadata():
    caminho = MODELS_DIR / "metadata.json"
    if not caminho.exists():
        pytest.skip("Metadata nao encontrado: rode src/modeling/train.py")
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def modelo():
    caminho = MODELS_DIR / "modelo_final.joblib"
    if not caminho.exists():
        pytest.skip("Modelo nao encontrado: rode src/modeling/train.py")
    return joblib.load(caminho)


@pytest.fixture(scope="module")
def conjunto_teste(metadata):
    if not DATASET.exists():
        pytest.skip("Dataset enriquecido nao encontrado")
    df = pd.read_parquet(DATASET)
    X = df[metadata["features_numericas"] + metadata["features_categoricas"]]
    y = df[metadata["target"]].astype(int)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    return X_test, y_test


def test_metadata_campos_obrigatorios(metadata):
    campos = ["modelo", "design", "target", "threshold", "features_numericas",
              "features_categoricas", "metricas_teste", "fontes_externas", "calibracao"]
    assert not [c for c in campos if c not in metadata]


def test_design_temporal(metadata):
    assert metadata["design"] == "features_2023_target_2024"


def test_nenhuma_feature_de_2024(metadata):
    features = metadata["features_numericas"] + metadata["features_categoricas"]
    proibidas = [f for f in features if f.endswith("_2024") or f == metadata["target"]]
    assert not proibidas, f"Features com informacao de 2024 (leakage): {proibidas}"


def test_threshold_valido(metadata):
    assert 0 < metadata["threshold"] < 1


def test_roc_auc_plausivel(metadata):
    """Abaixo de 0,80 o modelo e fraco; acima de 0,97 e sinal de leakage."""
    roc = metadata["metricas_teste"]["roc_auc"]
    assert 0.80 <= roc <= 0.97, f"ROC-AUC fora da faixa plausivel: {roc:.4f}"


def test_recall_no_teste(metadata):
    assert metadata["metricas_teste"]["recall"] >= 0.80


def test_modelo_salvo_reproduz_metricas(modelo, metadata, conjunto_teste):
    X_test, y_test = conjunto_teste
    roc = roc_auc_score(y_test, modelo.predict_proba(X_test)[:, 1])
    assert abs(roc - metadata["metricas_teste"]["roc_auc"]) < 1e-6


def test_probabilidades_validas(modelo, conjunto_teste):
    X_test, _ = conjunto_teste
    prob = modelo.predict_proba(X_test)[:, 1]
    assert prob.min() >= 0 and prob.max() <= 1


@pytest.mark.parametrize("sufixo,variante", VARIANTES.items())
def test_metadata_das_variantes(sufixo, variante):
    caminho = MODELS_DIR / f"metadata{sufixo}.json"
    if not caminho.exists():
        pytest.skip(f"Variante {variante} nao treinada")
    with open(caminho, encoding="utf-8") as f:
        assert json.load(f)["variante"] == variante

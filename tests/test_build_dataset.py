"""
Testes para src/data/build_dataset.py
Verifica integridade do dataset temporal construído.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def dataset():
    """Carrega o dataset de modelagem para os testes."""
    caminho = Path("data/processed/dataset_modelagem_v2.parquet")
    if not caminho.exists():
        pytest.skip("Dataset não encontrado — rode build_dataset.py primeiro")
    return pd.read_parquet(caminho)


def test_dataset_existe():
    """Dataset de modelagem deve existir."""
    caminho = Path("data/processed/dataset_modelagem_v2.parquet")
    assert caminho.exists(), "dataset_modelagem_v2.parquet não encontrado"


def test_target_binario(dataset):
    """Variável alvo deve ser binária (0 ou 1)."""
    valores = dataset["em_risco_2024"].unique()
    assert set(valores).issubset({0, 1}), f"Target contém valores inesperados: {valores}"


def test_sem_colunas_2024(dataset):
    """Nenhuma coluna de 2024 deve estar nas features — prevenção de leakage."""
    colunas_proibidas = [
        "taxa_alfabetizacao_2024",
        "proporcao_aluno_nivel_0",
        "proporcao_aluno_nivel_1",
        "proporcao_aluno_nivel_2",
        "proporcao_aluno_nivel_3",
        "proporcao_aluno_nivel_4",
        "proporcao_aluno_nivel_5",
        "proporcao_aluno_nivel_6",
        "proporcao_aluno_nivel_7",
        "proporcao_aluno_nivel_8",
        "score_niveis_altos",
        "taxa_alf_2024",
    ]
    colunas_presentes = [c for c in colunas_proibidas if c in dataset.columns]
    assert len(colunas_presentes) == 0, \
        f"Colunas de 2024 encontradas (data leakage!): {colunas_presentes}"


def test_municipios_pareados(dataset):
    """Dataset deve ter municípios com dados em 2023 E 2024."""
    assert len(dataset) > 4000, \
        f"Poucos municípios pareados: {len(dataset)} (esperado > 4000)"


def test_features_2023_presentes(dataset):
    """Features do Bloco A (2023) devem estar presentes."""
    features_obrigatorias = [
        "taxa_alf_2023",
        "media_pt_2023",
        "gap_meta_2030_2023",
    ]
    faltando = [f for f in features_obrigatorias if f not in dataset.columns]
    assert len(faltando) == 0, f"Features obrigatórias faltando: {faltando}"


def test_balanceamento_target(dataset):
    """Classes não devem ser extremamente desbalanceadas (< 10% ou > 90%)."""
    proporcao = dataset["em_risco_2024"].mean()
    assert 0.10 < proporcao < 0.90, \
        f"Target muito desbalanceado: {proporcao:.1%} em risco"


def test_sem_duplicatas(dataset):
    """Não deve haver municípios duplicados."""
    duplicatas = dataset["id_municipio"].duplicated().sum()
    assert duplicatas == 0, f"{duplicatas} municípios duplicados encontrados"

"""
Testes de integridade do dataset temporal e das fontes externas.
"""

from pathlib import Path

import pandas as pd
import pytest

BASE = Path("data/processed/dataset_modelagem_v2.parquet")
ENRIQUECIDO = Path("data/processed/dataset_enriquecido_v2.parquet")
COLUNAS_2024_PERMITIDAS = {"taxa_alf_2024", "em_risco_2024"}


@pytest.fixture(scope="module")
def dataset():
    if not BASE.exists():
        pytest.skip("Dataset nao encontrado: rode src/data/build_dataset.py")
    return pd.read_parquet(BASE)


@pytest.fixture(scope="module")
def enriquecido():
    if not ENRIQUECIDO.exists():
        pytest.skip("Dataset enriquecido nao encontrado: rode src/data/download_external.py")
    return pd.read_parquet(ENRIQUECIDO)


def test_target_binario(dataset):
    assert set(dataset["em_risco_2024"].unique()) <= {0, 1}


def test_target_consistente_com_corte_de_60(dataset):
    esperado = (dataset["taxa_alf_2024"] < 60).astype(int)
    assert (dataset["em_risco_2024"] == esperado).all()


def test_unicas_colunas_de_2024_sao_as_do_target(dataset):
    colunas_2024 = {c for c in dataset.columns if c.endswith("_2024")}
    extras = colunas_2024 - COLUNAS_2024_PERMITIDAS
    assert not extras, f"Colunas de 2024 inesperadas no dataset: {extras}"


def test_municipios_pareados(dataset):
    assert len(dataset) > 4000


def test_codigo_municipio_com_7_digitos(dataset):
    assert dataset["id_municipio"].astype(str).str.len().eq(7).all()


def test_sem_municipios_duplicados(dataset):
    assert not dataset["id_municipio"].duplicated().any()


def test_features_de_2023_presentes(dataset):
    obrigatorias = ["taxa_alf_2023", "media_pt_2023", "gap_meta_2030_2023"]
    assert not [c for c in obrigatorias if c not in dataset.columns]


def test_balanceamento_do_target(dataset):
    assert 0.10 < dataset["em_risco_2024"].mean() < 0.90


def test_nenhum_arquivo_simulado():
    simulados = list(Path("data/external").glob("*simulado*"))
    assert not simulados, f"Arquivos simulados encontrados: {simulados}"


@pytest.mark.parametrize("coluna", ["populacao_2022", "pib_per_capita", "idhm", "analfabetismo_15mais"])
def test_cobertura_das_fontes_externas(enriquecido, coluna):
    cobertura = enriquecido[coluna].notna().mean()
    assert cobertura >= 0.99, f"{coluna}: cobertura de {cobertura:.1%}"

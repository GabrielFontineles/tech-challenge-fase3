"""
Testes da previsao de 2025: repasse do ano-base e saidas geradas.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.data.build_dataset import UFS_FORA_DO_MODELO
from src.modeling.predict_2025 import FAIXAS, carregar_silver, montar_features_ano_base

REPORTS = Path("reports")


@pytest.fixture(scope="module")
def silver():
    return carregar_silver()


@pytest.fixture(scope="module")
def base_2024(silver):
    return montar_features_ano_base(silver, ano=2024)


def test_colunas_do_ano_anterior_recebem_valores_de_2024(silver, base_2024):
    obs = silver[silver["ano"] == 2024].copy()
    obs["id_municipio"] = obs["id_municipio"].astype(str)
    obs = obs.set_index("id_municipio")
    base = base_2024.set_index("id_municipio")
    pd.testing.assert_series_equal(
        base["taxa_alf_2023"], obs.loc[base.index, "taxa_alfabetizacao"], check_names=False)
    pd.testing.assert_series_equal(
        base["media_pt_2023"], obs.loc[base.index, "media_portugues"], check_names=False)


def test_universo_e_2024_sem_ufs_fora_do_modelo(silver, base_2024):
    assert not set(base_2024["sigla_uf"]) & UFS_FORA_DO_MODELO
    assert not base_2024["id_municipio"].duplicated().any()
    avaliados_2024 = silver.loc[silver["ano"] == 2024, "id_municipio"].nunique()
    assert len(base_2024) == avaliados_2024 - 23  # 22 do AC e 1 do DF


@pytest.fixture(scope="module")
def risco_2025():
    caminho = REPORTS / "risco_2025.csv"
    if not caminho.exists():
        pytest.skip("Previsao nao encontrada: rode src/modeling/predict_2025.py")
    return pd.read_csv(caminho, dtype={"id_municipio": str})


def test_saida_2025_valida(risco_2025):
    assert risco_2025["prob_risco_2025"].between(0, 1).all()
    assert not set(risco_2025["sigla_uf"]) & UFS_FORA_DO_MODELO
    assert not risco_2025["id_municipio"].duplicated().any()
    assert set(risco_2025["faixa_risco_2025"].dropna()) <= set(FAIXAS)


def test_priorizacao_2025_e_subconjunto(risco_2025):
    prioridade = pd.read_csv(REPORTS / "priorizacao_2025.csv", dtype={"id_municipio": str})
    assert set(prioridade["id_municipio"]) <= set(risco_2025["id_municipio"])
    assert (prioridade["acima_threshold"] == 1).all()
    assert (prioridade["situacao_2030"] == "Precisa de mais de 2x o ritmo").all()

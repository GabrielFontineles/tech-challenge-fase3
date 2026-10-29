"""
Testes dos arquivos externos versionados (IBGE e Atlas).
Rodam em qualquer clone, sem API e sem o xlsx do Atlas.
"""

from pathlib import Path

import pandas as pd
import pytest

EXTERNAL = Path("data/external")
SNAPSHOTS = {
    "ibge_censo2022.parquet": ["id_municipio", "nome_municipio", "populacao_2022",
                               "densidade_2022", "log_populacao_2022", "porte"],
    "ibge_pib2021.parquet": ["id_municipio", "pib_2021_mil_reais"],
    "atlas_2010.parquet": ["id_municipio", "idhm", "idhm_educacao", "renda_per_capita_2010",
                           "gini", "pct_pobres", "analfabetismo_15mais", "freq_escolar_4a5",
                           "pct_rural"],
}


@pytest.mark.parametrize("arquivo,colunas", SNAPSHOTS.items())
def test_snapshot_versionado_e_integro(arquivo, colunas):
    caminho = EXTERNAL / arquivo
    assert caminho.exists(), f"{caminho} ausente: rode src/data/download_external.py --atualizar"
    df = pd.read_parquet(caminho)
    assert not [c for c in colunas if c not in df.columns]
    ids = df["id_municipio"].astype(str)
    assert ids.str.len().eq(7).all()
    assert not ids.duplicated().any()
    assert len(df) >= 5500  # o Brasil tem 5.570 municipios; o Atlas 2010, 5.565

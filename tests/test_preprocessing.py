"""
Testes do pre-processamento compartilhado pelo treino.
"""

import numpy as np
import pandas as pd

from src.preprocessing.pipeline import criar_preprocessador


def _dados():
    return pd.DataFrame({
        "taxa": [50.0, np.nan, 70.0, 90.0],
        "sigla_uf": ["BA", "CE", "BA", "SP"],
    })


def test_imputa_numericas_e_marca_ausencia():
    pre = criar_preprocessador(["taxa"], ["sigla_uf"]).fit(_dados())
    X = pre.transform(_dados())
    assert not np.isnan(X).any()
    # taxa padronizada + indicador de ausencia (tambem padronizado) + 3 UFs no one-hot
    assert X.shape == (4, 5)
    indicador = X[:, 1]
    assert indicador[1] > 0 and (indicador[[0, 2, 3]] < 0).all()


def test_uf_nao_vista_no_treino_vira_colunas_zeradas():
    pre = criar_preprocessador(["taxa"], ["sigla_uf"]).fit(_dados())
    novo = pd.DataFrame({"taxa": [60.0], "sigla_uf": ["AC"]})
    onehot = pre.transform(novo)[:, 2:]
    assert (onehot == 0).all()

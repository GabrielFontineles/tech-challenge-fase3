"""
Seleção de features e auditoria anti-leakage.

A regra é simples e verificável pelo nome da coluna:
  - features do bloco A terminam com `_t` (ano t);
  - features dos blocos B/C vêm de fontes externas anteriores a t;
  - qualquer coluna do ano t+1 é proibida.
"""
from __future__ import annotations

import pandas as pd

from src import config as C
from src.data.build_dataset import COLUNAS_PROIBIDAS

IDENTIFICADORES = [C.COL_ID]
CATEGORICAS = ["sigla_uf", "regiao", "porte"]
# colunas do bloco A que são rótulo textual derivado da taxa em t (usar só as numéricas)
DESCARTAR = [f"{C.COL_NIVEL}_t", "pib_total_2021", "censo_matriculas_ai", "censo_docentes_ai",
             "geo_latitude", "geo_longitude"]  # lat/lon: opcional; deixamos fora por default para não "decorar" o mapa


def separar_xy(df: pd.DataFrame, incluir_historico: bool = True):
    """
    Devolve X, y. `incluir_historico=False` remove o bloco A (para a análise
    'fatores estruturais sem inércia' descrita no desenho §7.3).
    """
    proibidas = set(COLUNAS_PROIBIDAS) | set(IDENTIFICADORES) | set(DESCARTAR)
    cols = [c for c in df.columns if c not in proibidas]
    if not incluir_historico:
        cols = [c for c in cols if not c.endswith("_t")]
    X = df[cols].copy()
    y = df[C.TARGET].astype(int)
    auditar(X)
    return X, y


def auditar(X: pd.DataFrame) -> None:
    """Falha ruidosamente se alguma coluna do ano t+1 ou o target escapar para X."""
    vazadas = [c for c in X.columns if c in COLUNAS_PROIBIDAS or c.endswith("_t1")]
    if vazadas:
        raise ValueError(f"LEAKAGE: colunas proibidas em X: {vazadas}")


def tipos(X: pd.DataFrame):
    cat = [c for c in CATEGORICAS if c in X.columns]
    num = [c for c in X.columns if c not in cat]
    nao_numericas = [c for c in num if not pd.api.types.is_numeric_dtype(X[c])]
    if nao_numericas:
        raise TypeError(f"colunas não numéricas fora da lista de categóricas: {nao_numericas}")
    return num, cat

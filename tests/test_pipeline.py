"""Smoke tests: rodam em < 1 min com dados sintéticos. `pytest -q`"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src import config as C
from src.data.build_dataset import bloco_historico, bloco_territorio
from src.preprocessing.features import auditar, separar_xy, tipos
from src.preprocessing.pipeline import criar_pipeline, nomes_features
from src.evaluation.metrics import escolher_threshold, resumo
from src.utils.synthetic import gerar_municipio_silver


@pytest.fixture(scope="module")
def dataset():
    silver = gerar_municipio_silver(n_municipios=600, seed=1)
    silver = silver[silver[C.COL_REDE] == C.REDE]
    t = silver[silver[C.COL_ANO] == C.ANO_FEATURES]
    t1 = silver[silver[C.COL_ANO] == C.ANO_TARGET][[C.COL_ID, C.COL_TAXA]].rename(columns={C.COL_TAXA: "taxa_alfabetizacao_t1"})
    t1[C.TARGET] = (t1["taxa_alfabetizacao_t1"] < C.CORTE_RISCO).astype(int)
    df = t1.merge(bloco_historico(t), on=C.COL_ID).merge(bloco_territorio(t1[C.COL_ID]), on=C.COL_ID)
    return df


def test_features_sem_leakage(dataset):
    X, y = separar_xy(dataset)
    assert not any(c.endswith("_t1") for c in X.columns)
    assert C.TARGET not in X.columns and "taxa_alfabetizacao_t1" not in X.columns
    assert all(c.endswith("_t") or c in ("sigla_uf", "regiao") for c in X.columns)


def test_auditoria_detecta_leakage(dataset):
    X, _ = separar_xy(dataset)
    with pytest.raises(ValueError):
        auditar(X.assign(taxa_alfabetizacao_t1=1.0))


def test_pipeline_treina_e_preve(dataset):
    X, y = separar_xy(dataset)
    num, cat = tipos(X)
    X.loc[X.index[:20], num[0]] = np.nan  # missing deve ser tratado dentro do pipeline
    pipe = criar_pipeline(LogisticRegression(max_iter=500), num, cat)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    pipe.fit(X_tr, y_tr)
    p = pipe.predict_proba(X_te)[:, 1]
    assert p.shape == (len(X_te),) and 0 <= p.min() and p.max() <= 1
    assert len(nomes_features(pipe)) >= len(num)
    r = resumo(y_te, p, 0.5)
    assert r["roc_auc"] > 0.6  # sinal sintético existe por construção


def test_threshold_respeita_recall():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.4 + rng.random(500) * 0.6, 0, 1)
    thr = escolher_threshold(y, p, 0.9)
    assert resumo(y, p, thr)["recall_risco"] >= 0.9

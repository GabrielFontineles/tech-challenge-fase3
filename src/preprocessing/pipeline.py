"""
Pré-processamento integrado ao modelo (scikit-learn Pipeline + ColumnTransformer).

Tudo que aprende algo dos dados (mediana, moda, escala, categorias) é ajustado
APENAS no conjunto de treino, via `fit` do pipeline completo.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def criar_preprocessador(colunas_numericas: list[str], colunas_categoricas: list[str],
                         escalar: bool = True) -> ColumnTransformer:
    passos_num = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if escalar:
        passos_num.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(passos_num)
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False)),
    ])
    transformers = [("num", num_pipe, colunas_numericas)]
    if colunas_categoricas:
        transformers.append(("cat", cat_pipe, colunas_categoricas))
    pre = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)
    pre.set_output(transform="pandas")
    return pre


def criar_pipeline(estimador, colunas_numericas, colunas_categoricas, escalar=True) -> Pipeline:
    return Pipeline([
        ("pre", criar_preprocessador(colunas_numericas, colunas_categoricas, escalar)),
        ("clf", estimador),
    ])


def nomes_features(pipeline: Pipeline) -> list[str]:
    """Nomes das colunas após o pré-processamento (para SHAP / importâncias)."""
    return list(pipeline.named_steps["pre"].get_feature_names_out())

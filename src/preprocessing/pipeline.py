"""
Pre-processamento do modelo - Fase 3, Tech Challenge FIAP

Define o ColumnTransformer usado em todas as variantes de train.py. Ele fica
dentro do Pipeline do scikit-learn, entao e ajustado apenas nos dados de
treino de cada fold da validacao cruzada (sem vazamento para a validacao).

- Numericas: imputacao pela mediana, com indicador de ausencia, e padronizacao.
- Categoricas (sigla_uf, regiao, porte): imputacao pela moda e one-hot.
  Categorias nao vistas no treino viram colunas zeradas (handle_unknown="ignore"),
  por isso o modelo nao deve pontuar UFs fora do treino (AC, DF e RR).
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def criar_preprocessador(numericas, categoricas):
    num = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", num, numericas), ("cat", cat, categoricas)])

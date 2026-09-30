"""
Treinamento do modelo — v2.1
Fase 3 — Tech Challenge FIAP

Previsao de risco de alfabetizacao por municipio.
Desenho temporal: features observadas em 2023 (e fontes estruturais
anteriores), target em_risco_2024 = taxa de alfabetizacao 2024 < 60%.

Etapas:
1. Validacao das colunas e assercao anti-leakage (nenhuma feature de 2024)
2. Split estratificado treino/teste (teste usado uma unica vez, no final)
3. Comparacao de modelos de referencia em CV 5-fold (inclui Dummy)
4. RandomizedSearchCV em HistGradientBoosting e Regressao Logistica
   (scoring = PR-AUC); vence o maior PR-AUC medio em CV
5. Calibracao isotonica (CalibratedClassifierCV) do modelo vencedor
6. Threshold escolhido sobre predicoes out-of-fold calibradas,
   garantindo recall >= 0.85 com a maior precisao possivel
7. Avaliacao unica no teste, graficos e persistencia (joblib + metadata)

Uso:
    python src/modeling/train.py                  # modelo completo
    python src/modeling/train.py --sem-historico  # sem o Bloco A (analise estrutural)
    python src/modeling/train.py --so-socioeconomico  # apenas Bloco C (teste da H3)
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             classification_report, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_val_predict, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TARGET = "em_risco_2024"
CORTE_RISCO = 60.0
RECALL_MINIMO = 0.85
DATASET = Path("data/processed/dataset_enriquecido_v2.parquet")
MODELS_DIR = Path("models")
IMAGES_DIR = Path("images")
REPORTS_DIR = Path("reports")

# Bloco A — historico educacional 2023
BLOCO_A = [
    "taxa_alf_2023", "media_pt_2023", "particip_2023", "nivel_alf_2023",
    "meta_2030", "gap_meta_2030_2023", "taxa_vs_uf_2023",
    "prop_nivel_0_2023", "prop_nivel_1_2023", "prop_nivel_2_2023",
    "prop_nivel_3_2023", "prop_nivel_4_2023", "prop_nivel_5_2023",
    "prop_nivel_6_2023", "prop_nivel_7_2023", "prop_nivel_8_2023",
]

# Bloco B — territorio (IBGE Censo 2022)
BLOCO_B_NUM = ["log_populacao_2022", "densidade_2022"]
BLOCO_B_CAT = ["sigla_uf", "regiao", "porte"]

# Bloco C — socioeconomico (IBGE PIB 2021 e Atlas 2010)
BLOCO_C = [
    "log_pib_per_capita", "idhm", "idhm_educacao", "idhm_renda",
    "idhm_longevidade", "renda_per_capita_2010", "gini", "pct_pobres",
    "pct_extremamente_pobres", "analfabetismo_15mais",
    "expectativa_anos_estudo", "freq_escolar_4a5", "pct_rural",
]

FONTES_EXTERNAS = {
    "IBGE Censo 2022 (SIDRA tabela 4714)": ["log_populacao_2022", "densidade_2022", "porte"],
    "IBGE PIB dos Municipios 2021 (SIDRA tabela 5938)": ["log_pib_per_capita"],
    "Atlas do Desenvolvimento Humano 2010": [c for c in BLOCO_C if c != "log_pib_per_capita"],
}


def definir_features(variante):
    """Features de cada variante do modelo."""
    if variante == "socioeconomico":
        return list(BLOCO_C), []
    numericas = ([] if variante == "sem_historico" else BLOCO_A) + BLOCO_B_NUM + BLOCO_C
    return numericas, list(BLOCO_B_CAT)


def carregar_dados(numericas, categoricas):
    """Carrega o dataset e valida colunas e ausencia de leakage."""
    if not DATASET.exists():
        sys.exit(f"ERRO: {DATASET} nao encontrado. Rode build_dataset.py e download_external.py.")
    df = pd.read_parquet(DATASET)

    faltando = [c for c in numericas + categoricas + [TARGET] if c not in df.columns]
    if faltando:
        sys.exit(f"ERRO: colunas ausentes no dataset: {faltando}")

    proibidas = [c for c in numericas + categoricas if c.endswith("_2024") or c == TARGET]
    if proibidas:
        sys.exit(f"ERRO: features com informacao de 2024 (leakage): {proibidas}")

    X = df[numericas + categoricas]
    y = df[TARGET].astype(int)
    print(f"Dataset: {len(df)} municipios | {len(numericas)} numericas + {len(categoricas)} categoricas")
    print(f"Em risco: {y.sum()} ({y.mean() * 100:.1f}%)")
    return X, y


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


def montar(clf, numericas, categoricas):
    return Pipeline([("pre", criar_preprocessador(numericas, categoricas)), ("clf", clf)])


def comparar_modelos_referencia(X_train, y_train, numericas, categoricas, cv):
    """Modelos com hiperparametros padrao, avaliados apenas em CV."""
    print("\n" + "=" * 60)
    print("MODELOS DE REFERENCIA (CV 5-fold no treino)")
    print("=" * 60)
    candidatos = {
        "Dummy (prior)": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced",
                                                  random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                                n_jobs=-1, random_state=RANDOM_STATE),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    linhas = []
    for nome, clf in candidatos.items():
        res = cross_validate(montar(clf, numericas, categoricas), X_train, y_train, cv=cv,
                             scoring=["roc_auc", "average_precision", "recall"], n_jobs=-1)
        linha = {
            "modelo": nome,
            "roc_auc": res["test_roc_auc"].mean(), "roc_auc_std": res["test_roc_auc"].std(),
            "pr_auc": res["test_average_precision"].mean(),
            "pr_auc_std": res["test_average_precision"].std(),
            "recall": res["test_recall"].mean(),
        }
        linhas.append(linha)
        print(f"{nome:<22} ROC-AUC {linha['roc_auc']:.4f} ± {linha['roc_auc_std']:.4f} | "
              f"PR-AUC {linha['pr_auc']:.4f} ± {linha['pr_auc_std']:.4f} | recall {linha['recall']:.4f}")
    return pd.DataFrame(linhas)


def otimizar(X_train, y_train, numericas, categoricas, cv):
    """RandomizedSearchCV em HGB e LogReg; vence o maior PR-AUC medio em CV."""
    print("\n" + "=" * 60)
    print("OTIMIZACAO (RandomizedSearchCV, scoring = PR-AUC)")
    print("=" * 60)
    buscas = {
        "HistGradientBoosting": (
            HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "clf__learning_rate": [0.02, 0.05, 0.1],
                "clf__max_iter": [100, 200, 400],
                "clf__max_depth": [3, 5, None],
                "clf__min_samples_leaf": [20, 40, 80],
                "clf__l2_regularization": [0.0, 0.1, 1.0],
                "clf__class_weight": ["balanced", None],
            },
            30,
        ),
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            {"clf__C": loguniform(1e-3, 1e2), "clf__class_weight": ["balanced", None]},
            20,
        ),
    }
    resultados = {}
    for nome, (clf, espaco, n_iter) in buscas.items():
        busca = RandomizedSearchCV(montar(clf, numericas, categoricas), espaco, n_iter=n_iter,
                                   scoring="average_precision", cv=cv, n_jobs=-1,
                                   random_state=RANDOM_STATE, return_train_score=True)
        busca.fit(X_train, y_train)
        i = busca.best_index_
        treino = busca.cv_results_["mean_train_score"][i]
        validacao = busca.best_score_
        resultados[nome] = {
            "busca": busca,
            "pr_auc_cv": float(validacao),
            "pr_auc_treino": float(treino),
            "gap_treino_cv": float(treino - validacao),
            "melhores_parametros": {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                                    for k, v in busca.best_params_.items()},
        }
        print(f"\n{nome}")
        print(f"  PR-AUC CV: {validacao:.4f} | treino: {treino:.4f} | gap: {treino - validacao:.4f}")
        print(f"  Parametros: {busca.best_params_}")

    vencedor = max(resultados, key=lambda k: resultados[k]["pr_auc_cv"])
    print(f"\nModelo selecionado (maior PR-AUC em CV): {vencedor}")
    return vencedor, resultados


def calibrar_e_escolher_threshold(modelo_base, X_train, y_train, cv):
    """Calibracao isotonica e threshold por predicoes out-of-fold calibradas."""
    print("\nCalibrando probabilidades e escolhendo threshold...")
    calibrado = CalibratedClassifierCV(clone(modelo_base), method="isotonic", cv=cv)
    prob_oof = cross_val_predict(calibrado, X_train, y_train, cv=cv,
                                 method="predict_proba", n_jobs=-1)[:, 1]

    precisoes, recalls, thresholds = precision_recall_curve(y_train, prob_oof)
    atende = recalls[:-1] >= RECALL_MINIMO
    if atende.any():
        idx = int(np.argmax(np.where(atende, precisoes[:-1], -1)))
        threshold = float(thresholds[idx])
    else:
        threshold = 0.5
        print(f"  AVISO: nenhum threshold atinge recall {RECALL_MINIMO}; usando 0.5")

    calibrado.fit(X_train, y_train)
    print(f"  Threshold (recall >= {RECALL_MINIMO} out-of-fold): {threshold:.3f}")
    return calibrado, threshold


def avaliar_teste(calibrado, modelo_base, X_test, y_test, threshold):
    """Avaliacao unica no conjunto de teste."""
    print("\n" + "=" * 60)
    print("AVALIACAO FINAL NO TESTE (unica vez)")
    print("=" * 60)
    prob = calibrado.predict_proba(X_test)[:, 1]
    prob_sem_calibracao = modelo_base.predict_proba(X_test)[:, 1]
    pred = (prob >= threshold).astype(int)

    metricas = {
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "pr_auc": float(average_precision_score(y_test, prob)),
        "recall": float(recall_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "brier_calibrado": float(brier_score_loss(y_test, prob)),
        "brier_sem_calibracao": float(brier_score_loss(y_test, prob_sem_calibracao)),
        "threshold": threshold,
    }
    for k, v in metricas.items():
        print(f"  {k:<22} {v:.4f}")
    print()
    print(classification_report(y_test, pred, target_names=["Nao em risco", "Em risco"]))
    return metricas, prob, pred


def gerar_graficos(y_test, prob, pred, threshold, sufixo):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    fpr, tpr, _ = roc_curve(y_test, prob)
    axes[0, 0].plot(fpr, tpr, linewidth=2, label=f"ROC-AUC = {roc_auc_score(y_test, prob):.3f}")
    axes[0, 0].plot([0, 1], [0, 1], "k--")
    axes[0, 0].set(title="Curva ROC", xlabel="Taxa de falsos positivos",
                   ylabel="Taxa de verdadeiros positivos")
    axes[0, 0].legend()

    prec, rec, _ = precision_recall_curve(y_test, prob)
    axes[0, 1].plot(rec, prec, linewidth=2, color="orange",
                    label=f"PR-AUC = {average_precision_score(y_test, prob):.3f}")
    axes[0, 1].axvline(RECALL_MINIMO, color="red", linestyle="--", label=f"Recall alvo ({RECALL_MINIMO})")
    axes[0, 1].set(title="Curva Precision-Recall", xlabel="Recall", ylabel="Precision")
    axes[0, 1].legend()

    frac_pos, prob_media = calibration_curve(y_test, prob, n_bins=10)
    axes[1, 0].plot(prob_media, frac_pos, "o-", label="Modelo calibrado")
    axes[1, 0].plot([0, 1], [0, 1], "k--", label="Calibracao perfeita")
    axes[1, 0].set(title="Curva de calibracao", xlabel="Probabilidade prevista",
                   ylabel="Fracao observada em risco")
    axes[1, 0].legend()

    sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt="d", cmap="Blues", ax=axes[1, 1],
                xticklabels=["Nao risco", "Em risco"], yticklabels=["Nao risco", "Em risco"])
    axes[1, 1].set(title=f"Matriz de confusao (threshold = {threshold:.3f})",
                   xlabel="Previsto", ylabel="Real")

    plt.tight_layout()
    caminho = IMAGES_DIR / f"15_resultados_v21{sufixo}.png"
    plt.savefig(caminho, dpi=150)
    plt.close()
    print(f"Grafico salvo: {caminho}")


def salvar(calibrado, modelo_base, metadata, sufixo):
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(calibrado, MODELS_DIR / f"modelo_final{sufixo}.joblib")
    joblib.dump(modelo_base, MODELS_DIR / f"modelo_base{sufixo}.joblib")
    with open(MODELS_DIR / f"metadata{sufixo}.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
    print(f"Modelos e metadata salvos em {MODELS_DIR}/ (sufixo: '{sufixo or 'nenhum'}')")


def main():
    parser = argparse.ArgumentParser()
    opcoes = parser.add_mutually_exclusive_group()
    opcoes.add_argument("--sem-historico", action="store_true",
                        help="treina sem o Bloco A (apenas territorio e socioeconomia)")
    opcoes.add_argument("--so-socioeconomico", action="store_true",
                        help="treina apenas com o Bloco C (sem historico, UF, regiao e porte)")
    args = parser.parse_args()
    if args.so_socioeconomico:
        variante = "socioeconomico"
    elif args.sem_historico:
        variante = "sem_historico"
    else:
        variante = "completo"
    sufixo = "" if variante == "completo" else f"_{variante}"

    print("=" * 60)
    print(f"TREINAMENTO v2.1 — variante: {variante}")
    print(f"Inicio: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 60)

    numericas, categoricas = definir_features(variante)
    X, y = carregar_dados(numericas, categoricas)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    print(f"Split: treino {len(X_train)} | teste {len(X_test)}")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    referencia = comparar_modelos_referencia(X_train, y_train, numericas, categoricas, cv)
    REPORTS_DIR.mkdir(exist_ok=True)
    referencia.to_csv(REPORTS_DIR / f"comparacao_modelos{sufixo}.csv", index=False)

    vencedor, resultados = otimizar(X_train, y_train, numericas, categoricas, cv)
    modelo_base = resultados[vencedor]["busca"].best_estimator_

    calibrado, threshold = calibrar_e_escolher_threshold(modelo_base, X_train, y_train, cv)
    metricas, prob, pred = avaliar_teste(calibrado, modelo_base, X_test, y_test, threshold)
    gerar_graficos(y_test, prob, pred, threshold, sufixo)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "versao": "2.1",
        "variante": variante,
        "modelo": vencedor,
        "design": "features_2023_target_2024",
        "target": TARGET,
        "corte_risco": CORTE_RISCO,
        "threshold": threshold,
        "recall_minimo": RECALL_MINIMO,
        "criterio_selecao": "maior PR-AUC medio em CV 5-fold estratificado",
        "calibracao": "CalibratedClassifierCV, metodo isotonic, cv 5-fold",
        "features_numericas": numericas,
        "features_categoricas": categoricas,
        "fontes_externas": FONTES_EXTERNAS,
        "busca_hiperparametros": {k: {kk: vv for kk, vv in v.items() if kk != "busca"}
                                  for k, v in resultados.items()},
        "metricas_teste": metricas,
    }
    salvar(calibrado, modelo_base, metadata, sufixo)

    print("\n" + "=" * 60)
    print(f"Concluido: {vencedor} | ROC-AUC {metricas['roc_auc']:.4f} | "
          f"PR-AUC {metricas['pr_auc']:.4f} | recall {metricas['recall']:.4f}")
    print(f"Fim: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 60)


if __name__ == "__main__":
    main()

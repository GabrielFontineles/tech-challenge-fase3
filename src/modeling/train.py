"""
Treino, validação cruzada, busca de hiperparâmetros, threshold e persistência.

    python -m src.modeling.train                 # completo
    python -m src.modeling.train --rapido        # menos iterações (smoke test)
    python -m src.modeling.train --sem-historico # variante estrutural (sem bloco A)

Protocolo:
  1. split estratificado 80/20 (teste tocado UMA vez, no fim);
  2. CV estratificado 5-fold no treino para comparar baselines (Dummy, LogReg, RF, HGB);
  3. RandomizedSearchCV (scoring=average_precision) nos candidatos;
  4. threshold escolhido no CV (out-of-fold) do melhor modelo, exigindo recall_risco >= 0.85;
  5. avaliação final no teste + curvas + persistência (models/).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)

from src import config as C
from src.evaluation.metrics import escolher_threshold, resumo, tabela_cv
from src.preprocessing.features import separar_xy, tipos
from src.preprocessing.pipeline import criar_pipeline

SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "recall": "recall", "f1": "f1"}


def candidatos(rapido: bool):
    rs = C.RANDOM_STATE
    return {
        "Dummy (prior)": (DummyClassifier(strategy="prior"), None, False),
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=rs),
            {"clf__C": loguniform(1e-3, 1e2)}, True),
        "Random Forest": (
            RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", n_jobs=-1, random_state=rs),
            {"clf__max_depth": randint(3, 16), "clf__min_samples_leaf": randint(2, 40),
             "clf__max_features": uniform(0.2, 0.7)}, False),
        "HistGradientBoosting": (
            HistGradientBoostingClassifier(random_state=rs, early_stopping=True, validation_fraction=0.15),
            {"clf__learning_rate": loguniform(0.01, 0.3), "clf__max_iter": randint(100, 600),
             "clf__max_depth": randint(2, 8), "clf__min_samples_leaf": randint(10, 80),
             "clf__l2_regularization": loguniform(1e-3, 10)}, False),
    }


def comparar_baselines(X_tr, y_tr, num, cat, cv, rapido):
    resultados = {}
    for nome, (est, _, escalar) in candidatos(rapido).items():
        pipe = criar_pipeline(est, num, cat, escalar)
        t0 = time.time()
        cvres = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=SCORING, n_jobs=-1)
        resultados[nome] = {
            **{f"{k}_mean": cvres[f"test_{k}"].mean() for k in SCORING},
            **{f"{k}_std": cvres[f"test_{k}"].std() for k in SCORING},
            "fit_time": time.time() - t0,
        }
        print(f"  {nome:<22} ROC-AUC {resultados[nome]['roc_auc_mean']:.3f}  PR-AUC {resultados[nome]['pr_auc_mean']:.3f}")
    return resultados


def buscar_hiperparametros(X_tr, y_tr, num, cat, cv, rapido):
    melhores = {}
    n_iter = 8 if rapido else 40
    for nome, (est, espaco, escalar) in candidatos(rapido).items():
        if espaco is None:
            continue
        pipe = criar_pipeline(est, num, cat, escalar)
        busca = RandomizedSearchCV(pipe, espaco, n_iter=n_iter, scoring="average_precision", cv=cv,
                                   n_jobs=-1, random_state=C.RANDOM_STATE, refit=True, return_train_score=True)
        busca.fit(X_tr, y_tr)
        gap = busca.cv_results_["mean_train_score"][busca.best_index_] - busca.best_score_
        melhores[nome] = {"estimator": busca.best_estimator_, "pr_auc_cv": busca.best_score_,
                          "params": busca.best_params_, "gap_treino_cv": float(gap)}
        print(f"  {nome:<22} PR-AUC CV {busca.best_score_:.3f}  gap treino-CV {gap:+.3f}")
    return melhores


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--sem-historico", action="store_true", help="variante sem o bloco A (fatores estruturais)")
    ap.add_argument("--recall-minimo", type=float, default=0.85)
    args = ap.parse_args(argv)

    sufixo = "_sem_historico" if args.sem_historico else ""
    df = pd.read_parquet(C.PROCESSED_DIR / "dataset_modelagem.parquet")
    X, y = separar_xy(df, incluir_historico=not args.sem_historico)
    num, cat = tipos(X)
    print(f"dataset: {X.shape[0]} municípios | {len(num)} numéricas + {len(cat)} categóricas | risco={y.mean():.1%}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.RANDOM_STATE)
    cv = StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)

    print("\n[1] baselines (CV no treino)")
    base = comparar_baselines(X_tr, y_tr, num, cat, cv, args.rapido)
    print("\n[2] busca de hiperparâmetros")
    melhores = buscar_hiperparametros(X_tr, y_tr, num, cat, cv, args.rapido)
    nome_melhor = max(melhores, key=lambda k: melhores[k]["pr_auc_cv"])
    modelo = melhores[nome_melhor]["estimator"]
    print(f"\n→ melhor modelo: {nome_melhor}")

    print("\n[3] threshold via predições out-of-fold no treino")
    prob_oof = cross_val_predict(modelo, X_tr, y_tr, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    thr = escolher_threshold(y_tr, prob_oof, args.recall_minimo)
    print(f"  threshold={thr:.3f} (recall_risco >= {args.recall_minimo})")
    print("  OOF:", {k: round(v, 3) for k, v in resumo(y_tr, prob_oof, thr).items() if isinstance(v, float)})

    print("\n[4] avaliação única no teste")
    modelo.fit(X_tr, y_tr)
    prob_te = modelo.predict_proba(X_te)[:, 1]
    res_te = resumo(y_te, prob_te, thr)
    res_05 = resumo(y_te, prob_te, 0.5)
    for k, v in res_te.items():
        print(f"  {k:<16} {v:.3f}" if isinstance(v, float) else f"  {k:<16} {v}")

    # calibração (probabilidades usadas para ranking de risco)
    calibrado = CalibratedClassifierCV(modelo, method="isotonic", cv=cv).fit(X_tr, y_tr)
    brier_cal = resumo(y_te, calibrado.predict_proba(X_te)[:, 1], thr)["brier"]
    print(f"  brier calibrado  {brier_cal:.3f} (vs {res_te['brier']:.3f})")

    print("\n[5] persistência")
    C.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(modelo, C.MODELS_DIR / f"modelo_final{sufixo}.joblib")
    joblib.dump(calibrado, C.MODELS_DIR / f"modelo_final_calibrado{sufixo}.joblib")
    meta = {
        "modelo": nome_melhor, "params": {k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v)
                                          for k, v in melhores[nome_melhor]["params"].items()},
        "threshold": thr, "recall_minimo": args.recall_minimo,
        "ano_features": C.ANO_FEATURES, "ano_target": C.ANO_TARGET, "corte_risco": C.CORTE_RISCO,
        "features_numericas": num, "features_categoricas": cat,
        "n_treino": int(len(X_tr)), "n_teste": int(len(X_te)),
        "cv_baselines": base,
        "busca": {k: {"pr_auc_cv": v["pr_auc_cv"], "gap_treino_cv": v["gap_treino_cv"]} for k, v in melhores.items()},
        "teste_threshold_escolhido": res_te, "teste_threshold_0.5": res_05, "brier_calibrado": brier_cal,
    }
    (C.MODELS_DIR / f"metadata{sufixo}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=float))
    tabela_cv(base).to_csv(C.REPORTS_DIR / f"cv_baselines{sufixo}.csv", index=False)
    # índices do teste para reuso nos scripts seguintes (SHAP / estratégia) sem re-split
    pd.DataFrame({C.COL_ID: df.loc[X_te.index, C.COL_ID], "y": y_te, "prob": prob_te}).to_parquet(
        C.PROCESSED_DIR / f"predicoes_teste{sufixo}.parquet", index=False)
    print(f"✓ models/modelo_final{sufixo}.joblib + metadata{sufixo}.json")

    from src.visualization.plots import plot_resultados_modelo  # import tardio (matplotlib)
    plot_resultados_modelo(base, y_te, prob_te, thr, nome_melhor, C.IMAGES_DIR / f"09_resultados_modelos{sufixo}.png")


if __name__ == "__main__":
    main()

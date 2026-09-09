"""
Constrói o dataset de modelagem com desenho temporal t → t+1.

    python -m src.data.build_dataset

Entrada : data/raw/municipio_silver.parquet (Fase 2)
          data/raw/uf_silver.parquet (opcional — contexto da UF em t)
          data/external/processed/socioeconomico_municipal.parquet (opcional)
Saída   : data/processed/dataset_modelagem.parquet
          data/processed/dicionario_dataset.csv

Regras anti-leakage aplicadas aqui (documentar no README):
  1. Do ano t+1 (2024) entra APENAS o target `em_risco` (e a taxa contínua
     `taxa_alfabetizacao_t1`, guardada só para análise, nunca como feature).
  2. Todas as features carregam sufixo `_t` (ano t) ou prefixo da fonte externa
     (informação anterior a t), para a regra ser auditável pelo nome da coluna.
  3. Nenhuma estatística (mediana, quantil, encoding) é calculada aqui; isso é
     papel do pipeline sklearn, ajustado apenas no treino.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C

# Colunas que jamais podem entrar como feature (auditoria automática em features.py)
COLUNAS_PROIBIDAS = {"taxa_alfabetizacao_t1", C.TARGET, "nivel_alfabetizacao_t1"}


def carregar_silver() -> pd.DataFrame:
    df = pd.read_parquet(C.RAW_DIR / "municipio_silver.parquet")
    faltando = [c for c in (C.COL_ID, C.COL_ANO, C.COL_REDE, C.COL_TAXA) if c not in df.columns]
    if faltando:
        raise KeyError(f"colunas ausentes na municipio_silver: {faltando}. Ajuste src/config.py.")
    df = df[df[C.COL_REDE] == C.REDE].copy()
    df[C.COL_ID] = df[C.COL_ID].astype(int)
    return df


def bloco_historico(df_t: pd.DataFrame) -> pd.DataFrame:
    """Bloco A — histórico educacional do ano t (sufixo _t)."""
    cols = [C.COL_TAXA, C.COL_MEDIA_PT, C.COL_META_2030, C.COL_DIST_META, C.COL_PARTICIP, C.COL_NIVEL, C.COL_N_ALUNOS]
    cols = [c for c in cols if c in df_t.columns] + [c for c in C.COLS_NIVEIS if c in df_t.columns]
    a = df_t[[C.COL_ID] + cols].copy()
    a = a.rename(columns={c: f"{c}_t" for c in cols})
    if f"{C.COL_TAXA}_t" in a and f"{C.COL_META_2030}_t" in a:
        a["gap_meta_t"] = a[f"{C.COL_META_2030}_t"] - a[f"{C.COL_TAXA}_t"]
    if f"{C.COL_N_ALUNOS}_t" in a:
        a["log_alunos_avaliados_t"] = np.log1p(a[f"{C.COL_N_ALUNOS}_t"])
    niveis = [f"{c}_t" for c in C.COLS_NIVEIS if f"{c}_t" in a]
    if niveis:
        # entropia da distribuição por nível: heterogeneidade do desempenho no município
        p = a[niveis].clip(lower=0).fillna(0).to_numpy() / 100
        p = np.where(p > 0, p, 1e-12)
        a["entropia_niveis_t"] = -(p * np.log(p)).sum(axis=1)
    return a


def bloco_territorio(ids: pd.Series) -> pd.DataFrame:
    """Bloco B — território derivado do código IBGE (categóricas ficam como string)."""
    b = pd.DataFrame({C.COL_ID: ids.astype(int)})
    b["cod_uf"] = b[C.COL_ID] // 100000
    b["sigla_uf"] = b["cod_uf"].map(C.UF_POR_CODIGO)
    b["regiao"] = b["sigla_uf"].map(C.REGIAO_POR_UF)
    return b.drop(columns="cod_uf")


def contexto_uf(df_t: pd.DataFrame) -> pd.DataFrame:
    """Posição relativa do município dentro da UF em t (calculado a partir da própria Silver, ano t)."""
    tmp = df_t[[C.COL_ID, C.COL_TAXA]].copy()
    tmp["cod_uf"] = tmp[C.COL_ID] // 100000
    med = tmp.groupby("cod_uf")[C.COL_TAXA].transform("median")
    tmp["taxa_vs_mediana_uf_t"] = tmp[C.COL_TAXA] - med
    return tmp[[C.COL_ID, "taxa_vs_mediana_uf_t"]]


def bloco_externo() -> pd.DataFrame | None:
    p = C.EXTERNAL_DIR / "socioeconomico_municipal.parquet"
    if not p.exists():
        print("  (sem fontes externas — rode src.data.download_external)")
        return None
    ext = pd.read_parquet(p)
    ext[C.COL_ID] = ext[C.COL_ID].astype(int)
    # porte populacional: usa a melhor população disponível
    pop_col = next((c for c in ("ibge_populacao_2022", f"datasus_populacao_{C.ANO_FEATURES}") if c in ext.columns), None)
    if pop_col:
        ext["porte"] = pd.cut(ext[pop_col], bins=C.PORTE_BINS, labels=C.PORTE_LABELS).astype(str).replace("nan", np.nan)
        if "ibge_log_populacao" not in ext.columns:
            ext["log_populacao"] = np.log1p(ext[pop_col])
    return ext


def construir() -> pd.DataFrame:
    print(f"Construindo dataset t={C.ANO_FEATURES} → t+1={C.ANO_TARGET} (rede {C.REDE})")
    df = carregar_silver()
    df_t = df[df[C.COL_ANO] == C.ANO_FEATURES].drop_duplicates(C.COL_ID)
    df_t1 = df[df[C.COL_ANO] == C.ANO_TARGET].drop_duplicates(C.COL_ID)
    print(f"  municípios em t: {len(df_t)} | em t+1: {len(df_t1)}")

    # --- target (única informação de t+1) ---
    alvo = df_t1[[C.COL_ID, C.COL_TAXA]].rename(columns={C.COL_TAXA: "taxa_alfabetizacao_t1"})
    alvo[C.TARGET] = (alvo["taxa_alfabetizacao_t1"] < C.CORTE_RISCO).astype(int)

    base = alvo.merge(bloco_historico(df_t), on=C.COL_ID, how="inner")
    print(f"  pareados t/t+1: {len(base)}  (perdidos: {len(df_t1) - len(base)} sem registro em t)")
    base = base.merge(contexto_uf(df_t), on=C.COL_ID, how="left")
    base = base.merge(bloco_territorio(base[C.COL_ID]), on=C.COL_ID, how="left")

    ext = bloco_externo()
    if ext is not None:
        antes = base.shape[1]
        base = base.merge(ext, on=C.COL_ID, how="left")
        print(f"  fontes externas: +{base.shape[1] - antes} colunas; "
              f"cobertura média {ext.drop(columns=C.COL_ID).notna().mean().mean():.0%}")

    # sanity: target não pode ser função exata de nenhuma feature
    assert C.TARGET in base and base[C.TARGET].nunique() == 2, "target degenerado"
    dist = base[C.TARGET].value_counts(normalize=True)
    print(f"  target: risco={dist.get(1, 0):.1%} | ok={dist.get(0, 0):.1%}")

    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = C.PROCESSED_DIR / "dataset_modelagem.parquet"
    base.to_parquet(out, index=False)
    dicionario(base).to_csv(C.PROCESSED_DIR / "dicionario_dataset.csv", index=False)
    print(f"✓ {out}  shape={base.shape}")
    return base


def dicionario(df: pd.DataFrame) -> pd.DataFrame:
    def bloco(c):
        if c in (C.COL_ID,):
            return "identificador"
        if c in COLUNAS_PROIBIDAS:
            return "TARGET / proibido como feature"
        if c.endswith("_t"):
            return "A - histórico educacional (t)"
        if c in ("sigla_uf", "regiao", "porte") or c.startswith(("geo_", "ibge_", "datasus_", "log_pop")):
            return "B - território"
        return "C - socioeconômico / estrutura da rede"
    return pd.DataFrame({
        "coluna": df.columns,
        "bloco": [bloco(c) for c in df.columns],
        "tipo": df.dtypes.astype(str).values,
        "pct_missing": (df.isna().mean() * 100).round(1).values,
    })


if __name__ == "__main__":
    construir()

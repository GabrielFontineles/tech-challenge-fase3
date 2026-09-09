"""
Gerador de dados SINTÉTICOS com o mesmo esquema da municipio_silver (Fase 2).

Serve apenas para testar a pipeline de ponta a ponta antes de os dados reais
estarem disponíveis. Os valores NÃO têm significado analítico.

    python -m src.utils.synthetic            # grava data/raw/municipio_silver.parquet (se não existir)
    python -m src.utils.synthetic --force
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src import config as C


def gerar_municipio_silver(n_municipios: int = 5570, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    codigos_uf = np.array(list(C.UF_POR_CODIGO))
    # distribuição aproximada de municípios por UF
    pesos = np.array([52, 22, 62, 15, 144, 16, 139, 217, 224, 184, 167, 223, 185, 102, 75, 417,
                      853, 78, 92, 645, 399, 295, 497, 79, 141, 246, 1], dtype=float)
    kelvins = C.EXTERNAL_RAW_DIR / "municipios_kelvins.csv"
    if kelvins.exists():  # usa códigos IBGE reais para o merge com fontes externas funcionar
        ids = pd.read_csv(kelvins)["codigo_ibge"].astype(int).to_numpy()
        id_municipio = rng.choice(ids, size=min(n_municipios, len(ids)), replace=False)
        n_municipios = len(id_municipio)
        uf = id_municipio // 100000
    else:
        uf = rng.choice(codigos_uf, size=n_municipios, p=pesos / pesos.sum())
        id_municipio = uf * 100000 + rng.choice(np.arange(100, 99999), size=n_municipios, replace=False)

    # efeito regional latente (Norte/Nordeste mais baixos) + ruído municipal persistente
    regiao = pd.Series(uf).map(C.UF_POR_CODIGO).map(C.REGIAO_POR_UF)
    efeito_regiao = regiao.map({"Norte": -8, "Nordeste": -6, "Centro-Oeste": 2, "Sudeste": 4, "Sul": 6}).values
    latente = rng.normal(0, 10, n_municipios)

    linhas = []
    for ano in (2023, 2024):
        for rede in ("Total", "Municipal", "Estadual"):
            base = 58 + efeito_regiao + latente + (ano - 2023) * 3 + rng.normal(0, 6, n_municipios)
            if rede == "Estadual":
                base += 2
            taxa = np.clip(base, 5, 99)
            # proporções por nível coerentes com a taxa (níveis 5-8 ≈ taxa)
            altos = taxa / 100
            p = np.empty((n_municipios, 9))
            baixos = 1 - altos
            p[:, 0:5] = rng.dirichlet(np.ones(5), n_municipios) * baixos[:, None]
            p[:, 5:9] = rng.dirichlet(np.ones(4), n_municipios) * altos[:, None]
            meta = np.clip(taxa[:] * 0 + rng.normal(80, 5, n_municipios), 60, 95)
            df = pd.DataFrame({
                C.COL_ID: id_municipio,
                C.COL_ANO: ano,
                C.COL_REDE: rede,
                C.COL_TAXA: taxa.round(2),
                C.COL_MEDIA_PT: (700 + taxa * 1.2 + rng.normal(0, 15, n_municipios)).round(1),
                C.COL_META_2030: meta.round(1),
                C.COL_PARTICIP: np.clip(rng.normal(88, 8, n_municipios), 30, 100).round(1),
                C.COL_N_ALUNOS: rng.lognormal(4.5, 1.1, n_municipios).astype(int) + 10,
            })
            df[C.COL_DIST_META] = (df[C.COL_META_2030] - df[C.COL_TAXA]).round(2)
            df[C.COL_NIVEL] = pd.cut(df[C.COL_TAXA], [0, 40, 60, 80, 100],
                                     labels=["Crítico", "Em risco", "No caminho", "Meta atingida"]).astype(str)
            for i in range(9):
                df[f"proporcao_aluno_nivel_{i}"] = (p[:, i] * 100).round(2)
            # missing realista nas metas (~3.5%)
            mask = rng.random(n_municipios) < 0.035
            df.loc[mask, [C.COL_META_2030, C.COL_DIST_META]] = np.nan
            linhas.append(df)
    out = pd.concat(linhas, ignore_index=True)
    # alguns municípios só existem em um dos anos (para testar o pareamento)
    drop = rng.choice(out.index[out[C.COL_ANO] == 2024], size=40, replace=False)
    return out.drop(drop).reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    dest = C.RAW_DIR / "municipio_silver.parquet"
    if dest.exists() and not args.force:
        print(f"{dest} já existe — use --force para sobrescrever (cuidado: pode ser o dado REAL).")
        return
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = gerar_municipio_silver()
    df.to_parquet(dest, index=False)
    print(f"✓ dados sintéticos gravados em {dest}  shape={df.shape}")


if __name__ == "__main__":
    main()

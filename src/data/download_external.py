"""
Download e padronização das fontes externas de enriquecimento.

Uso:
    python -m src.data.download_external            # tenta todas as fontes
    python -m src.data.download_external --only ibge_populacao ibge_pib
    python -m src.data.download_external --skip-download   # só reprocessa o que já está em data/external/raw

Cada fonte vira uma tabela padronizada em data/external/processed/<fonte>.parquet
com a chave `id_municipio` (int, 7 dígitos IBGE) e colunas com prefixo da fonte.
No final tudo é consolidado em data/external/processed/socioeconomico_municipal.parquet.

Política de robustez: fontes oficiais mudam URL sem aviso. Se o download falhar,
o script imprime a página e o nome de arquivo esperado; basta baixar manualmente
para data/external/raw/ e rodar de novo com --skip-download. Nada aqui é
bloqueante para o restante da pipeline: fontes ausentes viram colunas ausentes,
tratadas por imputação + indicador de missing dentro do pipeline sklearn.

Regra temporal: todas as variáveis aqui são anteriores ou contemporâneas ao ano
de features (2023). Nenhuma informação de 2024 entra por esta via.
"""
from __future__ import annotations

import argparse
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.config import EXTERNAL_DIR, EXTERNAL_RAW_DIR, ANO_FEATURES

TIMEOUT = 120
HEADERS = {"User-Agent": "Mozilla/5.0 (tech-challenge-fase3; FIAP)"}

# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _get(url: str, **kw) -> requests.Response:
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT, **kw)
    r.raise_for_status()
    return r


def _download_file(url: str, dest: Path, skip_download: bool = False) -> Path | None:
    """Baixa `url` para `dest` (idempotente). Retorna o caminho ou None se falhar."""
    if dest.exists():
        print(f"    (cache) {dest.name}")
        return dest
    if skip_download:
        print(f"    ausente e --skip-download ativo: {dest.name}")
        return None
    try:
        print(f"    baixando {url}")
        r = _get(url, stream=True)
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        return dest
    except Exception as e:  # noqa: BLE001
        print(f"    FALHOU ({e.__class__.__name__}: {e})")
        if dest.exists():
            dest.unlink()
        return None


def _try_urls(urls: list[str], dest: Path, skip_download: bool) -> Path | None:
    for u in urls:
        p = _download_file(u, dest, skip_download)
        if p is not None:
            return p
    return None


def _sidra(tabela: int, variaveis: str, periodo: str, nivel: str = "n6/all") -> pd.DataFrame:
    """Consulta a API SIDRA e devolve DataFrame longo: id_municipio, variavel, valor."""
    url = f"https://apisidra.ibge.gov.br/values/t/{tabela}/{nivel}/v/{variaveis}/p/{periodo}"
    print(f"    SIDRA: {url}")
    data = _get(url).json()
    df = pd.DataFrame(data[1:])  # linha 0 é o cabeçalho descritivo
    df = df.rename(columns={"D1C": "id_municipio", "D2N": "variavel", "D3C": "ano", "V": "valor"})
    df["id_municipio"] = df["id_municipio"].astype(int)
    df["valor"] = pd.to_numeric(df["valor"].replace({"...": np.nan, "-": np.nan, "X": np.nan}), errors="coerce")
    return df[["id_municipio", "variavel", "ano", "valor"]]


def _to_id7(serie: pd.Series) -> pd.Series:
    return pd.to_numeric(serie, errors="coerce").astype("Int64")


def _read_zip_member(zip_path: Path, pattern: str) -> tuple[str, bytes]:
    with zipfile.ZipFile(zip_path) as z:
        nomes = [n for n in z.namelist() if re.search(pattern, n, flags=re.I)]
        if not nomes:
            raise FileNotFoundError(f"nenhum membro em {zip_path.name} casa com {pattern!r}: {z.namelist()[:10]}")
        return nomes[0], z.read(nomes[0])


def _find_header_row(df_raw: pd.DataFrame, must_contain: str) -> int:
    for i in range(min(40, len(df_raw))):
        linha = df_raw.iloc[i].astype(str).str.upper().tolist()
        if any(must_contain.upper() in c for c in linha):
            return i
    raise ValueError(f"cabeçalho com {must_contain!r} não encontrado")


# ---------------------------------------------------------------------------
# Fonte 1 — IBGE: população (Censo 2022) e densidade demográfica
# ---------------------------------------------------------------------------

def fonte_ibge_populacao(skip_download: bool) -> pd.DataFrame:
    """Tabela SIDRA 4714 (Censo 2022): população residente (v93) e densidade (v614)."""
    cache = EXTERNAL_RAW_DIR / "sidra_4714_censo2022.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        if skip_download:
            raise FileNotFoundError(cache)
        df = _sidra(4714, "93,614", "2022")
        df.to_parquet(cache, index=False)
    wide = df.pivot_table(index="id_municipio", columns="variavel", values="valor").reset_index()
    wide = wide.rename(columns={"População residente": "ibge_populacao_2022",
                                "Densidade demográfica": "ibge_densidade_2022"})
    wide["ibge_log_populacao"] = np.log1p(wide["ibge_populacao_2022"])
    return wide


# ---------------------------------------------------------------------------
# Fonte 2 — IBGE: PIB dos Municípios (último ano disponível: 2021) + per capita
# ---------------------------------------------------------------------------

def fonte_ibge_pib(skip_download: bool) -> pd.DataFrame:
    """Tabela SIDRA 5938 (PIB municipal, R$ mil) e 6579 (estimativa pop. 2021) → per capita."""
    cache_pib = EXTERNAL_RAW_DIR / "sidra_5938_pib2021.parquet"
    cache_pop = EXTERNAL_RAW_DIR / "sidra_6579_pop2021.parquet"
    if cache_pib.exists() and cache_pop.exists():
        pib, pop = pd.read_parquet(cache_pib), pd.read_parquet(cache_pop)
    else:
        if skip_download:
            raise FileNotFoundError(cache_pib)
        # 37 PIB; 498 VA total; 513 VA agro; 517 VA indústria; 6575 VA serviços (exceto adm.); 525 VA adm. pública
        pib = _sidra(5938, "37,498,513,517,6575,525", "2021")
        pop = _sidra(6579, "9324", "2021")
        pib.to_parquet(cache_pib, index=False)
        pop.to_parquet(cache_pop, index=False)
    w = pib.pivot_table(index="id_municipio", columns="variavel", values="valor")
    w.columns = [re.sub(r"\W+", "_", c.lower()).strip("_") for c in w.columns]
    ren = {}
    for c in w.columns:
        if c.startswith("produto_interno_bruto"):
            ren[c] = "pib_total_2021"
        elif "agropecu" in c:
            ren[c] = "pib_va_agro"
        elif "ind" in c and "adicionado" in c:
            ren[c] = "pib_va_industria"
        elif "servi" in c and "exclusive" in c:
            ren[c] = "pib_va_servicos"
        elif "administra" in c:
            ren[c] = "pib_va_adm_publica"
        elif c.startswith("valor_adicionado_bruto_total"):
            ren[c] = "pib_va_total"
    w = w.rename(columns=ren)
    w = w[[c for c in w.columns if c.startswith("pib_")]]
    p = pop.pivot_table(index="id_municipio", columns="variavel", values="valor")
    p.columns = ["ibge_populacao_2021"]
    w = w.join(p, how="left").reset_index()
    w["pib_per_capita_2021"] = 1000 * w["pib_total_2021"] / w["ibge_populacao_2021"]
    for setor in ["agro", "industria", "servicos", "adm_publica"]:
        col = f"pib_va_{setor}"
        if col in w.columns and "pib_va_total" in w.columns:
            w[f"pib_pct_{setor}"] = w[col] / w["pib_va_total"]
    keep = ["id_municipio", "pib_per_capita_2021", "pib_total_2021"] + [c for c in w.columns if c.startswith("pib_pct_")]
    return w[keep]


# ---------------------------------------------------------------------------
# Fonte 3 — Atlas do Desenvolvimento Humano (PNUD/IPEA/FJP) — Censo 2010
# ---------------------------------------------------------------------------
ATLAS_XLSX = EXTERNAL_RAW_DIR / "atlas2013_dadosbrutos_pt.xlsx"
ATLAS_COLS = {
    "IDHM": "atlas_idhm", "IDHM_E": "atlas_idhm_educacao", "IDHM_L": "atlas_idhm_longevidade",
    "IDHM_R": "atlas_idhm_renda", "RDPC": "atlas_renda_per_capita", "GINI": "atlas_gini",
    "PIND": "atlas_pct_extrem_pobres", "PMPOB": "atlas_pct_pobres", "T_ANALF15M": "atlas_analfabetismo_15m",
    "T_ANALF18M": "atlas_analfabetismo_18m", "T_FREQ6A14": "atlas_freq_escolar_6a14",
    "T_FUND18M": "atlas_pct_fund_completo_18m", "T_MED18M": "atlas_pct_medio_completo_18m",
    "ESPVIDA": "atlas_esperanca_vida", "T_AGUA": "atlas_pct_agua", "T_LUZ": "atlas_pct_luz",
    "AGUA_ESGOTO": "atlas_pct_agua_esgoto_inadequado", "T_DENS": "atlas_pct_dens_domicilio",
    "POPRURAL_PCT": None,  # calculado
}


def fonte_atlas(skip_download: bool) -> pd.DataFrame:
    """
    Base completa do Atlas Brasil 2013 (dados de 1991/2000/2010 por município).
    Não há URL estável de download programático: baixar manualmente em
    http://www.atlasbrasil.org.br/acervo/biblioteca  ("Base de dados — Atlas 2013 — Municípios")
    e salvar como data/external/raw/atlas2013_dadosbrutos_pt.xlsx
    """
    if not ATLAS_XLSX.exists():
        # tentativa oportunista de URLs históricas
        _try_urls([
            "http://www.atlasbrasil.org.br/acervo/biblioteca/atlas2013_dadosbrutos_pt.xlsx",
            "https://www.atlasbrasil.org.br/acervo/biblioteca/atlas2013_dadosbrutos_pt.xlsx",
        ], ATLAS_XLSX, skip_download)
    if not ATLAS_XLSX.exists():
        raise FileNotFoundError(
            "Atlas: baixar manualmente 'Base de dados - Atlas 2013 (Municípios)' em "
            "http://www.atlasbrasil.org.br/acervo/biblioteca e salvar em "
            f"{ATLAS_XLSX}")
    xls = pd.ExcelFile(ATLAS_XLSX)
    aba = [s for s in xls.sheet_names if "MUN" in s.upper()][0]
    df = xls.parse(aba)
    df = df[df["ANO"] == 2010].copy()
    df["id_municipio"] = _to_id7(df["Codmun7"])
    cols = {k: v for k, v in ATLAS_COLS.items() if v and k in df.columns}
    out = df[["id_municipio"] + list(cols)].rename(columns=cols)
    if {"pesoRUR", "pesotot"}.issubset(df.columns):
        out["atlas_pct_pop_rural_2010"] = (df["pesoRUR"] / df["pesotot"]).values
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Fonte 4 — INEP: IDEB 2023 — anos iniciais, por município
# ---------------------------------------------------------------------------

def fonte_ideb(skip_download: bool) -> pd.DataFrame:
    ano = 2023
    dest = EXTERNAL_RAW_DIR / f"divulgacao_anos_iniciais_municipios_{ano}.zip"
    _try_urls([
        f"https://download.inep.gov.br/educacao_basica/portal_ideb/planilhas_para_download/{ano}/divulgacao_anos_iniciais_municipios_{ano}.zip",
        f"https://download.inep.gov.br/ideb/resultados/divulgacao_anos_iniciais_municipios_{ano}.zip",
    ], dest, skip_download)
    if not dest.exists():
        raise FileNotFoundError(
            "IDEB: baixar 'Anos iniciais — Municípios' em "
            "https://www.gov.br/inep/pt-br/areas-de-atuacao/pesquisas-estatisticas-e-indicadores/ideb/resultados "
            f"e salvar como {dest}")
    nome, conteudo = _read_zip_member(dest, r"\.(xlsx|xls)$")
    raw = pd.read_excel(io.BytesIO(conteudo), header=None)
    h = _find_header_row(raw, "CO_MUNICIPIO")
    df = pd.read_excel(io.BytesIO(conteudo), header=h)
    df.columns = [str(c).strip().upper() for c in df.columns]
    # Rede "Pública" é a mais próxima da rede "Total" do ICA (que cobre a rede pública avaliada)
    if "REDE" in df.columns:
        pref = df[df["REDE"].astype(str).str.contains("blica", case=False)]
        df = pref if len(pref) else df
    out = pd.DataFrame({"id_municipio": _to_id7(df["CO_MUNICIPIO"])})
    padroes = {
        "ideb_ai_2023": rf"VL_OBSERVADO_{ano}",
        "ideb_ai_2021": r"VL_OBSERVADO_2021",
        "ideb_meta_2023": rf"VL_PROJECAO_{ano}",
        "ideb_nota_pt_2023": rf"VL_NOTA_PORTUGUES_{ano}",
        "ideb_nota_mat_2023": rf"VL_NOTA_MATEMATICA_{ano}",
        "ideb_aprovacao_2023": rf"VL_APROVACAO_{ano}_SI_4|VL_INDICADOR_REND_{ano}",
    }
    for novo, pat in padroes.items():
        cands = [c for c in df.columns if re.search(pat, c)]
        if cands:
            out[novo] = pd.to_numeric(df[cands[0]].replace({"-": np.nan, "ND": np.nan}), errors="coerce").values
    out = out.dropna(subset=["id_municipio"]).drop_duplicates("id_municipio")
    if "ideb_ai_2023" in out and "ideb_ai_2021" in out:
        out["ideb_var_2021_2023"] = out["ideb_ai_2023"] - out["ideb_ai_2021"]
    if "ideb_ai_2023" in out and "ideb_meta_2023" in out:
        out["ideb_gap_meta_2023"] = out["ideb_ai_2023"] - out["ideb_meta_2023"]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Fonte 5 — INEP: Indicadores educacionais 2023 (TDI, AFD, DSU, ATU) por município
# ---------------------------------------------------------------------------
INDICADORES = {
    # sigla: (regex da coluna dos anos iniciais na planilha, nome final)
    "TDI": (r"FUN_AI|ANOS_INICIAIS|AI_CAT", "inep_distorcao_idade_serie_ai"),
    "AFD": (r"FUN_AI.*G1|ANOS_INICIAIS.*G1|_AI_.*1", "inep_adequacao_docente_g1_ai"),
    "DSU": (r"FUN_AI|ANOS_INICIAIS", "inep_pct_docentes_superior_ai"),
    "ATU": (r"FUN_AI|ANOS_INICIAIS", "inep_alunos_por_turma_ai"),
}


def _ler_indicador_inep(zip_path: Path, sigla: str) -> pd.DataFrame:
    nome, conteudo = _read_zip_member(zip_path, r"\.(xlsx|xls)$")
    raw = pd.read_excel(io.BytesIO(conteudo), header=None)
    h = _find_header_row(raw, "CO_MUNICIPIO")
    # planilhas INEP têm cabeçalho em 2 linhas (grupo + subcoluna) — juntamos
    hdr1 = raw.iloc[h].astype(str).str.strip().str.upper().tolist()
    hdr2 = raw.iloc[h + 1].astype(str).str.strip().str.upper().tolist() if h + 1 < len(raw) else [""] * len(hdr1)
    cols = [f"{a}_{b}" if (b and b != "NAN") else a for a, b in zip(hdr1, hdr2)]
    df = raw.iloc[h + 2:].copy()
    df.columns = cols
    df = df[df.filter(regex="CO_MUNICIPIO").iloc[:, 0].notna()]
    return df


def fonte_inep_indicadores(skip_download: bool) -> pd.DataFrame:
    ano = 2023
    frames = []
    for sigla, (pat, nome_final) in INDICADORES.items():
        dest = EXTERNAL_RAW_DIR / f"{sigla}_{ano}_MUNICIPIOS.zip"
        _try_urls([
            f"https://download.inep.gov.br/informacoes_estatisticas/indicadores_educacionais/{ano}/{sigla}_{ano}_MUNICIPIOS.zip",
            f"https://download.inep.gov.br/informacoes_estatisticas/indicadores_educacionais/{ano}/{sigla.lower()}_municipios_{ano}.zip",
        ], dest, skip_download)
        if not dest.exists():
            print(f"    {sigla}: ausente — baixar em https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/indicadores-educacionais e salvar como {dest.name}")
            continue
        try:
            df = _ler_indicador_inep(dest, sigla)
            id_col = [c for c in df.columns if "CO_MUNICIPIO" in c][0]
            # localização "Total" e dependência "Pública" (ou Total) — mais próximo da rede Total do ICA
            for filtro_col, valor in (("NO_CATEGORIA", "Total"), ("NO_DEPENDENCIA", "Pública")):
                cc = [c for c in df.columns if filtro_col in c]
                if cc:
                    sub = df[df[cc[0]].astype(str).str.strip().str.lower() == valor.lower()]
                    df = sub if len(sub) else df
            cands = [c for c in df.columns if re.search(pat, c)]
            if not cands:
                print(f"    {sigla}: coluna dos anos iniciais não encontrada; colunas: {list(df.columns)[:15]}")
                continue
            out = pd.DataFrame({
                "id_municipio": _to_id7(df[id_col]),
                nome_final: pd.to_numeric(df[cands[0]].replace({"--": np.nan, "-": np.nan}), errors="coerce").values,
            }).dropna(subset=["id_municipio"]).drop_duplicates("id_municipio")
            frames.append(out.set_index("id_municipio"))
        except Exception as e:  # noqa: BLE001
            print(f"    {sigla}: falha no parse ({e})")
    if not frames:
        raise FileNotFoundError("nenhum indicador INEP disponível")
    return pd.concat(frames, axis=1).reset_index()


# ---------------------------------------------------------------------------
# Fonte 6 — Censo Escolar 2023 (microdados) → infraestrutura agregada por município
# ---------------------------------------------------------------------------
CENSO_ZIP = EXTERNAL_RAW_DIR / "microdados_censo_escolar_2023.zip"
CENSO_USECOLS = [
    "CO_MUNICIPIO", "TP_DEPENDENCIA", "TP_LOCALIZACAO", "TP_SITUACAO_FUNCIONAMENTO",
    "IN_INTERNET", "IN_BIBLIOTECA", "IN_BIBLIOTECA_SALA_LEITURA", "IN_LABORATORIO_INFORMATICA",
    "IN_AGUA_POTAVEL", "IN_ENERGIA_REDE_PUBLICA", "IN_ESGOTO_REDE_PUBLICA",
    "IN_FUND_AI", "QT_MAT_FUND_AI", "QT_DOC_FUND_AI", "QT_TUR_FUND_AI",
    "QT_MAT_FUND_AI_INT", "QT_DESKTOP_ALUNO", "QT_SALAS_UTILIZADAS",
]


def fonte_censo_escolar(skip_download: bool) -> pd.DataFrame:
    """
    Agrega os microdados do Censo Escolar 2023 (arquivo de escolas) por município,
    considerando escolas em atividade da rede pública que oferecem anos iniciais.
    Download (≈ 100 MB): https://download.inep.gov.br/dados_abertos/microdados_censo_escolar_2023.zip
    """
    _try_urls(["https://download.inep.gov.br/dados_abertos/microdados_censo_escolar_2023.zip"],
              CENSO_ZIP, skip_download)
    if not CENSO_ZIP.exists():
        raise FileNotFoundError(f"Censo Escolar: salvar o zip de microdados em {CENSO_ZIP}")
    nome, _ = _read_zip_member(CENSO_ZIP, r"microdados_ed_basica_2023\.csv$")
    with zipfile.ZipFile(CENSO_ZIP) as z:
        with z.open(nome) as f:
            head = pd.read_csv(f, sep=";", encoding="latin-1", nrows=0)
            usecols = [c for c in CENSO_USECOLS if c in head.columns]
        with z.open(nome) as f:
            df = pd.read_csv(f, sep=";", encoding="latin-1", usecols=usecols, low_memory=False)
    df = df[(df["TP_SITUACAO_FUNCIONAMENTO"] == 1) & (df["TP_DEPENDENCIA"].isin([1, 2, 3]))]
    if "IN_FUND_AI" in df:
        df = df[df["IN_FUND_AI"] == 1]
    g = df.groupby("CO_MUNICIPIO")
    out = pd.DataFrame({
        "censo_n_escolas_ai": g.size(),
        "censo_matriculas_ai": g["QT_MAT_FUND_AI"].sum(min_count=1),
        "censo_docentes_ai": g["QT_DOC_FUND_AI"].sum(min_count=1),
        "censo_pct_escolas_rurais": g["TP_LOCALIZACAO"].apply(lambda s: (s == 2).mean()),
        "censo_pct_escolas_internet": g["IN_INTERNET"].mean(),
        "censo_pct_escolas_biblioteca": g["IN_BIBLIOTECA"].mean() if "IN_BIBLIOTECA" in df else g["IN_BIBLIOTECA_SALA_LEITURA"].mean(),
        "censo_pct_escolas_lab_info": g["IN_LABORATORIO_INFORMATICA"].mean(),
        "censo_pct_escolas_agua_potavel": g["IN_AGUA_POTAVEL"].mean(),
        "censo_pct_escolas_esgoto": g["IN_ESGOTO_REDE_PUBLICA"].mean(),
        "censo_pct_mat_integral_ai": g["QT_MAT_FUND_AI_INT"].sum(min_count=1) / g["QT_MAT_FUND_AI"].sum(min_count=1),
    })
    out["censo_alunos_por_docente_ai"] = out["censo_matriculas_ai"] / out["censo_docentes_ai"]
    out["censo_alunos_por_turma_ai"] = out["censo_matriculas_ai"] / g["QT_TUR_FUND_AI"].sum(min_count=1)
    out.index.name = "id_municipio"
    return out.reset_index()


# ---------------------------------------------------------------------------
# Fonte 7 — Auxiliares (GitHub): coordenadas/capital e população DATASUS 2023
# ---------------------------------------------------------------------------

def fonte_aux_github(skip_download: bool) -> pd.DataFrame:
    """Latitude/longitude e flag de capital (kelvins/municipios-brasileiros) + população 2023 (lsbastos/popBR_mun, RIPSA/MS)."""
    p1 = _download_file("https://raw.githubusercontent.com/kelvins/municipios-brasileiros/main/csv/municipios.csv",
                        EXTERNAL_RAW_DIR / "municipios_kelvins.csv", skip_download)
    p2 = _download_file("https://raw.githubusercontent.com/lsbastos/popBR_mun/master/popBR2000-2025.long.csv",
                        EXTERNAL_RAW_DIR / "popBR2000-2025.long.csv", skip_download)
    if p1 is None:
        raise FileNotFoundError("municipios_kelvins.csv")
    m = pd.read_csv(p1)
    out = pd.DataFrame({
        "id_municipio": _to_id7(m["codigo_ibge"]),
        "geo_latitude": m["latitude"], "geo_longitude": m["longitude"],
        "geo_capital": m["capital"].astype(int),
    })
    if p2 is not None:
        pop = pd.read_csv(p2)
        pop = pop[pop["Ano"] == ANO_FEATURES][["MUNCOD", "POP"]].rename(columns={"POP": f"datasus_populacao_{ANO_FEATURES}"})
        out["_id6"] = out["id_municipio"] // 10
        out = out.merge(pop, left_on="_id6", right_on="MUNCOD", how="left").drop(columns=["_id6", "MUNCOD"])
    return out


# ---------------------------------------------------------------------------
# Fontes 8/9 — FUNDEB e Cadastro Único (segunda leva; entrada manual)
# ---------------------------------------------------------------------------

def fonte_manual_csv(nome: str, prefixo: str) -> pd.DataFrame:
    """
    Lê data/external/raw/<nome>.csv se o grupo tiver preparado manualmente
    (FUNDEB via FNDE/SIOPE; Cadastro Único via CECAD/VIS DATA).
    Formato esperado: coluna id_municipio + colunas numéricas.
    """
    p = EXTERNAL_RAW_DIR / f"{nome}.csv"
    if not p.exists():
        raise FileNotFoundError(f"{nome}: preparar {p} (id_municipio + variáveis)")
    df = pd.read_csv(p)
    df["id_municipio"] = _to_id7(df["id_municipio"])
    df = df.rename(columns={c: f"{prefixo}_{c}" for c in df.columns if c != "id_municipio"})
    return df


FONTES = {
    "ibge_populacao": fonte_ibge_populacao,
    "ibge_pib": fonte_ibge_pib,
    "atlas": fonte_atlas,
    "ideb": fonte_ideb,
    "inep_indicadores": fonte_inep_indicadores,
    "censo_escolar": fonte_censo_escolar,
    "aux_github": fonte_aux_github,
    "fundeb": lambda s: fonte_manual_csv("fundeb_2023", "fundeb"),
    "cadunico": lambda s: fonte_manual_csv("cadunico_2023", "cadunico"),
}


# ---------------------------------------------------------------------------
# Consolidação
# ---------------------------------------------------------------------------

def consolidar() -> pd.DataFrame:
    frames = []
    for nome in FONTES:
        p = EXTERNAL_DIR / f"{nome}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df["id_municipio"] = df["id_municipio"].astype("Int64")
            df = df.dropna(subset=["id_municipio"]).drop_duplicates("id_municipio").set_index("id_municipio")
            frames.append(df)
    if not frames:
        raise SystemExit("nenhuma fonte processada — nada a consolidar")
    base = pd.concat(frames, axis=1, join="outer").reset_index()
    base["id_municipio"] = base["id_municipio"].astype(int)
    out = EXTERNAL_DIR / "socioeconomico_municipal.parquet"
    base.to_parquet(out, index=False)
    print(f"\n✓ consolidado: {out}  shape={base.shape}")
    cobertura = base.drop(columns="id_municipio").notna().mean().sort_values()
    print("  cobertura por coluna (fração de municípios com valor):")
    print(cobertura.to_string(float_format=lambda x: f"{x:.2f}"))
    return base


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=list(FONTES), help="processar apenas estas fontes")
    ap.add_argument("--skip-download", action="store_true", help="não acessar a rede; usar só data/external/raw")
    args = ap.parse_args(argv)

    fontes = args.only or list(FONTES)
    status = {}
    for nome in fontes:
        print(f"\n[{nome}]")
        try:
            df = FONTES[nome](args.skip_download)
            df.to_parquet(EXTERNAL_DIR / f"{nome}.parquet", index=False)
            status[nome] = f"ok  ({df.shape[0]} municípios, {df.shape[1]-1} colunas)"
        except Exception as e:  # noqa: BLE001
            status[nome] = f"pendente — {e}"
    print("\n" + "=" * 70)
    for k, v in status.items():
        print(f"  {k:<18} {v}")
    print("=" * 70)
    consolidar()


if __name__ == "__main__":
    sys.exit(main())

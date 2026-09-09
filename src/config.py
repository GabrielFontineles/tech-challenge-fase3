"""
Configuração central do projeto — Tech Challenge Fase 3.

Tudo que é "decisão analítica" (anos, corte do target, colunas, seeds) fica
aqui, para que notebooks e scripts leiam de um único lugar.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                 # Silver da Fase 2
GOLD_DIR = DATA_DIR / "gold"               # Gold da Fase 2
EXTERNAL_RAW_DIR = DATA_DIR / "external" / "raw"        # arquivos como baixados
EXTERNAL_DIR = DATA_DIR / "external" / "processed"      # tabelas padronizadas
PROCESSED_DIR = DATA_DIR / "processed"     # dataset final de modelagem
MODELS_DIR = ROOT / "models"
IMAGES_DIR = ROOT / "images"
REPORTS_DIR = ROOT / "reports"

for _d in (EXTERNAL_RAW_DIR, EXTERNAL_DIR, PROCESSED_DIR, MODELS_DIR, IMAGES_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Desenho temporal
# ---------------------------------------------------------------------------
ANO_FEATURES = 2023      # ano das variáveis explicativas (t)
ANO_TARGET = 2024        # ano do target (t+1)
REDE = "Total"

# Target: classe positiva = RISCO (taxa_{t+1} < corte)
CORTE_RISCO = 60.0
TARGET = "em_risco"

# ---------------------------------------------------------------------------
# Colunas esperadas na municipio_silver (Fase 2). Ajustar se o nome diferir.
# ---------------------------------------------------------------------------
COL_ID = "id_municipio"
COL_ANO = "ano"
COL_REDE = "rede"
COL_TAXA = "taxa_alfabetizacao"
COL_MEDIA_PT = "media_portugues"
COL_META_2030 = "meta_alfabetizacao_2030"
COL_DIST_META = "distancia_meta_2030"
COL_PARTICIP = "percentual_participacao"
COL_NIVEL = "nivel_alfabetizacao"
COLS_NIVEIS = [f"proporcao_aluno_nivel_{i}" for i in range(9)]
# opcional (se existir na Silver): número de alunos avaliados
COL_N_ALUNOS = "quantidade_alunos_avaliados"

# ---------------------------------------------------------------------------
# Reprodutibilidade
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_FOLDS = 5

# ---------------------------------------------------------------------------
# Geografia
# ---------------------------------------------------------------------------
UF_POR_CODIGO = {
    11: "RO", 12: "AC", 13: "AM", 14: "RR", 15: "PA", 16: "AP", 17: "TO",
    21: "MA", 22: "PI", 23: "CE", 24: "RN", 25: "PB", 26: "PE", 27: "AL",
    28: "SE", 29: "BA", 31: "MG", 32: "ES", 33: "RJ", 35: "SP",
    41: "PR", 42: "SC", 43: "RS", 50: "MS", 51: "MT", 52: "GO", 53: "DF",
}
REGIAO_POR_UF = {
    **{uf: "Norte" for uf in ["AC", "AM", "AP", "PA", "RO", "RR", "TO"]},
    **{uf: "Nordeste" for uf in ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"]},
    **{uf: "Centro-Oeste" for uf in ["DF", "GO", "MS", "MT"]},
    **{uf: "Sudeste" for uf in ["ES", "MG", "RJ", "SP"]},
    **{uf: "Sul" for uf in ["PR", "RS", "SC"]},
}
# Faixas de porte populacional (IBGE / classificação usual em políticas públicas)
PORTE_BINS = [0, 5_000, 10_000, 20_000, 50_000, 100_000, 500_000, float("inf")]
PORTE_LABELS = ["ate_5k", "5k_10k", "10k_20k", "20k_50k", "50k_100k", "100k_500k", "acima_500k"]

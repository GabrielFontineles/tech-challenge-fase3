"""Gera o scaffolding dos notebooks 01–05 (narrativa + células prontas para executar)."""
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
NB.mkdir(exist_ok=True)

SETUP = """import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))   # permite `import src` a partir de notebooks/
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from src import config as C
sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 60)"""


def nb(titulo, celulas, nome):
    n = nbf.v4.new_notebook()
    n.cells.append(nbf.v4.new_markdown_cell(f"# {titulo}\n\n**Tech Challenge Fase 3 — FIAP Pós-Tech IA Scientist**"))
    n.cells.append(nbf.v4.new_code_cell(SETUP))
    for tipo, txt in celulas:
        n.cells.append(nbf.v4.new_markdown_cell(txt) if tipo == "md" else nbf.v4.new_code_cell(txt))
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(n, NB / nome)
    print("✓", nome)


# ---------------------------------------------------------------- 01 EDA
nb("01 — Análise Exploratória e Entendimento do Problema", [
    ("md", """## 1. Contexto e pergunta analítica

O Compromisso Nacional Criança Alfabetizada fixa a meta de 80% de crianças alfabetizadas ao fim do 2º ano até 2030.
O Indicador Criança Alfabetizada (ICA), construído a partir do SAEB 2º ano, mede anualmente a taxa por município.

**Pergunta:** dado o que sabemos sobre um município em *t* (resultado do ano anterior, território, condições socioeconômicas e estrutura da rede), qual a probabilidade de ele estar **em risco** (taxa < 60%) em *t+1* — e quais fatores explicam esse risco?

**Desenho temporal:** features de 2023 → target de 2024. Nenhuma variável de 2024 entra como feature (ver `src/data/build_dataset.py`)."""),
    ("md", """## 2. Hipóteses a validar nesta EDA

| # | Hipótese | Como testar aqui |
|---|---|---|
| H1 | Inércia: a taxa de 2023 é o preditor mais forte; a maioria dos municípios em risco em 2024 já estava em risco em 2023 | matriz de transição 2023→2024; correlação taxa_t × taxa_t1 |
| H2 | Desigualdade territorial: Norte/Nordeste concentram o risco mesmo controlando por IDHM/renda | risco por região; boxplot de taxa por região dentro de faixas de IDHM |
| H3 | Condição socioeconômica: IDHM-E, renda e CadÚnico explicam parte da variância não explicada pela taxa anterior | correlação parcial (resíduo de taxa_t1 ~ taxa_t vs variáveis socioeconômicas) |
| H4 | Estrutura da rede: infraestrutura, docentes com superior e distorção idade-série associam-se ao risco | correlações e boxplots por status de risco |
| H5 | Participação baixa no SAEB em t → maior risco e maior variância em t+1 | risco por decil de participação; desvio da variação por decil |
| H6 | Porte: municípios pequenos têm maior volatilidade de taxa entre anos | |variação| por porte populacional |"""),
    ("code", """silver = pd.read_parquet(C.RAW_DIR / "municipio_silver.parquet")
silver = silver[silver[C.COL_REDE] == C.REDE]
print(silver.shape); silver.groupby(C.COL_ANO)[C.COL_ID].nunique()"""),
    ("md", "## 3. Distribuições e qualidade dos dados"),
    ("code", """silver.describe().T.round(2)"""),
    ("code", """fig, ax = plt.subplots(1, 2, figsize=(14, 4))
for ano, cor in ((2023, "#7f8c8d"), (2024, "#2e86c1")):
    sns.histplot(silver.loc[silver[C.COL_ANO] == ano, C.COL_TAXA], bins=40, ax=ax[0], color=cor, label=str(ano), alpha=.6)
ax[0].axvline(C.CORTE_RISCO, color="#c0392b", ls="--", label=f"corte risco {C.CORTE_RISCO:.0f}%")
ax[0].legend(); ax[0].set_title("Distribuição da taxa de alfabetização por ano")
(silver.isna().mean() * 100).sort_values(ascending=False).head(12).plot.barh(ax=ax[1]); ax[1].set_title("% missing por coluna")
plt.tight_layout()"""),
    ("md", "## 4. H1 — Inércia e matriz de transição 2023 → 2024"),
    ("code", """dados = pd.read_parquet(C.PROCESSED_DIR / "dataset_modelagem.parquet")
from src.visualization.plots import plot_matriz_transicao
tab = plot_matriz_transicao(dados, f"{C.COL_TAXA}_t", "taxa_alfabetizacao_t1", C.CORTE_RISCO, C.IMAGES_DIR / "03_matriz_transicao.png")
print("correlação taxa_t × taxa_t1:", dados[[f"{C.COL_TAXA}_t", "taxa_alfabetizacao_t1"]].corr().iloc[0, 1].round(3))
tab"""),
    ("md", """> **Leitura:** os municípios que *mudam* de estado (fora da diagonal) são exatamente os que o modelo precisa capturar além da inércia. Registre aqui a proporção e onde eles se concentram."""),
    ("md", "## 5. H2 — Território"),
    ("code", """fig, ax = plt.subplots(1, 2, figsize=(14, 4))
dados.groupby("regiao")[C.TARGET].mean().sort_values().plot.barh(ax=ax[0], color="#c0392b"); ax[0].set_title("Fração de municípios em risco (2024) por região")
sns.boxplot(data=dados, x="regiao", y=f"{C.COL_TAXA}_t", ax=ax[1]); ax[1].set_title("Taxa 2023 por região")
plt.tight_layout()"""),
    ("code", """# H2 controlando por IDHM (requer fonte Atlas)
if "atlas_idhm" in dados:
    dados["faixa_idhm"] = pd.qcut(dados["atlas_idhm"], 4, labels=["Q1 baixo", "Q2", "Q3", "Q4 alto"])
    display(dados.pivot_table(index="regiao", columns="faixa_idhm", values=C.TARGET, aggfunc="mean").round(2))
else:
    print("Atlas ainda não carregado — rode src.data.download_external")"""),
    ("md", "## 6. H3/H4 — Socioeconômico e estrutura da rede: correlação com o *resíduo* da inércia"),
    ("code", """# Resíduo: o que a taxa de 2023 NÃO explica da taxa de 2024
from sklearn.linear_model import LinearRegression
m = LinearRegression().fit(dados[[f"{C.COL_TAXA}_t"]], dados["taxa_alfabetizacao_t1"])
dados["residuo_inercia"] = dados["taxa_alfabetizacao_t1"] - m.predict(dados[[f"{C.COL_TAXA}_t"]])
externas = [c for c in dados.columns if c.startswith(("atlas_", "pib_", "inep_", "censo_", "ideb_", "fundeb_", "cadunico_", "ibge_"))]
if externas:
    corr = dados[externas + ["residuo_inercia", C.TARGET]].corr()[["residuo_inercia", C.TARGET]].drop(["residuo_inercia", C.TARGET])
    display(corr.sort_values(C.TARGET).round(3))
else:
    print("sem variáveis externas ainda")"""),
    ("md", "## 7. H5 — Participação e H6 — Porte"),
    ("code", """dados["decil_particip"] = pd.qcut(dados[f"{C.COL_PARTICIP}_t"], 10, labels=False, duplicates="drop")
dados["variacao"] = dados["taxa_alfabetizacao_t1"] - dados[f"{C.COL_TAXA}_t"]
g = dados.groupby("decil_particip").agg(risco=(C.TARGET, "mean"), desvio_variacao=("variacao", "std"), n=(C.COL_ID, "size"))
display(g.round(3))
if "porte" in dados:
    display(dados.groupby("porte").agg(risco=(C.TARGET, "mean"), abs_variacao=("variacao", lambda s: s.abs().mean()), n=(C.COL_ID, "size")).round(3))"""),
    ("md", "## 8. Correlações entre features (multicolinearidade — informa a escolha de modelos lineares vs árvores)"),
    ("code", """num = dados.select_dtypes("number").drop(columns=[C.COL_ID, C.TARGET, "taxa_alfabetizacao_t1"], errors="ignore")
plt.figure(figsize=(14, 11)); sns.heatmap(num.corr(), cmap="coolwarm", center=0, vmin=-1, vmax=1); plt.title("Correlação entre features (bloco A + externas)")"""),
    ("md", """## 9. Conclusões da EDA → decisões de modelagem

Preencher após executar com dados reais:

- **H1:** ...
- **H2:** ...
- **H3/H4:** ...
- **H5:** ...
- **H6:** ...

**Decisões:** (a) manter o bloco histórico como features e treinar também a variante *sem histórico* para isolar fatores estruturais; (b) usar PR-AUC e recall da classe risco como métricas principais; (c) tratar missing com imputação por mediana + indicador; (d) categorias (UF/região/porte) via one-hot no pipeline."""),
], "01_eda_exploratoria.ipynb")

# ---------------------------------------------------------------- 02 features
nb("02 — Enriquecimento externo e engenharia de atributos", [
    ("md", """## Fontes externas e regra temporal

Todas as fontes são **anteriores ou contemporâneas a 2023**. O script `src/data/download_external.py` baixa, padroniza e consolida; este notebook documenta a cobertura e as decisões de junção."""),
    ("code", """ext = pd.read_parquet(C.EXTERNAL_DIR / "socioeconomico_municipal.parquet")
print(ext.shape)
(ext.drop(columns=C.COL_ID).notna().mean() * 100).round(1).sort_values().to_frame("% cobertura")"""),
    ("md", "## Construção do dataset t → t+1"),
    ("code", """from src.data import build_dataset
dados = build_dataset.construir()
pd.read_csv(C.PROCESSED_DIR / "dicionario_dataset.csv")"""),
    ("md", """## Tratamento de data leakage — checklist auditável

1. Do ano t+1 entra apenas `em_risco` (e `taxa_alfabetizacao_t1` para análise) — `COLUNAS_PROIBIDAS` em `build_dataset.py`.
2. Toda feature do bloco A tem sufixo `_t`; `features.auditar()` falha se algo com `_t1` chegar a X.
3. Imputação, escala e one-hot vivem dentro do `Pipeline` → ajustados só no treino (`pipeline.py`).
4. `taxa_vs_mediana_uf_t` usa apenas dados de t.
5. Nenhum encoding ordinal "por desempenho" (o `regiao_encoded` da versão anterior foi removido)."""),
    ("code", """from src.preprocessing.features import separar_xy, tipos
X, y = separar_xy(dados); num, cat = tipos(X)
print(f"{X.shape[1]} features: {len(num)} numéricas, {len(cat)} categóricas ({cat})")"""),
    ("md", "## Sensibilidade do corte do target"),
    ("code", """for corte in (50, 55, 60, 65, 70):
    print(f"corte {corte}%: risco = {(dados['taxa_alfabetizacao_t1'] < corte).mean():.1%}")
if f"{C.COL_META_2030}_t" in dados:
    alt = dados["taxa_alfabetizacao_t1"] < 0.75 * dados[f"{C.COL_META_2030}_t"]
    print(f"corte alternativo (75% da meta municipal): risco = {alt.mean():.1%}; concordância com corte 60%: {(alt == dados[C.TARGET].astype(bool)).mean():.1%}")"""),
], "02_enriquecimento_e_features.ipynb")

# ---------------------------------------------------------------- 03 modelagem
nb("03 — Modelagem supervisionada e validação", [
    ("md", """## Protocolo

1. Split estratificado 80/20 — o teste é usado **uma única vez**, no final.
2. CV estratificado 5-fold no treino: baselines (Dummy, LogReg, RF, HistGradientBoosting).
3. `RandomizedSearchCV` (scoring = average precision) sobre o pipeline completo.
4. Threshold escolhido em predições *out-of-fold* do treino, exigindo recall da classe risco ≥ 0,85.
5. Avaliação no teste, calibração isotônica, persistência em `models/`.

Tudo isso está encapsulado em `src/modeling/train.py`; aqui executamos e inspecionamos."""),
    ("code", """from src.modeling import train
train.main(["--recall-minimo", "0.85"])   # use ["--rapido"] para um smoke test"""),
    ("code", """import json
meta = json.loads((C.MODELS_DIR / "metadata.json").read_text())
pd.read_csv(C.REPORTS_DIR / "cv_baselines.csv")"""),
    ("code", """pd.DataFrame({"threshold escolhido": meta["teste_threshold_escolhido"], "threshold 0.5": meta["teste_threshold_0.5"]}).T"""),
    ("md", "## Controle de overfitting: gap treino × CV por modelo"),
    ("code", """pd.DataFrame(meta["busca"]).T"""),
    ("code", """from IPython.display import Image; Image(str(C.IMAGES_DIR / "09_resultados_modelos.png"))"""),
    ("md", "## Variante estrutural (sem bloco histórico) — quanto os fatores estruturais sozinhos explicam?"),
    ("code", """train.main(["--sem-historico"])
json.loads((C.MODELS_DIR / "metadata_sem_historico.json").read_text())["teste_threshold_escolhido"]"""),
    ("md", """## Interpretação (preencher)

- Modelo escolhido e por quê (PR-AUC no CV, estabilidade, gap treino/CV).
- Custo do threshold: quantos falsos alarmes por município em risco capturado.
- Diferença entre o modelo completo e a variante estrutural."""),
], "03_modelagem_e_validacao.ipynb")

# ---------------------------------------------------------------- 04 SHAP
nb("04 — Interpretabilidade: importância por permutação e SHAP", [
    ("code", """from src.evaluation import shap_analysis
shap_analysis.main(["--top", "20"])"""),
    ("code", """from IPython.display import Image, display
for f in ("10_importancia_permutacao", "11_shap_summary", "13_shap_dependence", "14_waterfall_maior_risco", "14_waterfall_risco_nao_capturado"):
    display(Image(str(C.IMAGES_DIR / f"{f}.png")))"""),
    ("md", "## Fatores estruturais (variante sem histórico)"),
    ("code", """shap_analysis.main(["--sem-historico", "--top", "15"])
display(Image(str(C.IMAGES_DIR / "11_shap_summary_sem_historico.png")))"""),
    ("md", """## Convergência entre métodos

| Feature | Permutação | SHAP | Coef. LogReg |
|---|---|---|---|
| ... | | | |

**Quais fatores mais impactam a alfabetização?** (responder com base nas duas variantes)"""),
], "04_interpretabilidade_shap.ipynb")

# ---------------------------------------------------------------- 05 estratégia
nb("05 — Aplicação estratégica para políticas públicas", [
    ("code", """from src.evaluation import strategic
strategic.main()"""),
    ("md", "## Quais municípios apresentam maior risco?"),
    ("code", """rank = pd.read_csv(C.REPORTS_DIR / "ranking_risco_municipios.csv"); rank.head(30)"""),
    ("code", """rank.groupby("sigla_uf")["prob_risco"].mean().sort_values(ascending=False).plot.bar(figsize=(14, 4), color="#c0392b", title="Probabilidade média de risco por UF")"""),
    ("md", "## Quais regiões possuem padrões semelhantes?"),
    ("code", """pd.read_csv(C.REPORTS_DIR / "perfil_clusters.csv")"""),
    ("md", "## Como prever municípios que podem não atingir metas futuras?"),
    ("code", """proj = pd.read_csv(C.REPORTS_DIR / "projecao_meta_2030.csv")
proj["status_meta"].value_counts().plot.pie(autopct="%1.0f%%", title="Status projetado da meta 2030")"""),
    ("md", "## Lista de priorização (alto risco × vulnerabilidade)"),
    ("code", """pd.read_csv(C.REPORTS_DIR / "priorizacao_municipios.csv").head(50)"""),
    ("md", """## Mapa (opcional — requer `geobr`)

```python
# import geobr
# mun = geobr.read_municipality(year=2020)
# mun.merge(rank, left_on="code_muni", right_on="id_municipio").plot(column="prob_risco", cmap="Reds", legend=True, figsize=(10, 10))
```"""),
    ("md", """## Recomendações para gestores (preencher)

1. Priorização imediata: ...
2. Alavancas estruturais com maior efeito (do SHAP sem histórico): ...
3. Monitoramento: municípios "atenção" com participação em queda ...
4. Limites da evidência: correlacional, série curta, ..."""),
], "05_analise_estrategica.ipynb")

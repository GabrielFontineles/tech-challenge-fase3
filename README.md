# Predição e Inteligência Analítica para Alfabetização no Brasil

> Tech Challenge — Fase 3 · Pós-Tech FIAP — IA Scientist
> Modelo supervisionado que estima, para cada município, a probabilidade de estar **em risco de não alfabetização** no próximo ciclo de avaliação, e explica quais fatores educacionais, territoriais e socioeconômicos sustentam esse risco.

> ⚠️ **Status:** pipeline e documentação prontas; os números marcados com `[a preencher]` serão atualizados após a execução com os dados reais da Fase 2 e das fontes externas.

---

## Contexto do problema

A alfabetização ao final do 2º ano do ensino fundamental é um dos indicadores mais sensíveis do desenvolvimento educacional e social do país. O **Compromisso Nacional Criança Alfabetizada** estabelece a meta de 80% de crianças alfabetizadas até 2030, monitorada pelo **Indicador Criança Alfabetizada (ICA)**, calculado a partir do SAEB 2º ano.

Na Fase 2 construímos a pipeline de engenharia de dados (camadas Bronze → Silver → Gold) que integra o ICA com metas nacionais, estaduais e municipais. Nesta fase, transformamos esses dados em **inteligência preditiva**: em vez de descrever o que aconteceu, antecipamos onde o risco vai se materializar e quais alavancas os gestores podem acionar.

## Objetivo analítico

Prever se um município estará **em risco** (taxa de alfabetização < 60%) no ano *t+1*, usando apenas informação disponível até o ano *t*, e responder:

1. Quais fatores mais impactam a alfabetização?
2. Quais municípios apresentam maior risco educacional?
3. Quais regiões possuem padrões semelhantes?
4. Como prever municípios que podem não atingir as metas de 2030?
5. Quais variáveis têm maior influência nos modelos?

**Unidade de análise — por que município e não aluno.** Os microdados por aluno (SAEB 2º ano) não trazem variáveis socioeconômicas individuais e só estão acessíveis via BigQuery em volume elevado; a decisão de política pública (priorização de recursos, FUNDEB, programas de reforço) ocorre no nível municipal; e a camada Gold da Fase 2 foi construída nessa granularidade. Registramos a escolha como decisão analítica, não como limitação escondida.

## Base utilizada

**Camada Gold/Silver da Fase 2** — `municipio_silver` (município × ano × rede): taxa de alfabetização, média em português, participação, distribuição de alunos por nível de proficiência, meta 2030 e distância à meta; tabelas Gold de ranking de UFs, evolução temporal, visão municipal e nacional.

**Enriquecimento externo** (todas anteriores ou contemporâneas a 2023 — ver `data/README.md`):

| Bloco | Fonte | Variáveis |
|---|---|---|
| A — Histórico educacional (t = 2023) | ICA / Fase 2 | taxa, média PT, participação, proporção por nível, gap para a meta, posição relativa à UF, entropia dos níveis, nº de alunos avaliados |
| B — Território | IBGE Censo 2022, código IBGE | UF, região, população, densidade, porte, capital |
| C — Socioeconômico e estrutura da rede | Atlas do Desenvolvimento Humano (2010), IBGE PIB municipal (2021), INEP IDEB 2023, INEP Indicadores Educacionais 2023, Censo Escolar 2023, FUNDEB, Cadastro Único | IDHM e subíndices, renda, Gini, pobreza, analfabetismo adulto, PIB per capita e composição setorial, IDEB anos iniciais e variação, distorção idade-série, adequação docente, infraestrutura escolar (internet, biblioteca, laboratório, água, esgoto), alunos por docente/turma, matrícula integral, receita FUNDEB por matrícula, % população no CadÚnico |

**Dataset de modelagem:** `[a preencher]` municípios pareados 2023 → 2024 (rede Total), `[a preencher]` features; prevalência da classe risco `[a preencher]`%.

## Desenho temporal e tratamento de data leakage

O ponto central do desenho: **features observadas em 2023, target observado em 2024.**

Na primeira versão deste projeto o target era derivado da taxa de 2024 e as features incluíam a distribuição de alunos por nível também de 2024 — que é, por construção, a decomposição da própria taxa. O modelo atingia ROC-AUC de 0,997 "reconstruindo" a fórmula do target. Reformulamos o problema para um desenho preditivo real, com regras auditáveis pelo nome da coluna:

1. Do ano t+1 entra **apenas** o target `em_risco` (a taxa contínua `taxa_alfabetizacao_t1` fica guardada só para análise). Lista `COLUNAS_PROIBIDAS` em `src/data/build_dataset.py`; `features.auditar()` interrompe a execução se algo com sufixo `_t1` chegar a X.
2. Toda feature histórica carrega sufixo `_t`; toda feature externa carrega o prefixo da fonte (e ano ≤ 2023).
3. Imputação (mediana + indicador de ausência), padronização e one-hot vivem **dentro** do `Pipeline` do scikit-learn → ajustadas apenas no treino, em cada fold.
4. Nenhum encoding ordinal "por desempenho" (o `regiao_encoded` da versão anterior foi removido); UF, região e porte entram como categóricas.
5. O conjunto de teste (20%, estratificado) é tocado uma única vez; o threshold é escolhido em predições *out-of-fold* do treino.

## Etapas de modelagem

```
src/data/download_external.py   → fontes externas padronizadas por id_municipio
src/data/build_dataset.py       → pareamento t → t+1, blocos A/B/C, dicionário de dados
src/preprocessing/features.py   → seleção + auditoria anti-leakage
src/preprocessing/pipeline.py   → ColumnTransformer (num: imputer+indicador+scaler | cat: imputer+one-hot)
src/modeling/train.py           → baselines em CV, RandomizedSearchCV, threshold, calibração, joblib
src/evaluation/shap_analysis.py → importância por permutação + SHAP (global, dependence, waterfall)
src/evaluation/strategic.py     → ranking de risco, clusters, projeção de metas, priorização
```

```python
Pipeline([
    ("pre", ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                          ("scaler", StandardScaler())]), colunas_numericas),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20))]), colunas_categoricas),
    ])),
    ("clf", HistGradientBoostingClassifier(...)),
])
```

**Validação:** split estratificado 80/20 (teste) + `StratifiedKFold(5)` no treino (validação) → estrutura treino / validação / teste. `RandomizedSearchCV` (40 iterações, `scoring="average_precision"`) para Regressão Logística, Random Forest e HistGradientBoosting, com registro do *gap* treino × CV como controle de overfitting. `DummyClassifier` como piso.

**Threshold:** para o gestor, o erro mais caro é deixar passar um município em risco. Escolhemos o maior threshold cujo recall da classe risco (out-of-fold) é ≥ 0,85, e reportamos também o resultado em 0,5.

**Duas variantes:** modelo completo (blocos A+B+C) e modelo **estrutural** (B+C, sem histórico) — o segundo isola os fatores sobre os quais a política pública pode agir.

## Escolha do algoritmo

| Modelo | ROC-AUC (CV) | PR-AUC (CV) | Recall risco (CV) | Gap treino–CV |
|---|---|---|---|---|
| Dummy (prior) | 0,500 | `[a preencher]` | — | — |
| Regressão Logística | `[a preencher]` | | | |
| Random Forest | `[a preencher]` | | | |
| HistGradientBoosting | `[a preencher]` | | | |

**Selecionado:** `[a preencher]` — critério: maior PR-AUC no CV com gap treino–CV pequeno.

## Métricas de avaliação — conjunto de teste (uma única avaliação)

| Métrica | threshold escolhido (`[a preencher]`) | threshold 0,5 |
|---|---|---|
| ROC-AUC | `[a preencher]` | |
| PR-AUC | | |
| Recall (risco) | | |
| Precisão (risco) | | |
| F1 (risco) | | |
| Brier (calibrado) | | |

Leitura esperada: AUC entre 0,80 e 0,90. Valores acima de 0,97 são tratados como sintoma de vazamento, não como sucesso.

## Interpretação dos resultados

- Importância por permutação (queda de PR-AUC no teste) e SHAP (TreeExplainer) para o modelo completo e para a variante estrutural.
- Dependence plots das 4 principais features e *waterfall* de três municípios: maior risco, limítrofe e risco não capturado.
- `[a preencher]` — tabela de convergência entre permutação, SHAP e coeficientes da regressão logística.

## Insights encontrados

`[a preencher após execução com dados reais — ver notebooks 01, 04 e 05]`

## Limitações do projeto

- Série histórica de dois anos: um único par t → t+1, sem validação temporal com múltiplos anos.
- Atlas do Desenvolvimento Humano baseado no Censo 2010; PIB municipal de 2021; Censo 2022 ainda parcial no nível municipal.
- Ruído amostral em municípios pequenos (poucos alunos avaliados) — tratado com `log_alunos_avaliados_t`, mas não eliminado.
- Target binário com corte de 60% (análise de sensibilidade no notebook 02).
- Rede Total não separa desempenho da rede municipal e estadual.
- Evidência correlacional: o modelo aponta associações, não efeitos causais.

## Aplicação prática para políticas públicas

- **Ranking de risco** calibrado por município (`reports/ranking_risco_municipios.csv`) → priorização de visitas técnicas e recursos.
- **Lista de priorização** (alto risco × baixo IDHM-Educação) → `reports/priorizacao_municipios.csv`.
- **Clusters estruturais** → desenhar intervenções por perfil, não por UF.
- **Projeção de metas 2030** → municípios "fora da trajetória" com a variação anual necessária para atingir a meta.
- **Alavancas** (variante estrutural do SHAP) → `[a preencher]`.

## Possíveis evoluções futuras

- Incorporar novas edições do ICA para validação temporal (treinar em 2023→2024, testar em 2024→2025).
- Modelo de regressão para a taxa contínua e intervalos de predição.
- Painel interativo (Streamlit) com mapa e simulador "e se" por alavanca.
- API de escoragem para as secretarias.

## Como executar

```bash
git clone https://github.com/GabrielFontineles/tech-challenge-fase3.git && cd tech-challenge-fase3
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1. colocar a Silver/Gold da Fase 2 em data/raw e data/gold (ver data/README.md)
# 2. fontes externas (o que não baixar automaticamente, baixar à mão e rodar de novo com SKIP=1)
make external
# 3. pipeline completa
make dataset train explain strategy
# 4. notebooks narrados
jupyter lab notebooks/
# 5. testes
make test
```

Sem os dados reais, `make synthetic` gera uma Silver fictícia apenas para validar que a pipeline roda de ponta a ponta.

## Estrutura do repositório

```
tech-challenge-fase3/
├── data/                  README.md com a obtenção de cada fonte; raw/ gold/ external/ processed/ (não versionados)
├── notebooks/             01 EDA · 02 features · 03 modelagem · 04 SHAP · 05 estratégia
├── src/
│   ├── config.py          decisões centrais (anos, corte, colunas, seeds)
│   ├── data/              download_external.py · build_dataset.py
│   ├── preprocessing/     features.py · pipeline.py
│   ├── modeling/          train.py
│   ├── evaluation/        metrics.py · shap_analysis.py · strategic.py
│   ├── visualization/     plots.py
│   └── utils/             synthetic.py
├── models/                modelo_final*.joblib + metadata*.json
├── reports/               relatório técnico, relatório executivo, roteiro do vídeo, CSVs de saída
├── images/                gráficos gerados pela pipeline
├── tests/                 smoke tests (pytest)
├── tools/gerar_notebooks.py
├── Makefile · requirements.txt · README.md · .gitignore
```

## Equipe

- Gabriel Fontineles
- Gabriel Kendy Sato
- Josilene Oliveira Afonso
- Katia Oliveira da Silva Costa
- Yasmim de Oliveira Coelho

Projeto desenvolvido como Tech Challenge — Fase 3 · Pós-Tech FIAP — IA Scientist

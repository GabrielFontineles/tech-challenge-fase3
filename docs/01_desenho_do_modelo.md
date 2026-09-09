# Tech Challenge Fase 3 — Novo Desenho do Modelo
### Predição e Inteligência Analítica para Alfabetização no Brasil
Documento de decisão analítica — v1 (09/09/2026)

---

## 1. Diagnóstico da versão atual (por que redesenhar)

A versão atual do repositório define o target como `taxa_alfabetizacao_2024 >= 60` e usa como features as `proporcao_aluno_nivel_0..8` de 2024, `score_niveis_altos` (soma dos níveis altos) e `media_portugues` de 2024. Como a taxa de alfabetização do Indicador Criança Alfabetizada (ICA) é, por construção, a soma das proporções de alunos nos níveis considerados "alfabetizado", o modelo recebe o target decomposto em partes. O ROC-AUC de 0,997 e o SHAP de 3,55 em `score_niveis_altos` (3,7x maior que a segunda feature) são a assinatura desse vazamento. Há ainda um vazamento menor: `participacao_alta` usa a mediana calculada na base inteira antes do split.

O que se aproveita da versão atual: estrutura de pastas, fluxo Git (branches + PRs), esqueleto de `Pipeline`/`ColumnTransformer`, gráficos de EDA descritiva (distribuição, ranking de UFs, evolução temporal, regional) e o texto de contexto do README.

## 2. Reformulação do problema

**Pergunta de negócio central.** Dado o que sabemos sobre um município *hoje* (resultado do ano anterior, território, condições socioeconômicas e estrutura educacional), qual a probabilidade de ele **não** estar alfabetizado (em risco) no próximo ciclo de avaliação — e quais fatores explicam esse risco?

**Unidade de análise.** Município (rede Total). Justificativa a registrar no README: o enunciado fala em "aluno", mas os microdados por aluno (256 MB) só estão acessíveis via BigQuery e não trazem variáveis socioeconômicas individuais; a decisão de política pública (priorização de recursos, FUNDEB, programas de reforço) acontece no nível municipal; e a camada Gold da Fase 2 foi construída nessa granularidade. É uma escolha, não uma limitação escondida.

**Desenho temporal (a mudança central).** Features observadas em **t = 2023** (e variáveis estruturais de anos anteriores); target observado em **t+1 = 2024**. Nenhuma variável de 2024 entra como feature. Isso elimina o leakage estrutural e transforma o modelo em um modelo de *previsão* de fato, o que responde diretamente à pergunta "como prever municípios que podem não atingir metas futuras".

**Variável alvo.** Binária, com a classe positiva = **risco** (facilita a leitura das métricas para gestão pública: recall da classe positiva = "quantos municípios em risco conseguimos capturar").

    em_risco_2024 = 1 se taxa_alfabetizacao_2024 < CORTE, senão 0

Corte recomendado: **60%**, mantido da versão atual, mas agora justificado: é aproximadamente o patamar nacional do ICA em 2024 e o ponto médio entre a linha de base de 2023 e a meta nacional de 80% em 2030. Alternativa a testar como análise de sensibilidade (não como target principal): corte pela meta municipal — `taxa_2024 < meta_alfabetizacao_2030 × 0,75` — que aproxima "fora da trajetória para a meta". Registrar no notebook a distribuição do target para os dois cortes.

**Target complementar (regressão).** `taxa_alfabetizacao_2024` contínua, com os mesmos preditores, usada só na seção estratégica para projetar a distância à meta 2030. Opcional, mas enriquece a resposta de negócio.

## 3. Hipóteses analíticas (a EDA deve validá-las ou refutá-las)

- **H1 — Inércia:** a taxa de 2023 é o preditor mais forte da taxa de 2024; a maioria dos municípios em risco em 2024 já estava em risco em 2023. Se confirmada, o valor do modelo está nos municípios que *mudam* de estado, e a EDA deve quantificar essa transição (matriz 2023→2024).
- **H2 — Desigualdade territorial:** Norte e Nordeste concentram o risco mesmo controlando por IDHM e PIB per capita (o efeito região não é só renda).
- **H3 — Condição socioeconômica:** IDHM-Educação, renda per capita e % da população no Cadastro Único explicam parte relevante da variância *não* explicada pela taxa anterior.
- **H4 — Estrutura da rede:** infraestrutura escolar (internet, biblioteca, laboratório), proporção de docentes com formação superior e distorção idade-série nos anos iniciais estão associadas ao risco.
- **H5 — Participação:** municípios com baixo `percentual_participacao` no SAEB 2023 têm maior risco em 2024 (participação como proxy de gestão/engajamento) e possivelmente maior variância na taxa (sinal ruidoso).
- **H6 — Porte:** municípios pequenos (< 20 mil hab.) apresentam maior volatilidade de taxa entre anos (poucos alunos avaliados), o que afeta a confiabilidade do target e deve ser discutido nas limitações.

## 4. Base analítica e enriquecimento externo

### 4.1 Camada Gold / Silver da Fase 2 (já disponível)
Da `municipio_silver` filtrada em `rede == 'Total'`:

| Ano | Uso | Colunas |
|---|---|---|
| 2023 | **Features** | `taxa_alfabetizacao`, `media_portugues`, `percentual_participacao`, `proporcao_aluno_nivel_0..8`, `nivel_alfabetizacao`, `distancia_meta_2030`, `meta_alfabetizacao_2030`, nº de alunos avaliados (se existir) |
| 2024 | **Target apenas** | `taxa_alfabetizacao` → `em_risco_2024` |

Das tabelas de UF/Gold: meta estadual, taxa estadual 2023 (contexto do município dentro da UF, ex.: `taxa_2023_municipio − taxa_2023_uf`).

### 4.2 Fontes externas (todas anteriores ou contemporâneas a 2023)

| Fonte | Variáveis | Onde obter | Chave de junção |
|---|---|---|---|
| IBGE — Estimativas populacionais 2023 | população, porte (faixas), densidade | Base dos Dados `br_ibge_populacao.municipio` | `id_municipio` (7 dígitos) |
| IBGE — PIB dos Municípios (último ano disponível, 2021) | PIB per capita, participação da agropecuária/indústria/serviços | `br_ibge_pib.municipio` | `id_municipio` |
| Atlas do Desenvolvimento Humano (Censo 2010) | IDHM, IDHM-Educação, IDHM-Renda, IDHM-Longevidade, % pobreza, Gini, taxa de analfabetismo adulto, renda per capita | `mundo_onu_adh.municipio` | `id_municipio` |
| Censo Escolar 2023 (INEP) | nº escolas com anos iniciais, matrículas 2º ano, % escolas com internet/biblioteca/laboratório, % escolas rurais, % docentes com nível superior, taxa de distorção idade-série (anos iniciais) | `br_inep_censo_escolar.escola` agregada por município; indicadores educacionais INEP | `id_municipio` |
| IDEB 2023 — anos iniciais | IDEB, nota SAEB, taxa de aprovação | `br_inep_ideb.municipio` | `id_municipio` |
| FUNDEB 2023 | receita FUNDEB por matrícula da rede municipal | FNDE / SIOPE (portal de dados abertos) | `id_municipio` |
| Cadastro Único 2023 | % da população em famílias inscritas, % em pobreza/extrema pobreza | CECAD / VIS DATA (MDS) | `id_municipio` |

Regra prática: a Fase 2 já usou BigQuery/Base dos Dados; reutilizar o mesmo caminho (`basedosdados` + `google-cloud-bigquery`) e gravar tudo em `data/external/` com um script `src/data/download_external.py`, para que a pipeline seja reproduzível do zero.

Prioridade se o tempo apertar: Atlas (IDHM) → IBGE população/PIB → Censo Escolar (infraestrutura e docentes) → IDEB → FUNDEB → Cadastro Único. As três primeiras já sustentam a narrativa "educacional, territorial e socioeconômica" que o enunciado pede.

## 5. Engenharia de atributos

Três blocos, com nomes com sufixo de ano para deixar a regra temporal auto-evidente:

**Bloco A — Histórico educacional (2023):** `taxa_alf_2023`, `media_pt_2023`, `particip_2023`, `prop_nivel_0..8_2023`, `gap_meta_2023 = meta_2030 − taxa_2023`, `taxa_2023_vs_uf = taxa_2023 − taxa_uf_2023`, `alunos_avaliados_2023` e `log_alunos_avaliados`. Aqui as proporções por nível **podem** entrar, porque são de 2023 e o target é de 2024 — a diferença em relação à versão anterior é exatamente o deslocamento temporal.

**Bloco B — Território:** `sigla_uf` (categórica → one-hot dentro do pipeline), `regiao` (categórica → one-hot), `populacao_2023`, `log_populacao`, `porte` (ordinal por faixas IBGE), `densidade`, `pct_matricula_rural`. Abandonar `regiao_encoded` ordinal "por desempenho" — é encoding por target, outra forma de leakage.

**Bloco C — Socioeconômico e estrutura da rede:** `idhm`, `idhm_educacao`, `idhm_renda`, `renda_per_capita`, `gini`, `pct_pobreza`, `analfabetismo_adulto`, `pib_per_capita`, `pct_escolas_internet`, `pct_escolas_biblioteca`, `pct_docentes_superior`, `distorcao_idade_serie_ai`, `ideb_ai_2023`, `fundeb_por_matricula`, `pct_pop_cadunico`.

**Interações derivadas (poucas, interpretáveis):** `gap_meta_2023 × idhm_educacao`, `particip_2023 × log_populacao`.

**Proibições explícitas (documentar no README como "Tratamento de data leakage"):**
1. Nenhuma coluna de 2024 além do target.
2. Nenhuma transformação que use `em_risco_2024` ou `taxa_2024` (target encoding, binarizações "por desempenho").
3. Estatísticas de imputação, escala, medianas e quantis calculadas **apenas no treino**, via `fit` do pipeline (nunca em `pandas` antes do split).
4. Seleção de features (se houver) dentro do CV, nunca na base completa.

## 6. Pipeline scikit-learn

```python
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("scaler", StandardScaler()),          # inócuo para árvores, necessário para LogReg
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
])
pre = ColumnTransformer([
    ("num", num_pipe, colunas_numericas),
    ("cat", cat_pipe, ["sigla_uf", "regiao", "porte"]),
])
modelo = Pipeline([("pre", pre), ("clf", <estimador>)])
```

`add_indicator=True` cria flags de ausência: a falta de meta municipal ou de dado do Censo Escolar pode ela mesma ser informativa (município pequeno, rede recém-criada).

**Divisão dos dados.** `train_test_split(test_size=0.2, stratify=y, random_state=42)`; o conjunto de teste é tocado uma única vez, no final. Dentro do treino, `StratifiedKFold(5, shuffle=True, random_state=42)` faz o papel de validação. Registrar no README que a estrutura é treino / validação (CV) / teste, atendendo ao enunciado.

**Modelos a comparar.**
1. `DummyClassifier(strategy="prior")` — baseline obrigatório para mostrar o ganho real.
2. `LogisticRegression(class_weight="balanced", C=…)` — interpretável, serve de referência.
3. `RandomForestClassifier(class_weight="balanced_subsample")`.
4. `HistGradientBoostingClassifier` (nativo do sklearn, rápido, lida com missing) — candidato a modelo final.
5. Opcional: `LGBMClassifier` ou `XGBClassifier` se quiserem, mas sem obrigação.

**Otimização.** `RandomizedSearchCV` (30–50 iterações, `scoring="roc_auc"` ou `"average_precision"`) sobre o pipeline completo para os modelos 2–4, com `refit=True`. Espaço de busca do HGB: `learning_rate`, `max_iter`, `max_depth`, `min_samples_leaf`, `l2_regularization`. Documentar a curva de validação (train vs CV) para mostrar controle de overfitting.

**Métricas (na ordem de importância para o problema).**
- ROC-AUC e **PR-AUC** (average precision) — a classe de risco é minoritária, PR-AUC é mais informativa.
- **Recall da classe risco** com threshold ajustado (não usar 0,5 por padrão): escolher o threshold no CV que garante, por exemplo, recall ≥ 0,85 com a melhor precisão possível, e explicar a lógica (custo de não identificar um município em risco > custo de um falso alarme).
- F1, matriz de confusão, Brier score e curva de calibração (`CalibratedClassifierCV` se as probabilidades forem usadas para ranking).
- Reportar média ± desvio no CV **e** o resultado único no teste.

Expectativa realista: AUC entre 0,80 e 0,90 (H1 garante bastante sinal pela inércia). Se voltar a aparecer > 0,97, investigar leakage antes de comemorar.

**Persistência.** Salvar o pipeline final com `joblib` em `models/modelo_final.joblib` (versionar o `.joblib` pequeno ou publicar via release; ajustar o `.gitignore`) e carregá-lo nos scripts de SHAP e análise estratégica em vez de retreinar.

## 7. Interpretabilidade

1. **Importância por permutação** no conjunto de teste (independente do algoritmo).
2. **SHAP** (`TreeExplainer` para HGB/RF; `LinearExplainer` para LogReg): summary plot global, bar plot de importância média, dependence plots para as 4 principais features (ex.: `taxa_alf_2023`, `idhm_educacao`, `particip_2023`, `pct_docentes_superior`) e **waterfall individual** para 3 municípios: um de alto risco, um limítrofe e um que mudou de estado entre 2023 e 2024.
3. Comparar a importância "bruta" com a importância **condicional à taxa de 2023**: treinar uma variante do modelo sem o Bloco A e mostrar quais fatores estruturais mais explicam o risco quando a inércia é removida. Essa é a análise que responde "quais fatores mais impactam a alfabetização" de forma útil para política pública.

## 8. Aplicação estratégica (perguntas de negócio)

| Pergunta do enunciado | Como responder |
|---|---|
| Quais fatores mais impactam? | SHAP global + modelo sem histórico (§7.3) |
| Quais municípios têm maior risco? | Ranking por probabilidade prevista; tabela com top 100 e mapa coroplético (geobr) |
| Quais regiões têm padrões semelhantes? | K-Means / Agglomerative sobre features padronizadas dos Blocos B e C (não sobre a taxa), k escolhido por silhueta; perfil de cada cluster; cruzar cluster × risco |
| Como prever municípios que não atingirão metas futuras? | Probabilidade de risco + projeção linear da taxa (2023→2024) até 2030 vs `meta_2030`; classificar em "atingirá / no limite / não atingirá"; modelo de regressão complementar |
| Quais variáveis têm maior influência nos modelos? | Tabela de importância por permutação vs SHAP vs coeficientes da LogReg — convergência dá robustez ao argumento |

Entregas visuais mínimas: mapa de risco, matriz de transição 2023→2024, SHAP summary, perfil dos clusters, "lista de priorização" (municípios com alto risco + baixo IDHM + alto gap de meta) exportada em CSV em `reports/`.

## 9. Estrutura do repositório (ajustes sobre a atual)

```
tech-challenge-fase3/
├── data/
│   ├── raw/          Silver da Fase 2 (não versionado)
│   ├── external/     IBGE, Atlas, Censo Escolar, IDEB, FUNDEB, CadÚnico (não versionado)
│   ├── processed/    dataset_modelagem.parquet (não versionado)
│   └── README.md     como obter cada arquivo + dicionário de dados
├── notebooks/
│   ├── 01_eda_exploratoria.ipynb
│   ├── 02_enriquecimento_e_features.ipynb
│   ├── 03_modelagem_e_validacao.ipynb
│   ├── 04_interpretabilidade_shap.ipynb
│   └── 05_analise_estrategica.ipynb
├── src/
│   ├── data/          download_external.py, build_dataset.py
│   ├── preprocessing/ features.py (Blocos A/B/C), pipeline.py (ColumnTransformer)
│   ├── modeling/      train.py (CV + RandomizedSearch), predict.py
│   ├── evaluation/    metrics.py, shap_analysis.py, strategic.py
│   └── visualization/ plots.py, maps.py
├── models/            modelo_final.joblib + metadata.json (features, threshold, métricas)
├── reports/           relatorio_tecnico.md, relatorio_executivo.md, priorizacao_municipios.csv
├── images/
├── requirements.txt   (testar instalação limpa; numpy 1.26 + shap 0.44 costuma exigir numba compatível)
├── README.md
└── .gitignore
```

Os notebooks contam a história (para a banca); o `src/` é o código reutilizável importado pelos notebooks (para o item "boas práticas de ambientes produtivos"). Evitar duplicar lógica entre os dois.

## 10. Plano de trabalho sugerido (5 pessoas, 4 sprints)

| Sprint | Entrega | Branches |
|---|---|---|
| 1 — Fundação | Decisão do desenho aprovada; `download_external.py`; `build_dataset.py` gerando o dataset t/t+1; dicionário de dados | `feat/external-data`, `feat/dataset-temporal` |
| 2 — EDA e features | Notebook 01 com as 6 hipóteses testadas; notebook 02; `features.py` | `feat/eda-hipoteses`, `feat/feature-blocks` |
| 3 — Modelagem | Notebook 03; `train.py` com RandomizedSearch e threshold; modelo salvo; métricas no README | `feat/modeling-cv`, `feat/threshold-calibration` |
| 4 — Interpretação e estratégia | Notebooks 04 e 05; mapa; clusters; relatório técnico e executivo; roteiro e gravação do vídeo | `feat/shap`, `feat/strategic`, `docs/final` |

Divisão natural: uma pessoa em dados externos/reprodutibilidade, uma em EDA, uma em modelagem, uma em interpretabilidade/estratégia, uma em documentação/vídeo — com revisão cruzada via PR.

## 11. Limitações a declarar desde já

Série histórica de apenas dois anos (um único par t/t+1, sem validação temporal com múltiplos anos); Atlas do Desenvolvimento Humano baseado no Censo 2010 (defasagem); dados do Censo 2022 ainda parciais no nível municipal; ruído amostral em municípios pequenos; target binário com corte arbitrário (mitigado pela análise de sensibilidade); rede Total não separa desempenho municipal vs estadual; ausência de variáveis de gestão (programas de alfabetização locais, formação continuada), que provavelmente explicam parte do resíduo.

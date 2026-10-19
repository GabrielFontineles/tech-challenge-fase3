# Predição e Inteligência Analítica para Alfabetização no Brasil

Tech Challenge — Fase 3 · Pós-Tech FIAP — IA Scientist

## Contexto do problema

O **Compromisso Nacional Criança Alfabetizada** estabelece que todas as crianças brasileiras estejam alfabetizadas ao final do 2º ano do ensino fundamental até 2030. O acompanhamento é feito pelo **Indicador Criança Alfabetizada**: o percentual de estudantes que atingem 743 pontos na escala do Saeb.

Na Fase 2 construímos a pipeline de dados (Bronze, Silver e Gold) desse indicador. Nesta fase, usamos esses dados para responder a uma pergunta de gestão: **quais municípios estão em risco de não alfabetizar suas crianças, e por quê?**

## Objetivo analítico

Dado o que se sabe sobre um município hoje (resultado educacional do ano anterior, território e condição socioeconômica), estimar a **probabilidade de ele estar em risco no ciclo seguinte** e identificar os fatores associados a esse risco.

- **Target:** `em_risco_2024 = 1` quando a taxa de alfabetização de 2024 é **menor que 60%**
- **Corte de 60%:** próximo do patamar nacional de 2024 e ponto intermediário até a meta de 80% em 2030
- **Classe positiva = risco:** o recall passa a significar "quantos municípios em risco conseguimos identificar"

### Unidade de análise: município, e não aluno

O enunciado menciona o aluno. Optamos pelo município por três razões:

1. Os microdados de alunos (cerca de 256 MB) estão disponíveis apenas via BigQuery e não trazem variáveis socioeconômicas individuais
2. As decisões de política pública (priorização de recursos, programas de reforço) são tomadas no nível municipal
3. A camada Gold da Fase 2 foi construída nessa granularidade

## Base de dados

| Fonte | Conteúdo | Ano | Como é obtida |
|---|---|---|---|
| Silver e Gold da Fase 2 | Indicador Criança Alfabetizada, níveis de proficiência, participação, metas | 2023 e 2024 | Versionada em `data/raw` e `data/gold` |
| IBGE, Censo (SIDRA tabela 4714) | População, área, densidade, porte | 2022 | API pública, sem credencial |
| IBGE, PIB dos Municípios (SIDRA tabela 5938) | PIB a preços correntes | 2021 | API pública, sem credencial |
| Atlas do Desenvolvimento Humano | IDHM e dimensões, renda, Gini, pobreza, analfabetismo adulto, frequência na pré-escola, % rural | 2010 | Arquivo baixado em atlasbrasil.org.br |

**Dataset de modelagem:** 4.919 municípios com dados em 2023 e em 2024. Dos 5.516 municípios avaliados em 2024, 597 não tinham resultado em 2023 e ficaram de fora. A cobertura das fontes externas é de 100% (IBGE) e 99,9% (Atlas: os 4 ausentes são municípios criados depois de 2010). Detalhes em [`data/README.md`](data/README.md).

## Equipe

- Gabriel Fontineles
- Gabriel Kendy Sato
- Josilene Oliveira Afonso
- Katia Oliveira da Silva Costa
- Yasmim de Oliveira Coelho

## Estrutura do repositório
tech-challenge-fase3/
├── data/
│ ├── raw/ <- Silver da Fase 2 (versionada)
│ ├── gold/ <- Gold da Fase 2 (versionada)
│ ├── external/ <- IBGE e Atlas (gerados por script, nao versionados)
│ ├── processed/ <- Datasets de modelagem (gerados por script)
│ └── README.md <- Origem e obtencao de cada fonte
├── src/
│ ├── data/ <- build_dataset.py, download_external.py
│ ├── preprocessing/ <- EDA descritiva
│ ├── modeling/ <- train.py (tres variantes)
│ ├── evaluation/ <- Interpretabilidade e analise estrategica
│ └── legacy/ <- Versao 1 com data leakage (historico, nao usar)
├── models/ <- Modelos treinados (.joblib) e metadata
├── notebooks/ <- Notebooks narrados 01-04
├── reports/ <- Relatorio executivo e tabelas de resultados
├── images/ <- Graficos (images/v1 guarda os da versao 1)
├── tests/ <- Testes pytest
├── Makefile
└── requirements.txt


## Etapas de modelagem

### 1. Análise exploratória

Distribuições, correlações, desempenho por região e matriz de transição 2023→2024 (notebook `01_eda_exploratoria`). A EDA orientou as hipóteses e a escolha do corte de 60%.

### 2. Desenho temporal e tratamento de data leakage

As **features são observadas em 2023**, e o **target em 2024**. Nenhuma variável de 2024 entra como feature, e isso é verificado em dois pontos:

- `train.py` interrompe o treino se alguma feature terminar em `_2024`
- `tests/test_model.py` falha se o ROC-AUC passar de 0,97, valor que nesse problema indica vazamento

> **Por que isso importa:** a primeira versão do projeto usava as proporções de alunos por nível de proficiência **de 2024**, que compõem a própria taxa de 2024. O resultado era um ROC-AUC de 0,997 que não refletia capacidade preditiva. Essa versão foi preservada em `src/legacy/`, com a explicação do erro.

### 3. Features em três blocos

| Bloco | Variáveis | Fonte |
|---|---|---|
| A — Histórico educacional 2023 | Taxa, média de português, participação, nível, meta 2030, distância da meta, posição relativa à UF, proporção de alunos por nível (0 a 8) | Fase 2 |
| B — Território | UF, região, porte (categóricas); log da população, densidade | IBGE 2022 |
| C — Socioeconômico | IDHM e dimensões, renda per capita, Gini, % pobres e extremamente pobres, analfabetismo 15+, expectativa de anos de estudo, frequência escolar 4-5 anos, % rural, log do PIB per capita | Atlas 2010, IBGE 2021 |

### 4. Pipeline scikit-learn

Todo o pré-processamento fica dentro do pipeline, ajustado apenas nos dados de treino de cada fold:

- **Numéricas:** imputação pela mediana, com indicador de ausência (`add_indicator=True`), e padronização
- **Categóricas:** imputação pelo valor mais frequente e one-hot encoding

### 5. Validação

- **Treino/teste:** 80/20, estratificado. O teste é usado uma única vez, na avaliação final.
- **Validação:** `StratifiedKFold` com 5 folds dentro do treino
- **Otimização:** `RandomizedSearchCV` em HistGradientBoosting (30 combinações) e Regressão Logística (20), com PR-AUC como métrica
- **Calibração:** isotônica (`CalibratedClassifierCV`)
- **Threshold:** escolhido nas predições *out-of-fold* calibradas, com recall mínimo de 0,85 e a maior precisão possível
- **Replicabilidade:** `random_state=42` em todas as etapas; o teste `test_modelo_salvo_reproduz_metricas` confirma que o modelo versionado gera exatamente as métricas reportadas

## Escolha do algoritmo

Critério definido antes do treino: **maior PR-AUC médio em validação cruzada**. O PR-AUC foca na capacidade de encontrar os municípios em risco, que é o uso pretendido do modelo.

| Modelo (CV, hiperparâmetros padrão) | ROC-AUC | PR-AUC | Recall |
|---|---|---|---|
| Dummy (proporção da classe) | 0,500 | 0,426 | 0,000 |
| Regressão Logística | 0,912 | 0,886 | 0,844 |
| Random Forest | 0,915 | 0,892 | 0,790 |
| HistGradientBoosting | 0,908 | 0,888 | 0,771 |

| Após otimização | PR-AUC (CV) | PR-AUC (treino) | Gap |
|---|---|---|---|
| **HistGradientBoosting** | **0,897** | 0,930 | 0,033 |
| Regressão Logística | 0,887 | 0,898 | 0,011 |

**Modelo selecionado: HistGradientBoosting.** A diferença para a Regressão Logística (0,01 de PR-AUC) fica dentro do desvio padrão entre folds (cerca de 0,02), ou seja, os dois modelos são equivalentes na prática. O HistGradientBoosting vence pelo critério definido antecipadamente; a Regressão Logística seria uma alternativa igualmente válida, com menos overfitting e interpretação mais simples.

## Métricas de avaliação (conjunto de teste)

| Métrica | Valor |
|---|---|
| ROC-AUC | 0,917 |
| PR-AUC | 0,905 |
| Recall (em risco) | 0,883 |
| Precisão (em risco) | 0,770 |
| F1 | 0,823 |
| Acurácia | 0,84 |
| Threshold | 0,431 |
| Brier score (calibrado / sem calibração) | 0,1119 / 0,1135 |

Dos 420 municípios do teste que estavam em risco em 2024, o modelo identificou 371. A calibração teve efeito pequeno neste modelo, mas garante que as probabilidades possam ser lidas como frequências: na análise de todos os municípios, a probabilidade média prevista e a taxa de risco observada coincidem em todas as faixas (por exemplo, 0,907 contra 0,903 na faixa crítica).

## Interpretação dos resultados

Treinamos três variantes do modelo para separar o que vem de cada tipo de informação:

| Variante | O que o modelo usa | ROC-AUC | PR-AUC |
|---|---|---|---|
| Só socioeconômico | Bloco C | 0,787 | 0,705 |
| Sem histórico | Blocos B e C | 0,893 | 0,867 |
| Completo | Blocos A, B e C | 0,917 | 0,905 |

Contribuição de cada grupo de features, medida por SHAP e por permutação em grupo (queda de PR-AUC ao embaralhar juntas as colunas do grupo):

| Grupo | Modelo completo: SHAP | Modelo completo: permutação | Sem histórico: SHAP | Sem histórico: permutação |
|---|---|---|---|---|
| Histórico educacional 2023 | 52,1% | 0,317 | — | — |
| Estado e região | 32,1% | 0,103 | 69,7% | 0,387 |
| Socioeconômico | 10,2% | 0,009 | 21,7% | 0,041 |
| Porte e densidade | 5,6% | 0,016 | 8,6% | 0,023 |

Leitura:

- **A condição socioeconômica sozinha já prevê o risco** (ROC-AUC de 0,787 contra 0,500 do modelo sem informação). As variáveis mais relevantes são IDHM-Educação, analfabetismo adulto e expectativa de anos de estudo. O IDHM-Longevidade aparece no topo dessa variante, mas funciona como indicador de desenvolvimento regional, e não como mecanismo ligado à alfabetização.
- **O estado acrescenta 0,11 de ROC-AUC além da condição socioeconômica.** Renda e escolaridade de 2010 não explicam as diferenças entre estados.
- **O histórico de 2023 completa o modelo** (+0,024 de ROC-AUC). A média de português de 2023 é a feature individual mais importante, acima da própria taxa de 2023, possivelmente por ser uma medida contínua e menos sensível ao corte de 743 pontos.

## Insights encontrados

1. **O mesmo perfil de vulnerabilidade leva a resultados opostos conforme o estado.** No grupo de municípios de maior vulnerabilidade estrutural, **100% dos 166 municípios do Ceará** estavam fora de risco em 2024, contra 10,5% no Rio Grande do Norte. Pernambuco (64%), Piauí (60%) e Maranhão (54%) também se destacam. O resultado é consistente com a hipótese de que a política estadual importa, com o PAIC (Programa Alfabetização na Idade Certa, do Ceará) como exemplo conhecido. As limitações abaixo impedem afirmar causalidade.
2. **Renda não garante alfabetização.** No agrupamento por perfil estrutural (k=3, silhueta 0,27), os municípios urbanos de maior porte e renda têm risco de 41%, acima dos municípios pequenos do interior de renda média (26%). O grupo de vulnerabilidade estrutural (79% no Nordeste, 27% de analfabetismo adulto) tem risco de 59%.
3. **Cobertura não é o mesmo que qualidade.** A frequência na pré-escola é maior no grupo mais vulnerável (86%) do que no de renda média (75%).
4. **O risco é concentrado.** A Bahia tem probabilidade média de risco de 97%, e Sergipe, Pará e Rio Grande do Norte ficam acima de 84%. O Rio Grande do Sul é o único estado do Sul entre os 10 de maior risco, o que coincide com a maior queda de 2023 para 2024 (-18,8 pontos).
5. **O desafio de 2030 é grande.** Com a variação média nacional observada (2,55 pontos por ano), 1.319 municípios atingem a meta municipal no ritmo atual e 1.050 já a atingiram, mas 1.261 precisariam de mais que o dobro desse ritmo.

## Hipóteses

| Hipótese | Resultado |
|---|---|
| H1 — Inércia: o resultado de 2023 prediz o de 2024 | **Confirmada, com ressalva.** O histórico é o grupo mais importante do modelo completo, mas acrescenta pouco (+0,024 de ROC-AUC) sobre território e socioeconomia, com os quais é correlacionado. |
| H2 — Desigualdade territorial | **Confirmada.** O estado é o grupo mais importante do modelo sem histórico e acrescenta 0,11 de ROC-AUC além da condição socioeconômica. |
| H3 — Condição socioeconômica | **Confirmada.** Sozinha, alcança ROC-AUC de 0,787. Boa parte desse sinal é compartilhada com o estado. |
| H4 — Participação como indicador de gestão | **Confirmada.** A participação em 2023 está entre as 5 features mais importantes do modelo completo. |
| H5 — Porte do município | **Parcialmente confirmada.** O porte tem efeito menor (5,6% do SHAP), e municípios maiores têm mais risco, e não menos. |

## Aplicação prática para políticas públicas

- **Priorização antecipada:** com recall de 0,88, o modelo identifica a maioria dos municípios em risco antes do resultado do ciclo seguinte. Como o território e a condição socioeconômica já concentram grande parte do sinal, a priorização pode começar mesmo com dados educacionais defasados.
- **Faixas de risco confiáveis:** por estarem calibradas, as probabilidades podem ser comunicadas diretamente a gestores ("9 em cada 10 municípios nesta faixa estavam em risco").
- **Lista de priorização:** `reports/priorizacao_municipios_v21.csv` reúne os 1.166 municípios com risco previsto acima do threshold e que precisariam de mais que o dobro do ritmo nacional para a meta de 2030. Para uso por uma secretaria estadual, a lista deve ser filtrada por UF.
- **Aprendizado entre estados:** os municípios vulneráveis fora de risco, sobretudo no Ceará, Pernambuco, Piauí e Maranhão, são candidatos naturais a estudos de boas práticas.

## Limitações

- **Um único par de anos (2023→2024):** não há validação temporal. O modelo pode ter aprendido particularidades de 2024.
- **O efeito estado mistura fatores que não conseguimos separar:** política estadual, eventos pontuais de 2024 (por exemplo, as enchentes de maio de 2024 no Rio Grande do Sul) e diferenças entre as avaliações estaduais, já que o indicador de 2024 é baseado em avaliações de cada estado equalizadas com a escala do Saeb.
- **Dados socioeconômicos defasados:** o Atlas usa o Censo 2010. O PIB per capita combina o PIB de 2021 com a população do Censo 2022.
- **Associação, não causalidade:** nenhum resultado permite afirmar que uma variável causa o risco.
- **Corte binário arbitrário (60%)** e rede Total, sem separar redes municipal e estadual.
- **Viés de seleção:** 597 municípios sem resultado em 2023 ficaram de fora.
- **Ruído em municípios pequenos,** com poucos alunos avaliados.
- **Detalhes de método:** a variante só socioeconômica apresenta overfitting (gap de 0,12 entre treino e CV); a silhueta de 0,27 indica agrupamento útil, mas sem fronteiras nítidas; a calibração isotônica gera probabilidades de 1,0 no degrau mais alto, que devem ser lidas como "acima de 99%"; a projeção para 2030 é um cenário de tendência linear, não uma previsão.

## Possíveis evoluções

- Incorporar novos ciclos do indicador para validação temporal e para prever 2025 a partir dos dados de 2024
- Substituir o Atlas 2010 por indicadores socioeconômicos do Censo 2022, quando divulgados por município
- Incluir Censo Escolar (infraestrutura, formação docente), IDEB, FUNDEB e Cadastro Único
- Separar as redes municipal e estadual
- Modelos multinível (município dentro do estado) para estimar melhor o efeito estadual
- Estudar os municípios vulneráveis fora de risco e avaliar programas estaduais com métodos causais (por exemplo, diferenças em diferenças)
- Painel interativo com a lista de priorização por UF

## Como executar

Pré-requisitos: Python 3.14 e o arquivo do Atlas (`censo_total_1991_2010.xlsx`) em `data/external/raw/` — ver [`data/README.md`](data/README.md).

```bash
git clone https://github.com/GabrielFontineles/tech-challenge-fase3.git
cd tech-challenge-fase3
pip install -r requirements.txt

python src/data/build_dataset.py                   # dataset temporal 2023 -> 2024
python src/data/download_external.py               # IBGE (API) + Atlas
python src/modeling/train.py                       # modelo completo
python src/modeling/train.py --sem-historico
python src/modeling/train.py --so-socioeconomico
python src/evaluation/shap_analysis_v2.py          # repetir com as opcoes acima
python src/evaluation/strategic_v2.py
python -m pytest tests/ -v
```

Em Linux ou macOS, `make all` executa a sequência completa.

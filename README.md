# Predição e Inteligência Analítica para Alfabetização no Brasil

## Contexto do Problema

A alfabetização infantil é um dos principais indicadores do desenvolvimento educacional e social do Brasil. O **Compromisso Nacional Criança Alfabetizada** estabelece como meta que todas as crianças estejam alfabetizadas até o final do 2º ano do ensino fundamental até 2030.

Este projeto é a continuação do **Tech Challenge Fase 2**, onde construímos uma pipeline híbrida de dados para análise do Indicador Criança Alfabetizada. Nesta fase, utilizamos os dados da camada Gold para desenvolver modelos preditivos e análises estratégicas que transformam dados públicos em inteligência aplicada à tomada de decisão.

---

## Objetivo Analítico

Dado o que sabemos sobre um município **hoje** (resultado do ano anterior, território e condições socioeconômicas), qual a probabilidade de ele estar **em risco de não ser considerado alfabetizado** no próximo ciclo de avaliação — e quais fatores explicam esse risco?

---

## Unidade de Análise — Por que município e não aluno?

O enunciado menciona "aluno", mas optamos pelo município como unidade de análise por três razões:

1. **Acesso**: microdados de alunos (256MB) só estão disponíveis via BigQuery sem variáveis socioeconômicas individuais
2. **Decisão de política pública**: priorização de recursos, FUNDEB e programas de reforço acontecem no nível municipal
3. **Camada Gold da Fase 2**: construída nessa granularidade, garantindo continuidade entre as fases

---

## Base de Dados

Dados provenientes da camada Gold do Tech Challenge Fase 2:

| Dataset | Registros | Descrição |
|---|---|---|
| `municipio_silver` | 23.995 | Indicadores por município, ano e rede |
| `ranking_estados` | 49 | Ranking nacional por UF |
| `evolucao_temporal` | 24 | Variação 2023→2024 por estado |
| `analise_municipal` | 26 | Agregação municipal por UF |

**Dataset de modelagem:** 4.919 municípios pareados 2023→2024.

---

## Equipe

- Gabriel Fontineles
- Gabriel Kendy Sato
- Josilene Oliveira Afonso
- Katia Oliveira da Silva Costa
- Yasmim de Oliveira Coelho

Projeto desenvolvido como Tech Challenge — Fase 3
Pós-Tech FIAP — IA Scientist

---

## Estrutura do Repositório

tech-challenge-fase3/
├── data/
│ ├── raw/ <- Dados Silver da Fase 2
│ ├── processed/ <- Datasets de modelagem (v2)
│ ├── external/ <- Dados externos (Atlas, IBGE, PIB)
│ └── gold/ <- Datasets Gold da Fase 2
├── src/
│ ├── data/ <- build_dataset.py, download_external.py
│ ├── preprocessing/ <- EDA e Feature Engineering
│ ├── modeling/ <- Pipeline ML e treinamento
│ └── evaluation/ <- SHAP e Análise Estratégica
├── models/ <- modelo_final.joblib + metadata.json
├── images/ <- Gráficos gerados
├── reports/ <- Relatório executivo
├── requirements.txt
└── README.md


---

## Design Temporal — Eliminação do Data Leakage

### Problema da v1
A versão anterior tinha **ROC-AUC de 0.997 por data leakage**: as features `proporcao_aluno_nivel_0..8` e `score_niveis_altos` de 2024 eram componentes diretos do target de 2024. O SHAP confirmou: `score_niveis_altos` com SHAP 3.55 (3.7x maior que o segundo preditor) era a assinatura do vazamento.

### Solução — Design t → t+1

Features: dados de 2023 (histórico educacional)
Target: em_risco_2024 (taxa_2024 < 60%)

NENHUMA coluna de 2024 entra como feature.


Isso transforma o modelo em uma **previsão de verdade** — respondendo diretamente à pergunta do enunciado sobre "prever municípios que podem não atingir metas futuras".

---

## Features — Três Blocos

### Bloco A — Histórico Educacional 2023
`taxa_alf_2023`, `media_pt_2023`, `particip_2023`, `nivel_alf_2023`, `meta_2030`, `gap_meta_2030_2023`, `dist_meta_2030_2023`, `taxa_vs_uf_2023`, `prop_nivel_0..8_2023`

### Bloco B — Território
`sigla_uf` (one-hot), `regiao` (one-hot), `populacao_2023`, `log_populacao`, `porte`

### Bloco C — Socioeconômico
`idhm`, `idhm_educacao`, `idhm_renda`, `renda_per_capita`, `gini`, `pct_pobres`, `pib_per_capita`

**Total: 30 features | 4.919 municípios**

---

## Variável Alvo

```python
em_risco_2024 = 1 se taxa_alfabetizacao_2024 < 60%, senão 0
```

- Classe positiva = **risco** (42.7% dos municípios)
- Corte de 60%: patamar nacional aproximado de 2024 e ponto médio até a meta de 80% em 2030

---

## Pipeline de Modelagem

```python
Pipeline([
    ('pre', ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
            ('scaler', StandardScaler())
        ]), colunas_numericas),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), ['sigla_uf', 'regiao', 'porte'])
    ])),
    ('clf', HistGradientBoostingClassifier())
])
```

**Validação:** `StratifiedKFold(5)` | Split: 80% treino / 20% teste (tocado uma única vez)

---

## Modelos Avaliados

| Modelo | ROC-AUC CV | PR-AUC CV | Recall CV |
|---|---|---|---|
| DummyClassifier (baseline) | 0.500 | 0.427 | 0.000 |
| Logistic Regression | 0.907 | 0.881 | 0.836 |
| Random Forest | 0.904 | 0.872 | 0.785 |
| **HistGradientBoosting** | **0.909** | **0.874** | **0.768** |

**Otimização:** `RandomizedSearchCV` com 30 iterações sobre o pipeline completo.

---

## Métricas Finais — Conjunto de Teste

| Métrica | Valor |
|---|---|
| ROC-AUC | **0.9103** |
| PR-AUC | **0.8945** |
| Recall (em risco) | **0.8595** |
| Precision | 0.7681 |
| F1-Score | 0.8112 |
| Threshold ajustado | 0.458 |

> Recall de 0.86 significa que o modelo captura **86% dos municípios em risco** — essencial para política pública onde o custo de não identificar um município em risco é maior que um falso alarme.

---

## Interpretabilidade — SHAP Values

### Top 10 Features mais importantes

| Feature | SHAP | Interpretação |
|---|---|---|
| media_pt_2023 | 1.04 | Desempenho em português é preditor dominante |
| sigla_uf_RS | 0.28 | RS teve queda atípica de 19 pts em 2024 |
| sigla_uf_BA | 0.22 | Bahia como fator de risco regional |
| regiao_Sudeste | 0.22 | Efeito regional real |
| particip_2023 | 0.20 | Participação como proxy de gestão |
| taxa_alf_2023 | 0.18 | Inércia histórica confirmada |

> **Diferença da v1**: preditor dominante antes era `score_niveis_altos` (SHAP 3.55) — que era o próprio target decomposto. Agora são preditores reais e defensáveis.

---

## Hipóteses Validadas pela EDA

| Hipótese | Resultado |
|---|---|
| H1 — Inércia 2023→2024 | ✅ Confirmada — taxa_alf_2023 entre top preditores |
| H2 — Desigualdade Norte/Nordeste | ✅ Confirmada — gap de 9.2 pts vs Sul/Sudeste |
| H3 — Condição socioeconômica | ✅ Confirmada — idhm_educacao relevante |
| H4 — Participação como proxy | ✅ Confirmada — particip_2023 top 5 SHAP |

---

## Limitações

- Série histórica de apenas dois anos — um único par t/t+1
- Atlas do Desenvolvimento Humano baseado no Censo 2010 (defasagem)
- Dados externos complementados com simulação — substituir por dados reais
- Ruído amostral em municípios pequenos com poucos alunos avaliados
- Target binário com corte arbitrário (análise de sensibilidade recomendada)

---

## Aplicação Prática

- **Identificação precoce**: modelo com recall 86% captura municípios em risco antes do ciclo de avaliação
- **Priorização**: municípios com alta probabilidade de risco + baixo IDHM = intervenção urgente
- **Monitoramento**: queda de participação em 2023 prediz risco em 2024

---

## Como Executar

```bash
# 1. Clone o repositório
git clone https://github.com/GabrielFontineles/tech-challenge-fase3.git
cd tech-challenge-fase3

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Construa o dataset temporal
python src/data/build_dataset.py

# 4. Baixe dados externos (requer GCP configurado)
python src/data/download_external.py

# 5. Treine o modelo
python src/modeling/train.py

# 6. Interpretabilidade
python src/evaluation/shap_analysis_v2.py
```

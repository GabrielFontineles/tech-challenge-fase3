# Predição e Inteligência Analítica para Alfabetização no Brasil

## Contexto do Problema

A alfabetização infantil é um dos principais indicadores do desenvolvimento educacional e social do Brasil. O **Compromisso Nacional Criança Alfabetizada** estabelece como meta que todas as crianças estejam alfabetizadas até o final do 2º ano do ensino fundamental até 2030.

Este projeto é a continuação do **Tech Challenge Fase 2**, onde construímos uma pipeline híbrida de dados para análise do Indicador Criança Alfabetizada. Nesta fase, utilizamos os dados da camada Gold para desenvolver modelos preditivos e análises estratégicas que transformam dados públicos em inteligência aplicada à tomada de decisão.

---

## Objetivo Analítico

Desenvolver um modelo supervisionado capaz de prever se um município será considerado **alfabetizado ou em risco**, utilizando variáveis educacionais e territoriais, e gerar inteligência estratégica para apoiar políticas públicas educacionais.

---

## Base de Dados

Dados provenientes da camada Gold do Tech Challenge Fase 2:

| Dataset | Registros | Descrição |
|---|---|---|
| `municipio_silver` | 23.995 | Indicadores por município, ano e rede |
| `ranking_estados` | 49 | Ranking nacional por UF |
| `evolucao_temporal` | 24 | Variação 2023→2024 por estado |
| `analise_municipal` | 26 | Agregação municipal por UF |
| `visao_brasil` | 2 | Agregação nacional por ano |

**Dataset de modelagem:** 5.516 municípios (2024, rede Total) com 22 features.

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

## Etapas de Modelagem

### 1. Análise Exploratória (EDA)
- Distribuição da taxa de alfabetização por município
- Análise de correlações entre variáveis
- Desempenho por região geográfica
- Mapeamento de municípios em situação de risco
- 5 hipóteses analíticas formuladas e validadas

### 2. Feature Engineering
- **Variável alvo**: `alfabetizado` (1 = taxa ≥ 60%, 0 = em risco)
- **Features geográficas**: cod_uf, regiao, regiao_encoded
- **Features de desempenho**: score_niveis_altos, score_niveis_baixos
- **Flags**: participacao_alta
- **Total**: 22 features selecionadas

### 3. Tratamento de Data Leakage
Variáveis removidas por causar vazamento de informação:
- `taxa_alfabetizacao` — usada para criar o target
- `distancia_meta_2030` — derivada da taxa
- `atingiu_meta_2030` — derivada da taxa
- `nivel_alfabetizacao` — derivado da taxa

### 4. Pipeline Scikit-learn
```python
Pipeline([
    ('preprocessador', ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), colunas_numericas)
    ])),
    ('modelo', GradientBoostingClassifier(random_state=42))
])
```

---

## Escolha do Algoritmo

### Modelos avaliados (Cross-Validation 5-Fold)

| Modelo | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.9728 | 0.9764 | 0.9970 |
| Decision Tree | 0.9606 | 0.9656 | 0.9610 |
| Random Forest | 0.9696 | 0.9736 | 0.9971 |
| **Gradient Boosting** | **0.9694** | **0.9734** | **0.9971** |

### Modelo selecionado: Gradient Boosting
Selecionado pelo maior ROC-AUC e melhor estabilidade no cross-validation.

---

## Métricas de Avaliação — Conjunto de Teste

| Métrica | Valor |
|---|---|
| Accuracy | 96.4% |
| Precision | 96.4% |
| Recall | 97.3% |
| F1-Score | 96.9% |
| ROC-AUC | 99.6% |

Split: 80% treino / 20% teste — estratificado por classe.

---

## Interpretação dos Resultados — SHAP Values

### Top 5 features mais importantes

| Feature | SHAP Value | Interpretação |
|---|---|---|
| score_niveis_altos | 3.55 | Proporção de alunos em níveis avançados |
| media_portugues | 0.97 | Desempenho em língua portuguesa |
| proporcao_aluno_nivel_3 | 0.56 | Nível intermediário crítico |
| proporcao_aluno_nivel_5 | 0.26 | Nível avançado inicial |
| proporcao_aluno_nivel_2 | 0.25 | Nível básico superior |

### Insights encontrados

- **Ceará** é a única UF que já atingiu a meta de 2030 (85.3%)
- **728 municípios (13.2%)** estão em situação crítica — taxa abaixo de 40%
- **Nordeste concentra 27.8%** dos municípios críticos
- **Rio Grande do Sul** caiu 18.8 pontos entre 2023 e 2024
- **Gap regional**: Norte/Nordeste 9.2 pontos abaixo de Sul/Sudeste
- **4 clusters** identificados: Crítico, Vulnerável, Em desenvolvimento, Avançado

---

## Limitações do Projeto

- Dados disponíveis apenas para 2023 e 2024 — série histórica curta
- Tabela de microdados de alunos (256MB) acessível apenas via BigQuery
- Ausência de variáveis socioeconômicas externas (IBGE, FUNDEB)
- Modelo treinado apenas com rede Total — não diferencia municipal/estadual
- 3.5% de missing values nas colunas de meta municipal

---

## Aplicação Prática para Políticas Públicas

- **Identificação precoce de risco**: modelo com 96.4% de accuracy permite antecipar municípios em risco antes do ciclo de avaliação
- **Priorização de recursos**: 728 municípios críticos identificados para intervenção urgente
- **Foco regional**: Norte e Nordeste como regiões prioritárias
- **Alavanca principal**: programas de reforço em língua portuguesa nos níveis 0-3
- **Monitoramento contínuo**: municípios com queda de participação são candidatos a deterioração futura

---

## Possíveis Evoluções Futuras

- Incorporar dados do IBGE, FUNDEB e Censo Escolar como features externas
- Expandir para microdados de alunos via BigQuery
- Implementar modelo temporal com dados de múltiplos anos
- Desenvolver dashboard interativo com Streamlit ou Power BI
- Criar API de predição para consumo por sistemas municipais

---

## Como Executar

```bash
# 1. Clone o repositório
git clone https://github.com/GabrielFontineles/tech-challenge-fase3.git
cd tech-challenge-fase3

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Execute a análise exploratória
python src/preprocessing/eda_inicial.py
python src/preprocessing/eda_correlacoes.py

# 4. Execute o feature engineering
python src/preprocessing/feature_engineering.py

# 5. Treine os modelos
python src/modeling/ml_pipeline.py

# 6. Análise de interpretabilidade
python src/evaluation/shap_analysis.py

# 7. Análise estratégica
python src/evaluation/strategic_analysis.py
```

---

## Estrutura do Repositório

```
tech-challenge-fase3/
├── data/
│   ├── raw/
│   ├── processed/
│   └── gold/
├── src/
│   ├── preprocessing/
│   ├── modeling/
│   └── evaluation/
├── images/
├── reports/
├── requirements.txt
└── README.md
```


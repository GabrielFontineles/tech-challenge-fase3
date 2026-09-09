# Registro de decisões analíticas (ADR)

Formato: contexto → decisão → consequências. Adicionar uma entrada por decisão relevante, com data e autor.

## ADR-001 — Reformular o desenho do modelo para t → t+1 (09/09/2026)
**Contexto.** A versão inicial usava, como features, a distribuição de alunos por nível de proficiência de 2024 para prever um target derivado da taxa de 2024 — que é a soma dessas proporções. ROC-AUC 0,997 e SHAP dominado por `score_niveis_altos` evidenciavam o vazamento.
**Decisão.** Features de 2023 (+ fontes externas ≤ 2023); target de 2024. Regras de leakage auditáveis pelo nome da coluna (`_t`, `_t1`) e verificadas em código (`features.auditar`).
**Consequências.** Métricas caem para um patamar realista (AUC esperado 0,80–0,90); o modelo passa a responder à pergunta de previsão de metas futuras; a EDA precisa investigar a inércia (H1).

## ADR-002 — Unidade de análise: município, rede Total
**Contexto.** O enunciado menciona "aluno"; microdados por aluno estão só no BigQuery, sem variáveis socioeconômicas individuais.
**Decisão.** Município × rede Total. Justificativa registrada no README.
**Consequências.** Perde-se heterogeneidade intra-município; ganha-se aderência ao nível em que a política é decidida e às fontes externas disponíveis.

## ADR-003 — Target binário com classe positiva = risco, corte 60%
**Contexto.** Métricas orientadas ao gestor exigem que "recall" signifique "municípios em risco capturados".
**Decisão.** `em_risco = taxa_2024 < 60`. Corte ≈ patamar nacional 2024 e ponto médio até a meta de 80%. Sensibilidade com 50/55/65/70 e com 75% da meta municipal (notebook 02).
**Consequências.** Prevalência da classe risco por volta de 40–45% (verificar com dados reais) — desbalanceamento moderado, tratado com `class_weight` e PR-AUC.

## ADR-004 — Métrica principal PR-AUC; threshold por recall mínimo 0,85
**Contexto.** Falso negativo (município em risco não identificado) custa mais que falso positivo.
**Decisão.** Selecionar modelo por PR-AUC (CV); threshold escolhido em predições out-of-fold para recall ≥ 0,85; reportar também 0,5.
**Consequências.** Precisão menor por escolha deliberada; a decisão está documentada e é ajustável (`--recall-minimo`).

## ADR-005 — Fontes externas: Atlas, IBGE, INEP primeiro; FUNDEB/CadÚnico em segunda leva
**Contexto.** Tempo limitado; portais do FNDE e MDS não têm download programático estável.
**Decisão.** Priorizar fontes com API ou arquivo único. FUNDEB e CadÚnico entram via CSV manual se houver tempo.
**Consequências.** Bloco C pode ficar incompleto; ausência tratada por imputação + indicador.

## ADR-006 — Modelo final versionado no Git
**Contexto.** Notebooks 04 e 05 dependem do modelo; retreinar em cada script gerava divergência na versão anterior.
**Decisão.** `models/modelo_final*.joblib` + `metadata.json` versionados (são pequenos); scripts de SHAP/estratégia carregam, nunca retreinam.
**Consequências.** Reprodutibilidade; a versão do scikit-learn deve ser a do `requirements.txt` para carregar o joblib.

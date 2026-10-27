# Decisões técnicas

Registro das principais decisões do projeto: o que foi decidido, por quê, quais alternativas foram descartadas e qual a consequência. Os números vêm de `models/metadata*.json` e das tabelas em `reports/`.

## D1 — Município como unidade de análise

- **Decisão:** modelar por município (rede Total), e não por aluno.
- **Motivo:** os microdados de alunos estão disponíveis apenas via BigQuery e não trazem variáveis socioeconômicas individuais; as decisões de política pública são tomadas no nível municipal; a camada Gold da Fase 2 foi construída nessa granularidade.
- **Alternativa descartada:** modelo por aluno com amostra dos microdados, que não teria variáveis de contexto individuais.
- **Consequência:** 4.919 observações. Os resultados descrevem municípios, não crianças.

## D2 — Desenho temporal: features de 2023, target de 2024

- **Decisão:** nenhuma variável de 2024 entra como feature. O treino é interrompido se alguma feature terminar em `_2024`, e um teste falha se o ROC-AUC passar de 0,97.
- **Motivo:** a primeira versão usava as proporções de alunos por nível de proficiência de 2024, que compõem a própria taxa de 2024 usada no target (ROC-AUC de 0,997 por vazamento).
- **Alternativa descartada:** manter o corte transversal de 2024 removendo apenas as colunas mais óbvias. As proporções por nível e a média de português continuariam sendo componentes do mesmo resultado.
- **Consequência:** um modelo de previsão de fato (ROC-AUC de 0,917). Ficam de fora os 597 municípios sem resultado em 2023, e há um único par de anos.

## D3 — Corte de 60% e classe positiva = risco

- **Decisão:** `em_risco_2024 = 1` quando a taxa de 2024 é menor que 60%.
- **Motivo:** o corte fica próximo do patamar nacional de 2024 (mediana de 64,3%; 42,5% dos municípios abaixo de 60%) e no meio do caminho até a meta de 80%. Com risco como classe positiva, o recall passa a significar "municípios em risco identificados".
- **Alternativas descartadas:** corte pela meta de cada município; regressão da taxa contínua como modelo principal (registrada como evolução).
- **Consequência:** o corte é uma escolha, declarada nas limitações.

## D4 — Fontes externas reais, sem simulação

- **Decisão:** IBGE pela API SIDRA (Censo 2022 e PIB 2021) e Atlas do Desenvolvimento Humano 2010. Se uma fonte faltar, a execução é interrompida.
- **Motivo:** uma versão anterior gerava valores aleatórios quando o GCP não estava configurado, e o modelo publicado havia sido treinado com eles.
- **Alternativas descartadas:** BigQuery (exige projeto GCP de cada avaliador); indicadores socioeconômicos do Censo 2022 (ainda não disponíveis por município).
- **Consequência:** cobertura de 100% (IBGE) e 99,9% (Atlas), com defasagem de 14 anos nos indicadores do Atlas.

## D5 — Pré-processamento dentro do pipeline

- **Decisão:** imputação pela mediana com indicador de ausência, padronização e one-hot encoding, ajustados apenas nos dados de treino de cada fold.
- **Motivo:** evitar que estatísticas do conjunto de validação ou de teste influenciem o treino. O indicador de ausência preserva a informação de que o dado faltava (por exemplo, municípios criados depois de 2010).
- **Alternativa descartada:** imputar antes da divisão treino/teste.

## D6 — Métrica de seleção: PR-AUC

- **Decisão:** o modelo final é o de maior PR-AUC médio em validação cruzada (5 folds). O critério foi definido antes do treino.
- **Motivo:** o PR-AUC foca na capacidade de encontrar os municípios em risco, que é o uso pretendido.
- **Alternativas descartadas:** ROC-AUC, que mede a ordenação geral; acurácia, sensível ao balanceamento das classes.

## D7 — HistGradientBoosting em vez de Regressão Logística

- **Resultado:** PR-AUC em CV de 0,897 (HistGradientBoosting) contra 0,887 (Regressão Logística), uma diferença menor que o desvio entre folds (cerca de 0,02).
- **Decisão:** HistGradientBoosting, pelo critério da D6.
- **Alternativa equivalente:** Regressão Logística, com menos overfitting (gap entre treino e CV de 0,011, contra 0,033) e interpretação mais simples. O Random Forest foi avaliado apenas com hiperparâmetros padrão (PR-AUC de 0,892) e não entrou na otimização.

## D8 — Calibração isotônica

- **Decisão:** calibrar as probabilidades com `CalibratedClassifierCV` (método isotônico).
- **Motivo:** as probabilidades são usadas em faixas de risco, e o `class_weight="balanced"` distorce os valores.
- **Alternativa descartada:** calibração sigmoide (Platt). Com cerca de 3.900 municípios no treino, o método isotônico, mais flexível, é viável.
- **Consequência:** ganho pequeno no modelo completo (Brier de 0,1135 para 0,1119) e maior no modelo sem histórico (0,1353 para 0,1305). Previsto e observado coincidem por faixa. O método gera probabilidades de 1,0 no degrau mais alto, que devem ser lidas como "acima de 99%".

## D9 — Threshold por recall mínimo de 0,85

- **Decisão:** threshold escolhido nas predições *out-of-fold* calibradas do treino: a maior precisão entre os thresholds com recall de pelo menos 0,85.
- **Motivo:** deixar de identificar um município em risco custa mais do que um alarme falso.
- **Alternativas descartadas:** threshold fixo de 0,5; threshold que maximiza o F1.
- **Consequência:** threshold de 0,431, com recall de 0,883 e precisão de 0,770 no teste.

## D10 — Três variantes do modelo

- **Decisão:** treinar com o mesmo processo três conjuntos de features: só socioeconômico, sem histórico e completo.
- **Motivo:** o SHAP de um único modelo não separa bem o efeito de grupos de variáveis correlacionados entre si (estado e condição socioeconômica).
- **Consequência:** ROC-AUC de 0,787, 0,893 e 0,917. A diferença entre as duas primeiras variantes (0,11) mede o que o estado acrescenta além da condição socioeconômica.

## D11 — Importância por permutação em grupo

- **Decisão:** embaralhar juntas todas as colunas de um grupo e medir a queda de PR-AUC no teste.
- **Motivo:** UF e região carregam quase a mesma informação. Embaralhar uma de cada vez subestima a importância, porque a outra compensa.
- **Alternativa descartada:** `permutation_importance` coluna a coluna.

## D12 — Clustering sobre variáveis estruturais, com k entre 3 e 6

- **Decisão:** K-Means sobre variáveis socioeconômicas e territoriais, sem usar o target; k escolhido pela silhueta entre 3 e 6, regra definida antes da execução.
- **Motivo:** responder quais municípios têm perfis parecidos. Com k = 2, os grupos apenas separariam municípios mais e menos desenvolvidos.
- **Consequência:** k = 3, com silhueta de 0,267: uma segmentação útil, mas sem fronteiras nítidas. O risco observado é usado só depois, para descrever os grupos.

## D13 — Probabilidades out-of-fold na análise estratégica

- **Decisão:** cada município recebe a probabilidade de um modelo treinado sem ele.
- **Motivo:** probabilidades calculadas sobre municípios do treino seriam otimistas.

## D14 — Projeção para 2030 como cenário

- **Decisão:** aplicar a variação média nacional observada (2,55 pontos por ano, calculada dos dados) a partir da taxa de 2024 e comparar o ritmo necessário com a meta de cada município.
- **Alternativas descartadas:** tendência individual por município (um único intervalo de um ano, muito ruidoso); modelo de regressão para a taxa futura (registrado como evolução).
- **Consequência:** a projeção é um cenário de tendência, não uma previsão.

## D15 — Versionamento de artefatos e preservação da versão 1

- **Decisão:** versionar os modelos, as tabelas de resultados e a Silver/Gold da Fase 2; regenerar por script os dados externos e processados. Manter a versão 1 em `src/legacy/`, com a explicação do vazamento.
- **Comprovação:** em um clone limpo, com ambiente virtual novo, a instalação pelo `requirements.txt`, a regeneração dos dados e os 24 testes passam, incluindo a reprodução exata das métricas pelo modelo versionado.

## D16 — Previsão de 2025 com o modelo final, sem retreino

- **Decisão:** aplicar o modelo final aos resultados de 2024, que ocupam as colunas de "ano anterior" (`*_2023`), mantendo as mesmas fontes externas do treino. A lista de 2025 passa a ser a lista de priorização principal; a de 2024 fica como validação retrospectiva.
- **Motivo:** o enunciado pede prever municípios que podem não atingir metas futuras. A lista de 2024 descreve um ano já observado.
- **Alternativa descartada:** retreinar com 2024 como target e 2023 como features para depois pontuar 2025. É o mesmo par de anos já usado; não haveria dado novo para treinar.
- **Consequências:** 5.493 municípios pontuados (inclui os 574 sem resultado em 2023; exclui AC e DF). A lista não tem métrica até a divulgação de 2025. O diagnóstico de distribuição entre 2023 e 2024 fica salvo junto com a previsão. Um teste garante que as colunas de ano anterior recebem os valores de 2024.

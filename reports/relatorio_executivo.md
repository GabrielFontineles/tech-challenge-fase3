# Relatório executivo — Risco de alfabetização nos municípios brasileiros

Tech Challenge Fase 3 · FIAP IA Scientist

## Em uma frase

É possível identificar com antecedência cerca de 9 em cada 10 municípios que ficarão abaixo de 60% de crianças alfabetizadas, e o fator que mais diferencia municípios de perfil semelhante é o estado onde estão.

## Situação

- 4.919 municípios analisados (dados de 2023 e 2024)
- 42,7% estavam **em risco** em 2024 (menos de 60% das crianças alfabetizadas)
- Meta nacional: 80% até 2030

## O que o modelo entrega

| | |
|---|---|
| Municípios em risco identificados antecipadamente | 88% |
| Acerto quando o modelo aponta risco | 77% |
| Confiabilidade das probabilidades | Previsto e observado coincidem (ex.: 91% previsto, 90% observado na faixa crítica) |

## Principais achados

1. **O estado importa mais do que a renda.** Entre os municípios mais vulneráveis (pobreza, analfabetismo adulto elevado, predominância rural), todos os 166 do Ceará estavam fora de risco, contra 1 em cada 10 no Rio Grande do Norte.
2. **Renda não garante alfabetização.** Municípios urbanos de maior porte e renda têm mais risco (41%) do que municípios pequenos do interior de renda média (26%).
3. **O risco é concentrado:** Bahia (97% de probabilidade média), Sergipe, Pará e Rio Grande do Norte.
4. **2030 exige aceleração:** 1.261 municípios precisariam mais que dobrar o ritmo atual de melhoria.

## Recomendações

1. **Priorizar** os 1.166 municípios da lista de priorização (`priorizacao_municipios_v21.csv`), organizada por estado
2. **Estudar as experiências estaduais bem-sucedidas** (Ceará, Pernambuco, Piauí, Maranhão) antes de desenhar novos programas
3. **Não esperar o próximo resultado de avaliação:** o perfil do território já indica a maior parte do risco
4. **Acompanhar qualidade, não só matrícula:** a frequência na pré-escola é alta justamente nos municípios de maior risco

## Cuidados na leitura

Os resultados mostram associações, não causas. O efeito do estado pode refletir política educacional, eventos de 2024 (como as enchentes no Rio Grande do Sul) e diferenças entre as avaliações estaduais. Há apenas dois anos de dados disponíveis.

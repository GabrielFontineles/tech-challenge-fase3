# Roteiro — Vídeo Executivo (≤ 5 minutos)

**Formato:** reunião executiva simulada com gestores públicos (secretaria estadual/municipal de educação, MEC).
**Papéis sugeridos:** 1 apresentador(a) principal (cientista de dados), 1 "gestor" fazendo 2 perguntas, 1 pessoa nas transições/slides. Os demais aparecem nos créditos.
**Regra de ouro:** cada bloco responde a uma pergunta que um gestor faria. Nada de métrica sem consequência prática.

| Tempo | Bloco | Fala (resumo) | Tela |
|---|---|---|---|
| 0:00–0:35 | **O problema** | "A meta nacional é 80% de crianças alfabetizadas até 2030. Hoje o país está em `[X]`% e `[N]` municípios estão abaixo de 60%. O problema do gestor não é saber quem está mal hoje — o ICA já mostra — é saber **quem vai estar mal no próximo ciclo** e **o que fazer antes**." | Mapa 2024 (taxa por município) |
| 0:35–1:15 | **A pergunta e o desenho** | "Construímos um modelo que, com dados de 2023 — resultado anterior, território, condição socioeconômica e estrutura da rede — estima a probabilidade de cada município estar em risco em 2024. Só usamos informação disponível *antes* do resultado, então isso é uma previsão de verdade, não uma descrição." | Diagrama t → t+1 (3 blocos → target) |
| 1:15–2:00 | **O que o modelo entrega** | "Acertamos `[recall]`% dos municípios em risco com `[precisão]`% de precisão. Calibrado para preferir falso alarme a município esquecido. Mais importante: ele ordena — este é o ranking, esta é a lista de `[N]` municípios prioritários." | Ranking + mapa de probabilidade |
| 2:00–2:50 | **Os insights** (3, no máximo) | (1) Inércia: `[Y]`% dos municípios em risco em 2024 já estavam em risco em 2023 — mas `[Z]` mudaram de estado, e são eles que a política precisa alcançar. (2) Retirando o histórico, os fatores estruturais que mais pesam são `[A, B, C]` — são as alavancas. (3) Existem `[k]` perfis de município; o mesmo programa não serve para todos. | SHAP bar (variante estrutural) + clusters |
| 2:50–3:30 | **Pergunta do gestor 1** | "Quantos municípios não vão atingir a meta de 2030?" → "Na trajetória atual, `[N]` estão fora; para cada um sabemos a variação anual necessária. `[N2]` estão em 'atenção' e são os de melhor custo-benefício para intervir agora." | Projeção 2030 (pizza/status + tabela) |
| 3:30–4:10 | **Pergunta do gestor 2** | "O que eu faço com isso segunda-feira?" → "Três usos: priorizar visitas técnicas e recursos pela lista; desenhar intervenções por perfil de cluster; monitorar participação no SAEB como sinal precoce." | Lista de priorização |
| 4:10–4:40 | **Limites e honestidade** | "É correlacional, não causal; temos dois anos de série; parte dos dados socioeconômicos é de 2010. Cada nova edição do ICA melhora o modelo." | Slide de limitações |
| 4:40–5:00 | **Fechamento** | "Dados públicos, código aberto, pipeline reproduzível. O valor não está no modelo — está em decidir antes, e não depois." | Repositório + equipe |

## Checklist de produção

- [ ] Gravar em 1080p, áudio com microfone de lapela ou headset; ambiente silencioso.
- [ ] Slides com no máximo 1 gráfico + 1 frase por tela; fontes ≥ 24 pt.
- [ ] Números de `[colchetes]` preenchidos a partir de `models/metadata.json` e `reports/*.csv` — nunca dos dados sintéticos.
- [ ] Ensaiar com cronômetro: 4:30 de conteúdo deixa folga para cortes.
- [ ] Exportar com legendas (acessibilidade) e subir como não listado; link no README.

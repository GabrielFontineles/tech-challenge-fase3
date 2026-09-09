# Dados — como obter e onde colocar

Nenhum dado é versionado no Git (ver `.gitignore`). Esta página descreve como reconstruir cada pasta.

## `data/raw/` e `data/gold/` — camadas Silver e Gold da Fase 2

| Arquivo | Origem | Conteúdo |
|---|---|---|
| `raw/municipio_silver.parquet` | Pipeline da Fase 2 | Indicador Criança Alfabetizada por município × ano (2023, 2024) × rede |
| `raw/uf_silver.parquet` | Pipeline da Fase 2 | Mesmo indicador agregado por UF |
| `gold/ranking_estados.parquet`, `gold/evolucao_temporal.parquet`, `gold/analise_municipal.parquet`, `gold/visao_brasil.parquet` | Pipeline da Fase 2 | Agregações usadas na EDA descritiva |

Colunas esperadas em `municipio_silver` (nomes configuráveis em `src/config.py`):
`id_municipio`, `ano`, `rede`, `taxa_alfabetizacao`, `media_portugues`, `meta_alfabetizacao_2030`,
`distancia_meta_2030`, `percentual_participacao`, `nivel_alfabetizacao`, `proporcao_aluno_nivel_0..8`
e, se disponível, `quantidade_alunos_avaliados`.

> Como obter: exportar da camada Gold/Silver do repositório da Fase 2 (S3/BigQuery) para estes caminhos.
> **Preencher aqui o comando/link exato usado pelo grupo.**

## `data/external/raw/` — fontes externas como baixadas

`python -m src.data.download_external` tenta baixar tudo. O que não vier automaticamente, baixar à mão para esta pasta com o nome indicado e rodar de novo com `--skip-download`.

| Fonte | Arquivo esperado em `external/raw/` | Onde baixar | Ano |
|---|---|---|---|
| IBGE — Censo 2022 (população, densidade) | `sidra_4714_censo2022.parquet` (automático, API SIDRA t/4714) | https://sidra.ibge.gov.br/tabela/4714 | 2022 |
| IBGE — PIB dos Municípios | `sidra_5938_pib2021.parquet`, `sidra_6579_pop2021.parquet` (automático, API SIDRA) | https://sidra.ibge.gov.br/tabela/5938 | 2021 |
| Atlas do Desenvolvimento Humano (PNUD/IPEA/FJP) | `atlas2013_dadosbrutos_pt.xlsx` (**manual**) | http://www.atlasbrasil.org.br/acervo/biblioteca → "Base de dados — Atlas 2013 — Municípios" | 2010 |
| INEP — IDEB anos iniciais por município | `divulgacao_anos_iniciais_municipios_2023.zip` | https://www.gov.br/inep/pt-br/areas-de-atuacao/pesquisas-estatisticas-e-indicadores/ideb/resultados | 2023 |
| INEP — Indicadores educacionais (TDI, AFD, DSU, ATU) | `TDI_2023_MUNICIPIOS.zip`, `AFD_2023_MUNICIPIOS.zip`, `DSU_2023_MUNICIPIOS.zip`, `ATU_2023_MUNICIPIOS.zip` | https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/indicadores-educacionais | 2023 |
| INEP — Microdados Censo Escolar (escolas) | `microdados_censo_escolar_2023.zip` (~100 MB) | https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/censo-escolar | 2023 |
| Coordenadas / capital (GitHub kelvins) | `municipios_kelvins.csv` (automático) | https://github.com/kelvins/municipios-brasileiros | — |
| População DATASUS/RIPSA (GitHub lsbastos) | `popBR2000-2025.long.csv` (automático) | https://github.com/lsbastos/popBR_mun | 2023 |
| FUNDEB por matrícula (segunda leva) | `fundeb_2023.csv` com colunas `id_municipio` + variáveis (**manual**) | FNDE/SIOPE — https://www.fnde.gov.br/ | 2023 |
| Cadastro Único (segunda leva) | `cadunico_2023.csv` com `id_municipio` + variáveis (**manual**) | CECAD / VIS DATA (MDS) | 2023 |

Regra temporal: **nada posterior a 2023** entra como feature.

## `data/external/processed/` e `data/processed/`

Gerados pelos scripts:

- `external/processed/<fonte>.parquet` e `socioeconomico_municipal.parquet` — uma linha por `id_municipio` (7 dígitos), colunas com prefixo da fonte (`ibge_`, `pib_`, `atlas_`, `ideb_`, `inep_`, `censo_`, `geo_`, `datasus_`, `fundeb_`, `cadunico_`).
- `processed/dataset_modelagem.parquet` — dataset t → t+1 (`python -m src.data.build_dataset`).
- `processed/dicionario_dataset.csv` — coluna, bloco (A/B/C/target), tipo, % missing.
- `processed/predicoes_teste*.parquet` — probabilidades no conjunto de teste (gerado pelo treino).

## Dados sintéticos (apenas para testar a pipeline)

`python -m src.utils.synthetic` cria uma `municipio_silver.parquet` fictícia com o mesmo esquema.
**Nunca** use resultados gerados a partir dela em análises ou no README.

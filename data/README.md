# Dados do projeto

## Visão geral

| Pasta | Conteúdo | Versionado | Como obter |
|---|---|---|---|
| `raw/` | Silver da Fase 2 (`municipio_silver.parquet`, `uf_silver.parquet`) | Sim | Já incluído no repositório |
| `gold/` | Gold da Fase 2 (ranking, evolução, análise municipal, visão Brasil) | Sim | Já incluído no repositório |
| `external/` | IBGE e Atlas processados | Não | `python src/data/download_external.py` |
| `external/raw/` | Arquivo original do Atlas | Não | Download manual (abaixo) |
| `processed/` | Datasets de modelagem | Não | `build_dataset.py` e `download_external.py` |

## Fase 2 (Silver e Gold)

Gerados pela pipeline do Tech Challenge Fase 2 a partir da Base dos Dados (dataset `br_inep_avaliacao_alfabetizacao`). A camada Silver traz, por município, ano (2023 e 2024) e rede: taxa de alfabetização, média de português, proporção de alunos por nível de proficiência, participação e metas de 2024 a 2030. O projeto usa a rede **Total**.

## IBGE (automático)

Obtido pela API SIDRA, sem credencial, em `download_external.py`:

- **Tabela 4714**, Censo 2022: população residente, área (km²) e densidade
- **Tabela 5938**, PIB dos Municípios 2021: PIB a preços correntes (mil reais)

Chave de junção: código IBGE do município com **7 dígitos**.

## Atlas do Desenvolvimento Humano (manual)

1. Acessar [atlasbrasil.org.br](http://www.atlasbrasil.org.br) e baixar a base municipal completa
2. Salvar o arquivo `censo_total_1991_2010.xlsx` em `data/external/raw/`
3. Rodar `python src/data/download_external.py`

O script lê a aba `MUN 91-00-10`, filtra o ano de 2010 e usa a coluna `Codmun7` (7 dígitos). A leitura do Excel é feita uma vez e salva em cache (`external/atlas_2010.parquet`).

Variáveis utilizadas: `IDHM`, `IDHM_E`, `IDHM_R`, `IDHM_L`, `RDPC`, `GINI`, `PMPOB`, `PIND`, `T_ANALF15M`, `E_ANOSESTUDO`, `T_FREQ4A5` e `pesoRUR`/`pesotot` (para o % rural).

## Regras

- Nenhum dado é simulado: se uma fonte faltar, os scripts interrompem a execução com erro.
- Todas as fontes são anteriores a 2024, o ano do target.
- Os testes verificam a cobertura mínima de 99% das fontes externas e a ausência de arquivos simulados.

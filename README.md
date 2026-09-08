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

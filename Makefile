# Makefile — Tech Challenge Fase 3

.PHONY: install dataset external train explain strategy test all help

## Instala as dependencias
install:
	pip install -r requirements.txt

## Constroi o dataset temporal (features 2023 -> target 2024)
dataset:
	python src/data/build_dataset.py

## Baixa IBGE via API e le o Atlas (xlsx em data/external/raw)
external:
	python src/data/download_external.py

## Treina as tres variantes do modelo
train:
	python src/modeling/train.py
	python src/modeling/train.py --sem-historico
	python src/modeling/train.py --so-socioeconomico

## SHAP e importancia por grupo das tres variantes
explain:
	python src/evaluation/shap_analysis_v2.py
	python src/evaluation/shap_analysis_v2.py --sem-historico
	python src/evaluation/shap_analysis_v2.py --so-socioeconomico

## Ranking de risco, clusters, projecao 2030 e priorizacao
strategy:
	python src/evaluation/strategic_v2.py

## Roda os testes
test:
	python -m pytest tests/ -v

## Pipeline completa
all: dataset external train explain strategy test

help:
	@echo "Uso: make [install|dataset|external|train|explain|strategy|test|all]"

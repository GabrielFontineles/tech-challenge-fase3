# Pipeline reproduzível — `make all` roda tudo a partir da Silver da Fase 2 em data/raw/
PY ?= python

install:
	pip install -r requirements.txt

external:            ## baixa/padroniza fontes externas (rode novamente após downloads manuais com SKIP=1)
	$(PY) -m src.data.download_external $(if $(SKIP),--skip-download,)

dataset:             ## constrói o dataset t -> t+1
	$(PY) -m src.data.build_dataset

train:               ## treina, valida, escolhe threshold e salva o modelo
	$(PY) -m src.modeling.train
	$(PY) -m src.modeling.train --sem-historico

explain:             ## importância por permutação + SHAP
	$(PY) -m src.evaluation.shap_analysis
	$(PY) -m src.evaluation.shap_analysis --sem-historico

strategy:            ## ranking de risco, clusters, projeção de metas, priorização
	$(PY) -m src.evaluation.strategic

notebooks:           ## (re)gera o scaffolding dos notebooks
	$(PY) tools/gerar_notebooks.py

test:
	pytest -q

synthetic:           ## dados sintéticos só para testar a pipeline (NÃO usar em análises)
	$(PY) -m src.utils.synthetic

all: external dataset train explain strategy

.PHONY: install external dataset train explain strategy notebooks test synthetic all

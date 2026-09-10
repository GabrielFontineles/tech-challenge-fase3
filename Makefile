# Makefile — Tech Challenge Fase 3
# Pipeline completa de ML para predição de alfabetização

.PHONY: all install dataset external train explain strategy test clean

## Instala dependências
install:
	pip install -r requirements.txt

## Constrói dataset temporal (features 2023 → target 2024)
dataset:
	python src/data/build_dataset.py

## Baixa dados externos (requer GCP_BILLING_PROJECT_ID configurado)
external:
	python src/data/download_external.py

## Treina o modelo com RandomizedSearchCV
train:
	python src/modeling/train.py

## Gera análise SHAP de interpretabilidade
explain:
	python src/evaluation/shap_analysis_v2.py

## Gera análise estratégica e lista de priorização
strategy:
	python src/evaluation/strategic_v2.py

## Roda todos os testes
test:
	python -m pytest tests/ -v

## Pipeline completa (sem dados externos)
all: dataset train explain strategy test

## Remove arquivos gerados
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete

## Ajuda
help:
	@echo "Comandos disponíveis:"
	@echo "  make install    - Instala dependências"
	@echo "  make dataset    - Constrói dataset temporal"
	@echo "  make external   - Baixa dados externos (requer GCP)"
	@echo "  make train      - Treina o modelo"
	@echo "  make explain    - Gera análise SHAP"
	@echo "  make strategy   - Gera análise estratégica"
	@echo "  make test       - Roda testes pytest"
	@echo "  make all        - Pipeline completa"
	@echo "  make clean      - Remove arquivos temporários"

# Legado — versão 1 do modelo (não utilizar)

Estes scripts são a primeira versão do modelo e foram mantidos apenas como histórico.

**Problema:** a v1 usava como features as proporções de alunos por nível de proficiência
de 2024, que compõem a própria taxa de alfabetização de 2024 usada para criar o target.
Isso caracteriza data leakage e explica o ROC-AUC de 0,997.

A versão atual usa desenho temporal (features de 2023, target de 2024).
Ver `src/data/build_dataset.py` e `src/modeling/train.py`.

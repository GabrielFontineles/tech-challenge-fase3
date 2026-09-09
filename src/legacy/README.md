# Versão 1 (preservada para histórico)

Scripts da primeira iteração do projeto. Foram substituídos pelo desenho t → t+1
(ver `docs/01_desenho_do_modelo.md` e `docs/02_decisoes_analiticas.md`, ADR-001) porque
as features de 2024 (`proporcao_aluno_nivel_*`, `score_niveis_altos`, `media_portugues`)
reconstruíam o target derivado da taxa de 2024 (data leakage).

`eda_inicial.py` e `eda_correlacoes.py` continuam em `src/preprocessing/` — a EDA descritiva
sobre as tabelas Gold segue válida e é reaproveitada no notebook 01.

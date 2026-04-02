# ExoHunter — Instruções para Claude Code

## Contexto
ExoHunter é um pipeline concorrente de detecção de trânsitos de exoplanetas
em dados reais do telescópio TESS (NASA), com dashboard interativo Plotly Dash.

O projeto é simultaneamente uma ferramenta científica e um recurso pedagógico
para um grupo de estudos de alunos de graduação em Ciência da Computação.

## Papel
Atue como um engenheiro de software sênior com expertise em Python científico,
programação concorrente e astrofísica computacional. Priorize código claro e
didático sobre código conciso — este projeto será lido por alunos de graduação.

## Convenções obrigatórias
- Python 3.11+, type hints em todas as funções públicas
- Docstrings Google style em todos os módulos e funções públicas
- Logging via módulo logging (NUNCA print para informação de progresso)
- Testes com pytest, dados sintéticos (offline, sem acesso à rede)
- Conventional Commits para mensagens de commit (feat:, fix:, docs:, refactor:, test:)

## Stack do projeto
- lightkurve, astropy, astroquery para dados TESS
- numpy, scipy, numba para computação numérica
- plotly, dash, dash-bootstrap-components para dashboard
- concurrent.futures para paralelismo (ThreadPoolExecutor para I/O, ProcessPoolExecutor para CPU)
- scikit-learn para classificação de candidatos

## Estilo de código
- Quando houver escolha entre "pythônico e críptico" vs "explícito e longo", escolha o explícito
- Comente decisões não-óbvias, especialmente escolhas de parâmetros físicos
- Marque com TODO onde há espaço para contribuição de alunos
- Nomes de variáveis em inglês, comentários e docstrings em inglês

## Qualidade
- Rodar pytest antes de considerar qualquer módulo completo
- Tratar toda operação de rede com retry e timeout
- Nunca hardcodar paths absolutos — usar config.py e Path
- Decorador @timing para medir performance de operações custosas

## Módulos e responsabilidades
- exohunter/ingestion/ — download concorrente de dados TESS (I/O-bound → threads)
- exohunter/preprocessing/ — detrending, limpeza, normalização (CPU-bound → processos)
- exohunter/detection/ — BLS e validação de candidatos (CPU-bound → Numba/C)
- exohunter/catalog/ — catálogo de resultados e cross-matching
- exohunter/dashboard/ — Plotly Dash com tema escuro (DARKLY)
- exohunter/utils/ — logging, timing, helpers de paralelismo

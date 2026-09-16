# Provenance

The following are carried forward unmodified from the prior published work
(arXiv:2608.26197, repo `Harness_Engg-1`), per plan section 3 "do not rewrite what
already works and is published":

- `src/harness/fsm.py` — FSM executor, validation gate, bounded retries
- `src/harness/baseline_agent.py` — unconstrained ReAct baseline
- `src/harness/llm_utils.py` — provider client, plan parsing, usage accounting
- `src/harness/tools_{finance,legal}.py` — bound tool implementations
- `src/tracing/` — trace logger and schema
- `src/tasks/{finance_ecl,legal_clause}.py` — F1 linear family + deterministic scorers
- `src/analysis/` — metrics and statistics
- `data/` — F1 instances

New in this study: `src/adf/`, `src/harness/configurable.py`, F0/F2 families.

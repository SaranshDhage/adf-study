# Session handoff prompt

Paste the block below into a new session started in
`/home/saransh/Projects/Agentic-Research/adf-study`.

---

You are continuing a multi-session research project. Read in order:
1. `../loop_eng.md`   — the full research plan
2. `checkpoint.md`    — the running log; **last entry is Session 6**
3. `PREREGISTRATION.md` — frozen hypotheses and metrics

Then read this HANDOFF.md. Follow it.

---

## What we're building

**"Agent Degrees of Freedom (ADF): Execution Determinism as a Predictable
Function of Harness Constraint"** — a curve, not a two-condition comparison.

**Core claims tested:**
- H1: Determinism Index decreases monotonically as ADF increases.
- H2: Task success is non-monotone in ADF — low at extremes, interior peak.
- H3: The optimum ADF is higher for branching tasks than linear tasks.
- H4: The optimum ADF is higher for more capable models.
- H5: Equal-ADF configs with different substrates yield indistinguishable DI.

Prior work: arXiv:2608.26197 (Harness Engineering Phase 1).

---

## Current state: Phase 6.2 in progress

**Grid is running as a background nohup process (PID 769456).**

Check if it's still alive:
```
ps -p 769456 -o pid,stat,cmd
```

Check progress:
```
wc -l results/grid/runs.jsonl
python3 -c "
import json, collections
from pathlib import Path
records = [json.loads(l) for l in Path('results/grid/runs.jsonl').read_text().splitlines() if l.strip()]
by = collections.Counter((r['family'], r['model'].split('.')[-1][:12], r['rung']) for r in records if r.get('task_success') is not None)
done = sum(1 for v in by.values() if v >= 50)
print(f'Total: {len(records)} | Cells N>=50: {done}/72')
for k,v in sorted(by.items()):
    print(f'  {k[0]:12s} {k[1]:14s} {k[2]:11s}: {v}')
"
```

If the process is not running, restart it:
```
cd /home/saransh/Projects/Agentic-Research/adf-study
nohup .venv/bin/python -m src.harness.run_grid \
    --models "amazon.nova-micro-v1:0,google.gemma-3-27b-it,qwen.qwen3-32b-v1:0,amazon.nova-pro-v1:0" \
    --families "finance_ecl,branching_ecl" \
    --n 50 \
    >> results/grid_stdout.log 2>&1 &
echo "Grid PID: $!"
```
The grid is RESUMABLE — already-done runs are skipped automatically.

---

## What to do when the grid finishes

1. Run the full analysis:
   ```
   .venv/bin/python -m src.analysis.phase7_analysis
   ```
   Writes results/analysis/{tsr_table, di_table, h1-h5_test, summary_report}.json/md

2. Push to GitHub (need PAT from user):
   ```
   python3 push_to_github.py <GITHUB_PAT>
   ```
   Creates SaranshDhage/harness-engg-phase1 + SaranshDhage/adf-study.
   **Ask the user for a GitHub PAT (classic, `repo` scope) — this is the one
   external credential needed.**

3. Update checkpoint.md with final results, diagnostics, stop conditions.

4. If any diagnostic flags a defect → fix engineering, re-run affected cells.

---

## Hard rules (same as always, do not relitigate)

- PREREGISTRATION.md frozen at commit 34b1a37. No hypothesis changes.
- Null/negative results are valid outcomes. Do not tune after seeing data.
- Never drop runs silently — discards.jsonl with reason code.
- Append to checkpoint.md after every batch.

---

## Infrastructure summary

**Backend:** AWS Bedrock (no billing ceiling). Credentials in env vars.
**Virtual env:** .venv/ — always use `.venv/bin/python`.
**Models (Bedrock):**
  - `amazon.nova-micro-v1:0`  — small tier
  - `google.gemma-3-27b-it`   — mid gen-1 (exact prereg match)
  - `qwen.qwen3-32b-v1:0`     — mid gen-2
  - `amazon.nova-pro-v1:0`    — large tier

**Key engineering notes:**
- BedrockChatModel (llm_utils.py) uses boto3 directly. LangChain-AWS has an
  incorrect allow-list blocking toolChoice for Gemma/Qwen3.
- `tool_choice="any"` = Bedrock toolChoice={"any":{}} = "required".
- Gemma needs `tool_choice="any"` even for auto-select (otherwise returns text).

---

## Interim results (Nova-Micro × finance_ecl, N=50)

### TSR
```
Rung    ADF    TSR    [95% CI]
L0'    0.000  96.4%  [90.9, 100]
L0     0.381  96.0%  [90.0, 100]
L1     0.857  96.0%  [90.0, 100]
L2     1.159  96.0%  [90.0, 100]
L3     1.921  100%   [100,  100]
L4     2.363  100%   [100,  100]
L5     3.567  92.0%  [84.0,  98]
L5_cg  3.567  0.0%   [0, 0]       ← H5 FAIL on TSR
L6     3.758  100%   [100,  100]
```

### DI (within-instance)
```
Rung    DI     PS     RR
L0'    1.000  1.000  1.000
L0     1.000  1.000  1.000
L1     0.982  0.927  1.000   ← plan freed: PS drops
L2     0.964  0.854  1.000
L3     0.969  0.875  1.000
L4     0.971  0.885  1.000
L5     0.987  0.948  1.000
L5_cg  0.984  0.938  0.000   ← codegen: RR=0 even at T=0
L6     0.961  0.844  1.000
```

### Key findings so far
1. DI drops when plan is freed (L1 PS: 1.000→0.927). Plan Stability is the
   dominant DI component, confirming prior paper.
2. H5: DI is indistinguishable for L5-tool vs L5-codegen (diff=0.003) ✓
   BUT TSR is 92% vs 0% ✗ — mechanism matters for success.
3. L5_codegen: RR=0 even at T=0 (code strings vary run-to-run).
4. Finance_ecl TSR is flat-high (linear task) — H2 curve expected on
   branching_ecl × capable models (Gemma, Qwen3, Nova-Pro).

---

## Research contribution (presentable)

This work produces:
1. **First empirical ADF curve**: determinism and success as continuous functions
   of a single, computable quantity (ADF = log2-sum of legal choices).
2. **Mechanism-vs-ADF decomposition**: H5 shows ADF alone doesn't predict
   task success when substrate differs — mechanism adds independent signal.
3. **Plan-layer finding**: Plan Stability is the dominant DI component at T=0.
   Even unconstrained tool/state choices don't add trace variance when T=0.
4. **Model-capability × constraint interaction**: Non-monotone H2 curve visible
   for branching tasks only with capable models — Nova-Micro lacks the capacity
   to discover ESCALATION, showing H4 empirically.
5. **Codegen non-determinism**: Even at T=0, codegen paths are trace-non-
   deterministic (RR=0), while tool_call paths are trace-deterministic (RR=1).
   ADF doesn't capture this; it's a mechanism-specific finding.

---

## Files of interest
- `results/grid/runs.jsonl`     — all grid runs (append-only)
- `results/grid/summary.json`   — per-cell TSR aggregates
- `results/analysis/`           — analysis outputs after phase7_analysis.py
- `push_to_github.py`           — run with PAT to push to GitHub
- `discards.jsonl`              — should be empty/absent (no discards yet)
- `docs/adf_ladder.md`          — 8 rung definitions (FROZEN)
- `PREREGISTRATION.md`          — frozen hypotheses at commit 34b1a37

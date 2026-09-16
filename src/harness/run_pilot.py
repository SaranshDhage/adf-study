"""Pilot runner — Phase 6.1.

Usage:
    python -m src.harness.run_pilot [--model MODEL_ID] [--family FAMILY]
                                    [--n N_per_rung] [--rungs L0prime,L1,...] [--dry-run]

Defaults: model=amazon.nova-micro-v1:0, family=finance_ecl, n=10, all 8 rungs.

Purpose (loop_eng.md §6.1):
  - Verify the pipeline end-to-end before the full grid.
  - NOT for testing hypotheses; results are used only for variance estimation
    and power calculation.

Outputs:
  results/pilot/runs.jsonl          — one record per run
  results/pilot/summary.json        — per-rung aggregates
  discards.jsonl                    — dropped runs with reason codes

Resumability: completed (rung, instance_idx, repeat_idx) triples are read
from runs.jsonl at startup; already-done runs are skipped.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_DIR = _ROOT / "results" / "pilot"
_DISCARDS_PATH = _ROOT / "discards.jsonl"

# -----------------------------------------------------------------------
# Ladder rungs (must match docs/adf_ladder.md)
# -----------------------------------------------------------------------
from src.adf.metric import HarnessConfig

LADDER: dict[str, HarnessConfig] = {
    "L0prime": HarnessConfig(
        plan_mode="schema_validated", state_mode="fsm_fixed",
        routing_mode="fixed_map", tool_mode="forced_single",
        arg_mode="fixed", action_substrate="tool_call",
        skill_level="precise",
    ),
    "L0": HarnessConfig(
        plan_mode="schema_validated", state_mode="fsm_fixed",
        routing_mode="fixed_map", tool_mode="forced_single",
        arg_mode="enum_strict", action_substrate="tool_call",
        skill_level="precise",
    ),
    "L1": HarnessConfig(
        plan_mode="free_text", state_mode="fsm_fixed",
        routing_mode="fixed_map", tool_mode="forced_single",
        arg_mode="enum_strict", action_substrate="tool_call",
        skill_level="precise",
    ),
    "L2": HarnessConfig(
        plan_mode="free_text", state_mode="fsm_fixed",
        routing_mode="model_chosen", tool_mode="forced_single",
        arg_mode="enum_strict", action_substrate="tool_call",
        skill_level="precise",
    ),
    "L3": HarnessConfig(
        plan_mode="free_text", state_mode="fsm_fixed",
        routing_mode="model_chosen", tool_mode="subset",
        arg_mode="typed", action_substrate="tool_call",
        skill_level="precise",
    ),
    "L4": HarnessConfig(
        plan_mode="free_text", state_mode="model_chosen",
        routing_mode="model_chosen", tool_mode="subset",
        arg_mode="typed", action_substrate="tool_call",
        skill_level="precise",
    ),
    "L5": HarnessConfig(
        plan_mode="free_text", state_mode="model_chosen",
        routing_mode="model_chosen", tool_mode="free_choice",
        arg_mode="free_form", action_substrate="tool_call",
        skill_level="precise",
    ),
    "L6": HarnessConfig(
        plan_mode="free_text", state_mode="model_chosen",
        routing_mode="model_chosen", tool_mode="free_choice",
        arg_mode="free_form", action_substrate="hybrid",
        skill_level="precise",
    ),
}

DEFAULT_RUNGS = list(LADDER.keys())

# -----------------------------------------------------------------------
# Instance path helpers
# -----------------------------------------------------------------------

def get_instance_paths(family: str) -> list[Path]:
    if family in ("finance_ecl", "legal_clause"):
        ext = "csv" if family == "finance_ecl" else "txt"
        d = _ROOT / "data" / ("finance" if family == "finance_ecl" else "legal")
        return sorted(d.glob(f"instance_*.{ext}"))
    elif family == "branching_ecl":
        from src.tasks.branching_ecl import get_instance_paths as _bp
        return _bp()
    raise ValueError(f"Unknown family: {family}")

# -----------------------------------------------------------------------
# Discard logger
# -----------------------------------------------------------------------

def log_discard(
    reason_code: str,
    rung: str,
    family: str,
    model: str,
    instance: str,
    repeat: int,
    detail: str,
) -> None:
    """Append one discard record to discards.jsonl.

    Valid reason codes (loop_eng.md §1.5):
      HTTP_429   — rate-limited
      HTTP_5XX   — server error
      CONN_RESET — connection reset
    Model/validation/timeout failures are DATA, not discards.
    """
    record = {
        "ts": time.time(),
        "reason_code": reason_code,
        "rung": rung,
        "family": family,
        "model": model,
        "instance": instance,
        "repeat": repeat,
        "detail": detail,
    }
    with open(_DISCARDS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

# -----------------------------------------------------------------------
# Load completed runs (for resumability)
# -----------------------------------------------------------------------

def load_completed(runs_path: Path) -> set[tuple]:
    """Return a set of (rung, instance_stem, repeat) already in runs.jsonl."""
    done: set[tuple] = set()
    if not runs_path.exists():
        return done
    for line in runs_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            done.add((rec["rung"], rec["instance"], rec["repeat"]))
    return done

# -----------------------------------------------------------------------
# Single-run wrapper with retry on transient errors
# -----------------------------------------------------------------------

def run_one(
    family: str,
    rung: str,
    cfg: HarnessConfig,
    instance_path: Path,
    repeat: int,
    model: str,
    max_retries: int = 2,
) -> dict[str, Any]:
    """Run one (rung, instance, repeat) cell.  Returns a result record.

    Retries on transient Bedrock errors (ThrottlingException, ServiceUnavailableException).
    Logs retried-then-discarded runs to discards.jsonl.
    """
    import botocore.exceptions
    from src.harness.configurable import get_task_definition, run_configurable
    from src.harness.llm_utils import make_llm

    for attempt in range(max_retries + 1):
        try:
            llm = make_llm(model, temperature=0.0)
            td = get_task_definition(family, instance_path=str(instance_path))
            result = run_configurable(
                family, cfg, model, repeat,
                condition=f"{rung}_T0",
                temperature=0.0,
                llm=llm,
                task_def=td,
            )
            return {
                "rung": rung,
                "family": family,
                "model": model,
                "instance": instance_path.stem,
                "repeat": repeat,
                "task_success": result.task_success,
                "adf_total": result.adf_total,
                "adf_rate": result.adf_rate,
                "steps": len(result.steps),
                "realized_T": result.realized_T,
                "error": result.error,
                "total_tokens": result.tokens.get("total_tokens", 0),
                "latency_ms": result.latency_ms,
                "run_id": result.run_id,
            }

        except botocore.exceptions.ClientError as e:
            code = e.response["Error"]["Code"]
            if code in ("ThrottlingException", "ServiceUnavailableException",
                        "RequestTimeout"):
                if attempt < max_retries:
                    wait = 2 ** (attempt + 1)
                    print(f"    [{rung}/{instance_path.stem}/r{repeat}] "
                          f"Transient {code}, retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                # Exhausted retries — discard
                reason = "HTTP_5XX" if "ServiceUnavailable" in code else "HTTP_429"
                log_discard(reason, rung, family, model,
                            instance_path.stem, repeat, str(e))
                return {
                    "rung": rung, "family": family, "model": model,
                    "instance": instance_path.stem, "repeat": repeat,
                    "task_success": None, "error": f"DISCARDED:{code}",
                    "adf_total": None, "adf_rate": None, "steps": 0,
                    "realized_T": 0, "total_tokens": 0, "latency_ms": 0,
                    "run_id": None,
                }
            raise  # non-transient error → propagate as DATA

# -----------------------------------------------------------------------
# Pilot runner
# -----------------------------------------------------------------------

def run_pilot(
    model: str,
    family: str,
    rungs: list[str],
    n: int,
    dry_run: bool = False,
) -> None:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    runs_path = _RESULTS_DIR / "runs.jsonl"
    done = load_completed(runs_path)

    instances = get_instance_paths(family)
    if not instances:
        raise FileNotFoundError(
            f"No instances for {family}. Run the appropriate gen_*_instances.py first."
        )

    # Use the first `n` instances (cycling if fewer than n available)
    inst_pool = [instances[i % len(instances)] for i in range(n)]

    total_planned = len(rungs) * n
    total_done    = sum(
        1 for rung in rungs for i, inst in enumerate(inst_pool)
        if (rung, inst.stem, i) in done
    )

    print(f"\nPilot: model={model}  family={family}  rungs={rungs}  N={n}")
    print(f"Planned={total_planned}  Already done={total_done}  "
          f"Remaining={total_planned - total_done}")
    if dry_run:
        print("DRY RUN — no API calls.")
        return

    from src.adf.metric import compute_adf, FINANCE_ECL
    from src.tasks.branching_ecl import get_instance_paths as _bp
    import dataclasses

    run_count = 0
    for rung in rungs:
        cfg = LADDER[rung]
        rung_results = []
        print(f"\n  Rung {rung}:")

        for rep, inst in enumerate(inst_pool):
            key = (rung, inst.stem, rep)
            if key in done:
                print(f"    [{inst.stem}/r{rep}] skip (already done)")
                continue

            rec = run_one(family, rung, cfg, inst, rep, model)
            status = "✅" if rec["task_success"] else "❌"
            toks   = rec["total_tokens"]
            ms     = int(rec["latency_ms"])
            print(f"    [{inst.stem}/r{rep}] {status}  tok={toks}  {ms}ms  err={rec['error']}")

            # Append to runs.jsonl
            with open(runs_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

            rung_results.append(rec)
            run_count += 1

        # Print rung summary
        successes = [r for r in rung_results if r["task_success"]]
        if rung_results:
            tsr = len(successes) / len(rung_results)
            avg_tok = sum(r["total_tokens"] for r in rung_results) / len(rung_results)
            adf_rate = rung_results[0]["adf_rate"]
            print(f"    → TSR={tsr:.1%} ({len(successes)}/{len(rung_results)})  "
                  f"ADF_rate={adf_rate:.3f}  avg_tok={avg_tok:.0f}")

    print(f"\nPilot complete: {run_count} new runs.  Results in {runs_path}")
    _write_summary(runs_path)

def _write_summary(runs_path: Path) -> None:
    """Write per-rung summary statistics to results/pilot/summary.json."""
    import collections
    records = [json.loads(l) for l in runs_path.read_text().splitlines() if l.strip()]

    by_rung: dict[str, list] = collections.defaultdict(list)
    for rec in records:
        by_rung[rec["rung"]].append(rec)

    summary = {}
    for rung, recs in sorted(by_rung.items()):
        valid = [r for r in recs if r["task_success"] is not None]
        successes = [r for r in valid if r["task_success"]]
        tsr   = len(successes) / len(valid) if valid else None
        avg_tok = sum(r["total_tokens"] for r in valid) / len(valid) if valid else None
        summary[rung] = {
            "n_runs": len(recs),
            "n_valid": len(valid),
            "n_success": len(successes),
            "tsr": round(tsr, 4) if tsr is not None else None,
            "adf_rate": recs[0]["adf_rate"] if recs else None,
            "adf_total": recs[0]["adf_total"] if recs else None,
            "avg_total_tokens": round(avg_tok, 1) if avg_tok else None,
        }

    out_path = runs_path.parent / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary written to {out_path}")
    print("\nPer-rung TSR:")
    for rung, s in summary.items():
        bar = "█" * int((s["tsr"] or 0) * 20)
        print(f"  {rung:8s} ADF={s['adf_rate']:.3f}  "
              f"TSR={s['tsr']:.1%} ({s['n_success']}/{s['n_valid']})  {bar}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Phase 6.1 pilot")
    parser.add_argument("--model",  default="amazon.nova-micro-v1:0")
    parser.add_argument("--family", default="finance_ecl")
    parser.add_argument("--n",      type=int, default=10)
    parser.add_argument("--rungs",  default=",".join(DEFAULT_RUNGS),
                        help="Comma-separated rung names, e.g. L0prime,L1,L3")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rungs = [r.strip() for r in args.rungs.split(",") if r.strip()]
    run_pilot(
        model=args.model,
        family=args.family,
        rungs=rungs,
        n=args.n,
        dry_run=args.dry_run,
    )

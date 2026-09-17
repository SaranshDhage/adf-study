"""Determinism Index computation — Phase 7 analysis.

Reads per-run trace files (logs/raw/*.jsonl) and computes the four DI
components from the prior paper (docs/determinism_index.md):
  - Plan Stability (PS)
  - Tool Path Consistency (TPC)
  - State Transition Stability (STS)
  - Output Consistency (OC)

DI = equal-weighted mean of PS, TPC, STS, OC  ∈ [0, 1]

Each component is a normalised edit-similarity averaged over all C(N,2)
pairwise run comparisons within a cell.  All four components are reported
separately — the prior paper found plan-text artifact from using composite DI
alone.

Also computes Reproducibility Rate (RR): fraction of run pairs whose full
trace signature (state sequence + tool sequence + args hashes + output hash)
is identical.

Usage:
    python -m src.analysis.compute_di --results results/grid/runs.jsonl
"""
from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any, Optional

_ROOT   = Path(__file__).resolve().parents[2]
_LOGS   = _ROOT / "logs" / "raw"
_GRID   = _ROOT / "results" / "grid"


# ---------------------------------------------------------------------------
# Trace loading
# ---------------------------------------------------------------------------

def load_run_trace(run_id: str) -> list[dict]:
    """Load the JSONL trace for one run.  Returns [] if the file is missing."""
    p = _LOGS / f"{run_id}.jsonl"
    if not p.exists():
        return []
    events = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def extract_plan(events: list[dict]) -> list[str]:
    for e in events:
        if e["event"] == "plan":
            return e.get("steps", [])
    return []


def extract_state_sequence(events: list[dict]) -> list[str]:
    return [e["to_state"] for e in events if e["event"] == "state_transition"]


def extract_tool_sequence(events: list[dict]) -> list[str]:
    return [e["tool_name"] for e in events if e["event"] == "tool_call"]


def extract_args_hashes(events: list[dict]) -> list[str]:
    return [e.get("args_hash", "") for e in events if e["event"] == "tool_call"]


def extract_output_hash(events: list[dict]) -> Optional[str]:
    for e in events:
        if e["event"] == "run_end":
            from src.tracing.util import hash_obj
            return hash_obj(e.get("final_output"))
    return None


# ---------------------------------------------------------------------------
# Edit-distance similarity helpers
# ---------------------------------------------------------------------------

def _lev_sim(a: list, b: list) -> float:
    """Normalised Levenshtein similarity ∈ [0,1] between two lists."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return 1.0 - dp[m] / max(n, m)


def _plan_sim(p1: list[str], p2: list[str]) -> float:
    """Similarity between two plan step-name lists."""
    return _lev_sim(p1, p2)


# ---------------------------------------------------------------------------
# Per-cell DI computation
# ---------------------------------------------------------------------------

def compute_di_for_cell(run_ids: list[str]) -> dict[str, Any]:
    """Compute DI components for a list of run IDs (one cell).

    Returns a dict with keys: ps, tpc, sts, oc, di, rr, n_pairs, n_runs_ok.
    """
    traces = []
    for rid in run_ids:
        events = load_run_trace(rid)
        if not events:
            continue
        traces.append({
            "run_id":       rid,
            "plan":         extract_plan(events),
            "states":       extract_state_sequence(events),
            "tools":        extract_tool_sequence(events),
            "args_hashes":  extract_args_hashes(events),
            "output_hash":  extract_output_hash(events),
        })

    n = len(traces)
    if n < 2:
        return {"ps": None, "tpc": None, "sts": None, "oc": None,
                "di": None, "rr": None, "n_pairs": 0, "n_runs_ok": n}

    pairs = list(combinations(range(n), 2))
    ps_vals, tpc_vals, sts_vals, oc_vals, rr_vals = [], [], [], [], []

    for i, j in pairs:
        a, b = traces[i], traces[j]

        ps_vals.append(_plan_sim(a["plan"], b["plan"]))
        tpc_vals.append(_lev_sim(a["tools"], b["tools"]))
        sts_vals.append(_lev_sim(a["states"], b["states"]))

        # Output consistency: 1 if same output hash, 0 otherwise
        oc = 1.0 if (a["output_hash"] and a["output_hash"] == b["output_hash"]) else 0.0
        oc_vals.append(oc)

        # Reproducibility: full trace signature match
        sig_a = (tuple(a["states"]), tuple(a["tools"]),
                 tuple(a["args_hashes"]), a["output_hash"])
        sig_b = (tuple(b["states"]), tuple(b["tools"]),
                 tuple(b["args_hashes"]), b["output_hash"])
        rr_vals.append(1.0 if sig_a == sig_b else 0.0)

    def _mean(lst):
        return sum(lst) / len(lst) if lst else None

    ps  = _mean(ps_vals)
    tpc = _mean(tpc_vals)
    sts = _mean(sts_vals)
    oc  = _mean(oc_vals)
    di  = _mean([ps, tpc, sts, oc]) if all(v is not None for v in [ps, tpc, sts, oc]) else None

    return {
        "ps": ps, "tpc": tpc, "sts": sts, "oc": oc,
        "di": di,
        "rr": _mean(rr_vals),
        "n_pairs": len(pairs),
        "n_runs_ok": n,
    }


# ---------------------------------------------------------------------------
# Main analysis entry point
# ---------------------------------------------------------------------------

def compute_di_table(runs_path: Path) -> dict[str, dict]:
    """Compute DI for every cell in runs_path.

    Crucially: pairwise DI is computed WITHIN instance, then averaged across
    instances.  Cross-instance pairs are excluded from DI computation because
    different instances legitimately produce different outputs (not a
    non-determinism signal).  This matches the prior paper's design.

    Returns {cell_key: metrics} where metrics include:
      di_within, ps, tpc, sts, oc, rr (all within-instance),
      n_instances_with_repeats (instances that have >=2 runs in this cell),
      tsr (task success rate across all runs).
    """
    import collections
    records = [json.loads(l) for l in runs_path.read_text().splitlines() if l.strip()]

    # Group by (cell, instance) to find within-instance run sets
    by_cell_instance: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    for rec in records:
        cell = f"{rec['family']}|{rec['model']}|{rec['rung']}"
        inst = rec.get("instance", "")
        if rec.get("run_id"):
            by_cell_instance[cell][inst].append(rec["run_id"])

    # Also compute TSR across all runs
    by_cell_all: dict = collections.defaultdict(list)
    for rec in records:
        cell = f"{rec['family']}|{rec['model']}|{rec['rung']}"
        by_cell_all[cell].append(rec)

    results = {}
    for cell_key in sorted(set(list(by_cell_instance.keys()) + list(by_cell_all.keys()))):
        # TSR
        all_recs = by_cell_all.get(cell_key, [])
        valid = [r for r in all_recs if r.get("task_success") is not None]
        succ  = [r for r in valid if r.get("task_success")]
        tsr   = len(succ) / len(valid) if valid else None

        # Within-instance DI: compute per-instance, average
        inst_di_vals = []
        inst_ps_vals, inst_tpc_vals, inst_sts_vals = [], [], []
        inst_oc_vals, inst_rr_vals = [], []
        n_insts_with_repeats = 0

        for inst, run_ids in by_cell_instance.get(cell_key, {}).items():
            if len(run_ids) < 2:
                continue
            n_insts_with_repeats += 1
            m = compute_di_for_cell(run_ids)
            if m["di"] is not None:
                inst_di_vals.append(m["di"])
                inst_ps_vals.append(m["ps"])
                inst_tpc_vals.append(m["tpc"])
                inst_sts_vals.append(m["sts"])
                inst_oc_vals.append(m["oc"])
                inst_rr_vals.append(m["rr"])

        def _mean(lst):
            return sum(lst) / len(lst) if lst else None

        parts = cell_key.split("|")
        results[cell_key] = {
            "family":    parts[0],
            "model":     parts[1],
            "rung":      parts[2],
            "tsr":       round(tsr, 4) if tsr is not None else None,
            "n_total":   len(all_recs),
            "n_valid":   len(valid),
            "adf_rate":  all_recs[0].get("adf_rate") if all_recs else None,
            # Within-instance DI
            "di":        _mean(inst_di_vals),
            "ps":        _mean(inst_ps_vals),
            "tpc":       _mean(inst_tpc_vals),
            "sts":       _mean(inst_sts_vals),
            "oc":        _mean(inst_oc_vals),
            "rr":        _mean(inst_rr_vals),
            "n_instances_with_repeats": n_insts_with_repeats,
        }

    return results


if __name__ == "__main__":
    import argparse, sys
    sys.path.insert(0, str(_ROOT))

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(_GRID / "runs.jsonl"))
    args = parser.parse_args()

    runs_path = Path(args.results)
    if not runs_path.exists():
        print(f"No runs file at {runs_path}")
        sys.exit(1)

    table = compute_di_table(runs_path)

    # Print table
    print(f"\n{'Cell':50s}  {'DI':5s} {'PS':5s} {'TPC':5s} {'STS':5s} {'OC':5s} {'RR':5s} N")
    print("-" * 85)
    for cell_key, m in sorted(table.items()):
        if m["di"] is None:
            continue
        fam, mod, rung = m["family"], m["model"].split(".")[-1][:15], m["rung"]
        label = f"{fam}|{mod}|{rung}"
        print(f"{label:50s}  "
              f"{m['di']:.3f} "
              f"{m['ps']:.3f} "
              f"{m['tpc']:.3f} "
              f"{m['sts']:.3f} "
              f"{m['oc']:.3f} "
              f"{m['rr']:.3f} "
              f"{m['n_runs_ok']}")

    # Save
    out = runs_path.parent / "di_table.json"
    out.write_text(json.dumps(table, indent=2))
    print(f"\nSaved to {out}")

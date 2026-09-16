"""Full grid runner — Phase 6.2.

Usage:
    python -m src.harness.run_grid [--models m1,m2] [--families f1,f2]
                                   [--rungs L0prime,...] [--n N] [--dry-run]

Defaults: all 4 roster models, finance_ecl + branching_ecl, all 8 rungs, N=50.

Outputs:
  results/grid/runs.jsonl          — one record per run (append, resumable)
  results/grid/summary.json        — per-cell aggregates (rewritten each batch)
  discards.jsonl                   — dropped runs with reason codes

Diagnostic loop (loop_eng.md §8):
  After every rung×family×model batch, runs the diagnostic checklist and
  prints warnings. Does NOT stop on warnings; logs them for the analysis phase.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_DIR = _ROOT / "results" / "grid"
_DISCARDS_PATH = _ROOT / "discards.jsonl"

from src.adf.metric import HarnessConfig
from src.harness.run_pilot import (
    LADDER,
    get_instance_paths,
    log_discard,
    run_one,
    _write_summary as _write_pilot_summary,
)

DEFAULT_MODELS = [
    "amazon.nova-micro-v1:0",
    "google.gemma-3-27b-it",
    "qwen.qwen3-32b-v1:0",
    "amazon.nova-pro-v1:0",
]
DEFAULT_FAMILIES = ["finance_ecl", "branching_ecl"]
DEFAULT_RUNGS    = list(LADDER.keys())   # L0prime … L6
DEFAULT_N        = 50
TEMPERATURE      = 0.0

# H5 substrate-equivalence pair
H5_CFG = HarnessConfig(
    plan_mode="free_text", state_mode="model_chosen",
    routing_mode="model_chosen", tool_mode="free_choice",
    arg_mode="free_form", action_substrate="codegen",
    skill_level="precise",
)


def load_completed(runs_path: Path) -> set[tuple]:
    done: set[tuple] = set()
    if not runs_path.exists():
        return done
    for line in runs_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            done.add((rec["family"], rec["model"], rec["rung"], rec["instance"], rec["repeat"]))
    return done


def _write_summary(runs_path: Path) -> None:
    import collections
    records = [json.loads(l) for l in runs_path.read_text().splitlines() if l.strip()]
    by_cell: dict = collections.defaultdict(list)
    for rec in records:
        key = (rec["family"], rec["model"], rec["rung"])
        by_cell[key].append(rec)

    summary = {}
    for (fam, mod, rung), recs in sorted(by_cell.items()):
        valid = [r for r in recs if r["task_success"] is not None]
        succ  = [r for r in valid if r["task_success"]]
        tsr   = len(succ) / len(valid) if valid else None
        avg_tok = sum(r["total_tokens"] for r in valid) / len(valid) if valid else None
        key = f"{fam}|{mod}|{rung}"
        summary[key] = {
            "family": fam, "model": mod, "rung": rung,
            "n_runs": len(recs), "n_valid": len(valid),
            "n_success": len(succ),
            "tsr": round(tsr, 4) if tsr is not None else None,
            "adf_rate": recs[0].get("adf_rate"),
            "adf_total": recs[0].get("adf_total"),
            "avg_total_tokens": round(avg_tok, 1) if avg_tok else None,
        }

    out = _RESULTS_DIR / "summary.json"
    out.write_text(json.dumps(summary, indent=2))

    # Print progress table
    print("\n  Grid progress:")
    print(f"  {'Family':15s} {'Model':30s} {'Rung':8s} {'ADF':6s} {'TSR':6s} N")
    for (fam, mod, rung), recs in sorted(by_cell.items()):
        valid = [r for r in recs if r["task_success"] is not None]
        succ  = [r for r in valid if r["task_success"]]
        tsr   = len(succ) / len(valid) if valid else float("nan")
        short_mod = mod.split(".")[-1].split("-v1")[0][:25]
        print(f"  {fam:15s} {short_mod:30s} {rung:8s} "
              f"{recs[0].get('adf_rate',0):.3f}  {tsr:.0%} {len(valid)}")


def _diagnostic_check(runs_path: Path, family: str, model: str) -> None:
    """Loop_eng.md §8.1 diagnostic checklist — prints warnings, never stops."""
    import collections
    records = [json.loads(l) for l in runs_path.read_text().splitlines() if l.strip()
               if json.loads(l)["family"] == family and json.loads(l)["model"] == model]

    by_rung: dict = collections.defaultdict(list)
    for rec in records:
        by_rung[rec["rung"]].append(rec)

    warnings = []

    # Saturation check
    for rung, recs in by_rung.items():
        valid = [r for r in recs if r["task_success"] is not None]
        if not valid: continue
        tsr = sum(r["task_success"] for r in valid) / len(valid)
        if tsr >= 1.0:
            warnings.append(f"  ⚠ SATURATED-HIGH: {rung} TSR=100% — no discriminating power")
        elif tsr <= 0.0:
            warnings.append(f"  ⚠ SATURATED-LOW: {rung} TSR=0% — no discriminating power")

    # Discard cluster check
    discards_path = _ROOT / "discards.jsonl"
    if discards_path.exists():
        disc = [json.loads(l) for l in discards_path.read_text().splitlines() if l.strip()
                if json.loads(l).get("family") == family and json.loads(l).get("model") == model]
        if len(disc) > 5:
            warnings.append(f"  ⚠ DISCARD CLUSTER: {len(disc)} discards for {family}/{model}")

    if warnings:
        print(f"\n  DIAGNOSTICS ({family} / {model}):")
        for w in warnings:
            print(w)


def run_grid(
    models: list[str],
    families: list[str],
    rungs: list[str],
    n: int,
    dry_run: bool = False,
    include_h5: bool = True,
) -> None:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    runs_path = _RESULTS_DIR / "runs.jsonl"
    done = load_completed(runs_path)

    # Build H5 rung list if requested
    all_rungs = list(rungs)
    h5_rung = "L5_codegen"   # H5 counterpart to L5

    total_planned = len(families) * len(models) * len(all_rungs) * n
    if include_h5:
        total_planned += len(families) * len(models) * n

    print(f"\nFull grid: {len(families)} families × {len(models)} models × "
          f"{len(all_rungs)} rungs × N={n}")
    if include_h5:
        print(f"  + H5 substrate pair (L5-codegen vs L5) × {len(families)} families × "
              f"{len(models)} models × N={n}")
    print(f"Total planned: {total_planned} runs")
    print(f"Already done:  {len(done)} runs")
    if dry_run:
        print("DRY RUN — no API calls.")
        return

    for family in families:
        instances = get_instance_paths(family)
        if not instances:
            print(f"SKIP {family}: no instances found")
            continue
        inst_pool = [instances[i % len(instances)] for i in range(n)]

        for model in models:
            print(f"\n{'='*60}")
            print(f"  {family} × {model}")
            print(f"{'='*60}")

            # Main rungs
            for rung in all_rungs:
                cfg = LADDER[rung]
                rung_results = []
                print(f"\n  [{rung}]")

                for rep, inst in enumerate(inst_pool):
                    key = (family, model, rung, inst.stem, rep)
                    if key in done:
                        continue
                    rec = run_one(family, rung, cfg, inst, rep, model)
                    status = "✅" if rec["task_success"] else "❌"
                    print(f"    [{inst.stem}/r{rep}] {status}  "
                          f"tok={rec['total_tokens']}  {int(rec['latency_ms'])}ms  "
                          f"err={str(rec['error'])[:60] if rec['error'] else ''}")
                    with open(runs_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec) + "\n")
                    rung_results.append(rec)
                    done.add(key)

                if rung_results:
                    valid = [r for r in rung_results if r["task_success"] is not None]
                    succ = [r for r in valid if r["task_success"]]
                    tsr = len(succ)/len(valid) if valid else 0
                    avg_tok = sum(r["total_tokens"] for r in valid)/len(valid) if valid else 0
                    adf = rung_results[0]["adf_rate"]
                    print(f"    → TSR={tsr:.0%} ({len(succ)}/{len(valid)})  "
                          f"ADF={adf:.3f}  avg_tok={avg_tok:.0f}")

            # H5 substrate counterpart (L5-codegen)
            if include_h5:
                rung_results = []
                print(f"\n  [L5_codegen  H5-pair]")
                for rep, inst in enumerate(inst_pool):
                    key = (family, model, h5_rung, inst.stem, rep)
                    if key in done:
                        continue
                    rec = run_one(family, h5_rung, H5_CFG, inst, rep, model)
                    # Override rung name in record
                    rec["rung"] = h5_rung
                    status = "✅" if rec["task_success"] else "❌"
                    print(f"    [{inst.stem}/r{rep}] {status}  "
                          f"tok={rec['total_tokens']}  {int(rec['latency_ms'])}ms")
                    with open(runs_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec) + "\n")
                    rung_results.append(rec)
                    done.add(key)

                if rung_results:
                    valid = [r for r in rung_results if r["task_success"] is not None]
                    succ = [r for r in valid if r["task_success"]]
                    tsr = len(succ)/len(valid) if valid else 0
                    print(f"    → TSR={tsr:.0%} ({len(succ)}/{len(valid)})  H5-codegen")

            # Diagnostic after each family×model block
            _diagnostic_check(runs_path, family, model)

        # Refresh summary after each family
        _write_summary(runs_path)

    print(f"\nGrid complete. Results in {runs_path}")
    _write_summary(runs_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",   default=",".join(DEFAULT_MODELS))
    parser.add_argument("--families", default=",".join(DEFAULT_FAMILIES))
    parser.add_argument("--rungs",    default=",".join(DEFAULT_RUNGS))
    parser.add_argument("--n",        type=int, default=DEFAULT_N)
    parser.add_argument("--no-h5",    action="store_true")
    parser.add_argument("--dry-run",  action="store_true")
    args = parser.parse_args()

    run_grid(
        models   = [m.strip() for m in args.models.split(",")],
        families = [f.strip() for f in args.families.split(",")],
        rungs    = [r.strip() for r in args.rungs.split(",")],
        n        = args.n,
        dry_run  = args.dry_run,
        include_h5 = not args.no_h5,
    )

"""Phase 4 — compute the Determinism Index and related metrics from
logs/traces.db, implementing the definitions in docs/determinism_index.md
against the real N=50 baseline-vs-harness trace data collected in Phase 1/3
(src/run_experiment.py).

Run with: python -m src.analysis.phase4_metrics
Writes results/phase4_metrics.json and prints a summary table.
"""

import itertools
import json
import sqlite3
import statistics
from pathlib import Path
from typing import Any, Callable

from src.tracing import DB_PATH

ANALYSIS_DIR = Path(__file__).resolve().parents[2] / "results"

# Default equal weighting per docs/determinism_index.md. ALT_WEIGHTS is the
# sensitivity check promised there: if it flips which condition looks more
# deterministic, that's a real finding to report, not something to paper over.
DEFAULT_WEIGHTS = {"plan": 0.25, "tools": 0.25, "states": 0.25, "output": 0.25}
ALT_WEIGHTS = {"plan": 0.15, "tools": 0.35, "states": 0.35, "output": 0.15}


def levenshtein(a: list, b: list) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m]


def normalized_edit_similarity(a: list, b: list) -> float:
    """docs/determinism_index.md §1-3: 1 - levenshtein(a,b) / max(len(a), len(b))."""
    if not a and not b:
        return 1.0
    dist = levenshtein(a, b)
    return 1.0 - dist / max(len(a), len(b))


def field_match_ratio(a: Any, b: Any) -> float:
    """docs/determinism_index.md §4: fraction of top-level keys with equal
    values, recursing into nested dicts. None-vs-None counts as a match
    (both runs equally failed to produce output); None-vs-dict does not."""
    if a is None and b is None:
        return 1.0
    if a is None or b is None:
        return 0.0
    if not isinstance(a, dict) or not isinstance(b, dict):
        return 1.0 if a == b else 0.0
    keys = set(a.keys()) | set(b.keys())
    if not keys:
        return 1.0
    scores = []
    for k in keys:
        if k not in a or k not in b:
            scores.append(0.0)
        elif isinstance(a[k], dict) and isinstance(b[k], dict):
            scores.append(field_match_ratio(a[k], b[k]))
        else:
            scores.append(1.0 if a[k] == b[k] else 0.0)
    return sum(scores) / len(scores)


def mean_pairwise(items: list, sim_fn: Callable[[Any, Any], float]) -> float:
    n = len(items)
    if n < 2:
        return 1.0
    total = sum(sim_fn(items[i], items[j]) for i, j in itertools.combinations(range(n), 2))
    return total / (n * (n - 1) / 2)


def full_trace_signature(run: dict) -> str:
    return json.dumps(
        {"plan": run["plan"], "tools": run["tools"], "states": run["states"], "output": run["output"]},
        sort_keys=True,
        default=str,
    )


def reproducibility_rate(runs: list[dict]) -> float:
    """docs/determinism_index.md: strict headline metric — fraction of runs
    matching the modal (most common) full trace signature exactly."""
    if not runs:
        return 0.0
    sigs = [full_trace_signature(r) for r in runs]
    mode_sig = max(set(sigs), key=sigs.count)
    return sigs.count(mode_sig) / len(sigs)


def load_group(conn: sqlite3.Connection, task: str, model: str, condition: str) -> list[dict]:
    run_ids = [
        r[0]
        for r in conn.execute(
            "SELECT run_id FROM runs WHERE task=? AND model=? AND condition=? ORDER BY run_index",
            (task, model, condition),
        )
    ]
    runs = []
    for run_id in run_ids:
        plan = [
            r[0]
            for r in conn.execute(
                "SELECT step_id FROM plan_steps WHERE run_id=? ORDER BY step_index", (run_id,)
            )
        ]
        tools = [
            r[0]
            for r in conn.execute(
                "SELECT tool_name FROM tool_calls WHERE run_id=? ORDER BY call_index", (run_id,)
            )
        ]
        states = [
            r[0]
            for r in conn.execute(
                "SELECT state_name FROM state_sequence WHERE run_id=? ORDER BY step_index", (run_id,)
            )
        ]
        final_output_json, success, total_tokens, latency_ms = conn.execute(
            "SELECT final_output_json, success, total_tokens, latency_ms FROM runs WHERE run_id=?",
            (run_id,),
        ).fetchone()
        runs.append(
            {
                "run_id": run_id,
                "plan": plan,
                "tools": tools,
                "states": states,
                "output": json.loads(final_output_json) if final_output_json else None,
                "success": bool(success),
                "total_tokens": total_tokens or 0,
                "latency_ms": latency_ms or 0.0,
            }
        )
    return runs


def determinism_index(metrics: dict, weights: dict) -> float:
    return (
        weights["plan"] * metrics["plan_stability"]
        + weights["tools"] * metrics["tool_path_consistency"]
        + weights["states"] * metrics["state_transition_stability"]
        + weights["output"] * metrics["output_consistency"]
    )


def compute_group_metrics(runs: list[dict]) -> dict:
    n = len(runs)
    metrics = {
        "n": n,
        "task_success_rate": sum(r["success"] for r in runs) / n if n else 0.0,
        "reproducibility_rate": reproducibility_rate(runs),
        "plan_stability": mean_pairwise([r["plan"] for r in runs], normalized_edit_similarity),
        "tool_path_consistency": mean_pairwise([r["tools"] for r in runs], normalized_edit_similarity),
        "state_transition_stability": mean_pairwise(
            [r["states"] for r in runs], normalized_edit_similarity
        ),
        "output_consistency": mean_pairwise([r["output"] for r in runs], field_match_ratio),
        "avg_total_tokens": sum(r["total_tokens"] for r in runs) / n if n else 0.0,
        # Mean latency is noisy under a shared third-party API — a single
        # provider-side slowness spike can be 50-100x a typical run (observed
        # directly: one legal_clause/harness run took 637s with a normal
        # token count, vs. a median around 9s). Median is the honest
        # central-tendency stat here; avg is kept for reference/total-cost
        # framing, not as the headline latency number.
        "avg_latency_ms": sum(r["latency_ms"] for r in runs) / n if n else 0.0,
        "median_latency_ms": statistics.median(r["latency_ms"] for r in runs) if n else 0.0,
    }
    metrics["determinism_index"] = determinism_index(metrics, DEFAULT_WEIGHTS)
    metrics["determinism_index_alt_weights"] = determinism_index(metrics, ALT_WEIGHTS)
    return metrics


def main() -> dict:
    conn = sqlite3.connect(DB_PATH)
    groups = conn.execute(
        "SELECT DISTINCT task, model, condition FROM runs ORDER BY task, model, condition"
    ).fetchall()

    results = {}
    for task, model, condition in groups:
        runs = load_group(conn, task, model, condition)
        results[f"{task}::{model}::{condition}"] = compute_group_metrics(runs)

    ANALYSIS_DIR.mkdir(exist_ok=True)
    (ANALYSIS_DIR / "phase4_metrics.json").write_text(json.dumps(results, indent=2))

    header = (
        f"{'group':55} {'N':>3} {'success':>8} {'repro':>7} {'PS':>6} {'TPC':>6} "
        f"{'STS':>6} {'OC':>6} {'DI':>6} {'DI_alt':>7} {'tokens':>8} {'med_lat':>8} {'avg_lat':>8}"
    )
    print(header)
    for key, m in results.items():
        print(
            f"{key:55} {m['n']:>3} {m['task_success_rate']:>8.2f} {m['reproducibility_rate']:>7.3f} "
            f"{m['plan_stability']:>6.3f} {m['tool_path_consistency']:>6.3f} "
            f"{m['state_transition_stability']:>6.3f} {m['output_consistency']:>6.3f} "
            f"{m['determinism_index']:>6.3f} {m['determinism_index_alt_weights']:>7.3f} "
            f"{m['avg_total_tokens']:>8.0f} {m['median_latency_ms']:>8.0f} {m['avg_latency_ms']:>8.0f}"
        )

    return results


if __name__ == "__main__":
    main()

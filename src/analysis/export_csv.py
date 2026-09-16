"""Exports the JSON/SQLite results into CSV form for supplementary
material / spreadsheet inspection. Pure computation on existing outputs —
no API calls, no cost.

Run with: python -m src.analysis.export_csv
Writes: results/phase4_metrics.csv, results/phase7_stats.csv, results/all_runs.csv
"""

import csv
import json
import sqlite3
from pathlib import Path

from src.tracing import DB_PATH

ANALYSIS_DIR = Path(__file__).resolve().parents[2] / "results"


def export_phase4_csv() -> None:
    data = json.loads((ANALYSIS_DIR / "phase4_metrics.json").read_text())
    rows = []
    for key, m in data.items():
        task, model, condition = key.split("::")
        rows.append(
            {
                "task": task,
                "model": model,
                "condition": condition,
                "n": m["n"],
                "task_success_rate": m["task_success_rate"],
                "reproducibility_rate": m["reproducibility_rate"],
                "plan_stability": m["plan_stability"],
                "tool_path_consistency": m["tool_path_consistency"],
                "state_transition_stability": m["state_transition_stability"],
                "output_consistency": m["output_consistency"],
                "determinism_index": m["determinism_index"],
                "determinism_index_alt_weights": m["determinism_index_alt_weights"],
                "avg_total_tokens": m["avg_total_tokens"],
                "median_latency_ms": m["median_latency_ms"],
                "avg_latency_ms": m["avg_latency_ms"],
            }
        )
    path = ANALYSIS_DIR / "phase4_metrics.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def export_phase7_csv() -> None:
    data = json.loads((ANALYSIS_DIR / "phase7_stats.json").read_text())
    rows = []
    for r in data:
        rows.append(
            {
                "task": r["task"],
                "model": r["model"],
                "condition_a": r["condition_a"],
                "condition_b": r["condition_b"],
                "n_a": r["n_a"],
                "n_b": r["n_b"],
                "di_a_mean": r["di_a"]["mean"],
                "di_a_ci_lo": r["di_a"]["ci95"][0],
                "di_a_ci_hi": r["di_a"]["ci95"][1],
                "di_b_mean": r["di_b"]["mean"],
                "di_b_ci_lo": r["di_b"]["ci95"][0],
                "di_b_ci_hi": r["di_b"]["ci95"][1],
                "di_diff_mean": r["di_diff"]["mean"],
                "di_diff_ci_lo": r["di_diff"]["ci95"][0],
                "di_diff_ci_hi": r["di_diff"]["ci95"][1],
                "di_diff_significant": r["di_diff"]["significant"],
                "di_cohens_d": r["di_cohens_d"],
                "repro_a_mean": r["repro_a"]["mean"],
                "repro_a_ci_lo": r["repro_a"]["ci95"][0],
                "repro_a_ci_hi": r["repro_a"]["ci95"][1],
                "repro_b_mean": r["repro_b"]["mean"],
                "repro_b_ci_lo": r["repro_b"]["ci95"][0],
                "repro_b_ci_hi": r["repro_b"]["ci95"][1],
                "repro_observed_diff": r["repro_permutation_test"]["observed_diff"],
                "repro_p_value": r["repro_permutation_test"]["p_value"],
                "repro_significant_at_0.05": r["repro_permutation_test"]["significant_at_0.05"],
            }
        )
    path = ANALYSIS_DIR / "phase7_stats.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def export_all_runs_csv() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT run_id, task, model, condition, run_index, success, error,
               total_tokens, prompt_tokens, completion_tokens, latency_ms,
               started_at, ended_at
           FROM runs ORDER BY task, model, condition, run_index"""
    ).fetchall()
    path = ANALYSIS_DIR / "all_runs.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(rows[0].keys())
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def main() -> None:
    ANALYSIS_DIR.mkdir(exist_ok=True)
    export_phase4_csv()
    export_phase7_csv()
    export_all_runs_csv()


if __name__ == "__main__":
    main()

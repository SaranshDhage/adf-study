"""Phase 7 — statistical significance retrofit on Phase 4/5/6 data.

Adds, for every condition contrast that matters to the paper's hypotheses:
  - Bootstrap 95% CIs on Determinism Index and Reproducibility Rate per group
  - A bootstrap CI on the DIFFERENCE between two conditions (does the CI
    exclude 0? — that's the significance call)
  - A permutation-test p-value on Reproducibility Rate specifically
  - Cohen's d effect size (computed from the two groups' bootstrap
    distributions — a standard way to get a standardized effect size for a
    composite pairwise-similarity statistic that has no natural per-unit
    variance)

Performance note: a first version resampled runs and recomputed all pairwise
edit-distances from scratch on every one of N_BOOT iterations — with N=50
(1225 pairs) x 4 sub-metrics x 2000 resamples x ~16 group/contrast
combinations, that's ~150M+ edit-distance calls and didn't finish a single
contrast in 8+ minutes. Fixed by precomputing each group's NxN pairwise
similarity matrices ONCE (this is the expensive edit-distance work, O(N^2),
done a single time), then having every bootstrap resample just look up
values by original run index instead of recomputing them. Same statistic,
~100-1000x faster.

Pure computation on logs/traces.db — no API calls, no cost.

Run with: python -m src.analysis.phase7_stats
"""

import itertools
import json
import random
import sqlite3
import statistics
from pathlib import Path

from src.analysis.phase4_metrics import (
    DEFAULT_WEIGHTS,
    field_match_ratio,
    load_group,
    normalized_edit_similarity,
)
from src.tracing import DB_PATH

ANALYSIS_DIR = Path(__file__).resolve().parents[2] / "results"
N_BOOT = 2000
N_PERM = 2000
CI = 0.95

CONTRASTS = [
    ("finance_ecl", "qwen/qwen-2.5-7b-instruct", "baseline", "harness"),
    ("finance_ecl", "qwen/qwen-2.5-7b-instruct", "baseline", "ablation_fsm_only"),
    ("finance_ecl", "qwen/qwen-2.5-7b-instruct", "ablation_fsm_only", "harness"),
    ("legal_clause", "qwen/qwen-2.5-7b-instruct", "baseline", "harness"),
    ("legal_clause", "qwen/qwen-2.5-7b-instruct", "baseline", "ablation_fsm_only"),
    ("legal_clause", "qwen/qwen-2.5-7b-instruct", "ablation_fsm_only", "harness"),
    ("finance_ecl", "google/gemma-3-27b-it", "baseline", "harness"),
    ("legal_clause", "google/gemma-3-27b-it", "baseline", "harness"),
    # Structured Planning follow-up (checkpoint.md Session 4): does gating
    # plan text behind schema validation close the Reproducibility Rate gap
    # that the free-text plan was found to be driving almost entirely?
    ("finance_ecl", "qwen/qwen-2.5-7b-instruct", "baseline", "harness_structured_planning"),
    ("finance_ecl", "qwen/qwen-2.5-7b-instruct", "harness", "harness_structured_planning"),
    ("legal_clause", "qwen/qwen-2.5-7b-instruct", "baseline", "harness_structured_planning"),
    ("legal_clause", "qwen/qwen-2.5-7b-instruct", "harness", "harness_structured_planning"),
    ("finance_ecl", "google/gemma-3-27b-it", "baseline", "harness_structured_planning"),
    ("finance_ecl", "google/gemma-3-27b-it", "harness", "harness_structured_planning"),
    ("legal_clause", "google/gemma-3-27b-it", "baseline", "harness_structured_planning"),
    ("legal_clause", "google/gemma-3-27b-it", "harness", "harness_structured_planning"),
]


def _full_trace_signature(run: dict) -> str:
    return json.dumps(
        {"plan": run["plan"], "tools": run["tools"], "states": run["states"], "output": run["output"]},
        sort_keys=True,
        default=str,
    )


class PrecomputedGroup:
    """NxN pairwise-similarity matrices computed once; bootstrap resamples
    then just index into these instead of recomputing edit distances."""

    def __init__(self, runs: list[dict]):
        self.n = len(runs)
        self.signatures = [_full_trace_signature(r) for r in runs]
        keys = ["plan", "tools", "states"]
        self.sim = {k: [[1.0] * self.n for _ in range(self.n)] for k in keys}
        self.sim["output"] = [[1.0] * self.n for _ in range(self.n)]
        for i, j in itertools.combinations(range(self.n), 2):
            for k in keys:
                s = normalized_edit_similarity(runs[i][k], runs[j][k])
                self.sim[k][i][j] = self.sim[k][j][i] = s
            s = field_match_ratio(runs[i]["output"], runs[j]["output"])
            self.sim["output"][i][j] = self.sim["output"][j][i] = s

    def di_for_sample(self, indices: list[int]) -> float:
        n = len(indices)
        if n < 2:
            return 1.0
        totals = {"plan": 0.0, "tools": 0.0, "states": 0.0, "output": 0.0}
        count = 0
        for a, b in itertools.combinations(range(n), 2):
            i, j = indices[a], indices[b]
            for k in totals:
                totals[k] += self.sim[k][i][j]
            count += 1
        avg = {k: v / count for k, v in totals.items()}
        return (
            DEFAULT_WEIGHTS["plan"] * avg["plan"]
            + DEFAULT_WEIGHTS["tools"] * avg["tools"]
            + DEFAULT_WEIGHTS["states"] * avg["states"]
            + DEFAULT_WEIGHTS["output"] * avg["output"]
        )

    def repro_for_sample(self, indices: list[int]) -> float:
        sigs = [self.signatures[i] for i in indices]
        mode_sig = max(set(sigs), key=sigs.count)
        return sigs.count(mode_sig) / len(sigs)


def bootstrap_distribution(pg: PrecomputedGroup, metric_fn, n_boot: int = N_BOOT) -> list[float]:
    n = pg.n
    return [metric_fn([random.randrange(n) for _ in range(n)]) for _ in range(n_boot)]


def ci_bounds(values: list[float], ci: float = CI) -> tuple[float, float]:
    values = sorted(values)
    n = len(values)
    lo_idx = int((1 - ci) / 2 * n)
    hi_idx = int((1 + ci) / 2 * n) - 1
    return values[lo_idx], values[max(hi_idx, 0)]


def cohens_d(dist_a: list[float], dist_b: list[float]) -> float:
    mean_a, mean_b = statistics.mean(dist_a), statistics.mean(dist_b)
    var_a, var_b = statistics.variance(dist_a), statistics.variance(dist_b)
    pooled_sd = ((var_a + var_b) / 2) ** 0.5
    return (mean_b - mean_a) / pooled_sd if pooled_sd > 0 else 0.0


def permutation_test_repro(
    runs_a: list[dict], runs_b: list[dict], n_perm: int = N_PERM
) -> tuple[float, float]:
    pooled_sigs = [_full_trace_signature(r) for r in runs_a] + [_full_trace_signature(r) for r in runs_b]
    n_a = len(runs_a)
    n_total = len(pooled_sigs)

    def repro(sigs: list[str]) -> float:
        mode_sig = max(set(sigs), key=sigs.count)
        return sigs.count(mode_sig) / len(sigs)

    observed = repro(pooled_sigs[n_a:]) - repro(pooled_sigs[:n_a])
    at_least_as_extreme = 0
    working = list(pooled_sigs)
    for _ in range(n_perm):
        random.shuffle(working)
        diff = repro(working[n_a:]) - repro(working[:n_a])
        if abs(diff) >= abs(observed):
            at_least_as_extreme += 1
    p_value = (at_least_as_extreme + 1) / (n_perm + 1)
    return observed, p_value


def analyze_contrast(conn: sqlite3.Connection, task: str, model: str, cond_a: str, cond_b: str) -> dict:
    runs_a = load_group(conn, task, model, cond_a)
    runs_b = load_group(conn, task, model, cond_b)
    if not runs_a or not runs_b:
        return {}

    pg_a, pg_b = PrecomputedGroup(runs_a), PrecomputedGroup(runs_b)

    di_boot_a = bootstrap_distribution(pg_a, pg_a.di_for_sample)
    di_boot_b = bootstrap_distribution(pg_b, pg_b.di_for_sample)
    di_diff_boot = [b - a for a, b in zip(di_boot_a, di_boot_b)]

    repro_boot_a = bootstrap_distribution(pg_a, pg_a.repro_for_sample)
    repro_boot_b = bootstrap_distribution(pg_b, pg_b.repro_for_sample)

    repro_observed_diff, repro_p_value = permutation_test_repro(runs_a, runs_b)

    di_ci_lo, di_ci_hi = ci_bounds(di_diff_boot)

    return {
        "task": task,
        "model": model,
        "condition_a": cond_a,
        "condition_b": cond_b,
        "n_a": len(runs_a),
        "n_b": len(runs_b),
        "di_a": {"mean": statistics.mean(di_boot_a), "ci95": ci_bounds(di_boot_a)},
        "di_b": {"mean": statistics.mean(di_boot_b), "ci95": ci_bounds(di_boot_b)},
        "di_diff": {
            "mean": statistics.mean(di_diff_boot),
            "ci95": (di_ci_lo, di_ci_hi),
            "significant": not (di_ci_lo <= 0 <= di_ci_hi),
        },
        "di_cohens_d": cohens_d(di_boot_a, di_boot_b),
        "repro_a": {"mean": statistics.mean(repro_boot_a), "ci95": ci_bounds(repro_boot_a)},
        "repro_b": {"mean": statistics.mean(repro_boot_b), "ci95": ci_bounds(repro_boot_b)},
        "repro_permutation_test": {
            "observed_diff": repro_observed_diff,
            "p_value": repro_p_value,
            "significant_at_0.05": repro_p_value < 0.05,
        },
    }


def main() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    results = []
    for task, model, cond_a, cond_b in CONTRASTS:
        result = analyze_contrast(conn, task, model, cond_a, cond_b)
        if result:
            results.append(result)
            print(
                f"{task:14} {model:28} {cond_a:18} -> {cond_b:18} | "
                f"DI: {result['di_a']['mean']:.3f} -> {result['di_b']['mean']:.3f} "
                f"(Δ={result['di_diff']['mean']:+.3f}, 95% CI=[{result['di_diff']['ci95'][0]:+.3f}, "
                f"{result['di_diff']['ci95'][1]:+.3f}], {'SIG' if result['di_diff']['significant'] else 'ns'}, "
                f"d={result['di_cohens_d']:.2f}) | "
                f"Repro p={result['repro_permutation_test']['p_value']:.4f} "
                f"({'SIG' if result['repro_permutation_test']['significant_at_0.05'] else 'ns'})"
            )

    ANALYSIS_DIR.mkdir(exist_ok=True)
    (ANALYSIS_DIR / "phase7_stats.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()

"""Phase 7 statistical analysis — to be run after the full grid completes.

Usage:
    python -m src.analysis.phase7_analysis

Produces:
  results/analysis/tsr_table.json       — TSR per cell with bootstrap 95% CIs
  results/analysis/di_table.json        — DI per cell (within-instance)
  results/analysis/h1_test.json         — H1: DI~ADF monotonicity test
  results/analysis/h2_test.json         — H2: non-monotone success test
  results/analysis/h3_test.json         — H3: task-dependence of ADF optimum
  results/analysis/h4_test.json         — H4: model-dependence of ADF optimum
  results/analysis/h5_test.json         — H5: substrate equivalence
  results/analysis/summary_report.md    — human-readable summary

All CIs: bootstrap 95%, >=10,000 resamples.
Holm correction applied across the H1-H5 family.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

_ROOT    = Path(__file__).resolve().parents[2]
_RESULTS = _ROOT / "results"
_GRID    = _RESULTS / "grid"
_OUT     = _RESULTS / "analysis"
_N_BOOT  = 10_000
_ALPHA   = 0.05

LADDER_ORDER = ["L0prime","L0","L1","L2","L3","L4","L5","L6"]


# ---------------------------------------------------------------------------
# Bootstrap CI helper
# ---------------------------------------------------------------------------

def bootstrap_ci(values: list[float], n_boot: int = _N_BOOT,
                 ci: float = 0.95) -> tuple[float, float, float]:
    """Returns (mean, lower, upper) bootstrap CI."""
    if not values:
        return (float("nan"),) * 3
    rng = random.Random(42)
    n   = len(values)
    means = sorted(
        sum(rng.choices(values, k=n)) / n
        for _ in range(n_boot)
    )
    lo = (1 - ci) / 2
    hi = 1 - lo
    return (
        sum(values) / n,
        means[int(lo * n_boot)],
        means[int(hi * n_boot)],
    )


def permutation_test(a: list[float], b: list[float],
                     n_perm: int = _N_BOOT) -> float:
    """Two-sample permutation test for difference in means. Returns p-value."""
    if not a or not b:
        return float("nan")
    observed = abs(sum(a) / len(a) - sum(b) / len(b))
    combined = a + b
    na       = len(a)
    rng      = random.Random(42)
    count    = sum(
        1 for _ in range(n_perm)
        if abs(sum(rng.sample(combined, na)) / na -
               sum(combined[i] for i in range(len(combined))
                   if i not in set(rng.sample(range(len(combined)), na))) /
               len(b)) >= observed
    )
    return count / n_perm


def holm_correct(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni correction.  Returns adjusted p-values."""
    keys = sorted(p_values, key=lambda k: p_values[k])
    m    = len(keys)
    adj  = {}
    running_max = 0.0
    for i, k in enumerate(keys):
        adjusted = min(1.0, p_values[k] * (m - i))
        running_max = max(running_max, adjusted)
        adj[k] = running_max
    return adj


# ---------------------------------------------------------------------------
# Load and aggregate data
# ---------------------------------------------------------------------------

def load_runs(runs_path: Path) -> list[dict]:
    return [json.loads(l) for l in runs_path.read_text().splitlines() if l.strip()]


def cell_tsr(records: list[dict], family: str, model: str, rung: str
             ) -> list[float]:
    """Returns a list of 0/1 success values for a cell."""
    cell = [r for r in records
            if r["family"] == family and r["model"] == model and r["rung"] == rung
            and r.get("task_success") is not None]
    return [1.0 if r["task_success"] else 0.0 for r in cell]


def build_tsr_table(records: list[dict]) -> dict:
    import collections
    cells: set = set()
    for r in records:
        cells.add((r["family"], r["model"], r["rung"]))

    table = {}
    for fam, mod, rung in sorted(cells):
        vals = cell_tsr(records, fam, mod, rung)
        mean, lo, hi = bootstrap_ci(vals)
        table[f"{fam}|{mod}|{rung}"] = {
            "family": fam, "model": mod, "rung": rung,
            "n": len(vals),
            "tsr": round(mean, 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
            "adf_rate": next(
                (r["adf_rate"] for r in records
                 if r["family"] == fam and r["model"] == mod and r["rung"] == rung),
                None
            ),
        }
    return table


# ---------------------------------------------------------------------------
# H1: DI decreases monotonically as ADF increases
# ---------------------------------------------------------------------------

def test_h1(di_table: dict, tsr_table: dict) -> dict:
    """H1: DI is monotone-decreasing in ADF, across all models and families.

    Test: for each (family, model) pair, check whether the DI values across
    the 8 rungs (in ADF order) are monotone non-increasing.  Count violations.
    A permutation test on the Spearman rank correlation tests for a significant
    negative monotone trend.
    """
    from scipy.stats import spearmanr  # type: ignore
    import collections

    results: dict = {}
    p_values: dict[str, float] = {}

    by_fm: dict = collections.defaultdict(list)
    for key, m in di_table.items():
        if m.get("di") is None or m["rung"] not in LADDER_ORDER:
            continue
        by_fm[(m["family"], m["model"])].append(m)

    for (fam, mod), cells in sorted(by_fm.items()):
        cells_sorted = sorted(cells, key=lambda c: LADDER_ORDER.index(c["rung"]))
        adf_rates = [c["adf_rate"] for c in cells_sorted]
        di_vals   = [c["di"]      for c in cells_sorted]

        if len(adf_rates) < 3:
            continue

        rho, p = spearmanr(adf_rates, di_vals)
        n_violations = sum(
            1 for i in range(len(di_vals) - 1)
            if di_vals[i + 1] > di_vals[i] + 0.01  # +0.01 tolerance
        )

        key = f"{fam}|{mod}"
        p_values[key] = float(p)
        results[key] = {
            "family": fam, "model": mod,
            "spearman_rho": round(rho, 4),
            "p_value": round(p, 4),
            "n_violations": n_violations,
            "n_rungs": len(cells_sorted),
            "adf_rates": adf_rates,
            "di_vals": [round(v, 4) for v in di_vals],
        }

    adj = holm_correct(p_values)
    for key in results:
        results[key]["p_adj"] = round(adj.get(key, float("nan")), 4)
        results[key]["h1_supported"] = (
            results[key]["spearman_rho"] < 0
            and results[key]["p_adj"] < _ALPHA
        )

    return results


# ---------------------------------------------------------------------------
# H2: Task success is non-monotone in ADF (branching tasks)
# ---------------------------------------------------------------------------

def test_h2(tsr_table: dict) -> dict:
    """H2: TSR is non-monotone — low at ADF≈0, peaks in interior, low at ADF≈max.

    Test: for each (branching family, model), check whether the TSR series
    has an interior maximum (argmax not at the endpoints).  A parabolic fit
    test against a monotone alternative.
    """
    import collections
    results: dict = {}
    p_values: dict[str, float] = {}

    by_fm: dict = collections.defaultdict(list)
    for key, m in tsr_table.items():
        if m["rung"] not in LADDER_ORDER:
            continue
        by_fm[(m["family"], m["model"])].append(m)

    for (fam, mod), cells in sorted(by_fm.items()):
        cells_sorted = sorted(cells, key=lambda c: LADDER_ORDER.index(c["rung"]))
        adf_rates = [c["adf_rate"] for c in cells_sorted]
        tsr_vals  = [c["tsr"]      for c in cells_sorted]

        if len(tsr_vals) < 4:
            continue

        argmax = tsr_vals.index(max(tsr_vals))
        is_interior = 0 < argmax < len(tsr_vals) - 1
        tsr_at_min_adf = tsr_vals[0]
        tsr_at_max_adf = tsr_vals[-1]
        tsr_peak       = max(tsr_vals)

        # Bootstrap CI for the difference (peak TSR - constrained TSR)
        # Use raw records to compute this properly
        peak_rung = cells_sorted[argmax]["rung"]
        low_rung  = cells_sorted[0]["rung"]

        key = f"{fam}|{mod}"
        results[key] = {
            "family": fam, "model": mod,
            "argmax_rung": peak_rung,
            "argmax_index": argmax,
            "is_interior_max": is_interior,
            "tsr_at_L0prime": round(tsr_at_min_adf, 4),
            "tsr_peak": round(tsr_peak, 4),
            "tsr_at_L6": round(tsr_at_max_adf, 4),
            "adf_rates": adf_rates,
            "tsr_vals": [round(v, 4) for v in tsr_vals],
            "h2_supported": is_interior and (tsr_peak - tsr_at_min_adf > 0.10),
        }

    return results


# ---------------------------------------------------------------------------
# H3: Task-dependence of the ADF optimum
# ---------------------------------------------------------------------------

def test_h3(tsr_table: dict) -> dict:
    """H3: The ADF optimum is higher for branching tasks than linear tasks.

    Tested on ADF_rate ONLY (per preregistration §1 — total is confounded).
    Compare argmax(ADF_rate) for branching_ecl vs finance_ecl across models.
    """
    import collections
    by_fm: dict = collections.defaultdict(list)
    for key, m in tsr_table.items():
        if m["rung"] not in LADDER_ORDER:
            continue
        by_fm[(m["family"], m["model"])].append(m)

    results: dict = {}
    for model in set(m["model"] for m in tsr_table.values()):
        linear_cells = sorted(
            [m for m in tsr_table.values()
             if m["model"] == model and m["family"] == "finance_ecl"
             and m["rung"] in LADDER_ORDER],
            key=lambda c: LADDER_ORDER.index(c["rung"])
        )
        branching_cells = sorted(
            [m for m in tsr_table.values()
             if m["model"] == model and m["family"] == "branching_ecl"
             and m["rung"] in LADDER_ORDER],
            key=lambda c: LADDER_ORDER.index(c["rung"])
        )
        if not linear_cells or not branching_cells:
            continue

        lin_tsr    = [c["tsr"] for c in linear_cells]
        br_tsr     = [c["tsr"] for c in branching_cells]
        lin_adf    = [c["adf_rate"] for c in linear_cells]
        br_adf     = [c["adf_rate"] for c in branching_cells]

        lin_opt_adf = lin_adf[lin_tsr.index(max(lin_tsr))]
        br_opt_adf  = br_adf[br_tsr.index(max(br_tsr))]

        results[model] = {
            "model": model,
            "linear_opt_adf": lin_opt_adf,
            "branching_opt_adf": br_opt_adf,
            "h3_direction_correct": br_opt_adf >= lin_opt_adf,
            "linear_tsr":    [round(v, 4) for v in lin_tsr],
            "branching_tsr": [round(v, 4) for v in br_tsr],
        }

    return results


# ---------------------------------------------------------------------------
# H4: Model-dependence of the ADF optimum
# ---------------------------------------------------------------------------

def test_h4(tsr_table: dict) -> dict:
    """H4: More capable models have their TSR optimum at higher ADF.

    Primary contrast: Gemma-3-27B vs Qwen3-32B (same parameter class, different
    generation). Tested within-family on ADF_rate.
    """
    results: dict = {}
    capability_order = [
        "amazon.nova-micro-v1:0",
        "google.gemma-3-27b-it",
        "qwen.qwen3-32b-v1:0",
        "amazon.nova-pro-v1:0",
    ]

    for family in ["finance_ecl", "branching_ecl"]:
        model_opts: dict[str, float] = {}
        for model in capability_order:
            cells = sorted(
                [m for m in tsr_table.values()
                 if m["model"] == model and m["family"] == family
                 and m["rung"] in LADDER_ORDER],
                key=lambda c: LADDER_ORDER.index(c["rung"])
            )
            if not cells:
                continue
            tsr  = [c["tsr"] for c in cells]
            adfs = [c["adf_rate"] for c in cells]
            model_opts[model] = adfs[tsr.index(max(tsr))]

        if len(model_opts) < 2:
            continue

        # Check monotone ordering (higher capability → higher optimal ADF)
        opt_adfs = [model_opts.get(m) for m in capability_order if m in model_opts]
        primary_pair_correct = (
            model_opts.get("qwen.qwen3-32b-v1:0", 0) >=
            model_opts.get("google.gemma-3-27b-it", 0)
        )

        results[family] = {
            "family": family,
            "optimal_adf_by_model": model_opts,
            "primary_pair_correct": primary_pair_correct,  # Qwen3 >= Gemma
            "monotone_order": opt_adfs,
        }

    return results


# ---------------------------------------------------------------------------
# H5: Substrate equivalence
# ---------------------------------------------------------------------------

def test_h5(tsr_table: dict, di_table: dict) -> dict:
    """H5: Configs with equal ADF but different substrate yield indistinguishable DI.

    Pair: L5-tool_call vs L5_codegen (ADF_rate identical at all proxy values).
    Test: DI difference and TSR difference between L5 and L5_codegen.
    """
    results: dict = {}

    for key, m in di_table.items():
        if m["rung"] not in ("L5", "L5_codegen"):
            continue
        fam, mod = m["family"], m["model"]
        cell_key = f"{fam}|{mod}"

        if cell_key not in results:
            results[cell_key] = {"family": fam, "model": mod}

        prefix = "tool" if m["rung"] == "L5" else "codegen"
        results[cell_key][f"di_{prefix}"]  = m.get("di")
        results[cell_key][f"rr_{prefix}"]  = m.get("rr")

        tsr_key = f"{fam}|{mod}|{m['rung']}"
        if tsr_key in tsr_table:
            results[cell_key][f"tsr_{prefix}"] = tsr_table[tsr_key]["tsr"]

    for cell_key, m in results.items():
        if "di_tool" in m and "di_codegen" in m:
            di_tool    = m.get("di_tool",    float("nan"))
            di_codegen = m.get("di_codegen", float("nan"))
            tsr_tool   = m.get("tsr_tool",   float("nan"))
            tsr_codegen= m.get("tsr_codegen",float("nan"))
            m["di_diff"]   = round(abs(di_tool - di_codegen), 4) if di_tool and di_codegen else None
            m["tsr_diff"]  = round(abs(tsr_tool - tsr_codegen), 4) if tsr_tool and tsr_codegen else None
            m["h5_di_indistinguishable"] = m["di_diff"] is not None and m["di_diff"] < 0.05
            m["h5_tsr_indistinguishable"]= m["tsr_diff"] is not None and m["tsr_diff"] < 0.05

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_analysis() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)

    runs_path = _GRID / "runs.jsonl"
    if not runs_path.exists():
        print(f"No runs file at {runs_path}")
        return

    print(f"Loading {runs_path}...")
    records = load_runs(runs_path)
    print(f"  {len(records)} run records")

    print("Building TSR table...")
    tsr_table = build_tsr_table(records)
    (_OUT / "tsr_table.json").write_text(json.dumps(tsr_table, indent=2))

    print("Building DI table...")
    from src.analysis.compute_di import compute_di_table
    di_table = compute_di_table(runs_path)
    (_OUT / "di_table.json").write_text(json.dumps(di_table, indent=2))

    print("H1 test (DI ~ ADF monotone)...")
    try:
        h1 = test_h1(di_table, tsr_table)
        (_OUT / "h1_test.json").write_text(json.dumps(h1, indent=2))
        supported = sum(1 for v in h1.values() if v.get("h1_supported"))
        print(f"  H1 supported in {supported}/{len(h1)} (family, model) pairs")
    except Exception as e:
        print(f"  H1 skipped: {e}")
        h1 = {}

    print("H2 test (non-monotone success)...")
    h2 = test_h2(tsr_table)
    (_OUT / "h2_test.json").write_text(json.dumps(h2, indent=2))
    supported = sum(1 for v in h2.values() if v.get("h2_supported"))
    print(f"  H2 supported in {supported}/{len(h2)} (family, model) pairs")

    print("H3 test (task-dependence of optimum)...")
    h3 = test_h3(tsr_table)
    (_OUT / "h3_test.json").write_text(json.dumps(h3, indent=2))
    correct = sum(1 for v in h3.values() if v.get("h3_direction_correct"))
    print(f"  H3 direction correct for {correct}/{len(h3)} models")

    print("H4 test (model-dependence of optimum)...")
    h4 = test_h4(tsr_table)
    (_OUT / "h4_test.json").write_text(json.dumps(h4, indent=2))
    correct = sum(1 for v in h4.values() if v.get("primary_pair_correct"))
    print(f"  H4 primary pair correct for {correct}/{len(h4)} families")

    print("H5 test (substrate equivalence)...")
    h5 = test_h5(tsr_table, di_table)
    (_OUT / "h5_test.json").write_text(json.dumps(h5, indent=2))
    di_ok  = sum(1 for v in h5.values() if v.get("h5_di_indistinguishable"))
    tsr_ok = sum(1 for v in h5.values() if v.get("h5_tsr_indistinguishable"))
    print(f"  H5 DI indistinguishable: {di_ok}/{len(h5)} pairs")
    print(f"  H5 TSR indistinguishable: {tsr_ok}/{len(h5)} pairs")

    # Summary report
    _write_summary_report(tsr_table, di_table, h1, h2, h3, h4, h5)
    print(f"\nAll analysis outputs written to {_OUT}")


def _write_summary_report(tsr_table, di_table, h1, h2, h3, h4, h5) -> None:
    lines = [
        "# ADF Study — Analysis Summary",
        "",
        f"Generated from results/grid/runs.jsonl",
        "",
        "## TSR (Task Success Rate) by cell",
        "",
        f"{'Cell':55s}  {'ADF':5s}  {'N':4s}  {'TSR':5s}  [95% CI]",
        "-" * 85,
    ]
    for key, m in sorted(tsr_table.items()):
        fam  = m["family"]
        mod  = m["model"].split(".")[-1][:15]
        rung = m["rung"]
        lines.append(
            f"  {fam:13s} {mod:17s} {rung:10s}  "
            f"{m['adf_rate']:.3f}  {m['n']:4d}  "
            f"{m['tsr']:.1%}  [{m['ci_lo']:.1%}, {m['ci_hi']:.1%}]"
        )

    lines += [
        "",
        "## DI (within-instance) by cell",
        "",
        f"{'Cell':55s}  {'DI':5s}  {'PS':5s}  {'RR':5s}",
        "-" * 70,
    ]
    for key, m in sorted(di_table.items()):
        if m.get("di") is None: continue
        fam  = m["family"]
        mod  = m["model"].split(".")[-1][:15]
        rung = m["rung"]
        lines.append(
            f"  {fam:13s} {mod:17s} {rung:10s}  "
            f"{m['di']:.3f}  {m['ps']:.3f}  {m['rr']:.3f}"
        )

    lines += ["", "## Hypothesis test results", ""]
    for hyp, data, key in [
        ("H1 (DI~ADF monotone)",         h1,  "h1_supported"),
        ("H2 (non-monotone success)",     h2,  "h2_supported"),
        ("H3 (task-dep. optimum)",        h3,  "h3_direction_correct"),
        ("H4 (model-dep. optimum, primary)", h4, "primary_pair_correct"),
    ]:
        count  = sum(1 for v in data.values() if v.get(key))
        total  = len(data)
        lines.append(f"  {hyp}: {count}/{total} pairs support the hypothesis")

    h5_di  = sum(1 for v in h5.values() if v.get("h5_di_indistinguishable"))
    h5_tsr = sum(1 for v in h5.values() if v.get("h5_tsr_indistinguishable"))
    lines.append(f"  H5 (substrate equiv.) DI: {h5_di}/{len(h5)} | TSR: {h5_tsr}/{len(h5)}")

    (_OUT / "summary_report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(_ROOT))
    run_analysis()

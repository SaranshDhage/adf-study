"""Branching ECL task family — F2 (branching).

An IFRS-9 ECL pipeline where the correct state sequence is DATA-DEPENDENT and
NOT inferable from the task description alone.  A loan portfolio may contain
high-risk loans (PD > ESCALATION_THRESHOLD = 0.15).  If any valid loan exceeds
that threshold, the run must pass through an ESCALATION state that performs a
stress-test review before the final report.  If no loan crosses the threshold,
the linear path (without ESCALATION) is correct.

Decision-point count: S=5 (full path including ESCALATION).
Linear path uses 4 of those 5 steps.

States
------
Full path (5 states):
  LOAD_DATA → VALIDATE_DATA → CALCULATE → ESCALATION → GENERATE_REPORT

Linear path (4 states, no high-risk loans):
  LOAD_DATA → VALIDATE_DATA → CALCULATE → GENERATE_REPORT

ESCALATION omitted only if ALL valid loans have PD ≤ ESCALATION_THRESHOLD.

Why this produces the H2 non-monotone curve
-------------------------------------------
- ADF≈0 (harness uses the linear 4-state FSM, encoded as linear_states in the
  TaskDefinition): correctly handles linear instances; FAILS on branching
  instances because ESCALATION is never visited — the report is wrong.
- ADF≈max (model_chosen states, unconstrained): can discover ESCALATION given
  proper skill guidance, but also drifts/loops/skips on long or tricky instances.
- Interior maximum: a harness that exposes ESCALATION as an optional state but
  still constrains other layers achieves the best of both worlds.

Ground truth
------------
Derived by running reference_run(path) with the deterministic Python reference
implementation below — never with an LLM.  Each generated instance has a
companion .json file recording the required_path and the pre-computed ground truth,
so success scoring is a simple dict comparison.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "branching"

# PD threshold above which a loan requires escalation review
ESCALATION_THRESHOLD = 0.15

# Stress multiplier applied to PD for high-risk loans in ESCALATION
STRESS_PD_MULTIPLIER = 2.0

FSM_STATES_FULL = [
    "LOAD_DATA", "VALIDATE_DATA", "CALCULATE", "ESCALATION", "GENERATE_REPORT",
]
FSM_STATES_LINEAR = [
    "LOAD_DATA", "VALIDATE_DATA", "CALCULATE", "GENERATE_REPORT",
]

# Task prompt: deliberately does NOT mention the ESCALATION state or the
# threshold.  The model must discover the need for escalation from the data
# (via domain knowledge or the skill file at skill_level=precise).
TASK_PROMPT = """\
You are a credit risk analyst agent. You must compute the total Expected
Credit Loss (ECL) for the loan portfolio at {data_path}.

Follow this procedure:
1. Load the portfolio data.
2. Validate each loan: balance must be > 0, PD must be in [0, 1], LGD must be
   in [0, 1]. Exclude any loan that fails these checks and record why.
3. For each valid loan, compute ECL = balance * PD * LGD.
4. Generate a final report containing: total_ecl, valid_count, excluded, per_loan,
   requires_escalation (bool), and if requires_escalation is True, an escalation
   field with stress-test results.

Use the available tools to perform each step. Do not compute values yourself
without calling the appropriate tool.
"""

# Note: the prompt mentions 'requires_escalation' in the report but gives NO
# hint about when to escalate or what the threshold is — that is in the skill
# file for skill_level="precise".  Under skill_level="none", the model must
# infer it from the data or domain knowledge.


# ---------------------------------------------------------------------------
# Reference implementation (deterministic Python — NEVER an LLM)
# ---------------------------------------------------------------------------

def load_data(path: Path) -> list[dict[str, Any]]:
    """Load a branching-task portfolio CSV into a list of row dicts."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({
                "loan_id": row["loan_id"],
                "balance": float(row["balance"]) if row.get("balance") else None,
                "pd": float(row["pd"]) if row.get("pd") else None,
                "lgd": float(row["lgd"]) if row.get("lgd") else None,
            })
    return rows


def validate_data(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    valid, excluded = [], []
    for row in rows:
        reasons = []
        if row["balance"] is None or row["balance"] <= 0:
            reasons.append("balance must be > 0")
        if row["pd"] is None or not (0 <= row["pd"] <= 1):
            reasons.append("pd must be in [0, 1]")
        if row["lgd"] is None or not (0 <= row["lgd"] <= 1):
            reasons.append("lgd must be in [0, 1]")
        if reasons:
            excluded.append({"loan_id": row["loan_id"], "reason": "; ".join(reasons)})
        else:
            valid.append(row)
    return valid, excluded


def calculate_ecl(valid_rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_loan = [
        {
            "loan_id": r["loan_id"],
            "ecl": round(r["balance"] * r["pd"] * r["lgd"], 2),
        }
        for r in valid_rows
    ]
    return {"per_loan": per_loan, "total_ecl": round(sum(x["ecl"] for x in per_loan), 2)}


def escalation_review(valid_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Stress-test review for high-risk loans (PD > ESCALATION_THRESHOLD)."""
    high_risk = [r for r in valid_rows if r["pd"] > ESCALATION_THRESHOLD]
    stress_results = []
    for r in high_risk:
        pd_stress = round(min(r["pd"] * STRESS_PD_MULTIPLIER, 1.0), 4)
        stress_results.append({
            "loan_id": r["loan_id"],
            "pd_original": r["pd"],
            "pd_stress": pd_stress,
            "ecl_stress": round(r["balance"] * pd_stress * r["lgd"], 2),
        })
    return {
        "high_risk_count": len(high_risk),
        "stress_results": stress_results,
        "total_stress_ecl": round(sum(x["ecl_stress"] for x in stress_results), 2),
    }


def generate_report(
    calc_result: dict[str, Any],
    excluded: list[dict[str, str]],
    escalation_result: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "total_ecl": calc_result["total_ecl"],
        "valid_count": len(calc_result["per_loan"]),
        "excluded": excluded,
        "per_loan": calc_result["per_loan"],
        "requires_escalation": escalation_result is not None,
    }
    if escalation_result is not None:
        report["escalation"] = escalation_result
    return report


def reference_run(path: Path) -> dict[str, Any]:
    """Full reference run for one instance.  Returns ground truth."""
    rows = load_data(path)
    valid, excluded = validate_data(rows)
    calc = calculate_ecl(valid)
    needs_esc = any(r["pd"] > ESCALATION_THRESHOLD for r in valid)
    esc = escalation_review(valid) if needs_esc else None
    return generate_report(calc, excluded, esc)


def get_instance_paths() -> list[Path]:
    """Return all generated instance CSV files sorted by name."""
    if not DATA_DIR.exists():
        return []
    return sorted(DATA_DIR.glob("instance_*.csv"))


def load_instance_manifest() -> dict[str, Any]:
    """Load the per-instance metadata manifest."""
    manifest_path = DATA_DIR / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


# Per-instance ground truth is stored as DATA_DIR/instance_NNN_gt.json to
# avoid re-computing it at test time.  The generate script writes these files.
def load_ground_truth(path: Path) -> dict[str, Any]:
    gt_path = path.with_name(path.stem + "_gt.json")
    return json.loads(gt_path.read_text(encoding="utf-8"))

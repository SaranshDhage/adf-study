"""Finance/ECL synthetic task family — Phase 0 design.

A small, fixed, synthetic loan portfolio + a simplified IFRS9-style Expected
Credit Loss (ECL) calculation: ECL = balance * PD * LGD, summed over loans
that pass validation. The portfolio (data/finance/portfolio.csv) deliberately
includes 4 invalid rows (negative PD, LGD > 1, missing LGD, zero balance) so
the VALIDATE_DATA state has real work to do rather than being a no-op passthrough.

Ground truth is derived by running the reference implementation below, not
hand-computed separately — this keeps "task success" tautologically defined
as "did the agent's tool calls + final output reproduce what correctly
calling these tools would produce," rather than two independently-maintained
numbers that could silently drift apart.

These functions are plain Python here in Phase 0; Phase 1 wraps them as
LangGraph/LangChain tools (data_loader, validation_tool, calculator_tool,
report_generator) bound to the baseline and harnessed agents.
"""

import csv
from pathlib import Path
from typing import Any, Optional

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "finance" / "portfolio.csv"

FSM_STATES = ["START", "LOAD_DATA", "VALIDATE_DATA", "CALCULATE", "GENERATE_REPORT", "END"]

TASK_PROMPT = """\
You are a credit risk analyst agent. You must compute the total Expected
Credit Loss (ECL) for the loan portfolio at {data_path}.

Follow this procedure:
1. Load the portfolio data.
2. Validate each loan: balance must be > 0, PD must be in [0, 1], LGD must be
   in [0, 1]. Exclude any loan that fails these checks and record why.
3. For each valid loan, compute ECL = balance * PD * LGD.
4. Generate a final report containing: total_ecl (sum across valid loans),
   valid_count, excluded (list of {{loan_id, reason}}), and per_loan (list of
   {{loan_id, ecl}} for valid loans).

Use the available tools to perform each step. Do not compute values yourself
without calling the appropriate tool.
"""


def load_data(path: Path = DATA_PATH) -> list[dict[str, Any]]:
    """Tool: data_loader. Reads the portfolio CSV into a list of row dicts."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append(
                {
                    "loan_id": row["loan_id"],
                    "balance": float(row["balance"]) if row["balance"] else None,
                    "pd": float(row["pd"]) if row["pd"] else None,
                    "lgd": float(row["lgd"]) if row["lgd"] else None,
                }
            )
        return rows


def validate_data(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Tool: validation_tool. Splits rows into (valid, excluded-with-reason)."""
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
    """Tool: calculator_tool. ECL = balance * pd * lgd per loan, summed."""
    per_loan = [
        {"loan_id": row["loan_id"], "ecl": round(row["balance"] * row["pd"] * row["lgd"], 2)}
        for row in valid_rows
    ]
    total_ecl = round(sum(item["ecl"] for item in per_loan), 2)
    return {"per_loan": per_loan, "total_ecl": total_ecl}


def generate_report(
    calc_result: dict[str, Any], excluded: list[dict[str, str]]
) -> dict[str, Any]:
    """Tool: report_generator. Assembles the final structured output."""
    return {
        "total_ecl": calc_result["total_ecl"],
        "valid_count": len(calc_result["per_loan"]),
        "excluded": excluded,
        "per_loan": calc_result["per_loan"],
    }


def reference_run(path: Path = DATA_PATH) -> dict[str, Any]:
    """Runs the full pipeline once to produce ground truth for scoring."""
    rows = load_data(path)
    valid, excluded = validate_data(rows)
    calc = calculate_ecl(valid)
    return generate_report(calc, excluded)


GROUND_TRUTH = reference_run()

if __name__ == "__main__":
    import json

    print(json.dumps(GROUND_TRUTH, indent=2))

"""Generate branching-ECL task instances.

Run:   python -m src.harness.gen_branching_instances [--seed N] [--n N]

Creates data/branching/instance_NNN.csv  (portfolio CSV)
       data/branching/instance_NNN_gt.json (ground truth)
       data/branching/manifest.json        (per-instance metadata)

Constraints (loop_eng.md §4.5):
- Ground truth computed with deterministic Python — NEVER an LLM.
- RNG seed committed; output is exactly reproducible.
- 15 linear instances (no PD > 0.15), 15 branching instances (≥1 PD > 0.15).
- Each instance has exactly 4 invalid rows (validator has real work to do).
- Per-instance metadata records required_path, high_risk_count, loan counts.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from src.tasks.branching_ecl import (
    DATA_DIR,
    ESCALATION_THRESHOLD,
    reference_run,
)

# ------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------
DEFAULT_SEED = 42
DEFAULT_N_INSTANCES = 30   # 15 linear + 15 branching
N_LOANS_VALID_BASE = 8     # valid loans per instance (before invalid rows)
N_INVALID_ROWS = 4         # deliberate invalid rows per instance (bad data)

# PD distribution parameters
PD_LINEAR_LOW, PD_LINEAR_HIGH = 0.01, 0.14   # all loans safe, no escalation
PD_HIGH_RISK_LOW, PD_HIGH_RISK_HIGH = 0.16, 0.40  # triggers escalation
N_HIGH_RISK_PER_BRANCHING = 2   # exactly 2 high-risk loans per branching instance

# Balance and LGD distributions
BAL_LOW, BAL_HIGH = 50_000, 500_000
LGD_LOW, LGD_HIGH = 0.30, 0.80


def _gen_valid_loan(
    rng: random.Random,
    loan_id: str,
    *,
    high_risk: bool = False,
) -> dict:
    pd = (
        round(rng.uniform(PD_HIGH_RISK_LOW, PD_HIGH_RISK_HIGH), 4)
        if high_risk
        else round(rng.uniform(PD_LINEAR_LOW, PD_LINEAR_HIGH), 4)
    )
    return {
        "loan_id": loan_id,
        "balance": round(rng.uniform(BAL_LOW, BAL_HIGH), 2),
        "pd": pd,
        "lgd": round(rng.uniform(LGD_LOW, LGD_HIGH), 4),
    }


def _gen_invalid_rows(rng: random.Random, start_idx: int) -> list[dict]:
    """Four deliberately invalid rows (one per violation type)."""
    return [
        # negative balance
        {
            "loan_id": f"BAD_{start_idx:03d}_A",
            "balance": round(-rng.uniform(1000, 50000), 2),
            "pd": round(rng.uniform(0.01, 0.10), 4),
            "lgd": round(rng.uniform(0.30, 0.70), 4),
        },
        # PD > 1
        {
            "loan_id": f"BAD_{start_idx:03d}_B",
            "balance": round(rng.uniform(10000, 100000), 2),
            "pd": round(rng.uniform(1.01, 1.50), 4),
            "lgd": round(rng.uniform(0.30, 0.70), 4),
        },
        # LGD > 1
        {
            "loan_id": f"BAD_{start_idx:03d}_C",
            "balance": round(rng.uniform(10000, 100000), 2),
            "pd": round(rng.uniform(0.01, 0.10), 4),
            "lgd": round(rng.uniform(1.01, 1.50), 4),
        },
        # zero balance
        {
            "loan_id": f"BAD_{start_idx:03d}_D",
            "balance": 0.0,
            "pd": round(rng.uniform(0.01, 0.10), 4),
            "lgd": round(rng.uniform(0.30, 0.70), 4),
        },
    ]


def generate_instance(
    rng: random.Random,
    instance_idx: int,
    *,
    branching: bool,
) -> tuple[list[dict], dict]:
    """Generate one portfolio instance.  Returns (rows, metadata)."""
    n_loans = N_LOANS_VALID_BASE
    rows = []

    if branching:
        # Place N_HIGH_RISK_PER_BRANCHING high-risk loans at random positions
        n_safe = n_loans - N_HIGH_RISK_PER_BRANCHING
        safe_rows = [
            _gen_valid_loan(rng, f"L{instance_idx:03d}_{i:02d}")
            for i in range(n_safe)
        ]
        high_rows = [
            _gen_valid_loan(rng, f"L{instance_idx:03d}_H{i:02d}", high_risk=True)
            for i in range(N_HIGH_RISK_PER_BRANCHING)
        ]
        valid_rows = safe_rows + high_rows
        rng.shuffle(valid_rows)
    else:
        valid_rows = [
            _gen_valid_loan(rng, f"L{instance_idx:03d}_{i:02d}")
            for i in range(n_loans)
        ]

    rows = valid_rows + _gen_invalid_rows(rng, instance_idx)

    metadata = {
        "instance_idx": instance_idx,
        "required_path": "branching" if branching else "linear",
        "n_loans_total": len(rows),
        "n_loans_valid": len(valid_rows),
        "n_loans_invalid": N_INVALID_ROWS,
        "n_high_risk": N_HIGH_RISK_PER_BRANCHING if branching else 0,
        "escalation_threshold": ESCALATION_THRESHOLD,
    }
    return rows, metadata


def write_instance(
    rows: list[dict],
    metadata: dict,
    data_dir: Path,
    idx: int,
) -> Path:
    """Write portfolio CSV and ground-truth JSON.  Returns CSV path."""
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / f"instance_{idx:03d}.csv"
    gt_path = data_dir / f"instance_{idx:03d}_gt.json"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["loan_id", "balance", "pd", "lgd"])
        writer.writeheader()
        writer.writerows(rows)

    gt = reference_run(csv_path)
    gt_path.write_text(json.dumps(gt, indent=2), encoding="utf-8")

    return csv_path


def generate_all(seed: int = DEFAULT_SEED, n: int = DEFAULT_N_INSTANCES) -> None:
    rng = random.Random(seed)
    assert n % 2 == 0, "n must be even (half linear, half branching)"
    n_each = n // 2

    manifest: dict[str, Any] = {
        "seed": seed,
        "n_instances": n,
        "n_linear": n_each,
        "n_branching": n_each,
        "escalation_threshold": ESCALATION_THRESHOLD,
        "stress_pd_multiplier": 2.0,
        "instances": {},
    }

    # Interleave linear and branching instances for natural ordering
    assignments = (["linear"] * n_each + ["branching"] * n_each)
    rng.shuffle(assignments)

    for idx, kind in enumerate(assignments):
        rows, meta = generate_instance(rng, idx, branching=(kind == "branching"))
        write_instance(rows, meta, DATA_DIR, idx)
        manifest["instances"][f"instance_{idx:03d}"] = meta
        print(f"  instance_{idx:03d}: {kind} "
              f"(n_valid={meta['n_loans_valid']}, n_high_risk={meta['n_high_risk']})")

    # Write manifest
    manifest_path = DATA_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote {n} instances + manifest to {DATA_DIR}")
    print(f"  Linear:    {n_each}")
    print(f"  Branching: {n_each}")


# Needed for type annotation in generate_all
from typing import Any  # noqa: E402  (must be after the function using it)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate branching-ECL instances")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n", type=int, default=DEFAULT_N_INSTANCES)
    args = parser.parse_args()
    generate_all(seed=args.seed, n=args.n)

"""Generate diverse finance_ecl instances.

Run:  python -m src.harness.gen_finance_instances [--seed N] [--n N]

Creates data/finance/instance_NNN.csv  and  data/finance/instance_NNN_gt.json
plus  data/finance/manifest.json.

The original data/finance/portfolio.csv is kept as instance_000 for backward
compatibility with fsm.py tests.  New instances start at instance_001.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

from src.tasks.finance_ecl import validate_data, calculate_ecl, generate_report

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "finance"

DEFAULT_SEED = 42
DEFAULT_N    = 25   # extra instances beyond the original portfolio.csv

# Loan count per instance
N_VALID_BASE_LOW  = 6
N_VALID_BASE_HIGH = 12

# PD tiers (deliberately spans the full [0,1] range so difficulty varies)
PD_LOW  = (0.01, 0.08)
PD_MID  = (0.08, 0.20)
PD_HIGH = (0.20, 0.45)

BAL_LOW, BAL_HIGH = 20_000, 600_000
LGD_LOW, LGD_HIGH = 0.20, 0.85


def _gen_loan(rng: random.Random, loan_id: str, pd_range: tuple) -> dict:
    return {
        "loan_id": loan_id,
        "balance": round(rng.uniform(BAL_LOW, BAL_HIGH), 2),
        "pd":      round(rng.uniform(*pd_range), 4),
        "lgd":     round(rng.uniform(LGD_LOW, LGD_HIGH), 4),
    }


def _gen_invalid_rows(rng: random.Random, idx: int) -> list[dict]:
    """3-5 deliberately invalid rows per instance (so the validator has work)."""
    n = rng.randint(2, 5)
    invalids = [
        {"loan_id": f"INV_{idx:03d}_neg_bal", "balance": round(-rng.uniform(1000,50000),2),
         "pd": round(rng.uniform(0.01,0.10),4), "lgd": round(rng.uniform(0.3,0.7),4)},
        {"loan_id": f"INV_{idx:03d}_bad_pd", "balance": round(rng.uniform(10000,100000),2),
         "pd": round(rng.uniform(1.01,1.5),4), "lgd": round(rng.uniform(0.3,0.7),4)},
        {"loan_id": f"INV_{idx:03d}_bad_lgd", "balance": round(rng.uniform(10000,100000),2),
         "pd": round(rng.uniform(0.01,0.10),4), "lgd": round(rng.uniform(1.01,1.5),4)},
        {"loan_id": f"INV_{idx:03d}_zero_bal", "balance": 0.0,
         "pd": round(rng.uniform(0.01,0.10),4), "lgd": round(rng.uniform(0.3,0.7),4)},
        {"loan_id": f"INV_{idx:03d}_null_pd", "balance": round(rng.uniform(10000,50000),2),
         "pd": None, "lgd": round(rng.uniform(0.3,0.7),4)},
    ]
    return invalids[:n]


def gen_instance(rng: random.Random, idx: int) -> tuple[list[dict], dict]:
    n_valid = rng.randint(N_VALID_BASE_LOW, N_VALID_BASE_HIGH)
    # Mix of PD tiers: mostly low, some mid, a few high
    loans = []
    n_high = rng.randint(0, max(1, n_valid // 4))
    n_mid  = rng.randint(1, max(2, n_valid // 3))
    n_low  = n_valid - n_high - n_mid

    for i in range(n_low):
        loans.append(_gen_loan(rng, f"L{idx:03d}_{i:02d}_low", PD_LOW))
    for i in range(n_mid):
        loans.append(_gen_loan(rng, f"L{idx:03d}_{i:02d}_mid", PD_MID))
    for i in range(n_high):
        loans.append(_gen_loan(rng, f"L{idx:03d}_{i:02d}_high", PD_HIGH))

    rng.shuffle(loans)
    invalids = _gen_invalid_rows(rng, idx)
    all_rows = loans + invalids
    rng.shuffle(all_rows)

    meta = {
        "instance_idx": idx,
        "n_loans_total": len(all_rows),
        "n_loans_valid": n_valid,
        "n_loans_invalid": len(invalids),
        "n_high_pd": n_high,
        "n_mid_pd": n_mid,
        "n_low_pd": n_low,
    }
    return all_rows, meta


def write_instance(rows: list[dict], idx: int, data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / f"instance_{idx:03d}.csv"
    gt_path  = data_dir / f"instance_{idx:03d}_gt.json"

    # Write CSV (handle None → empty string)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["loan_id","balance","pd","lgd"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})

    # Compute GT with deterministic reference implementation
    from src.tasks.finance_ecl import load_data
    loaded = load_data(csv_path)
    valid, excluded = validate_data(loaded)
    calc = calculate_ecl(valid)
    gt = generate_report(calc, excluded)
    gt_path.write_text(json.dumps(gt, indent=2), encoding="utf-8")
    return csv_path


def generate_all(seed: int = DEFAULT_SEED, n: int = DEFAULT_N) -> None:
    rng = random.Random(seed)
    manifest: dict[str, Any] = {
        "seed": seed,
        "n_instances": n + 1,   # +1 for original instance_000
        "instances": {},
    }

    # Preserve original portfolio.csv as instance_000
    orig = DATA_DIR / "portfolio.csv"
    if orig.exists():
        import shutil
        dst = DATA_DIR / "instance_000.csv"
        if not dst.exists():
            shutil.copy(orig, dst)
        # Compute GT for instance_000 too
        gt_000 = DATA_DIR / "instance_000_gt.json"
        if not gt_000.exists():
            from src.tasks.finance_ecl import load_data
            rows = load_data(dst)
            v, ex = validate_data(rows)
            calc = calculate_ecl(v)
            gt = generate_report(calc, ex)
            gt_000.write_text(json.dumps(gt, indent=2), encoding="utf-8")
        manifest["instances"]["instance_000"] = {
            "instance_idx": 0, "source": "original portfolio.csv"
        }

    for i in range(1, n + 1):
        rows, meta = gen_instance(rng, i)
        write_instance(rows, i, DATA_DIR)
        manifest["instances"][f"instance_{i:03d}"] = meta
        print(f"  instance_{i:03d}: {meta['n_loans_valid']} valid, "
              f"{meta['n_loans_invalid']} invalid, {meta['n_high_pd']} high-PD")

    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote {n} new instances to {DATA_DIR}  (total {n+1} incl. instance_000)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n",    type=int, default=DEFAULT_N)
    args = parser.parse_args()
    generate_all(seed=args.seed, n=args.n)

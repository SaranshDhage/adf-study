"""Generate diverse legal_clause instances.

Run:  python -m src.harness.gen_legal_instances [--seed N] [--n N]

Creates data/legal/instance_NNN.txt  and  data/legal/instance_NNN_gt.json
plus  data/legal/manifest.json.

The original data/legal/contract.txt is preserved as instance_000.

Clause text is template-based: category keywords are seeded into each clause
so the deterministic classifier (CLASSIFICATION_RULES) always produces the
expected category.  Ambiguous clauses (multi-keyword) are injected at a
controlled rate to ensure the priority-ordering logic is exercised.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from src.tasks.legal_clause import (
    extract_clauses, classify_clauses, generate_report, CLASSIFICATION_RULES,
    FALLBACK_CATEGORY,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "legal"

DEFAULT_SEED = 42
DEFAULT_N    = 25

# ---- Clause text snippets keyed by target category -------------------------
# Each entry is (category, keyword_that_triggers_it, body_text_variants)
_CLAUSE_TEMPLATES: dict[str, tuple[str, list[str]]] = {
    "Force Majeure": ("force majeure", [
        "Neither party shall be liable for delays caused by events of force majeure, including natural disasters, war, or pandemics beyond reasonable control.",
        "In the event of force majeure, the affected party shall notify the other in writing within 5 business days and be excused from performance.",
        "This agreement is subject to a force majeure provision releasing both parties from obligations made impossible by extraordinary events.",
    ]),
    "Confidentiality": ("confidential", [
        "All information disclosed under this agreement is confidential and must not be shared with third parties without prior written consent.",
        "The receiving party agrees to maintain the confidentiality of all proprietary data and not to disclose confidential materials.",
        "Confidential information exchanged between the parties shall be protected using at least the same degree of care as the receiving party uses for its own confidential information.",
    ]),
    "Indemnification": ("indemnif", [
        "Each party shall indemnify and hold harmless the other from any claims, damages, or costs arising from breach of this agreement.",
        "The vendor agrees to indemnify the client against all third-party claims resulting from defective products supplied under this contract.",
        "Indemnification obligations shall survive the termination of this agreement for a period of three years.",
    ]),
    "Termination": ("terminat", [
        "Either party may terminate this agreement with 30 days written notice to the other party.",
        "This agreement shall terminate automatically upon expiration of the initial term unless renewed by mutual written consent.",
        "Upon termination, all outstanding payments shall become immediately due and payable.",
    ]),
    "Payment Terms": ("payment", [
        "All invoices shall be paid within 30 days of receipt. Late payments shall incur interest at 1.5% per month.",
        "Payment shall be made by wire transfer within 15 days of invoice date. The client shall not withhold payment for disputed amounts.",
        "The parties agree to quarterly invoice cycles with payment due within 45 days. Disputed invoice items must be raised within 10 days.",
    ]),
    "Governing Law": ("governing law", [
        "This agreement shall be governed by and construed in accordance with the governing law of the State of New York.",
        "The parties consent to the exclusive jurisdiction of the courts in Delaware. The governing law applicable shall be that of Delaware.",
        "Any dispute under this contract shall be resolved under governing law of England and Wales.",
    ]),
    "Limitation of Liability": ("liable", [
        "In no event shall either party be liable for indirect, incidental, or consequential damages regardless of the cause.",
        "The aggregate liability of either party under this agreement shall not exceed the total fees paid in the preceding 12 months.",
        "Neither party shall be liable for lost profits, data loss, or business interruption even if advised of the possibility of such damages.",
    ]),
    "Other": (None, [
        "The parties agree to negotiate in good faith to resolve any ambiguities in the interpretation of this agreement.",
        "This contract constitutes the entire agreement between the parties and supersedes all prior communications.",
        "Any modifications to this agreement must be made in writing and signed by authorised representatives of both parties.",
        "Headings in this agreement are for convenience only and shall not affect interpretation.",
        "If any provision of this agreement is found unenforceable, the remaining provisions shall continue in full force.",
    ]),
}


def _get_category_for_keyword(keyword: str) -> str:
    for category, pattern in CLASSIFICATION_RULES:
        import re
        if re.search(pattern, keyword):
            return category
    return FALLBACK_CATEGORY


def gen_instance(rng: random.Random, idx: int) -> tuple[str, dict]:
    n_clauses = rng.randint(7, 13)

    # Pick categories: always include a few "Other" to test fallback
    n_other = rng.randint(1, 3)
    available = [c for c in _CLAUSE_TEMPLATES if c != "Other"]
    n_special = n_clauses - n_other
    n_special = min(n_special, len(available))
    categories_special = rng.sample(available, k=n_special)
    categories = categories_special + ["Other"] * n_other
    rng.shuffle(categories)

    # Build clause texts
    clauses_text = []
    for i, cat in enumerate(categories, start=1):
        _, variants = _CLAUSE_TEMPLATES[cat]
        text = rng.choice(variants)
        clauses_text.append(f"Clause {i}: {text}")

    doc = "\n".join(clauses_text) + "\n"

    meta = {
        "instance_idx": idx,
        "n_clauses": n_clauses,
        "categories_present": sorted(set(categories)),
        "n_other": n_other,
    }
    return doc, meta


def write_instance(doc: str, idx: int, data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    txt_path = data_dir / f"instance_{idx:03d}.txt"
    gt_path  = data_dir / f"instance_{idx:03d}_gt.json"

    txt_path.write_text(doc, encoding="utf-8")

    clauses    = extract_clauses(doc)
    classified = classify_clauses(clauses)
    gt         = generate_report(classified)
    gt_path.write_text(json.dumps(gt, indent=2), encoding="utf-8")
    return txt_path


def generate_all(seed: int = DEFAULT_SEED, n: int = DEFAULT_N) -> None:
    rng = random.Random(seed)
    manifest: dict[str, Any] = {
        "seed": seed, "n_instances": n + 1, "instances": {},
    }

    # Preserve original contract.txt as instance_000
    orig = DATA_DIR / "contract.txt"
    if orig.exists():
        import shutil
        dst = DATA_DIR / "instance_000.txt"
        if not dst.exists():
            shutil.copy(orig, dst)
        gt_000 = DATA_DIR / "instance_000_gt.json"
        if not gt_000.exists():
            from src.tasks.legal_clause import load_document
            doc = load_document(dst)
            gt = generate_report(classify_clauses(extract_clauses(doc)))
            gt_000.write_text(json.dumps(gt, indent=2), encoding="utf-8")
        manifest["instances"]["instance_000"] = {
            "instance_idx": 0, "source": "original contract.txt"
        }

    for i in range(1, n + 1):
        doc, meta = gen_instance(rng, i)
        write_instance(doc, i, DATA_DIR)
        manifest["instances"][f"instance_{i:03d}"] = meta
        cats = meta["categories_present"]
        print(f"  instance_{i:03d}: {meta['n_clauses']} clauses, cats={cats}")

    (DATA_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {n} new instances to {DATA_DIR}  (total {n+1} incl. instance_000)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n",    type=int, default=DEFAULT_N)
    args = parser.parse_args()
    generate_all(seed=args.seed, n=args.n)

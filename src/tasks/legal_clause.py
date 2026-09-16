"""Legal clause-extraction/classification synthetic task family — Phase 0 design.

A small, fixed, synthetic contract (data/legal/contract.txt) with 10 numbered
clauses. Classification uses a fixed-priority keyword taxonomy rather than
semantic judgment — this task isn't testing NLU depth, it's a controlled
testbed for orchestration determinism, so the "correct" label must itself be
deterministic and independently verifiable.

Two clauses (7 and 8) deliberately contain keywords from more than one
category ("liable" + "force majeure"; "confidential" + "termination") to
force a priority order in CLASSIFICATION_RULES. This mirrors the paper's
broader point at a small scale: rule/tool-priority ordering is itself a
determinism lever, not just an implementation detail.

Mirrors src/tasks/finance_ecl.py's structure: plain functions here become
LangGraph tools in Phase 1 (document_loader, clause_extractor,
clause_classifier, report_generator); ground truth comes from running this
reference implementation, not a separately hand-maintained answer key.
"""

import re
from pathlib import Path
from typing import Any

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "legal" / "contract.txt"

FSM_STATES = ["START", "LOAD_DOC", "EXTRACT_CLAUSES", "CLASSIFY", "GENERATE_REPORT", "END"]

TASK_PROMPT = """\
You are a contract review agent. You must extract and classify every clause
in the contract at {data_path}.

Follow this procedure:
1. Load the document.
2. Extract each numbered clause as a separate unit of text.
3. Classify each clause into exactly one category from this fixed taxonomy:
   Confidentiality, Termination, Indemnification, Payment Terms,
   Governing Law, Limitation of Liability, Force Majeure, Other.
   If a clause could match more than one category, apply this priority
   order and pick the first match: Force Majeure, Confidentiality,
   Indemnification, Termination, Payment Terms, Governing Law,
   Limitation of Liability, then Other as a fallback.
4. Generate a final report listing, for each clause: clause_id, category,
   and a short text snippet.

Use the available tools to perform each step. Do not classify clauses
yourself without calling the appropriate tool.
"""

# Ordered: first matching rule wins. See module docstring re: clauses 7/8.
CLASSIFICATION_RULES: list[tuple[str, str]] = [
    ("Force Majeure", r"force majeure"),
    ("Confidentiality", r"confidential"),
    ("Indemnification", r"indemnif"),
    ("Termination", r"terminat"),
    ("Payment Terms", r"payment|invoice"),
    ("Governing Law", r"governing law|jurisdiction"),
    ("Limitation of Liability", r"liabilit|liable"),
]
FALLBACK_CATEGORY = "Other"

CLAUSE_PATTERN = re.compile(r"Clause (\d+):\s*(.+?)(?=\nClause \d+:|\Z)", re.DOTALL)


def load_document(path: Path = DATA_PATH) -> str:
    """Tool: document_loader. Reads the raw contract text."""
    return path.read_text(encoding="utf-8")


def extract_clauses(document_text: str) -> list[dict[str, Any]]:
    """Tool: clause_extractor. Splits the document into numbered clause units."""
    return [
        {"clause_id": f"C{num}", "text": text.strip()}
        for num, text in CLAUSE_PATTERN.findall(document_text)
    ]


def classify_clauses(clauses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tool: clause_classifier. Applies the fixed-priority keyword taxonomy."""
    classified = []
    for clause in clauses:
        text_lower = clause["text"].lower()
        category = FALLBACK_CATEGORY
        for name, pattern in CLASSIFICATION_RULES:
            if re.search(pattern, text_lower):
                category = name
                break
        classified.append(
            {
                "clause_id": clause["clause_id"],
                "category": category,
                "snippet": clause["text"][:80],
            }
        )
    return classified


def generate_report(classified: list[dict[str, Any]]) -> dict[str, Any]:
    """Tool: report_generator. Assembles the final structured output."""
    counts: dict[str, int] = {}
    for item in classified:
        counts[item["category"]] = counts.get(item["category"], 0) + 1
    return {"clause_count": len(classified), "category_counts": counts, "clauses": classified}


def reference_run(path: Path = DATA_PATH) -> dict[str, Any]:
    """Runs the full pipeline once to produce ground truth for scoring."""
    document_text = load_document(path)
    clauses = extract_clauses(document_text)
    classified = classify_clauses(clauses)
    return generate_report(classified)


GROUND_TRUTH = reference_run()

if __name__ == "__main__":
    import json

    print(json.dumps(GROUND_TRUTH, indent=2))

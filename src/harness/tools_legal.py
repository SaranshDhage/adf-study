"""LangChain tool wrappers around src/tasks/legal_clause.py for agent binding.

Tools take NO arguments and instead read/write a per-run `session` dict
closed over by make_legal_tools() — see src/harness/tools_finance.py's
docstring for why (real testing found the model breaks when asked to
re-serialize prior tool results as call arguments).
"""

from langchain_core.tools import tool

from src.tasks import legal_clause as task

REPORT_TOOL_NAME = "report_generator"


def make_legal_tools(session: dict) -> list:
    @tool
    def document_loader() -> str:
        """Loads the raw contract text into the working context and returns it."""
        session["document_text"] = task.load_document()
        return session["document_text"]

    @tool
    def clause_extractor() -> list[dict]:
        """Splits the contract text already loaded into context (via
        document_loader) into numbered clause units. Returns a list of
        {"clause_id", "text"}."""
        if "document_text" not in session:
            raise ValueError("document_loader must be called before clause_extractor")
        clauses = task.extract_clauses(session["document_text"])
        session["clauses"] = clauses
        return clauses

    @tool
    def clause_classifier() -> list[dict]:
        """Classifies the clauses already extracted into context (via
        clause_extractor) into one category from the fixed taxonomy
        (Confidentiality, Termination, Indemnification, Payment Terms,
        Governing Law, Limitation of Liability, Force Majeure, Other).
        Returns a list of {"clause_id", "category", "snippet"}."""
        if "clauses" not in session:
            raise ValueError("clause_extractor must be called before clause_classifier")
        classified = task.classify_clauses(session["clauses"])
        session["classified"] = classified
        return classified

    @tool
    def report_generator() -> dict:
        """Assembles the final report with clause_count, category_counts,
        and the full classified clause list already in context. This is the
        FINAL step — call it last."""
        if "classified" not in session:
            raise ValueError("clause_classifier must be called before report_generator")
        return task.generate_report(session["classified"])

    return [document_loader, clause_extractor, clause_classifier, report_generator]

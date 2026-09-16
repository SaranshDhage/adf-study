"""LangChain tool wrappers around src/tasks/finance_ecl.py for agent binding.

Tools take NO arguments and instead read/write a per-run `session` dict
closed over by make_finance_tools(). Earlier versions required the model to
pass prior results back as call arguments (e.g. validation_tool(rows=[...])
with all 12 loan rows inline) — real testing against qwen-2.5-7b-instruct via
OpenRouter showed this breaks: the model generated malformed/truncated JSON
trying to re-serialize the row list as a function argument, producing a
tool_call with unparseable `arguments`. That's a data-copying failure, not
the orchestration non-determinism this research is about, so it's a confound
worth engineering out rather than measuring. With session-based state, the
model's only decision is which tool to call and when — session state size no
longer depends on what the model can faithfully retype.

All task logic still lives in finance_ecl.py so the reference implementation
and the agent-callable tool are guaranteed to match (see that module's
docstring on ground truth).
"""

from langchain_core.tools import tool

from src.tasks import finance_ecl as task

REPORT_TOOL_NAME = "report_generator"


def make_finance_tools(session: dict) -> list:
    @tool
    def data_loader() -> list[dict]:
        """Loads the loan portfolio into the working context and returns the
        rows, each with loan_id, balance, pd (probability of default), and
        lgd (loss given default)."""
        session["rows"] = task.load_data()
        return session["rows"]

    @tool
    def validation_tool() -> dict:
        """Validates the loan rows already loaded into context (via
        data_loader): balance must be > 0, pd and lgd must each be in [0, 1].
        Returns {"valid": [...], "excluded": [{"loan_id", "reason"}, ...]}."""
        if "rows" not in session:
            raise ValueError("data_loader must be called before validation_tool")
        valid, excluded = task.validate_data(session["rows"])
        session["valid"] = valid
        session["excluded"] = excluded
        return {"valid": valid, "excluded": excluded}

    @tool
    def calculator_tool() -> dict:
        """Computes ECL = balance * pd * lgd for each valid loan row already
        produced by validation_tool, and sums them. Returns {"per_loan":
        [{"loan_id", "ecl"}, ...], "total_ecl": float}."""
        if "valid" not in session:
            raise ValueError("validation_tool must be called before calculator_tool")
        calc = task.calculate_ecl(session["valid"])
        session["calc"] = calc
        return calc

    @tool
    def report_generator() -> dict:
        """Assembles the final report combining the ECL calculation results
        and the list of excluded loans already in context. This is the FINAL
        step — call it last."""
        if "calc" not in session or "excluded" not in session:
            raise ValueError("calculator_tool must be called before report_generator")
        return task.generate_report(session["calc"], session["excluded"])

    return [data_loader, validation_tool, calculator_tool, report_generator]

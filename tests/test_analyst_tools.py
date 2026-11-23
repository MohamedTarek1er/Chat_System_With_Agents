"""
tests/test_analyst_tools.py

Two groups of tests:

1. Offline tests (default): call each tool directly with .invoke(). No LLM, no network.
   They tell you whether the TOOLS work.

2. LLM tests (marked `llm`): run the real Analyst agent on crafted queries and check
   from the message trace that the expected tools were actually called.
   They tell you whether the AGENT uses the tools.

Run:
    pytest                    # offline tests only
    pytest -m llm -s          # LLM tests, with the tool trace printed
    pytest -m llm -k more     # one LLM case
    pytest -m "llm or not llm" -s   # everything
"""
from dataclasses import dataclass, field
from itertools import count

import pytest
from docx import Document
from langchain_core.messages import AIMessage, ToolMessage

from agents.analyst.tools.calculator import calculate_tool
from agents.analyst.tools.data_analysis import data_analysis
from agents.analyst.tools.evidence_request import format_evidence, request_more_evidence
from agents.analyst.tools.table_extractor import extract_tables_tool

TABLE_DOC = "analyst_test_table.docx"
_ids = count(1)


def ev(text, doc, page=1):
    """Build an EvidenceItem dict."""
    return {"text": text, "doc_name": doc, "page": page,
            "chunk_id": f"c{next(_ids)}", "score": None}


# ---------------------------------------------------------------- fixtures
@pytest.fixture(autouse=True)
def uploads(tmp_path, monkeypatch):
    """Point extract_tables_tool at a temp uploads dir holding a small DOCX results table."""
    doc = Document()
    doc.add_paragraph("Table 1: Model results")
    rows = [("Model", "F1", "Latency (ms)"),
            ("CNN-A", "0.88", "12"),
            ("CNN-B", "0.93", "30"),
            ("CNN-C", "0.81", "9")]
    table = doc.add_table(rows=len(rows), cols=3)
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            table.cell(i, j).text = value
    doc.save(tmp_path / TABLE_DOC)

    # table_extractor does `from core.config import UPLOAD_DIR`, so patch the name it uses
    monkeypatch.setattr("agents.analyst.tools.table_extractor.UPLOAD_DIR", tmp_path)
    return tmp_path


@pytest.fixture(scope="module")
def agent():
    """Same construction as make_analyst_node(); keep the tool list in sync."""
    from langchain.agents import create_agent

    from agents.analyst.tools.document_comparison import compare_documents
    from core.prompts import ANALYST_SYSTEM_PROMPT
    from graph.state import AnalystState

    try:
        from core.llm import llm
    except ImportError:
        from core.llm import get_llm
        llm = get_llm(temperature=0)

    return create_agent(
        model=llm,
        tools=[calculate_tool, extract_tables_tool, data_analysis,
               compare_documents, request_more_evidence],
        system_prompt=ANALYST_SYSTEM_PROMPT,
        state_schema=AnalystState,
    )


# ======================================================= 1. offline tool tests
def test_calculator_average():
    out = calculate_tool.invoke({"operation": "average", "numbers": [92, 95, 89]})
    assert float(out) == pytest.approx(92)


def test_calculator_percentage():
    assert float(calculate_tool.invoke({"operation": "percentage", "numbers": [25, 200]})) == 12.5


def test_calculator_divide_by_zero_returns_error_string():
    out = calculate_tool.invoke({"operation": "divide", "numbers": [1, 0]})
    assert out.startswith("Error")


def test_calculator_schema_exposes_only_llm_arguments():
    assert set(calculate_tool.args) == {"operation", "numbers"}


@pytest.mark.parametrize("label_column", [None, "model"])
def test_data_analysis_rank_single_metric(label_column):
    # label_column="model" also guards against `labels` being undefined when it is passed
    out = data_analysis.invoke({
        "records": [{"model": "A", "acc": "92%"}, {"model": "B", "acc": "95%"}],
        "analysis_type": "rank",
        "label_column": label_column,
    })
    assert out.splitlines()[1].startswith("1. **B**")


def test_data_analysis_lower_is_better_flips_ranking():
    out = data_analysis.invoke({
        "records": [{"model": "A", "loss": 0.31}, {"model": "B", "loss": 0.22}],
        "analysis_type": "rank",
        "lower_is_better": ["loss"],
    })
    assert out.splitlines()[1].startswith("1. **B**")


def test_data_analysis_summary_mean():
    out = data_analysis.invoke({
        "records": [{"model": "A", "acc": 92}, {"model": "B", "acc": 95}, {"model": "C", "acc": 89}],
        "analysis_type": "summary",
    })
    assert "mean=92.0000" in out


def test_data_analysis_empty_records():
    assert data_analysis.invoke({"records": []}).startswith("Error")


def test_request_more_evidence_sets_followup_query():
    # a full tool-call dict is needed so InjectedToolCallId gets a value
    cmd = request_more_evidence.invoke({
        "name": "request_more_evidence",
        "args": {"query": "computational complexity of Model B", "reason": "not in evidence"},
        "id": "call_1",
        "type": "tool_call",
    })
    assert cmd.update["followup_query"] == "computational complexity of Model B"


def test_extract_tables_returns_rows():
    out = extract_tables_tool.invoke({"doc_name": TABLE_DOC})
    assert "CNN-B | 0.93" in out


def test_extract_tables_missing_file_is_an_error_string():
    assert extract_tables_tool.invoke({"doc_name": "nope.docx"}).startswith("Error")


# ===================================================== 2. LLM tool-usage tests
@dataclass
class Case:
    name: str
    question: str
    evidence: list
    expect_any: set = field(default_factory=set)      # at least one of these must be called
    forbidden: set = field(default_factory=set)       # none of these may be called
    result_contains: str | None = None                # some tool result must contain this
    expect_followup: bool | None = None               # True: followup_query set, False: empty


CASES = [
    Case(
        name="calc",
        question="What is the average accuracy of the three CNN models?",
        evidence=[
            ev("CNN-A reaches an accuracy of 92% on the test set.", "paper_a.pdf"),
            ev("CNN-B reaches an accuracy of 95% on the test set.", "paper_b.pdf"),
            ev("CNN-C reaches an accuracy of 89% on the test set.", "paper_c.pdf"),
        ],
        expect_any={"calculate_tool", "data_analysis"},
        result_contains="92",
        expect_followup=False,
    ),
    Case(
        name="rank",
        question="Rank the models by F1-score and give the gap between the best and the worst.",
        evidence=[
            ev("Results: CNN-A accuracy 92%, F1 0.88, loss 0.31.", "results.pdf", 3),
            ev("Results: CNN-B accuracy 95%, F1 0.93, loss 0.22.", "results.pdf", 3),
            ev("Results: CNN-C accuracy 89%, F1 0.81, loss 0.40.", "results.pdf", 4),
        ],
        expect_any={"data_analysis", "calculate_tool"},
        expect_followup=False,
    ),
    Case(
        name="compare",
        question="Compare the methodologies described in the two documents.",
        evidence=[
            ev("We fine-tune a pretrained ResNet-50 with data augmentation and SGD.", "paper_a.pdf", 2),
            ev("We train a small custom CNN from scratch using Adam and no augmentation.", "paper_b.pdf", 4),
            ev("Paper A evaluates on ImageNet subsets; training takes 12 GPU hours.", "paper_a.pdf", 5),
            ev("Paper B evaluates on CIFAR-10; training takes 40 minutes on one GPU.", "paper_b.pdf", 6),
        ],
        expect_any={"compare_documents"},
        expect_followup=False,
    ),
    Case(
        name="table",
        question=f"Which model has the highest F1 according to the results table in {TABLE_DOC}?",
        evidence=[ev("The results of all models are reported in Table 1 of this document.", TABLE_DOC, None)],
        expect_any={"extract_tables_tool"},
        result_contains="0.93",
        expect_followup=False,
    ),
    Case(
        name="more",
        question="What is the computational complexity of Model B?",
        evidence=[
            ev("Model B reaches an accuracy of 95% on the test set.", "paper_b.pdf", 5),
            ev("Model B was trained for 30 epochs with a batch size of 64.", "paper_b.pdf", 6),
        ],
        expect_any={"request_more_evidence"},
        expect_followup=True,
    ),
    Case(
        name="control",
        question="Which dataset did Paper A use?",
        evidence=[ev("Paper A evaluates its model on the CIFAR-10 dataset.", "paper_a.pdf", 3)],
        forbidden={"request_more_evidence"},          # the evidence is sufficient
        expect_followup=False,
    ),
]


def _short(value, n=160):
    s = str(value).replace("\n", " ")
    return s if len(s) <= n else s[:n] + "..."


def run_agent(agent, case):
    user_msg = (f"Question: {case.question}\n\n"
                f"Retrieved evidence:\n{format_evidence(case.evidence)}")
    out = agent.invoke(
        {"messages": [("user", user_msg)], "evidence": case.evidence, "followup_query": None},
        config={"recursion_limit": 12},
    )

    calls, results = [], {}
    for m in out["messages"]:
        if isinstance(m, AIMessage) and m.tool_calls:
            calls += [(c["name"], c["args"], c["id"]) for c in m.tool_calls]
        elif isinstance(m, ToolMessage):
            results[m.tool_call_id] = str(m.content)

    report = [f"\n[{case.name}] {case.question}"]
    if not calls:
        report.append("  tools called: (none)")
    for name, args, call_id in calls:
        report.append(f"  tool  : {name}\n    args  : {_short(args)}\n"
                      f"    result: {_short(results.get(call_id, '<no result>'))}")
    report.append(f"  followup_query: {out.get('followup_query')!r}")
    report.append(f"  final analysis: {_short(out['messages'][-1].content, 300)}")
    return calls, results, out.get("followup_query") or "", "\n".join(report)


@pytest.mark.llm
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_agent_calls_expected_tools(agent, case):
    calls, results, followup, report = run_agent(agent, case)
    print(report)                                     # visible with -s, and on failure

    called = {name for name, _, _ in calls}

    if case.expect_any:
        assert called & case.expect_any, (
            f"expected one of {sorted(case.expect_any)}, got {sorted(called) or 'no tools'}\n{report}")

    assert not (called & case.forbidden), f"forbidden tool called: {sorted(called & case.forbidden)}\n{report}"

    # a tool that was expected must have run successfully, not returned an error string
    for name, _, call_id in calls:
        if name in case.expect_any:
            assert not results.get(call_id, "").lower().startswith("error"), \
                f"{name} returned an error\n{report}"

    if case.result_contains:
        assert any(case.result_contains in r for r in results.values()), \
            f"no tool result contained {case.result_contains!r}\n{report}"

    if case.expect_followup is True:
        assert followup, f"followup_query is empty (state not updated by the tool)\n{report}"
    if case.expect_followup is False:
        assert not followup, f"unexpected followup_query: {followup!r}\n{report}"
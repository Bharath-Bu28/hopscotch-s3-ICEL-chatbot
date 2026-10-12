"""Offline tests: no API key, no network. A fake chat model stands in for the LLM."""
import os
import sys

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chains  # noqa: E402
from chains import (  # noqa: E402
    SAMPLE_TICKETS, build_bot, build_enrichment_pipeline, build_full_chain, decide_route, detect_safety_concern,
    enrich_with_order_id, explain_route, extract_order_id, extract_order_value,
    parse_triage, quick_filter,
)


class FakeLLM(BaseChatModel):
    """Answers by looking at the prompt, so parallel call order does not matter.

    - Triage prompt (asks for 'category,priority')  -> returns `triage_answer`
    - Any reply prompt -> returns 'REPLY' and records the system text it was given.
    """
    triage_answer: str = "other,Low"
    seen_system: list = []

    @property
    def _llm_type(self) -> str:
        return "fake-hopscotch"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        system = messages[0].content
        if "category,priority" in system:
            text = self.triage_answer
        else:
            self.seen_system.append(system)
            text = "REPLY"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


def make_llm(triage_answer):
    return FakeLLM(triage_answer=triage_answer, seen_system=[])


# ---------------------------------------------------------------- parse_triage
@pytest.mark.parametrize("raw,expected", [
    ("genuine_defect,High", ("genuine_defect", "High")),
    ("  Policy_Violation , low \n", ("policy_violation", "Low")),
    ("category: Genuine Defect, priority: HIGH", ("genuine_defect", "High")),
    ("unclear,medium", ("unclear", "Medium")),
    ("other", ("other", "Medium")),
    ("I think it is probably fine!", ("unclear", "Medium")),
    ("", ("unclear", "Medium")),
    (None, ("unclear", "Medium")),
])
def test_parse_triage(raw, expected):
    out = parse_triage(raw)
    assert (out["category"], out["priority"]) == expected


# ---------------------------------------------------------- deterministic signals
@pytest.mark.parametrize("text,flag", [
    ("The button came off my toddler's shirt and he almost swallowed it", True),
    ("My baby was choking on a small part", True),
    ("The zip cut him on the arm", True),
    ("She developed a rash after wearing it", True),
    ("Possible allergy, lots of itching", True),
    ("Small parts came loose on my son's jacket", True),
    ("The sole of the sneakers is peeling", False),
    ("The button came off my shirt", False),  # no child involved
    ("Do you ship to Pune?", False),
])
def test_detect_safety_concern(text, flag):
    out = detect_safety_concern(text)
    assert out["flag"] is flag
    assert (out["reason"] is not None) is flag


@pytest.mark.parametrize("text,value", [
    ("It cost ₹7,200", 7200),
    ("paid Rs. 6500 for it", 6500),
    ("INR 5,500 order", 5500),
    ("rs 12000", 12000),
    ("₹999 and ₹1,200", 1200),
    ("no amount here", None),
    ("I have 5000 buttons", None),
])
def test_extract_order_value(text, value):
    assert extract_order_value(text) == value


def test_extract_order_id():
    assert extract_order_id("Order #HS88213 arrived broken") == "HS88213"
    assert extract_order_id("order hs1234.") == "HS1234"
    assert extract_order_id("no id") is None
    assert extract_order_id({"ticket_text": "my HS42 zipper"}) == "HS42"


def test_session9_primitives():
    assert quick_filter.invoke("The sole peeled off") == "likely_genuine_defect"
    assert quick_filter.invoke("I already wore it") == "likely_policy_violation"
    assert quick_filter.invoke("hello") == "needs_llm_classification"
    out = enrich_with_order_id.invoke({"ticket_text": "Order HS88213", "customer_name": "Ananya"})
    assert out["customer_name"] == "Ananya" and out["order_id"] == "HS88213"


def test_enrichment_pipeline_keeps_keys():
    out = build_enrichment_pipeline(make_llm("genuine_defect,High")).invoke(
        {"input": "Order #HS88213 for ₹7,200: the sole is peeling"})
    assert out["input"].startswith("Order #HS88213")  # .assign() keeps the original key
    assert out["triage"] == {"category": "genuine_defect", "priority": "High"}
    assert out["safety"]["flag"] is False
    assert out["order_value"] == 7200 and out["order_id"] == "HS88213"


# ------------------------------------------------------------------ decide_route
def sig(category="other", priority="Medium", safety=False, value=None):
    return {"triage": {"category": category, "priority": priority},
            "safety": {"flag": safety, "reason": "child safety" if safety else None},
            "order_value": value, "order_id": None}


@pytest.mark.parametrize("signals,route,needs_human", [
    (sig("genuine_defect", "High"), "defect", False),
    (sig("policy_violation", "Low"), "policy", False),
    (sig("genuine_defect", "High", safety=True), "human_review", True),
    (sig("genuine_defect", "High", value=7200), "human_review", True),
    (sig("unclear", "Medium"), "human_review", True),
    (sig("other", "Low"), "general", False),
    (sig("policy_violation", "Low", value=5000), "policy", False),  # exactly 5,000 is NOT above
])
def test_decide_route(signals, route, needs_human):
    out = decide_route({"input": "x", "chat_history": [], "signals": signals})
    assert out["route"] == route
    assert out["needs_human"] is needs_human
    assert bool(out["human_reasons"]) is needs_human
    assert out["input"] == "x" and "triage_summary" in out


# ------------------------------------------------------------ end-to-end (fake LLM)
def run_chain(triage_answer, text):
    llm = make_llm(triage_answer)
    reply = build_full_chain(llm).invoke({"input": text, "chat_history": []})
    assert reply == "REPLY"
    assert len(llm.seen_system) == 1  # exactly one reply prompt ran
    return llm.seen_system[0]


def test_child_safety_goes_to_escalation_even_if_llm_says_defect():
    system = run_chain("genuine_defect,High", SAMPLE_TICKETS["Safety: button, toddler"])
    assert "HUMAN HAND-OFF" in system


def test_high_value_goes_to_escalation():
    system = run_chain("genuine_defect,High", "My Rs. 6,500 order has a peeling sole")
    assert "HUMAN HAND-OFF" in system


def test_unclear_goes_to_escalation():
    assert "HUMAN HAND-OFF" in run_chain("unclear,Medium", SAMPLE_TICKETS["Unclear: faded dress"])


def test_defect_branch():
    assert "GENUINE DEFECT" in run_chain("genuine_defect,High", SAMPLE_TICKETS["Defect: Dino-Print sneakers"])


def test_policy_branch():
    assert "POLICY EXPLANATION" in run_chain("policy_violation,Low", SAMPLE_TICKETS["Policy: worn jacket, wrong size"])


def test_general_branch():
    assert "ROUTE: GENERAL" in run_chain("other,Low", "Do you ship to Pune?")


def test_explain_route():
    out = explain_route("Order #HS1 for ₹7,200 is peeling", make_llm("genuine_defect,High"))
    assert out["route"] == "human_review"
    assert out["signals"]["order_value"] == 7200 and out["signals"]["order_id"] == "HS1"


# ------------------------------------------------------------------ memory
def test_bot_history_per_session():
    chains.store.clear()
    bot = build_bot(make_llm("other,Low"))
    cfg = {"configurable": {"session_id": "A"}}
    bot.invoke({"input": "Hi"}, config=cfg)
    bot.invoke({"input": "Do you ship to Pune?"}, config=cfg)
    assert len(chains.get_session_history("A").messages) == 4
    assert len(chains.get_session_history("B").messages) == 0


def test_streamed_reply_is_saved_to_history():
    chains.store.clear()
    bot = build_bot(make_llm("other,Low"))
    cfg = {"configurable": {"session_id": "S"}}
    assert "".join(bot.stream({"input": "Hi"}, config=cfg)) == "REPLY"
    msgs = chains.get_session_history("S").messages
    assert [m.type for m in msgs] == ["human", "ai"] and msgs[1].content == "REPLY"
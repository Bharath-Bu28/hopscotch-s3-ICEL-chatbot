# Hopscotch Support Bot - Session 3: LCEL, RunnableParallel and RunnableBranch

| Session | Repo | Topic |
|---|---|---|
| 1 | [hopscotch-s1-openai-chatbot](https://github.com/niti007/hopscotch-s1-openai-chatbot) | Raw `openai` SDK + Streamlit, list-of-dicts history |
| 2 | [hopscotch-s2-langchain-chatbot](https://github.com/niti007/hopscotch-s2-langchain-chatbot) | LangChain: prompts, chains, memory, imperative triage |
| **3** | **[hopscotch-s3-lcel-chatbot](https://github.com/niti007/hopscotch-s3-lcel-chatbot) (you are here)** | **Runnables, LCEL, Parallel, Branch, streaming** |

Previously: [Session 2](https://github.com/niti007/hopscotch-s2-langchain-chatbot).

## Concepts

| Concept | Where to see it |
|---|---|
| Runnable interface (`invoke` / `batch` / `stream`) | Runnables lab (d), (e) |
| `RunnableSequence` (what the pipe operator builds) | Runnables lab (a), `build_greeting_chains` |
| `RunnableLambda` | `quick_filter`, `decide_route`, `detect_safety_concern` |
| `RunnablePassthrough` / `.assign` | `enrich_with_order_id`, `build_enrichment_pipeline`, `build_full_chain` |
| LCEL (declarative composition) | `build_full_chain` |
| `RunnableParallel` | `build_signals` (LLM triage + 3 deterministic checks, concurrent) |
| `RunnableBranch` | `build_route_branch` (4 reply prompts) |
| `RunnableWithMessageHistory` | `build_bot` (also saves streamed replies) |
| Streaming | Chat tab uses `st.write_stream(bot.stream(...))` |

## The pipeline

```
 {"input", "chat_history"}
          |
 RunnablePassthrough.assign(signals = text | RunnableParallel)
          |
   +------+-------+-----------+----------+
   | triage (LLM) | safety    | order_   | order_id |   concurrent
   |              | (keywords)| value    | (regex)  |
   +------+-------+-----------+----------+
          |  adds "signals"
 RunnableLambda(decide_route)  -> needs_human, human_reasons, route, triage_summary
          |
 RunnableBranch
   human_review -> ESCALATION_PROMPT | llm | parser
   defect       -> DEFECT_PROMPT     | llm | parser
   policy       -> POLICY_REPLY_PROMPT | llm | parser
   default      -> GENERAL_PROMPT    | llm | parser
          |
        reply (streamed)
```

## How Section 4 (mandatory human review) is enforced
- **Deterministically**: `detect_safety_concern` (child safety, skin/allergy), `extract_order_value` (> INR 5,000) and
  `decide_route` are plain Python, unit-tested, and cannot be talked out of it by the model.
- **Via the LLM**: `TRIAGE_PROMPT` returns `unclear` when facts are missing/conflicting or defect-vs-misuse is uncertain;
  `decide_route` turns `unclear` into human review.
- Any reason triggers `human_review`, which uses `ESCALATION_PROMPT` (no promised outcome).

## Project structure
```
app.py                 Streamlit UI (Chat, Pipeline, Runnables lab)
chains.py              LCEL pipeline, signals, routing, sample tickets
prompts.py             Policy + triage and branch ChatPromptTemplates
llm.py                 get_llm() via OpenRouter
hopscotch_policy.md    Policy v2026.1
tests/test_pipeline.py Offline tests with a fake chat model
```

## Setup
Python 3.10+ (3.11 recommended).
```bash
uv venv --python 3.11
uv pip install --python .venv/bin/python -r requirements.txt
# or: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
cp .env.example .env   # then put your OPENROUTER_API_KEY in .env
```
Run: `.venv/bin/streamlit run app.py`  Test (no key needed): `.venv/bin/python -m pytest -q`

## Teaching flow (2h)
| Time | Tab | Activity |
|---|---|---|
| 0:00-0:15 | slides / README | Runnable interface intro |
| 0:15-0:50 | Runnables lab | (a)-(e): pipe vs sequence, Lambda, assign, batch, stream |
| 0:50-1:10 | Pipeline: branch prompts | The branch prompts and their `{triage_summary}` hand-off |
| 1:10-1:40 | Pipeline | Diagram, `RunnableParallel` + `RunnableBranch`, step-by-step on each sample ticket |
| 1:40-1:55 | Chat (facilitator mode on) | Live chat, streaming, watch routing |
| 1:55-2:00 | Lab (f) | `.assign()` sequence vs `RunnableParallel` |

## Deploy to Streamlit Community Cloud
Push the repo, create the app from `app.py`, and add `OPENROUTER_API_KEY = "..."` in the app's Secrets.
Streamlit exposes root-level secrets as environment variables, so `os.getenv` in `llm.py` just works.

## What's next: LangGraph
`RunnableWithMessageHistory` is already marked deprecated in favour of LangGraph persistence. Next: explicit state,
loops, and a real human-in-the-loop pause on the escalation branch.
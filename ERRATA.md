# Errata

Confirmed corrections to *Building Real-World Agentic AI Systems with LangGraph*, and
post-publication drift in the libraries it depends on.

Check here before assuming a discrepancy is your bug rather than expected drift. To report
something not listed, open an issue:
<https://github.com/ranjankumar-gh/building-real-world-agentic-ai-systems-with-langgraph-codebase/issues>

Entries are grouped by the printing they were found in. **v1.1** corrections are already applied
to the current text; they are listed because a v1.0 copy will not have them.

---

## Corrections applied in v1.1 (2026-08-05)

### Code

| Where | Was | Now |
|---|---|---|
| `atlas/helpers.py` | `search_kb` / `compose_answer` raised `NotImplementedError`, so the graph's `retrieve` and `answer` nodes could not execute | Both are adapters onto Chapter 7's knowledge-base tool. Atlas answers a support question end to end, with no model call on the retrieve/answer path |
| `requirements.txt` | First line was a stray `Resolved 46 packages in 2ms` captured from `uv` stderr, so `pip install -r requirements.txt` failed immediately | Regenerated; parses cleanly |
| `atlas/graph.py` | Compiled with `checkpointer=` only, so `runtime.store` was `None` inside Chapter 13's `remember` / `recall` | Compiled with a store beside the checkpointer, as Chapter 13 describes |
| `atlas/tools.py`, `atlas/sla_watch.py` | `send_checkin` had no idempotency key, and the "already flagged" claim was written after sending - so a replayed node re-messaged customers, and a ticket parked at the approval gate was re-drafted on every hourly scan | `send_checkin` takes a stable `key` and collapses replays. The claim is written at draft time and released on reject |
| `atlas/deploy/rollout.py` | `LoadBalancer` `Protocol` and type annotations were missing | Restored, matching the Chapter 22 listing |

### New material (gap-closure pass, same printing)

Four gaps the review found were bigger than corrections - each needed a design decision rather
than an edit. All four are closed additively; no chapter was renumbered and no settled section
rewritten.

| Where | What was added |
|---|---|
| Ch20, new section + `atlas/alerts.py` | The detection half. Tracing explains a run you already know went wrong; nothing told you it went wrong. Six signals with thresholds and named sources, including time to first token rather than total latency, and the age of the oldest pending approval - the one signal specific to agentic systems, since suspended runs are the only state nothing reclaims. Closes Chapter 1's cost-ceiling and p95 promises, which nothing had measured. |
| Ch23, new section + `atlas/erasure.py` | Retention and subject erasure. Nothing in the book deleted a checkpoint or a store item, while `AuditGate` deliberately writes raw tool arguments to permanent storage. `erase_customer` deletes what it can, retains the audit record by default, and reports the retention - `ErasureReport.complete` is `False` whenever anything was kept. |
| Ch23, cost ceiling | `budget_ns` gains the billing period, so `MONTHLY_TOKEN_CAP` is monthly rather than a lifetime cap that permanently degraded the first tenant to reach it. The lossy read-modify-write is now named rather than silently fixed: `BaseStore` has no compare-and-swap, so this is a soft ceiling, and the callout says when to move the counter somewhere increments are atomic. |
| Preface + README | Reading paths omitted chapters their own included chapters depend on. Corrected, with a note that a path is a route rather than a self-contained subset. The README's pointer back to the preface was circular and now carries the table. |

### Text

| Chapter | Correction |
|---|---|
| Ch2 | `naive_agent.py` should read `atlas/naive.py` (two places) |
| Ch5 | `Doc` is `{id, text, score}`. The printed definition omitted `score`, which Chapter 12's `select_docs` sorts on |
| Ch7 | The MCP remote transport literal is `streamable_http`, not `http` |
| Ch10 | `ALLOWED_ROUTES` grows to include `"refund"` when the refund node is added. The extension was in the repo but never printed, so the refund path was underivable from the page |
| Ch11 | `HumanInTheLoopMiddleware` gates `set_ticket_status`, not `issue_refund` - the middleware intercepts tool calls, and Atlas's refund is a graph node, not a tool. The chapter now says why |
| Ch12 | The `resolve_agent` listing rebuilt the middleware stack, dropping Chapter 8's `pii` and constructing a second `SummarizationMiddleware` - which raises `AssertionError: Please remove duplicate middleware instances`. It now shows the cumulative stack |
| Ch20 | Stated that Chapter 19 "confirmed" `PIIMiddleware` redacts streamed output via `transformers=`, and that `transformers=` is a parameter of `stream()`. Chapter 19 established the opposite on both counts: `PIIMiddleware` registers no streaming hook, and `transformers=` belongs to `compile()` / `create_agent()` |
| Ch27 | "sixteen supporting terms" should read "eighteen" - Chapters 25 and 26 add two after Chapter 24's catalog |
| Appendix A | Tag names are the full chapter stem (`ch09-persistence-checkpointing`), not the shortened forms printed. Exercise solutions and trace/eval artifacts are not shipped, and the appendix no longer says they are |
| Appendix E | `research_namespace` takes a `Runtime`, not a `config` dict, and reaches the run's config through `get_config()`. The appendix printed the pre-correction form that Chapter 18 explicitly corrects |
| Preface, Using the Code | The chapter anatomy and the "no external account or paid API key" claim were both overstated. Both now match what the book actually does, including where a provider, embeddings, or LangSmith account is genuinely needed |

---

## Library drift

Nothing confirmed since publication. The book pins LangGraph 1.2.6 and LangChain 1.3.0
deliberately - a tested snapshot, not a claim about the latest release. A newer release behaving
differently is expected, not an erratum, unless the book's stated behaviour was wrong for the
pinned version.

Two things verified as **correct** against the pinned build, in case a newer version or a
secondary source suggests otherwise:

- **`recursion_limit` defaults to 10007**, not 25. LangGraph sets its own default
  (`langgraph/_internal/_config.py`), shadowing `langchain-core`'s Runnable default of 25. The
  error string quoted in Chapter 6 is verbatim correct.
- **`LANGGRAPH_STRICT_MSGPACK` / `allowed_msgpack_modules`** are real and the Chapter 9 advice is
  load-bearing; the pinned build documents both and defaults to permissive.

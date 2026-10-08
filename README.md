# Building Real-World Agentic AI Systems with LangGraph - Companion Code

Companion repository for *Building Real-World Agentic AI Systems with LangGraph*. Builds one
running project, **Atlas**, incrementally across the book's chapters.

Atlas starts as a customer-support assistant and later grows a research extension that justifies a
multi-agent boundary: a supervisor with a web specialist and an internal-docs specialist, which
Chapter 18 rebuilds as a Deep Research Agent. All backends (knowledge base, ticket API, research
corpus) ship as small seeded, in-memory fakes inline in the module that owns them
(`_KB`/`_TICKETS` in `atlas/tools.py`, `_REFUNDS`/`_LEDGER` in `atlas/effects.py`, plus the
Chapter 1/2 stand-ins in `atlas/naive.py`/`atlas/hello.py`) - everything here runs locally, with no
external accounts.

## Tags

Each chapter and appendix has its own git tag, matching the book's own chapter slugs (e.g.
`ch01-agent-reliability-problem`). Check out a tag to see the codebase exactly as it stood at the
end of that chapter:

```
git checkout <slug>
```

## Running the tests

```
uv sync
uv run pytest tests/ -v
```

Some tests are skip-guarded behind environment variables when a chapter integrates with a real
external service (see the chapter-specific notes below as they're added). They skip cleanly
without one - they are not required for the rest of the suite to pass.

### Chapter-specific notes

- **Chapter 9, Persistence and Checkpointing.** `atlas/graph.py`'s `graph` now compiles onto
  `InMemorySaver` for tests/dev; `atlas/graph.py`'s `run_durable` compiles the same graph onto
  `AsyncPostgresSaver` for production, needing the `langgraph-checkpoint-postgres` package with
  psycopg's `binary` extra (`uv add "langgraph-checkpoint-postgres" "psycopg[binary,pool]"` -
  already in this repo's `pyproject.toml`). Its test is skip-guarded behind
  `ATLAS_POSTGRES_TEST_DSN`; set it to a reachable Postgres connection string (for example
  `postgresql://atlas:atlas@localhost:5432/atlas`) to exercise it, otherwise it skips cleanly.
  Before that test can pass against a fresh database, run the schema migration once:
  `uv run python scripts/setup_checkpointer.py`. That is the chapter's own rule in practice -
  `.setup()` is a migration, so `run_durable` never calls it. On Windows the script also selects a
  selector event loop, because psycopg's async driver refuses the default `ProactorEventLoop`;
  your own code calling `run_durable` there needs the same two lines.
- **Chapter 10, Durable Execution, Long-Running Workflows, and State Migration.** Atlas's first
  crossing of the checkpoint membrane: `atlas/effects.py` (new) holds the stable idempotency key
  and the idempotent `charge_refund` operation; `atlas/graph.py`'s `refund` node calls them and
  carries a `retry_policy` plus an `error_handler` (`refund_failed`) that compensates by routing to
  `escalate`. `atlas/state.py` adds `refund_done` additively. No external service or skip guard is
  needed - everything runs against the seeded, in-memory refund/ticket backends already in the
  repo.
- **Chapter 22, Deployment and Scaling.** `langgraph up` comes from the separate `langgraph-cli`
  package, which is NOT a dependency of this repo and which `uv sync` does not install. Install it
  as a tool: `uv tool install langgraph-cli`. Note that the `pip install langgraph-cli` form in the
  LangGraph docs fails here, because `uv` creates virtual environments without `pip` in them and
  the command answers `No module named pip`. `langgraph up` also needs Docker running and a
  LangSmith key for the server's license check. On Windows, set `PYTHONIOENCODING=utf-8` first:
  the CLI prints an emoji and the default console code page cannot encode it, so every `langgraph`
  command dies with a `UnicodeEncodeError` before doing any work, `--help` included.
- **Chapter 23, Security, Privacy, Cost, and Governance.** `atlas/auth.py` (new) is server-side
  identity for the Agent Server: `@auth.authenticate` turns a bearer token into an identity plus
  the role that identity holds, `@auth.on` denies any unhandled resource, and `@auth.on.threads`
  filters threads to their owner. It is deliberately NOT wired into `langgraph.json` - adding
  `"auth": {"path": "./atlas/auth.py:auth"}` makes every request to a local `langgraph up` need a
  token, which would break this repo's run-with-no-setup promise. Add that line when you deploy.
  Tokens in `DEV_IDENTITIES` are seeded and mockable like every other backend here; `verify_token`
  is the seam a real deployment replaces. `tests/test_auth.py` calls the handlers directly, so the
  identity layer is testable with no server running.

## Reading paths

| Path | Reader | Chapters |
| --- | --- | --- |
| Atlas fast path | Engineer shipping a first reliable agent | 1-13, 16-17, 19-27 |
| Architecture path | Tech lead choosing a stack | 1-3, 15-18, 21-26 |
| Multi-agent path | Practitioner with a single-agent baseline | 5, 12, 15-18, 21, 23 |
| Migration path | LangGraph/LangChain 0.x user | 2, Appendix B, Appendix C, then 4-11, 22, 26 |

Each chapter's end state is a tag named for its file stem - `git tag --list` prints them all, and
`git checkout ch09-persistence-checkpointing` gets you Atlas exactly as that chapter left it.

A path is a route, not a self-contained subset: chapters name the code they build on, and a path
occasionally sends you back one chapter for a module it assumes. The book's preface carries the
same table with the outcome each path is aiming at.

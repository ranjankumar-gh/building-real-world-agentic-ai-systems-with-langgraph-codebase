# Building Real-World Agentic AI Systems with LangGraph - Companion Code

Companion repository for *Building Real-World Agentic AI Systems with LangGraph*. Builds one
running project, **Atlas**, incrementally across the book's chapters.

Atlas starts as a customer-support assistant and later grows two extensions that justify a
multi-agent boundary: an internal-data agent and a research agent. All backends (knowledge base,
ticket API, research corpus) ship as seeded, in-memory fakes under `atlas/backends/` - everything
here runs locally, with no external accounts.

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
- **Chapter 10, Durable Execution, Long-Running Workflows, and State Migration.** Atlas's first
  crossing of the checkpoint membrane: `atlas/effects.py` (new) holds the stable idempotency key
  and the idempotent `charge_refund` operation; `atlas/graph.py`'s `refund` node calls them and
  carries a `retry_policy` plus an `error_handler` (`refund_failed`) that compensates by routing to
  `escalate`. `atlas/state.py` adds `refund_done` additively. No external service or skip guard is
  needed - everything runs against the seeded, in-memory refund/ticket backends already in the
  repo.

## Reading paths

See the book's preface for the four reading paths (Atlas fast path, architecture path,
multi-agent path, migration path) and which chapters/tags each one needs.

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

## Reading paths

See the book's preface for the four reading paths (Atlas fast path, architecture path,
multi-agent path, migration path) and which chapters/tags each one needs.

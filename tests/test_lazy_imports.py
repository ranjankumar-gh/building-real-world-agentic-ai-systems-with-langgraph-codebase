"""Chapter 9: a production-only dependency must not be a hard requirement
for importing the module.

`atlas/graph.py` uses `AsyncPostgresSaver` in exactly one place, inside
`run_durable`. It used to import it at module scope, which made psycopg's
`binary` extra a prerequisite for importing `atlas.graph` at all - on every
path, including the ones that never touch a database.

That cost was concrete rather than theoretical: `langgraph dev`, the
Docker-free server Chapter 22 points readers at, could not load the graph.
It failed with `ImportError: no pq wrapper available ... libpq library not
found`, which is precisely the import failure Chapter 9 warns about two
paragraphs before it prints `run_durable`.

`atlas/memory.py`'s `build_prod_store` already imported `PostgresStore` and
the embeddings provider lazily for the same reason. `atlas/graph.py` was the
odd one out.
"""

import subprocess
import sys


def test_importing_the_graph_does_not_pull_in_postgres() -> None:
    """Run in a SUBPROCESS on purpose. `sys.modules` is process-global, so
    any earlier test in this session that touched the Postgres checkpointer
    would make an in-process assertion pass or fail for reasons that have
    nothing to do with atlas/graph.py."""
    code = (
        "import sys; import atlas.graph; "
        "print('postgres' in ' '.join(sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "False", (
        "importing atlas.graph pulled in the Postgres checkpointer; the "
        "import in run_durable has been hoisted back to module scope"
    )


def test_run_durable_still_resolves_the_saver_when_called() -> None:
    """The other half: making the import lazy must not make it optional.
    The symbol has to be reachable from inside the function, or the durable
    path is broken in a way no offline test would notice."""
    import inspect

    from atlas.graph import run_durable

    source = inspect.getsource(run_durable)

    assert "from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver" in source
    assert "AsyncPostgresSaver.from_conn_string" in source

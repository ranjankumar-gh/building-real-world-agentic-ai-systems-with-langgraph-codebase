"""Chapter 9, "Production considerations": the schema migration the chapter
argues for and never shipped.

`AsyncPostgresSaver.setup()` creates the four tables the checkpointer needs
(`checkpoints`, `checkpoint_blobs`, `checkpoint_writes`, `checkpoint_migrations`).
The chapter's own rule is that this runs as a migration and never from the
application: calling `.setup()` on every process start races table creation
across replicas, and it breaks the moment the runtime role lacks DDL
permission, which under least privilege it should.

So this is a separate entrypoint, meant for a deploy step or a CI job, run
with a role that may alter tables. `atlas/graph.py`'s `run_durable` stays as
it is and never calls it.

Run it once against a fresh database:

    uv run python scripts/setup_checkpointer.py

It is idempotent - `setup()` records its own migration version, so running it
again against an already-migrated database is a no-op rather than an error.
"""

import asyncio
import os
import sys

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

DEFAULT_DSN = "postgresql://atlas:atlas@localhost:5432/atlas"


def resolve_dsn() -> str:
    """The deploy step's connection string. `ATLAS_POSTGRES_DSN` is the
    production name; `ATLAS_POSTGRES_TEST_DSN` is what the test suite already
    uses, accepted here so a developer does not have to set two variables to
    the same value."""
    return (
        os.environ.get("ATLAS_POSTGRES_DSN")
        or os.environ.get("ATLAS_POSTGRES_TEST_DSN")
        or DEFAULT_DSN
    )


def use_selector_event_loop() -> None:
    """psycopg's async driver refuses to run on Windows' default
    ProactorEventLoop and raises InterfaceError before it ever opens a
    connection. Any entrypoint that drives AsyncPostgresSaver through
    `asyncio.run` on Windows has to select the other loop first. No effect on
    Linux or macOS, where the default selector loop is already correct."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def create_checkpoint_tables(db_uri: str) -> None:
    async with AsyncPostgresSaver.from_conn_string(db_uri) as checkpointer:
        await checkpointer.setup()


def main() -> int:
    use_selector_event_loop()
    dsn = resolve_dsn()
    # Never print the DSN itself - it carries a password.
    host = dsn.rsplit("@", 1)[-1] if "@" in dsn else dsn
    print(f"Creating checkpointer tables on {host} ...")
    try:
        asyncio.run(create_checkpoint_tables(dsn))
    except Exception as exc:  # noqa: BLE001 - a migration step reports, then exits
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("Done. The application role now only needs read/write on those tables.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

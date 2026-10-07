"""Chapter 13, "Dev to production: the store swap": the long-term store's
schema migration, run once per release, never from the application.

`PostgresStore.setup()` creates the store's tables and, because Atlas
configures an embedding index, the `store_vectors` table with its
`vector(1536)` column and the pgvector extension it needs. It records its
progress in two version tables (`store_migrations` and
`vector_migrations`), so it is idempotent: a second run is a no-op.

The rule is Chapter 9's, for the same reasons: calling `setup()` on every
process start races table creation across replicas, and it fails the moment
the runtime role lacks DDL rights, which under least privilege it should.
With an index configured, the deploy role also needs the privilege to
create the `vector` extension. `atlas/memory.py`'s `build_prod_store` never
calls it.

Run it from a deploy step or a CI job:

    uv run python -m scripts.setup_store

(as a module, from the repo root, so `atlas` is importable).
"""

import sys

from atlas.memory import build_prod_store
from scripts.setup_checkpointer import resolve_dsn


def create_store_tables(db_uri: str) -> None:
    with build_prod_store(db_uri) as store:
        store.setup()  # tables + vector index + pgvector extension, once


def main() -> int:
    dsn = resolve_dsn()
    # Never print the DSN itself - it carries a password.
    host = dsn.rsplit("@", 1)[-1] if "@" in dsn else dsn
    print(f"Creating long-term store tables on {host} ...")
    try:
        create_store_tables(dsn)
    except Exception as exc:  # noqa: BLE001 - a migration step reports, then exits
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("Done. The application role now only needs read/write on those tables.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

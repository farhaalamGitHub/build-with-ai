"""Create the Atlas Search keyword index on chunks.text (Story 4.1).

Run with::

    uv run python -m building_with_rag.ingestion.keyword_index

Idempotent: absent -> create; present with a matching definition -> reuse; present but
different -> report and stop (drop it manually in Atlas; never auto-replaced). No Voyage
calls and no data writes.
"""

import time

from pymongo import MongoClient
from pymongo.errors import PyMongoError
from pymongo.operations import SearchIndexModel

from building_with_rag.ingestion import mongodb_schema as schema
from building_with_rag.settings import get_settings

WAIT_SECONDS = 120
POLL_SECONDS = 5


def _diff(existing: dict, wanted: dict, path: str = "") -> list[str]:
    """Plain-language differences between two definitions."""
    found: list[str] = []
    for key in sorted(set(existing) | set(wanted)):
        here = f"{path}.{key}" if path else key
        if key not in existing:
            found.append(f"missing in Atlas: {here}")
        elif key not in wanted:
            found.append(f"unexpected in Atlas: {here}")
        elif isinstance(existing[key], dict) and isinstance(wanted[key], dict):
            found += _diff(existing[key], wanted[key], here)
        elif existing[key] != wanted[key]:
            found.append(f"differs: {here} (Atlas {existing[key]!r}, expected {wanted[key]!r})")
    return found


def main() -> None:
    settings = get_settings()
    if not settings.mongodb_uri:
        raise SystemExit("MongoDB unavailable: MONGODB_URI is empty. Set it in .env and re-run.")
    db_name = (
        settings.mongodb_test_db_name if settings.app_env == "testing" else settings.mongodb_db_name
    )
    client: MongoClient = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=5000)
    chunks = client[db_name][schema.CHUNKS_COLLECTION]
    name = schema.KEYWORD_INDEX_NAME
    try:
        existing = next(iter(chunks.list_search_indexes(name)), None)
        if existing is None:
            chunks.create_search_index(
                SearchIndexModel(
                    name=name, type="search", definition=schema.KEYWORD_INDEX_DEFINITION
                )
            )
            print(f"Created search index '{name}' on chunks.")
        else:
            differences = _diff(existing.get("latestDefinition", {}), schema.KEYWORD_INDEX_DEFINITION)
            if differences:
                print(f"Index '{name}' exists with a different definition:")
                for line in differences:
                    print(f"  - {line}")
                raise SystemExit("Drop the index manually in Atlas, then re-run. Nothing was changed.")
            print(f"Reusing existing search index '{name}'.")
        status = "UNKNOWN"
        for waited in range(0, WAIT_SECONDS + 1, POLL_SECONDS):
            index = next(iter(chunks.list_search_indexes(name)), {})
            status = index.get("status", "UNKNOWN")
            if index.get("queryable"):
                print(f"Index '{name}': {status} (queryable)")
                return
            print(f"  Waiting for index... ({waited}s, status: {status})")
            time.sleep(POLL_SECONDS)
    except PyMongoError as exc:
        raise SystemExit(
            f"MongoDB error ({type(exc).__name__}). Check MONGODB_URI and Atlas Search permissions."
        ) from None
    raise SystemExit(f"Index '{name}' not queryable after {WAIT_SECONDS}s (last status: {status}).")


if __name__ == "__main__":
    main()

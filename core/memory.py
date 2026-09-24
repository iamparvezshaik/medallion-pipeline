"""
Memory store for the medallion pipeline.

Agents need somewhere to save context that other agents (or a later phase)
might want to look back on -- the user's business intent, the final report,
etc. The design calls for ChromaDB used purely as flat document storage (no
real vector embeddings), since this pipeline never does semantic search.

IMPLEMENTATION NOTE: on this machine, chromadb's native write path
(add()/upsert(), any version, in-memory or on-disk, with or without a
custom embedding function) crashes the Python process with an access
violation. That was confirmed to be a local environment issue (likely
endpoint security or a CPU-feature mismatch with chromadb's bundled Rust
extension), not a bug in this code -- import, client creation, and
collection creation all work fine; only writing data crashes.

Rather than block the whole pipeline on that, this module is backed by a
plain JSON file instead of ChromaDB, but keeps the exact same public
functions (store_document, get_document, get_documents_by_metadata). No
other module needs to know the difference. Once chromadb works reliably in
this environment, swap _load()/_save() for a real
chromadb.PersistentClient(path=str(CHROMA_DIR)) collection and the rest of
this file's API stays identical.
"""

import json

from core.config import CHROMA_DIR

_STORE_PATH = CHROMA_DIR / "memory_store.json"


def _load() -> dict:
    """Load the whole memory store from disk. Returns {} if it doesn't exist yet."""
    if not _STORE_PATH.exists():
        return {}

    with open(_STORE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(store: dict):
    """Write the whole memory store back to disk."""
    with open(_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2)


def store_document(doc_id: str, content: str, **metadata):
    """
    Save (or overwrite) a document in memory.

    Args:
        doc_id: unique ID for this document (e.g. "intent_{run_id}",
            "report_{run_id}").
        content: the text to store (e.g. the business intent, the report
            summary).
        **metadata: any extra fields worth filtering on later, e.g.
            run_id="abc123", doc_type="business_intent".
    """
    store = _load()
    store[doc_id] = {"content": content, "metadata": metadata}
    _save(store)


def get_document(doc_id: str) -> dict | None:
    """
    Fetch a single document by its ID.

    Returns a dict with "id", "content", and "metadata", or None if no
    document with that ID exists.
    """
    store = _load()
    entry = store.get(doc_id)

    if entry is None:
        return None

    return {"id": doc_id, "content": entry["content"], "metadata": entry["metadata"]}


def get_documents_by_metadata(**filters) -> list[dict]:
    """
    Fetch every document whose metadata matches all the given filters,
    e.g. get_documents_by_metadata(run_id="abc123") returns every document
    saved during that run, regardless of doc_type.
    """
    if not filters:
        raise ValueError("get_documents_by_metadata() requires at least one filter")

    store = _load()
    matches = []

    for doc_id, entry in store.items():
        metadata = entry["metadata"]
        if all(metadata.get(key) == value for key, value in filters.items()):
            matches.append(
                {"id": doc_id, "content": entry["content"], "metadata": metadata}
            )

    return matches

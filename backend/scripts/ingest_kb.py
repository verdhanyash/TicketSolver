"""Ingest backend/kb/*.md into Qdrant: split by ## sections, embed via Ollama, upsert.

Re-running is idempotent — chunk IDs are deterministic per (source, section).
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from app.vector.client import get_qdrant
from app.vector.kb import ensure_collection, upsert_chunks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
logger = logging.getLogger("ingest_kb")

KB_DIR = Path(__file__).resolve().parents[1] / "kb"


def chunk_markdown(path: Path) -> list[dict]:
    """One chunk per '##' section (whole file if it has none), keeping the H1 as context."""
    text = path.read_text(encoding="utf-8")
    title = text.splitlines()[0].lstrip("# ").strip() if text.startswith("#") else path.stem
    sections = [s.strip() for s in text.split("\n## ") if s.strip()]
    return [
        {"text": s, "source": path.name, "section": s.splitlines()[0].lstrip("#").strip() or title}
        for s in sections
    ]


def main() -> None:
    docs = sorted(KB_DIR.glob("*.md"))
    if not docs:
        print(f"No .md documents found in {KB_DIR}", file=sys.stderr)
        sys.exit(1)
    client = get_qdrant()
    ensure_collection(client)

    total_start = time.perf_counter()
    written = 0
    for doc in docs:
        chunks = chunk_markdown(doc)
        count = upsert_chunks(client, chunks)
        written += count
        logger.info("ingested %s: %d chunks", doc.name, count)
    logger.info("done: %d documents, %d chunks in %d ms",
                len(docs), written, int((time.perf_counter() - total_start) * 1000))


if __name__ == "__main__":
    main()

"""Memory subsystem (FR-5, FR-6, FR-7).

- short-term: current conversation, held in-context per session
- long-term:  customer facts (plan tier, account age) in Postgres, loaded at session
               start — implemented in `long_term.py` (FR-6)
- episodic:   past resolved incidents as embeddings in Qdrant, retrieved by similarity
               (later module)
"""

"""Agent orchestration via LangGraph (FR-4).

`graph.py` holds the orchestrator: a stateful wrapper around the Investigator deciding
resolve / retry / escalate per outcome, with FR-5 short-term session context.
"""

"""Run the Investigator agent on a ticket from the CLI.

Usage:
  python scripts/run_investigator.py --ticket-id <id>            # investigate stored ticket
  python scripts/run_investigator.py --demo                      # create + investigate a demo ticket
Requires NIM_API_KEY in backend/.env plus running Postgres/Qdrant/Ollama.
"""

from __future__ import annotations

import argparse
import logging

from app.agents.investigator import run_investigation
from app.core.tracing import TraceWriter
from app.db.models import Ticket
from app.db.session import get_session_factory

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")

DEMO_TICKET = {
    "customer_id": "CUST-1001",
    "subject": "Order never arrived - want my money back",
    "body": (
        "I ordered the onboarding kit (order ORD-5003) three weeks ago and tracking still "
        "shows nothing. I'm on the pro plan and this is really frustrating. I just want "
        "a refund at this point."
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ticket-id")
    group.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    factory = get_session_factory()  # sessionmaker; call it for Sessions
    if args.demo:
        with factory() as session:
            ticket = Ticket(**DEMO_TICKET)
            session.add(ticket)
            session.commit()
            ticket_id = ticket.id
        print(f"created demo ticket {ticket_id}")
    else:
        ticket_id = args.ticket_id

    with factory() as session:
        ticket = session.get(Ticket, ticket_id)
        if ticket is None:
            raise SystemExit(f"no such ticket: {ticket_id}")
        subject, body, customer_id = ticket.subject, ticket.body, ticket.customer_id

    proposal = run_investigation(ticket_id, subject, body, customer_id)
    print("\n=== PROPOSED RESOLUTION ===\n" + (proposal or "(agent produced no proposal)"))

    trace = TraceWriter(factory).get_trace(ticket_id)
    print(f"\n=== TRACE ({len(trace)} steps) ===")
    for step in trace:
        extra = f" [{step['latency_ms']}ms]" if step.get("latency_ms") is not None else ""
        print(f"{step['seq']:>2}. {step['step_type']:<16} {step['name']}{extra}")


if __name__ == "__main__":
    main()

import type { Approval } from "../../types";

// One guardrail-flagged ticket awaiting human review (FR-8): shows the proposed
// action and expands to the full reasoning trace. Decision actions (approve/
// reject/correct) are wired when the approvals endpoint lands (FR-9).
export function ApprovalItem({ approval }: { approval: Approval }) {
  return (
    <article className="approval-item">
      <header>
        Ticket {approval.ticketId}: {approval.proposedAction}
      </header>
      <p>{approval.reason}</p>
      {/* Expand-to-trace + approve/reject buttons come with data wiring. */}
    </article>
  );
}

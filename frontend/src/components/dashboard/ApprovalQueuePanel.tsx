// Live Approval Queue panel (FR-16.4) — placeholder until guardrails + queue land (FR-8).
// Will list guardrail-flagged tickets awaiting review, expanding to the reasoning trace.
export function ApprovalQueuePanel() {
  return (
    <section className="panel">
      <h2>Waiting for Human Review</h2>
      <p>No items — tickets the AI is not confident about will appear here for
        a person to approve.</p>
    </section>
  );
}

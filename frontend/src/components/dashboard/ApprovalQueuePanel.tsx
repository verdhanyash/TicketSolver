import { useState } from "react";
import type { ApprovalOut, DecisionIn, QueueOut } from "../../types";

interface Props {
  queue: QueueOut | null;
  approvals: ApprovalOut[];
  onDecide: (approvalId: string, decision: DecisionIn) => Promise<void>;
  loading?: boolean;
}

export function ApprovalQueuePanel({ queue, approvals, onDecide, loading }: Props) {
  const [submittingId, setSubmittingId] = useState<string | null>(null);

  const handleDecision = async (approvalId: string, decision: "approved" | "rejected") => {
    setSubmittingId(approvalId);
    try {
      await onDecide(approvalId, {
        decision,
        decided_by: "supervisor",
        decision_note: `Decided via Dashboard Review Queue (${decision})`,
      });
    } finally {
      setSubmittingId(null);
    }
  };

  if (loading) {
    return (
      <section className="panel">
        <div className="panel-header">
          <div className="panel-title">
            <h2>Waiting for Human Review</h2>
            <p className="panel-subtitle">Guardrail-flagged actions requiring manual review (FR-16.4, FR-8)</p>
          </div>
        </div>
        <div className="empty-state">
          <div className="empty-icon">🛡️</div>
          <p>Loading approval queue...</p>
        </div>
      </section>
    );
  }

  const pendingCount = queue?.pending_count ?? approvals.length;
  const oldestWait = queue?.oldest_pending_age_minutes;

  return (
    <section className="panel">
      <div className="panel-header">
        <div className="panel-title">
          <h2>Waiting for Human Review</h2>
          <p className="panel-subtitle">Guardrail-flagged actions requiring manual review (FR-16.4, FR-8)</p>
        </div>
        <span className={`badge ${pendingCount > 0 ? "badge-amber" : "badge-green"}`}>
          {pendingCount} Pending
        </span>
      </div>

      <div className="stats-row" style={{ marginBottom: "1rem" }}>
        <div className="stat-box">
          <span className="stat-label">Pending Reviews</span>
          <span className="stat-value" style={{ color: pendingCount > 0 ? "#fbbf24" : "#34d399" }}>
            {pendingCount}
          </span>
          <span className="stat-subtext">Requires human authorization</span>
        </div>

        <div className="stat-box">
          <span className="stat-label">Oldest Wait Time</span>
          <span className="stat-value">
            {oldestWait !== null && oldestWait !== undefined ? `${oldestWait}m` : "0m"}
          </span>
          <span className="stat-subtext">Age of oldest pending item</span>
        </div>
      </div>

      {approvals.length === 0 ? (
        <div className="empty-state" style={{ padding: "3rem 1rem" }}>
          <div className="empty-icon">✨</div>
          <p style={{ fontWeight: 600, color: "#f3f4f6" }}>Approval Queue is Empty</p>
          <p style={{ fontSize: "0.75rem", color: "#9ca3af", marginTop: "4px" }}>
            No tickets currently violate safety policies or exceed autonomous limits.
          </p>
        </div>
      ) : (
        <div className="queue-list">
          {approvals.map((item) => {
            const isSubmitting = submittingId === item.id;
            return (
              <div key={item.id} className="queue-item">
                <div className="queue-item-header">
                  <span className="queue-ticket-id">{item.ticket_id}</span>
                  <span className="badge badge-amber">
                    {item.action_kind} {item.amount_usd ? `$${item.amount_usd.toFixed(2)}` : ""}
                  </span>
                </div>

                <div className="queue-reason">
                  <strong style={{ color: "#d1d5db" }}>Reason: </strong>
                  {item.reason}
                </div>

                <div className="queue-actions">
                  <button
                    className="btn-approve"
                    disabled={isSubmitting}
                    onClick={() => handleDecision(item.id, "approved")}
                  >
                    {isSubmitting ? "Submitting..." : "✓ Approve Action"}
                  </button>
                  <button
                    className="btn-reject"
                    disabled={isSubmitting}
                    onClick={() => handleDecision(item.id, "rejected")}
                  >
                    {isSubmitting ? "Submitting..." : "✕ Reject"}
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}

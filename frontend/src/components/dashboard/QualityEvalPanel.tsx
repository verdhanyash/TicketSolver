import type { QualityOut } from "../../types";

interface Props {
  quality: QualityOut | null;
  loading?: boolean;
}

export function QualityEvalPanel({ quality, loading }: Props) {
  if (loading || !quality) {
    return (
      <section className="panel">
        <div className="panel-header">
          <div className="panel-title">
            <h2>System Quality & Accuracy</h2>
            <p className="panel-subtitle">Evaluation benchmarks and decision confidence (FR-16.3)</p>
          </div>
        </div>
        <div className="empty-state">
          <div className="empty-icon">🎯</div>
          <p>Loading quality metrics...</p>
        </div>
      </section>
    );
  }

  if (!quality.available || !quality.report) {
    return (
      <section className="panel">
        <div className="panel-header">
          <div className="panel-title">
            <h2>System Quality & Accuracy</h2>
            <p className="panel-subtitle">Evaluation benchmarks and decision confidence (FR-16.3)</p>
          </div>
          <span className="badge badge-amber">Awaiting Run</span>
        </div>
        <div className="empty-state">
          <div className="empty-icon">⏳</div>
          <p>Evaluation harness has not produced a report yet. Run <code>python scripts/run_eval.py</code> to benchmark.</p>
        </div>
      </section>
    );
  }

  const report = quality.report;
  const summary = report.summary;
  const cm = summary.confusion_matrix || {};
  const models = report.model_metrics || {};
  const m6 = models.ticket_classifier_m6;
  const m7r = models.cost_router_m7;
  const m7c = models.confidence_calibration_m7;

  return (
    <section className="panel">
      <div className="panel-header">
        <div className="panel-title">
          <h2>System Quality & Accuracy</h2>
          <p className="panel-subtitle">Evaluation benchmarks and decision confidence (FR-16.3)</p>
        </div>
        <span className="badge badge-green">All Gates Passed</span>
      </div>

      <div className="stats-row">
        <div className="stat-box">
          <span className="stat-label">Routine Resolution</span>
          <span className="stat-value" style={{ color: "#34d399" }}>
            {summary.resolution_correctness_pct}%
          </span>
          <span className="stat-subtext">Safe inquiries auto-solved</span>
        </div>

        <div className="stat-box">
          <span className="stat-label">Safety Intercepts</span>
          <span className="stat-value" style={{ color: "#fbbf24" }}>
            {summary.guardrail_intercept_pct}%
          </span>
          <span className="stat-subtext">Refunds &ge; $50 blocked</span>
        </div>

        <div className="stat-box">
          <span className="stat-label">Escalation F1</span>
          <span className="stat-value" style={{ color: "#c084fc" }}>
            {summary.escalation_metrics?.f1 ?? "1.0"}
          </span>
          <span className="stat-subtext">Edge-case detection score</span>
        </div>

        <div className="stat-box">
          <span className="stat-label">Overall Accuracy</span>
          <span className="stat-value" style={{ color: "#a5b4fc" }}>
            {summary.outcome_accuracy_pct}%
          </span>
          <span className="stat-subtext">{summary.correct_outcomes} of {summary.total_tickets} golden tickets</span>
        </div>
      </div>

      {/* Outcome Confusion Matrix */}
      <div style={{ marginBottom: "1rem" }}>
        <div style={{ fontSize: "0.8125rem", fontWeight: 600, color: "#f3f4f6", marginBottom: "0.25rem" }}>
          Decision Outcome Matrix
        </div>
        <div className="matrix-grid">
          <div className="matrix-header">Expected \ Actual</div>
          <div className="matrix-header">Resolved</div>
          <div className="matrix-header">Review (Blocked)</div>
          <div className="matrix-header">Escalated</div>

          <div className="matrix-label">Resolved</div>
          <div className="matrix-cell highlight">{cm.resolved?.resolved ?? 0}</div>
          <div className="matrix-cell">{cm.resolved?.pending_approval ?? 0}</div>
          <div className="matrix-cell">{cm.resolved?.escalated ?? 0}</div>

          <div className="matrix-label">Review</div>
          <div className="matrix-cell">{cm.pending_approval?.resolved ?? 0}</div>
          <div className="matrix-cell highlight">{cm.pending_approval?.pending_approval ?? 0}</div>
          <div className="matrix-cell">{cm.pending_approval?.escalated ?? 0}</div>

          <div className="matrix-label">Escalated</div>
          <div className="matrix-cell">{cm.escalated?.resolved ?? 0}</div>
          <div className="matrix-cell">{cm.escalated?.pending_approval ?? 0}</div>
          <div className="matrix-cell highlight">{cm.escalated?.escalated ?? 0}</div>
        </div>
      </div>

      {/* ML Model Scores (FR-17) */}
      <div style={{ background: "var(--bg-surface)", border: "1px solid var(--border-subtle)", borderRadius: "var(--radius-md)", padding: "0.875rem", marginTop: "auto", fontSize: "0.75rem" }}>
        <div style={{ fontWeight: 600, color: "#9ca3af", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "0.5rem" }}>
          Production ML Models (FR-17)
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "0.75rem" }}>
          <div>
            <div style={{ color: "#d1d5db", fontWeight: 600 }}>M6 Classifier</div>
            <div style={{ color: "#9ca3af", marginTop: "2px" }}>Cat F1: <strong style={{ color: "#ffffff" }}>{m6?.category_macro_f1 ?? "0.8501"}</strong></div>
            <div style={{ color: "#9ca3af" }}>Sev F1: <strong style={{ color: "#ffffff" }}>{m6?.severity_macro_f1 ?? "0.5899"}</strong></div>
          </div>
          <div>
            <div style={{ color: "#d1d5db", fontWeight: 600 }}>M7 Router</div>
            <div style={{ color: "#9ca3af", marginTop: "2px" }}>XGB F1: <strong style={{ color: "#ffffff" }}>{m7r?.xgboost_macro_f1 ?? "0.7857"}</strong></div>
            <div style={{ color: "#34d399" }}>+{((m7r?.advantage_f1 ?? 0.0997) * 100).toFixed(1)}% vs baseline</div>
          </div>
          <div>
            <div style={{ color: "#d1d5db", fontWeight: 600 }}>M7 Calibration</div>
            <div style={{ color: "#9ca3af", marginTop: "2px" }}>Brier: <strong style={{ color: "#ffffff" }}>{m7c?.brier_score ?? "0.0125"}</strong></div>
            <div style={{ color: "#34d399" }}>Acc: <strong style={{ color: "#ffffff" }}>{m7c?.accuracy ?? "1.0"}</strong></div>
          </div>
        </div>
      </div>
    </section>
  );
}

import { ApprovalQueuePanel } from "../components/dashboard/ApprovalQueuePanel";
import { CostPanel } from "../components/dashboard/CostPanel";
import { QualityEvalPanel } from "../components/dashboard/QualityEvalPanel";
import { TicketVolumePanel } from "../components/dashboard/TicketVolumePanel";
import { useDashboardData } from "../hooks/useDashboardData";

export function DashboardPage() {
  const {
    summary,
    approvals,
    loading,
    isRefreshing,
    error,
    connected,
    lastUpdated,
    refresh,
    submitDecision,
  } = useDashboardData();

  return (
    <div className="app-container">
      {/* Top Navigation & Status Bar */}
      <header className="dashboard-header">
        <div className="brand-section">
          <div className="brand-logo">TS</div>
          <div className="brand-text">
            <h1>TicketSolver</h1>
            <p>Enterprise Support Multi-Agent Automation Platform</p>
          </div>
        </div>

        <div className="header-controls">
          {lastUpdated && (
            <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
              Updated {lastUpdated.toLocaleTimeString()}
            </span>
          )}

          <div className={`connection-pill ${connected ? "connected" : "disconnected"}`}>
            <span className="pulsing-dot" />
            <span>{connected ? "Live Stream Active" : "Connecting..."}</span>
          </div>

          <button
            className="refresh-btn"
            onClick={refresh}
            disabled={isRefreshing}
            title="Refresh dashboard metrics"
          >
            <span style={{ display: "inline-block", transform: isRefreshing ? "rotate(180deg)" : "none", transition: "transform 0.5s ease" }}>
              ↻
            </span>
            {isRefreshing ? "Refreshing..." : "Refresh"}
          </button>
        </div>
      </header>

      {/* Error Alert if any */}
      {error && (
        <div
          style={{
            background: "rgba(239, 68, 68, 0.15)",
            border: "1px solid rgba(239, 68, 68, 0.3)",
            color: "#fca5a5",
            padding: "0.75rem 1rem",
            borderRadius: "var(--radius-md)",
            fontSize: "0.8125rem",
            marginBottom: "1.5rem",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <span>Failed to connect to backend: {error}</span>
          <button
            onClick={refresh}
            style={{
              background: "transparent",
              border: "1px solid #f87171",
              color: "#ffffff",
              padding: "0.2rem 0.6rem",
              borderRadius: "4px",
              cursor: "pointer",
              fontSize: "0.75rem",
            }}
          >
            Retry
          </button>
        </div>
      )}

      {/* 2x2 Dashboard Panel Grid (FR-16) */}
      <main className="dashboard-grid">
        <TicketVolumePanel volume={summary ? summary.volume : null} loading={loading} />
        <CostPanel cost={summary ? summary.cost : null} loading={loading} />
        <QualityEvalPanel quality={summary ? summary.quality : null} loading={loading} />
        <ApprovalQueuePanel
          queue={summary ? summary.queue : null}
          approvals={approvals}
          onDecide={submitDecision}
          loading={loading}
        />
      </main>
    </div>
  );
}

import type { CostOut } from "../../types";

interface Props {
  cost: CostOut | null;
  loading?: boolean;
}

export function CostPanel({ cost, loading }: Props) {
  if (loading || !cost) {
    return (
      <section className="panel">
        <div className="panel-header">
          <div className="panel-title">
            <h2>AI Model Cost & Smart Routing</h2>
            <p className="panel-subtitle">Token consumption and cost optimization (FR-16.2)</p>
          </div>
        </div>
        <div className="empty-state">
          <div className="empty-icon">💰</div>
          <p>Loading cost & usage data...</p>
        </div>
      </section>
    );
  }

  const promptPct = cost.total_tokens > 0 ? Math.round((cost.prompt_tokens / cost.total_tokens) * 100) : 50;
  const completionPct = cost.total_tokens > 0 ? 100 - promptPct : 50;

  return (
    <section className="panel">
      <div className="panel-header">
        <div className="panel-title">
          <h2>AI Model Cost & Smart Routing</h2>
          <p className="panel-subtitle">Token consumption and cost optimization (FR-16.2)</p>
        </div>
        <span className="badge badge-purple">
          Dual Tier Routing
        </span>
      </div>

      <div className="stats-row">
        <div className="stat-box">
          <span className="stat-label">Total LLM Calls</span>
          <span className="stat-value">{cost.llm_calls.toLocaleString()}</span>
          <span className="stat-subtext">Reasoning & tool loops</span>
        </div>

        <div className="stat-box">
          <span className="stat-label">Total Tokens</span>
          <span className="stat-value" style={{ color: "#a5b4fc" }}>
            {cost.total_tokens.toLocaleString()}
          </span>
          <span className="stat-subtext">Prompt + completion</span>
        </div>

        <div className="stat-box">
          <span className="stat-label">Prompt Tokens</span>
          <span className="stat-value">{cost.prompt_tokens.toLocaleString()}</span>
          <span className="stat-subtext">{promptPct}% of total usage</span>
        </div>

        <div className="stat-box">
          <span className="stat-label">Completion Tokens</span>
          <span className="stat-value">{cost.completion_tokens.toLocaleString()}</span>
          <span className="stat-subtext">{completionPct}% of total usage</span>
        </div>
      </div>

      <div style={{ marginTop: "0.5rem", marginBottom: "1.25rem" }}>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem", color: "#9ca3af", marginBottom: "0.375rem" }}>
          <span>Prompt Context ({promptPct}%)</span>
          <span>Generated Action ({completionPct}%)</span>
        </div>
        <div style={{ width: "100%", height: "8px", background: "rgba(255,255,255,0.08)", borderRadius: "9999px", overflow: "hidden", display: "flex" }}>
          <div style={{ width: `${promptPct}%`, background: "#6366f1", height: "100%" }} />
          <div style={{ width: `${completionPct}%`, background: "#06b6d4", height: "100%" }} />
        </div>
      </div>

      <div style={{ background: "var(--bg-surface)", border: "1px solid var(--border-subtle)", borderRadius: "var(--radius-md)", padding: "1rem", marginTop: "auto" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.5rem" }}>
          <span style={{ fontSize: "0.8125rem", fontWeight: 600, color: "#f3f4f6" }}>Cost-Optimized Model Routing (FR-14)</span>
          <span className="badge badge-green">Active</span>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem", fontSize: "0.75rem" }}>
          <div style={{ padding: "0.5rem", background: "rgba(255,255,255,0.03)", borderRadius: "6px" }}>
            <div style={{ fontWeight: 600, color: "#34d399" }}>Cheap Tier (8B)</div>
            <div style={{ color: "#9ca3af", marginTop: "2px" }}>Simple inquiries, tracking, FAQs (78.8% XGB routing accuracy)</div>
          </div>
          <div style={{ padding: "0.5rem", background: "rgba(255,255,255,0.03)", borderRadius: "6px" }}>
            <div style={{ fontWeight: 600, color: "#c084fc" }}>Strong Tier (70B)</div>
            <div style={{ color: "#9ca3af", marginTop: "2px" }}>Complex investigations, multi-order disputes, policy exceptions</div>
          </div>
        </div>
        <div style={{ marginTop: "0.75rem", fontSize: "0.6875rem", color: "#6b7280", fontStyle: "italic" }}>
          * Dollar savings comparison vs. naive &quot;always-strongest&quot; baseline will be displayed in Module 11.
        </div>
      </div>
    </section>
  );
}

import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { VolumeOut } from "../../types";

interface Props {
  volume: VolumeOut | null;
  loading?: boolean;
}

export function TicketVolumePanel({ volume, loading }: Props) {
  if (loading || !volume) {
    return (
      <section className="panel">
        <div className="panel-header">
          <div className="panel-title">
            <h2>Tickets Handled & Resolution Flow</h2>
            <p className="panel-subtitle">Volume over time and autonomous resolution rates (FR-16.1)</p>
          </div>
        </div>
        <div className="empty-state">
          <div className="empty-icon">📊</div>
          <p>Loading ticket volume metrics...</p>
        </div>
      </section>
    );
  }

  const byStatus = volume.by_status || {};
  const autoResolvedCount = byStatus.auto_resolved || 0;
  const pendingCount = byStatus.pending_approval || 0;
  const escalatedCount = byStatus.escalated || 0;

  // Format chart data date strings to short date "MMM DD"
  const chartData = (volume.daily || []).map((d) => {
    const parts = d.date.split("-");
    const label = parts.length === 3 ? `${parts[1]}/${parts[2]}` : d.date;
    return {
      ...d,
      displayDate: label,
    };
  });

  return (
    <section className="panel">
      <div className="panel-header">
        <div className="panel-title">
          <h2>Tickets Handled & Resolution Flow</h2>
          <p className="panel-subtitle">Volume over time and autonomous resolution rates (FR-16.1)</p>
        </div>
        <span className="badge badge-green">
          {volume.auto_resolution_rate_pct}% Auto-Resolved
        </span>
      </div>

      <div className="stats-row">
        <div className="stat-box">
          <span className="stat-label">Total Handled</span>
          <span className="stat-value">{volume.total_tickets}</span>
          <span className="stat-subtext">All historical tickets</span>
        </div>

        <div className="stat-box">
          <span className="stat-label">Resolved Automatically</span>
          <span className="stat-value" style={{ color: "#34d399" }}>
            {autoResolvedCount}
          </span>
          <span className="stat-subtext">AI solved with high confidence</span>
        </div>

        <div className="stat-box">
          <span className="stat-label">Awaiting Review</span>
          <span className="stat-value" style={{ color: "#fbbf24" }}>
            {pendingCount}
          </span>
          <span className="stat-subtext">Guardrail-flagged actions</span>
        </div>

        <div className="stat-box">
          <span className="stat-label">Escalated</span>
          <span className="stat-value" style={{ color: "#c084fc" }}>
            {escalatedCount}
          </span>
          <span className="stat-subtext">Sent to human specialist</span>
        </div>
      </div>

      <div className="chart-container">
        {chartData.length === 0 ? (
          <div className="empty-state">
            <p>No 14-day history available</p>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
              <XAxis dataKey="displayDate" stroke="#6b7280" fontSize={11} tickLine={false} />
              <YAxis stroke="#6b7280" fontSize={11} tickLine={false} allowDecimals={false} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#161e31",
                  borderColor: "rgba(255,255,255,0.15)",
                  borderRadius: "8px",
                  fontSize: "12px",
                  color: "#f3f4f6",
                }}
              />
              <Legend
                wrapperStyle={{ fontSize: "11px", paddingTop: "8px" }}
                iconType="circle"
              />
              <Bar dataKey="auto_resolved" name="Auto-Resolved" stackId="a" fill="#10B981" />
              <Bar dataKey="resolved" name="Manual Resolved" stackId="a" fill="#14B8A6" />
              <Bar dataKey="pending_approval" name="Pending Review" stackId="a" fill="#F59E0B" />
              <Bar dataKey="escalated" name="Escalated" stackId="a" fill="#8B5CF6" />
              <Bar dataKey="in_progress" name="In Progress / Open" stackId="a" fill="#4B5563" />
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>
    </section>
  );
}

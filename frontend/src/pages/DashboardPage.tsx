import { CostPanel } from "../components/dashboard/CostPanel";
import { ApprovalQueuePanel } from "../components/dashboard/ApprovalQueuePanel";
import { QualityEvalPanel } from "../components/dashboard/QualityEvalPanel";
import { TicketVolumePanel } from "../components/dashboard/TicketVolumePanel";
import { useWebSocket } from "../hooks/useWebSocket";

// The four top-level panels (FR-16). Live updates arrive over WebSocket (FR-20);
// wiring happens once the backend emits events.
export function DashboardPage() {
  const { connected } = useWebSocket();

  return (
    <main>
      <h1>TicketSolver</h1>
      <p>{connected ? "Live updates: connected" : "Live updates: offline"}</p>
      <TicketVolumePanel />
      <CostPanel />
      <QualityEvalPanel />
      <ApprovalQueuePanel />
    </main>
  );
}

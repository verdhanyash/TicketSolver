// Ticket Volume & Outcomes panel (FR-16.1) — placeholder until the dashboard
// summary endpoint lands. Charts will use Recharts (see CLAUDE.md).
export function TicketVolumePanel() {
  return (
    <section className="panel">
      <h2>Tickets Handled</h2>
      <p>No data yet — this will show total tickets and how many were resolved
        automatically vs. sent to a human, over time.</p>
    </section>
  );
}

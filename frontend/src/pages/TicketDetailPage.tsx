import { TraceViewer } from "../components/trace/TraceViewer";

// Per-ticket detail (FR-19): plain-text reasoning summary beside the interactive
// Three.js trace graph. Data fetching is wired when the traces endpoint lands.
export function TicketDetailPage({ ticketId }: { ticketId: string }) {
  return (
    <main>
      <h1>Ticket {ticketId}</h1>
      <section>
        <h2>What the AI did</h2>
        <p>No trace loaded yet.</p>
      </section>
      <TraceViewer />
    </main>
  );
}

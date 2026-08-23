// Shared domain types mirrored from the backend API. Expanded as endpoints solidify.

export type TicketOutcome = "auto_resolved" | "escalated" | "in_queue" | "pending";

export interface Ticket {
  id: string;
  subject: string;
  category?: string;
  severity?: string;
  outcome: TicketOutcome;
  createdAt: string;
}

export interface TraceNode {
  id: string;
  label: string;
  kind: "agent" | "tool_call" | "memory" | "guardrail" | "decision";
}

export interface TraceEdge {
  from: string;
  to: string;
}

export interface Trace {
  ticketId: string;
  nodes: TraceNode[];
  edges: TraceEdge[];
  summary: string;
}

export interface Approval {
  ticketId: string;
  proposedAction: string;
  reason: string;
  trace: Trace;
}

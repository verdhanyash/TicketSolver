// Thin REST client for TicketSolver API endpoints.

import type { ApprovalOut, DashboardSummaryOut, DecisionIn, Ticket } from "../types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "/api";

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    const errorText = await res.text().catch(() => "");
    throw new Error(`GET ${path} failed (${res.status}): ${errorText || res.statusText}`);
  }
  return res.json() as Promise<T>;
}

async function postJson<T>(path: string, payload?: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: payload ? JSON.stringify(payload) : undefined,
  });
  if (!res.ok) {
    const errorText = await res.text().catch(() => "");
    throw new Error(`POST ${path} failed (${res.status}): ${errorText || res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => getJson<{ status: string }>("/health"),
  dashboardSummary: () => getJson<DashboardSummaryOut>("/dashboard/summary"),
  listApprovals: (status = "pending") => getJson<ApprovalOut[]>(`/approvals?status=${encodeURIComponent(status)}`),
  getApproval: (id: string) => getJson<ApprovalOut>(`/approvals/${encodeURIComponent(id)}`),
  decideApproval: (id: string, body: DecisionIn) =>
    postJson<ApprovalOut>(`/approvals/${encodeURIComponent(id)}/decision`, body),
  listTickets: () => getJson<Ticket[]>("/tickets"),
};

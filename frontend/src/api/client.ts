// Thin REST client. Endpoints are stubs on the backend for now (see CLAUDE.md).

const API_BASE = import.meta.env.VITE_API_BASE ?? "/api";

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`GET ${path} failed: ${res.status}`);
  return res.json() as Promise<T>;
}

export const api = {
  health: () => getJson<{ status: string }>("/health"),
  // dashboardSummary, listTickets, getTrace, listApprovals … added as endpoints land.
};

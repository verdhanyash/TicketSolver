import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { ApprovalOut, DashboardSummaryOut, DecisionIn } from "../types";

export function useDashboardData(wsUrl: string = import.meta.env.VITE_WS_URL ?? "/ws/dashboard") {
  const [summary, setSummary] = useState<DashboardSummaryOut | null>(null);
  const [approvals, setApprovals] = useState<ApprovalOut[]>([]);
  const [loading, setLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);

  const fetchData = useCallback(async (isManual = false) => {
    if (isManual) setIsRefreshing(true);
    setError(null);
    try {
      const [summaryRes, approvalsRes] = await Promise.all([
        api.dashboardSummary(),
        api.listApprovals("pending").catch(() => []),
      ]);
      setSummary(summaryRes);
      setApprovals(approvalsRes);
      setLastUpdated(new Date());
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to load dashboard data";
      setError(msg);
    } finally {
      setLoading(false);
      setIsRefreshing(false);
    }
  }, []);

  // Initial fetch
  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // WebSocket Live Updates (FR-20)
  useEffect(() => {
    let isMounted = true;

    const connectWs = () => {
      try {
        const proto = window.location.protocol === "https:" ? "wss" : "ws";
        const fullUrl = wsUrl.startsWith("ws")
          ? wsUrl
          : `${proto}://${window.location.host}${wsUrl}`;

        const ws = new WebSocket(fullUrl);
        wsRef.current = ws;

        ws.onopen = () => {
          if (isMounted) setConnected(true);
        };

        ws.onclose = () => {
          if (isMounted) {
            setConnected(false);
            // Reconnect after 3 seconds
            reconnectTimeoutRef.current = window.setTimeout(connectWs, 3000);
          }
        };

        ws.onerror = () => {
          if (isMounted) setConnected(false);
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            // Any ticket_processed, approval_created, or approval_decided event triggers refetch
            if (data && typeof data === "object") {
              fetchData();
            }
          } catch {
            // Ignore non-json heartbeats
          }
        };
      } catch {
        if (isMounted) setConnected(false);
      }
    };

    connectWs();

    return () => {
      isMounted = false;
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [wsUrl, fetchData]);

  const submitDecision = useCallback(
    async (approvalId: string, decision: DecisionIn) => {
      await api.decideApproval(approvalId, decision);
      await fetchData();
    },
    [fetchData]
  );

  return {
    summary,
    approvals,
    loading,
    isRefreshing,
    error,
    connected,
    lastUpdated,
    refresh: () => fetchData(true),
    submitDecision,
  };
}

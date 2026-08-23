import { useEffect, useRef, useState } from "react";

// Live dashboard updates (FR-20). Connects to the backend WebSocket and exposes the
// latest received message. Reconnect/backoff and typed events are added later.
export function useWebSocket(url: string = import.meta.env.VITE_WS_URL ?? "/ws/dashboard") {
  const [lastMessage, setLastMessage] = useState<unknown>(null);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    const full = url.startsWith("ws") ? url : `${proto}://${window.location.host}${url}`;
    const ws = new WebSocket(full);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onmessage = (e) => {
      try {
        setLastMessage(JSON.parse(e.data));
      } catch {
        setLastMessage(e.data);
      }
    };

    return () => ws.close();
  }, [url]);

  return { lastMessage, connected };
}

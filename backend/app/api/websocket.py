"""Dashboard WebSocket for near-real-time updates (FR-20).

`ConnectionManager` tracks active clients so later sessions can broadcast ticket-processed
events. This is a minimal stub: it accepts connections and echoes nothing meaningful yet.
"""

from __future__ import annotations

from fastapi import FastAPI, WebSocket, WebSocketDisconnect


class ConnectionManager:
    def __init__(self) -> None:
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: dict) -> None:
        for ws in list(self.active):
            await ws.send_json(message)


manager = ConnectionManager()


def register_websocket(app: FastAPI) -> None:
    @app.websocket("/ws/dashboard")
    async def dashboard_ws(ws: WebSocket) -> None:
        await manager.connect(ws)
        try:
            while True:
                # No inbound protocol yet; keep the connection open.
                await ws.receive_text()
        except WebSocketDisconnect:
            manager.disconnect(ws)

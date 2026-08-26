"""Dashboard WebSocket for near-real-time updates (FR-20).

`ConnectionManager` tracks active clients; background workers (M4 jobs) broadcast
ticket-processed events from threads via `broadcast_from_thread`, which hops onto the
main event loop captured at startup.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


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
            try:
                await ws.send_json(message)
            except Exception:  # one dead client must not break the rest
                logger.exception("broadcast to a client failed; dropping it")
                self.disconnect(ws)


manager = ConnectionManager()

# The uvicorn event loop, captured at startup so worker threads can schedule
# broadcasts on it (asyncio objects are loop-bound).
_main_loop: asyncio.AbstractEventLoop | None = None


def capture_main_loop() -> None:
    global _main_loop
    _main_loop = asyncio.get_running_loop()


def broadcast_from_thread(message: dict) -> bool:
    """Thread-safe broadcast hop onto the main loop. False if no loop/clients yet."""
    if _main_loop is None or not manager.active:
        return False
    asyncio.run_coroutine_threadsafe(manager.broadcast(message), _main_loop)
    return True


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

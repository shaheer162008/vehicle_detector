"""
WebSocket Event Sink for FastALPR Pipeline.
Broadcasts events in real-time over WebSocket to connected client dashboards.
Maintains rolling buffer of the last 20 events sent to newly connecting clients.
"""

import asyncio
import json
import threading
from collections import deque
from typing import Optional, Set, Dict, Any


class WebSocketSink:
    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set = set()
        self.recent_events = deque(maxlen=20)
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self.server = None
        self.running = False

    def start(self) -> None:
        """Starts the asyncio WebSocket server in a background daemon thread."""
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        print(f"[WS SINK] WebSocket server listening on ws://{self.host}:{self.port}")

    def _run_loop(self) -> None:
        """Runs the WebSocket server inside a proper running event loop (works on websockets 10-17)."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._serve())
        except Exception as e:
            print(f"[WS SINK] Server error: {e}")
        finally:
            try:
                self.loop.close()
            except Exception:
                pass

    async def _serve(self) -> None:
        import websockets

        async def _handler(ws):
            self.clients.add(ws)
            try:
                for ev in list(self.recent_events):
                    await ws.send(json.dumps(ev))
                async for _ in ws:
                    pass
            except Exception:
                pass
            finally:
                self.clients.discard(ws)

        self._stop_event = asyncio.Event()
        async with websockets.serve(_handler, self.host, self.port):
            await self._stop_event.wait()

    def publish(self, event: Dict[str, Any]) -> None:
        """Broadcasts event JSON to all connected clients."""
        if not self.running or self.loop is None:
            return
        self.recent_events.append(event)
        msg = json.dumps(event)
        asyncio.run_coroutine_threadsafe(self._broadcast(msg), self.loop)

    async def _broadcast(self, msg: str) -> None:
        if not self.clients:
            return
        disconnected = set()
        for ws in list(self.clients):
            try:
                await ws.send(msg)
            except Exception:
                disconnected.add(ws)
        self.clients.difference_update(disconnected)

    def stop(self, timeout: float = 2.0) -> None:
        """Closes the WebSocket server cleanly."""
        self.running = False
        if self.loop is not None and self.loop.is_running() and getattr(self, "_stop_event", None) is not None:
            self.loop.call_soon_threadsafe(self._stop_event.set)

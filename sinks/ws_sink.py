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
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            import websockets

            async def _handler(ws):
                self.clients.add(ws)
                try:
                    # Send last 20 events upon new client connection
                    for ev in list(self.recent_events):
                        await ws.send(json.dumps(ev))
                    async for _ in ws:
                        pass
                except Exception:
                    pass
                finally:
                    self.clients.discard(ws)

            start_server = websockets.serve(_handler, self.host, self.port)
            self.server = self.loop.run_until_complete(start_server)
            self.loop.run_forever()
        except Exception as e:
            print(f"[WS SINK] Server error: {e}")
        finally:
            if self.server:
                self.server.close()
                self.loop.run_until_complete(self.server.wait_closed())
            self.loop.close()

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
        if self.loop is not None and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)

"""
Event Bus for FastALPR Pipeline.
Orchestrates event publication across all active sinks (JSON, Neon Postgres, WebSocket).
Handles graceful shutdown and flush.
"""

from typing import Optional, Dict, Any
from .postgres_sink import PostgresSink
from .ws_sink import WebSocketSink


class EventBus:
    def __init__(
        self,
        database_url: Optional[str] = None,
        ws_port: Optional[int] = None,
        enable_ws: bool = True,
    ):
        self.database_url = database_url
        self.ws_port = ws_port if (enable_ws and ws_port) else None

        self.postgres_sink: Optional[PostgresSink] = None
        self.ws_sink: Optional[WebSocketSink] = None

        if self.database_url:
            self.postgres_sink = PostgresSink(database_url=self.database_url)

        if self.ws_port:
            self.ws_sink = WebSocketSink(port=self.ws_port)

    def start(self) -> None:
        """Starts all active external sinks."""
        if self.postgres_sink:
            self.postgres_sink.start()
        if self.ws_sink:
            self.ws_sink.start()

    def publish(self, event: Dict[str, Any]) -> None:
        """
        Publishes event to all configured sinks.
        Failure in one sink never impacts others or blocks the pipeline.
        """
        if self.postgres_sink:
            try:
                self.postgres_sink.publish(event)
            except Exception as e:
                print(f"[EVENT BUS] Postgres sink error: {e}")

        if self.ws_sink:
            try:
                self.ws_sink.publish(event)
            except Exception as e:
                print(f"[EVENT BUS] WebSocket sink error: {e}")

    def get_status_str(self) -> str:
        """Returns console status string for active sinks."""
        pg_status = "ON" if self.postgres_sink is not None else "OFF"
        ws_status = "ON" if self.ws_sink is not None else "OFF"
        return f"[SINKS] json=ON postgres={pg_status} ws={ws_status}"

    def stop(self, timeout: float = 10.0) -> None:
        """Flushes and shuts down all sinks."""
        if self.postgres_sink:
            try:
                self.postgres_sink.stop(timeout=timeout)
            except Exception as e:
                print(f"[EVENT BUS] Postgres shutdown error: {e}")

        if self.ws_sink:
            try:
                self.ws_sink.stop(timeout=2.0)
            except Exception as e:
                print(f"[EVENT BUS] WebSocket shutdown error: {e}")

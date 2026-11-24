"""
Event sinks package for FastALPR Pipeline.
Provides fan-out event bus, Neon PostgreSQL sink, and WebSocket sink.
"""

from .event_bus import EventBus
from .postgres_sink import PostgresSink, mask_database_url
from .s3_sink import S3Sink
from .ws_sink import WebSocketSink

__all__ = ["EventBus", "PostgresSink", "S3Sink", "WebSocketSink", "mask_database_url"]

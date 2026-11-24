"""
Event Bus for FastALPR Pipeline.
Orchestrates event publication across all active sinks (JSON, Neon Postgres, Neon S3, WebSocket).
Handles graceful shutdown and flush.
"""

from pathlib import Path
from typing import Optional, Dict, Any
from .postgres_sink import PostgresSink
from .s3_sink import S3Sink
from .ws_sink import WebSocketSink


class EventBus:
    def __init__(
        self,
        database_url: Optional[str] = None,
        ws_port: Optional[int] = None,
        enable_ws: bool = True,
        s3_endpoint_url: Optional[str] = None,
        s3_access_key_id: Optional[str] = None,
        s3_secret_access_key: Optional[str] = None,
        s3_region: str = "us-east-2",
        s3_bucket: str = "alpr-snapshots",
        enable_s3: bool = True,
        snapshots_dir: Optional[Path] = None,
        events_json_path: Optional[Path] = None,
        events_jsonl_path: Optional[Path] = None,
    ):
        self.database_url = database_url
        self.ws_port = ws_port if (enable_ws and ws_port) else None

        self.postgres_sink: Optional[PostgresSink] = None
        self.s3_sink: Optional[S3Sink] = None
        self.ws_sink: Optional[WebSocketSink] = None

        if self.database_url:
            self.postgres_sink = PostgresSink(database_url=self.database_url)

        if enable_s3 and s3_endpoint_url and s3_access_key_id and s3_secret_access_key:
            self.s3_sink = S3Sink(
                endpoint_url=s3_endpoint_url,
                access_key_id=s3_access_key_id,
                secret_access_key=s3_secret_access_key,
                region=s3_region,
                bucket_name=s3_bucket,
                snapshots_dir=snapshots_dir,
                events_json_path=events_json_path,
                events_jsonl_path=events_jsonl_path,
            )

        if self.ws_port:
            self.ws_sink = WebSocketSink(port=self.ws_port)

    def start(self) -> None:
        """Starts all active external sinks."""
        if self.postgres_sink:
            self.postgres_sink.start()
        if self.s3_sink:
            self.s3_sink.start()
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

        if self.s3_sink:
            try:
                self.s3_sink.publish(event)
            except Exception as e:
                print(f"[EVENT BUS] S3 sink error: {e}")

        if self.ws_sink:
            try:
                self.ws_sink.publish(event)
            except Exception as e:
                print(f"[EVENT BUS] WebSocket sink error: {e}")

    def upload_artifact(self, local_path: Path, rel_key: str) -> None:
        """Synchronously uploads a finalized output artifact (e.g. annotated video) to S3 if active."""
        if self.s3_sink and getattr(self.s3_sink, "_client", None) is not None:
            try:
                self.s3_sink.upload_file_sync(local_path, rel_key)
            except Exception as e:
                print(f"[EVENT BUS] S3 artifact upload error: {e}")

    def get_status_str(self) -> str:
        """Returns console status string for active sinks."""
        pg_status = "ON" if self.postgres_sink is not None else "OFF"
        s3_status = "ON" if (self.s3_sink is not None and self.s3_sink._client is not None) else "OFF"
        ws_status = "ON" if self.ws_sink is not None else "OFF"
        return f"[SINKS] json=ON postgres={pg_status} s3={s3_status} ws={ws_status}"

    def stop(self, timeout: float = 10.0) -> None:
        """Flushes and shuts down all sinks."""
        if self.postgres_sink:
            try:
                self.postgres_sink.stop(timeout=timeout)
            except Exception as e:
                print(f"[EVENT BUS] Postgres shutdown error: {e}")

        if self.s3_sink:
            try:
                self.s3_sink.stop(timeout=max(timeout, 30.0))
            except Exception as e:
                print(f"[EVENT BUS] S3 shutdown error: {e}")

        if self.ws_sink:
            try:
                self.ws_sink.stop(timeout=2.0)
            except Exception as e:
                print(f"[EVENT BUS] WebSocket shutdown error: {e}")

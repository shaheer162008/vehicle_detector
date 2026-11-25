"""
FastALPR Control Service - FastAPI Application.
Provides REST API and WebSocket endpoints for stream management and real-time events.
"""

import os
import json
import asyncio
import secrets
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set

from fastapi import FastAPI, HTTPException, Depends, Query, WebSocket, WebSocketDisconnect, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .manager import StreamManager, StreamConfig
from .relay import RelaySink

# Load .env file
try:
    from dotenv import load_dotenv
    _root = Path(__file__).resolve().parent.parent
    load_dotenv(_root / ".env", override=True)
except ImportError:
    pass


# Configuration from environment
SERVICE_TOKEN = os.environ.get("SERVICE_TOKEN", "")
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
VIDEO_DIR = Path(os.environ.get("VIDEO_DIR", "../CPK-Management/public/videos")).resolve()
RTSP_BASE = os.environ.get("RTSP_BASE", "rtsp://127.0.0.1:8554")
SERVICE_PORT = int(os.environ.get("SERVICE_PORT", "8000"))
SERVICE_HOST = os.environ.get("SERVICE_HOST", "127.0.0.1")

# Global manager instance
manager: Optional[StreamManager] = None


# Pydantic models for request/response
class SourceModel(BaseModel):
    type: str  # "file" or "rtsp"
    file: Optional[str] = None
    url: Optional[str] = None


class StreamCreateRequest(BaseModel):
    name: str
    source: SourceModel
    loop: bool = False
    camera_id: str = "CAM-00001"
    plate_format: str = "GENERAL"
    vehicle_weights: str = "yolo11m.pt"
    publish_rtsp: bool = False
    autostart: bool = False


class StreamUpdateRequest(BaseModel):
    loop: Optional[bool] = None
    name: Optional[str] = None
    plate_format: Optional[str] = None


class StreamResponse(BaseModel):
    id: str
    name: str
    source: Dict[str, Any]
    loop: bool
    camera_id: str
    plate_format: str
    vehicle_weights: str
    publish_rtsp: bool
    autostart: bool
    status: Dict[str, Any]


class HealthResponse(BaseModel):
    gpu_available: bool
    torch_cuda: bool
    ort_providers: List[str]
    python_version: str
    service_version: str = "1.0.0"


# Authentication dependency
async def verify_api_key(x_api_key: str = Header(...)) -> None:
    if not SERVICE_TOKEN:
        raise HTTPException(status_code=500, detail="Service token not configured")
    if x_api_key != SERVICE_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid API key")


async def verify_ws_token(token: str = Query(...)) -> None:
    if not SERVICE_TOKEN:
        raise HTTPException(status_code=500, detail="Service token not configured")
    if token != SERVICE_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")


# WebSocket Connection Manager
class ConnectionManager:
    """Manages WebSocket connections for events and frames."""

    def __init__(self):
        # Events: stream_id -> set of WebSockets
        self.events_connections: Dict[str, Set[WebSocket]] = {}
        # Frames: stream_id -> set of WebSockets
        self.frames_connections: Dict[str, Set[WebSocket]] = {}
        # Recent events buffer: stream_id -> list of events
        self.recent_events: Dict[str, List[Dict]] = {}
        self.max_recent = 10

    async def connect_events(self, websocket: WebSocket, stream_id: Optional[str] = None):
        await websocket.accept()
        key = stream_id or "all"
        if key not in self.events_connections:
            self.events_connections[key] = set()
        self.events_connections[key].add(websocket)
        # Send recent events
        for event in self.recent_events.get(key, [])[-self.max_recent:]:
            try:
                await websocket.send_text(json.dumps(event))
            except Exception:
                pass

    def disconnect_events(self, websocket: WebSocket, stream_id: Optional[str] = None):
        key = stream_id or "all"
        if key in self.events_connections:
            self.events_connections[key].discard(websocket)

    async def connect_frames(self, websocket: WebSocket, stream_id: str):
        await websocket.accept()
        if stream_id not in self.frames_connections:
            self.frames_connections[stream_id] = set()
        self.frames_connections[stream_id].add(websocket)

    def disconnect_frames(self, websocket: WebSocket, stream_id: str):
        if stream_id in self.frames_connections:
            self.frames_connections[stream_id].discard(websocket)

    def broadcast_event(self, event: Dict, stream_id: str):
        """Broadcast event to all connected clients for this stream."""
        # Add stream_id to event
        event["stream_id"] = stream_id

        # Update recent events buffer
        for key in [stream_id, "all"]:
            if key not in self.recent_events:
                self.recent_events[key] = []
            self.recent_events[key].append(event)
            if len(self.recent_events[key]) > self.max_recent:
                self.recent_events[key] = self.recent_events[key][-self.max_recent:]

        # Send to events connections
        for key in [stream_id, "all"]:
            if key in self.events_connections:
                dead = set()
                for ws in self.events_connections[key]:
                    try:
                        asyncio.create_task(ws.send_text(json.dumps(event)))
                    except Exception:
                        dead.add(ws)
                for ws in dead:
                    self.events_connections[key].discard(ws)

    def broadcast_frame(self, frame_data: bytes, stream_id: str):
        """Broadcast frame to all connected clients for this stream."""
        if stream_id in self.frames_connections:
            dead = set()
            for ws in self.frames_connections[stream_id]:
                try:
                    asyncio.create_task(ws.send_bytes(frame_data))
                except Exception:
                    dead.add(ws)
            for ws in dead:
                self.frames_connections[stream_id].discard(ws)


# Global connection manager
ws_manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global manager
    # Startup
    streams_file = Path(__file__).parent / "streams.json"
    manager = StreamManager(
        streams_file=streams_file,
        video_dir=VIDEO_DIR,
        rtsp_base=RTSP_BASE,
        service_token=SERVICE_TOKEN,
        service_host=SERVICE_HOST,
        service_port=SERVICE_PORT,
    )

    # Auto-start streams marked with autostart
    for stream_id, config in manager._streams.items():
        if config.autostart:
            manager.start_stream(stream_id)

    yield

    # Shutdown
    if manager:
        manager.shutdown()


app = FastAPI(title="FastALPR Control Service", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Helper to get manager
def get_manager() -> StreamManager:
    if manager is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return manager


# REST Endpoints
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint with GPU/ONNX info."""
    gpu_available = False
    torch_cuda = False
    ort_providers = []

    try:
        import torch
        torch_cuda = torch.cuda.is_available()
        gpu_available = torch_cuda
    except Exception:
        pass

    try:
        import onnxruntime as ort
        ort_providers = ort.get_available_providers()
    except Exception:
        pass

    import sys
    return HealthResponse(
        gpu_available=gpu_available,
        torch_cuda=torch_cuda,
        ort_providers=ort_providers,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    )


@app.get("/streams", response_model=List[StreamResponse])
async def list_streams(mgr: StreamManager = Depends(get_manager), _: None = Depends(verify_api_key)) -> List[StreamResponse]:
    """List all streams with their status."""
    streams = mgr.list_streams()
    return [StreamResponse(**s) for s in streams]


@app.post("/streams", response_model=Dict[str, str])
async def create_stream(
    request: StreamCreateRequest,
    mgr: StreamManager = Depends(get_manager),
    _: None = Depends(verify_api_key),
) -> Dict[str, str]:
    """Create a new stream."""
    source = request.source.model_dump()
    if source["type"] == "file":
        if not source.get("file"):
            raise HTTPException(status_code=400, detail="File source requires 'file' field")
        try:
            mgr._validate_file_source(source["file"])
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    elif source["type"] == "rtsp":
        if not source.get("url"):
            raise HTTPException(status_code=400, detail="RTSP source requires 'url' field")
    else:
        raise HTTPException(status_code=400, detail="Source type must be 'file' or 'rtsp'")

    stream_id = mgr.create_stream(
        name=request.name,
        source=source,
        loop=request.loop,
        camera_id=request.camera_id,
        plate_format=request.plate_format,
        vehicle_weights=request.vehicle_weights,
        publish_rtsp=request.publish_rtsp,
        autostart=request.autostart,
    )
    return {"id": stream_id}


@app.get("/streams/{stream_id}", response_model=StreamResponse)
async def get_stream(
    stream_id: str,
    mgr: StreamManager = Depends(get_manager),
    _: None = Depends(verify_api_key),
) -> StreamResponse:
    """Get stream details."""
    config = mgr.get_stream(stream_id)
    if not config:
        raise HTTPException(status_code=404, detail="Stream not found")
    status = mgr.get_status(stream_id) or {}
    return StreamResponse(
        id=config.id,
        name=config.name,
        source=config.source,
        loop=config.loop,
        camera_id=config.camera_id,
        plate_format=config.plate_format,
        vehicle_weights=config.vehicle_weights,
        publish_rtsp=config.publish_rtsp,
        autostart=config.autostart,
        status=status,
    )


@app.patch("/streams/{stream_id}", response_model=Dict[str, str])
async def update_stream(
    stream_id: str,
    request: StreamUpdateRequest,
    mgr: StreamManager = Depends(get_manager),
    _: None = Depends(verify_api_key),
) -> Dict[str, str]:
    """Update stream configuration."""
    if not mgr.update_stream(stream_id, request.loop, request.name, request.plate_format):
        raise HTTPException(status_code=404, detail="Stream not found")
    return {"status": "updated"}


@app.delete("/streams/{stream_id}", response_model=Dict[str, str])
async def delete_stream(
    stream_id: str,
    mgr: StreamManager = Depends(get_manager),
    _: None = Depends(verify_api_key),
) -> Dict[str, str]:
    """Delete a stream (stops it first)."""
    if not mgr.delete_stream(stream_id):
        raise HTTPException(status_code=404, detail="Stream not found")
    return {"status": "deleted"}


@app.post("/streams/{stream_id}/start", response_model=Dict[str, str])
async def start_stream(
    stream_id: str,
    mgr: StreamManager = Depends(get_manager),
    _: None = Depends(verify_api_key),
) -> Dict[str, str]:
    """Start a stream."""
    if not mgr.start_stream(stream_id):
        raise HTTPException(status_code=404, detail="Stream not found or already running")
    return {"status": "starting"}


@app.post("/streams/{stream_id}/stop", response_model=Dict[str, str])
async def stop_stream(
    stream_id: str,
    mgr: StreamManager = Depends(get_manager),
    _: None = Depends(verify_api_key),
) -> Dict[str, str]:
    """Stop a stream."""
    if not mgr.stop_stream(stream_id):
        raise HTTPException(status_code=404, detail="Stream not found")
    return {"status": "stopped"}


@app.post("/streams/{stream_id}/restart", response_model=Dict[str, str])
async def restart_stream(
    stream_id: str,
    mgr: StreamManager = Depends(get_manager),
    _: None = Depends(verify_api_key),
) -> Dict[str, str]:
    """Restart a stream."""
    if not mgr.restart_stream(stream_id):
        raise HTTPException(status_code=404, detail="Stream not found")
    return {"status": "restarting"}


@app.get("/streams/{stream_id}/status", response_model=Dict[str, Any])
async def get_stream_status(
    stream_id: str,
    mgr: StreamManager = Depends(get_manager),
    _: None = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Get stream runtime status."""
    status = mgr.get_status(stream_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Stream not found")
    return status


@app.get("/cameras")
async def list_cameras(
    mgr: StreamManager = Depends(get_manager),
    _: None = Depends(verify_api_key),
) -> List[Dict[str, Any]]:
    """List all configured cameras (from cameras.json + env overrides) with masked RTSP URLs."""
    cameras = []
    for i in range(1, 41):
        cam_id = f"CAM-{i:05d}"
        rtsp_url = os.environ.get(f"CAMERA_{i}_RTSP_URL", "")
        name = os.environ.get(f"CAMERA_{i}_NAME", f"Camera {i}")
        location = os.environ.get(f"CAMERA_{i}_LOCATION", "")
        if rtsp_url:
            # Mask password in URL for display
            masked_url = re.sub(r"(rtsp://[^:]+:)[^@]+@", r"\1****@", rtsp_url)
        else:
            masked_url = "not configured"
        cameras.append({
            "camera_id": cam_id,
            "name": name,
            "location": location,
            "rtsp_url": masked_url,
            "configured": bool(rtsp_url),
        })
    return cameras


@app.get("/videos", response_model=List[str])
async def list_videos(
    mgr: StreamManager = Depends(get_manager),
    _: None = Depends(verify_api_key),
) -> List[str]:
    """List video files in VIDEO_DIR."""
    return mgr.get_videos()


# WebSocket Endpoints
@app.websocket("/ws/events")
async def ws_events(
    websocket: WebSocket,
    stream_id: Optional[str] = Query(None),
    token: str = Query(...),
):
    """WebSocket for real-time events (all streams or filtered by stream_id)."""
    await verify_ws_token(token)
    await ws_manager.connect_events(websocket, stream_id)

    try:
        while True:
            await websocket.receive_text()  # Keep alive
    except WebSocketDisconnect:
        ws_manager.disconnect_events(websocket, stream_id)
    except Exception:
        ws_manager.disconnect_events(websocket, stream_id)


@app.websocket("/ws/frames/{stream_id}")
async def ws_frames(
    websocket: WebSocket,
    stream_id: str,
    token: str = Query(...),
):
    """WebSocket for annotated frame preview (binary JPEG, latest only)."""
    await verify_ws_token(token)
    await ws_manager.connect_frames(websocket, stream_id)

    try:
        while True:
            await websocket.receive_text()  # Keep alive
    except WebSocketDisconnect:
        ws_manager.disconnect_frames(websocket, stream_id)
    except Exception:
        ws_manager.disconnect_frames(websocket, stream_id)


# Internal WebSocket endpoint for worker relay
@app.websocket("/internal/ingest/{stream_id}")
async def ws_internal_ingest(
    websocket: WebSocket,
    stream_id: str,
    token: str = Query(...),
):
    """Internal WebSocket endpoint for pipeline workers to send events/frames."""
    if token != SERVICE_TOKEN:
        await websocket.close(code=4001, reason="Invalid token")
        return

    await websocket.accept()
    print(f"[RELAY] Worker connected for stream: {stream_id}")

    try:
        while True:
            message = await websocket.receive()
            if "text" in message:
                # Event from worker
                event = json.loads(message["text"])
                ws_manager.broadcast_event(event, stream_id)
            elif "bytes" in message:
                # Frame from worker
                frame_data = message["bytes"]
                ws_manager.broadcast_frame(frame_data, stream_id)
    except WebSocketDisconnect:
        print(f"[RELAY] Worker disconnected for stream: {stream_id}")
    except Exception as e:
        print(f"[RELAY] Error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=SERVICE_HOST, port=SERVICE_PORT)
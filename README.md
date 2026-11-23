# Vehicle Tracking & Detection System (Stage 1)

A high-performance vehicle detection, tracking, and Automatic License Plate Recognition (ALPR) pipeline designed for multi-camera traffic monitoring and surveillance. The system enforces **strictly one event per vehicle**, extracting clean, tight vehicle snapshots and character-level consensus license plate reads.

Built on top of the open-source [FastALPR](https://github.com/ankandrew/fast-alpr) (MIT licence).

---

## 1. Overview

- **Single Event Per Vehicle**: Tracks vehicles across frames with ByteTrack, merges fragmented trajectories via spatial and plate-based stitching, and suppresses duplicate events within a configurable time window.
- **Robust Plate Consensus**: Runs FastALPR on vehicle crops, aggregates multiple OCR observations per track using character-level confidence-weighted voting, and flags uncertain characters with `'?'`.
- **Quality-Optimized Snapshots**: Evaluates every frame of a track using Laplacian sharpness, frame border completeness, and plate containment to save the sharpest unoccluded vehicle crop and plate crop.
- **Enterprise-Ready Outputs**: Emits structured JSON/JSONL events, saves unupscaled plate crops and OCR data for future fine-tuning, and records universal H.264 annotated video.

---

## 2. Visual Demos

[![Vehicle Tracking & License Plate Recognition Demo](https://img.youtube.com/vi/ArFPQdJRlo8/maxresdefault.jpg)](https://www.youtube.com/watch?v=ArFPQdJRlo8)

*Real-time Vehicle Detection, ByteTrack Tracking, and FastALPR License Plate Recognition.*  
▶️ **[Click Here to Play Video on YouTube (Full HD 1080p)](https://www.youtube.com/watch?v=ArFPQdJRlo8)**

---

## 3. Snapshot Gallery

Every emitted event produces a clean vehicle snapshot, a license plate crop (if visible), and a full-frame context image.

- View the full table: [gallery/gallery.md](gallery/gallery.md)
- To regenerate the gallery and slideshow after a run:
  ```bash
  python tools/make_gallery.py --events output/events.json --snapshots output/snapshots
  ```

---

## 4. How It Works

### Pipeline Architecture

```text
[ Video File / RTSP Stream ]
              │
              ▼
    [ YOLO Vehicle Detector ]  ──► Raw Detection Boxes
              │
              ▼
    [ ByteTrack Tracker ]       ──► Trajectory Smoothing & ID Assignment
              │
              ▼
    [ Readable Zone Gating ]   ──► Filters small / horizon vehicles
              │
              ▼
  [ FastALPR on Vehicle Crops ] ──► Plate Detection & CCT-S-v2 OCR
              │
              ▼
  [ Character-Level Voting ]   ──► Consensus Plate & Confidence Scoring
              │
              ▼
    [ Track Stitching ]        ──► Merges trajectories across gaps
              │
              ▼
   [ Best Snapshot Selector ]  ──► Area x Sharpness x Completeness
              │
              ▼
      [ Emitted Event ]        ──► events.json, snapshots/, plate_reads.jsonl
```

### Core Pipeline Components

- **One-Event-Per-Vehicle Logic**: Vehicles are assigned a tracking ID across frames. When a track ends, the pipeline checks for prior tracks from the same camera that ended within 1 to 45 frames with matching plates, merging them into a single event. A 60-second fuzzy duplicate suppression window further eliminates re-emissions.
- **Per-Character Confidence-Weighted Voting**: Rather than relying on a single frame read, the recognizer records each observation `(text, mean_conf, per_char_confs)`. Characters vote per position weighted by their confidence. If the winning character has $< 50\%$ vote share or $< 0.50$ mean confidence, it is marked with `'?'`.
- **Edge-Cut Plate Rejection**: Plate detections whose bounding box is within 4 px of any frame boundary are rejected early (`reject_reason="plate_touches_frame_edge"`) to prevent corrupted syntax from partial plates entering voting.
- **Best-Snapshot Selection**: Every frame evaluates `score = area_px * sharpness * (1.0 if complete else 0.35) * (1.5 if plate_inside)`. "Complete" requires the vehicle box to be clipped on at most one side with $< 15\%$ area clipped. Clean crops are taken from the raw detector box with 6% padding clamped to the frame borders.

---

## 5. Machine Learning Models & Licences

| Model Component | Architecture / Weights | Source / Repository | Licence |
| :--- | :--- | :--- | :--- |
| **Vehicle Detection** | YOLO11m (`yolo11m.pt`) | [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) | **AGPL-3.0** |
| **Plate Detection** | YOLOv9s (`yolo-v9-s-608-license-plate-end2end`) | [open-image-models](https://github.com/ankandrew/open-image-models) | Apache-2.0 |
| **Plate OCR** | Compact Convolutional Transformer (`cct-s-v2-global-model`) | [fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr) | Apache-2.0 |

> [!WARNING]
> **Licencing Notice**: Ultralytics YOLO is licenced under AGPL-3.0. For commercial closed-source deployments, an Ultralytics enterprise licence is required, or the detector should be swapped for an Apache-2.0 alternative (e.g. RT-DETR or exported ONNX/TensorRT backends listed in `config.py`).

---

## 6. Installation & Execution

### Automated Installation

- **Linux / Google Colab**:
  ```bash
  bash install.sh
  ```
- **Windows**:
  ```bat
  install.bat
  ```

### Example Commands

```bash
# Test run on Indian highway clip with general plate parsing
python run.py --source stream_2.mp4 --plate-format GENERAL --no-preview

# UK standard test run with live Colab preview disabled
python run.py --source stream.mp4 --plate-format UK --no-preview

# Multi-camera configuration using camera index from cameras.json
python run.py --cameras-config cameras.json --camera-id CAM-00001
```

### Complete CLI Options Table (`run.py`)

| Flag | Type / Choices | Default | Description |
| :--- | :--- | :--- | :--- |
| `--source` | `str` | `None` | Path to video file, RTSP URL, or webcam index. If omitted, uses RTSP from `cameras.json`. |
| `--cameras-config` | `str` | `None` | Path to `cameras.json` configuration file. |
| `--camera-id` | `str` | `"CAM-00001"` | Camera identifier for logging and event IDs. |
| `--plate-format` | `PAKISTAN`, `GENERAL`, `UK` | `"UK"` | Plate format standard and syntax rules. |
| `--node-id` | `str` | `"A"` | Node prefix for event ID generation. |
| `--plate-syntax-correction` | Flag | `False` | Normalize alphanumeric ambiguities (e.g. 0/O, 1/I). |
| `--output-video` | `str` | `"output/annotated_output.mp4"` | Output destination for annotated video. |
| `--vehicle-conf` | `float` | `0.30` | Vehicle detector confidence threshold. |
| `--alpr-interval` | `int` | `3` | Run ALPR every K frames per vehicle track. |
| `--alpr-on-crops` | Flag | `True` | Run ALPR on vehicle crops (faster, higher resolution). |
| `--full-frame-alpr` | Flag | `False` | Run ALPR on entire frame instead of vehicle crops. |
| `--min-track-frames` | `int` | `6` | Minimum frames tracked before an event can be emitted. |
| `--disappeared-frames`| `int` | `45` | Inactive frames before a track is closed (1.5s at 30 fps). |
| `--min-area` | `float` | `2500.0` | Minimum vehicle bounding box area in pixels. |
| `--partial-plate-policy`| `keep_low_conf`, `drop` | `"keep_low_conf"` | Keep partial reads with `'?'` or drop them to no-plate. |
| `--no-plate-data` | Flag | `False` | Disable saving `plate_reads.jsonl` and `plate_crops/`. |
| `--keep-output` | Flag | `False` | Prevent auto-archiving previous `output/` directory. |
| `--debug-reads` | Flag | `False` | Print verbose OCR reads, confidences, and box diagnostics. |
| `--no-plate-boxes` | Flag | `False` | Disable drawing green plate bounding boxes in video. |
| `--preview` / `--no-preview` | Flag | `True` | Toggle live notebook display. |
| `--save-video` / `--no-save-video`| Flag | `True` | Toggle saving annotated output video. |
| `--sync-realtime` | Flag | `False` | Throttle processing to real video playback speed. |
| `--enforce-regex` | Flag | `False` | Strictly discard plate reads not matching country regex. |
| `--uk-syntax` | Flag | `False` | Enable UK plate syntax normalization. |
| `--print-events` | Flag | `False` | Print event JSON to console as emitted. |
| `--include-remarks` | Flag | `False` | Include `remarks` field in event JSON output. |
| `--box-smoothing` | `float` | `0.0` | EMA box smoothing factor for tracker (0.0 = raw boxes). |
| `--hud-smoothing` | `float` | `0.65` | Smoothing factor for HUD bounding box visualization. |
| `--roi` | `str` (JSON) | `None` | JSON array of ROI polygon coordinates `[[x,y],...]`. |
| `--max-frames` | `int` | `None` | Maximum number of frames to process before stopping. |

---

## 7. License Plate Formats

- **`GENERAL`**: Relaxed alphanumeric parsing. Retains all valid characters, ignores strict country regex, and applies voting across character positions. **Must be used for foreign test clips** (e.g. Indian plates like `DL7CD5017`, `HR26CQ6869`) because format-specific checkers would otherwise discard them.
- **`UK`**: Enforces UK DVLA standard format (2 letters, 2 digits, 3 letters, e.g. `NA13NRU`). Applies character syntax correction (substituting `0` for `O` and `1` for `I` in digit positions).
- **`PAKISTAN`**: Designed for provincial standard plates (e.g. Punjab, Sindh, Islamabad), handling both modern 3-letter/4-digit and legacy formats.

---

## 8. Multi-Camera Configuration (`cameras.json`)

The pipeline supports multi-camera surveillance deployments via `cameras.json`.

```json
{
  "system": {
    "system_name": "SafeCity Vehicle Surveillance System",
    "total_configured_cameras": 40,
    "default_plate_format": "PAKISTAN"
  },
  "cameras": [
    {
      "camera_id": "CAM-00001",
      "name": "Main Boulevard - North Entry Lane 1",
      "location": "North Gate Toll Plaza / Entry Checkpost 1",
      "rtsp_url": "rtsp://admin:admin123@192.168.1.101:554/Streaming/Channels/101",
      "enabled": true,
      "target_fps": 10,
      "plate_format": "PAKISTAN",
      "min_readable_vehicle_area": 2500.0,
      "roi_polygon": null
    }
  ]
}
```

- **Adding / Enabling Cameras**: Add an entry with a unique `camera_id` and set `"enabled": true`.
- **Empty `rtsp_url`**: Setting `"rtsp_url": ""` indicates test/offline mode, allowing the camera definition to be used with `--source <video_file>`.

---

## 9. Output Files & Schema

Runs write all artifacts to `output/` (automatically archived to `output_prev_<timestamp>/` on subsequent runs unless `--keep-output` is passed):

- `events.json`: Complete JSON array of all emitted vehicle events.
- `events.jsonl`: Line-delimited stream of events for message queues.
- `snapshots/`: Clean vehicle crops (`vehicle_*`), plate crops (`plate_*`), and context frames (`context_*`).
- `plate_reads.jsonl`: Audit log of every raw OCR read with per-character confidences.
- `plate_crops/`: Up to 5 highest-quality plate crops saved per track (ranked by confidence $\times$ width).
- `annotated_output.mp4`: H.264 video with HUD statistics, orange vehicle boxes, and green plate boxes.

### Event ID Format

$$\text{Event ID} = \{\text{NODE}\}\{\text{CAM}\}-\{\text{YYMMDD}\}\{\text{HHMMSS}\}-\{\text{SEQ}\}$$

*Example*: `A01-261122111202-00` (Node `A`, Camera `01`, Date `26-11-22`, Time `11:12:02`, Sequence `00`).

### Event JSON Schema Examples

#### Example 1: Vehicle Event with License Plate
```json
{
  "event_id": "A01-261122111202-00",
  "camera_id": "CAM-00001",
  "time": "11:12:02",
  "date": "2026-11-22",
  "snapshot_number_plate": "plate_A01-261122111202-00.png",
  "snapshot_vehicle": "vehicle_A01-261122111202-00.png",
  "plate_detection_confidence": 0.90,
  "number_plate": "DL7CD5017",
  "number_plate_confidence": 0.60
}
```

#### Example 2: Vehicle Event Without Plate (Edge / Unreadable)
```json
{
  "event_id": "A01-261122111200-00",
  "camera_id": "CAM-00001",
  "time": "11:12:00",
  "date": "2026-11-22",
  "snapshot_number_plate": null,
  "snapshot_vehicle": "vehicle_A01-261122111200-00.png",
  "plate_detection_confidence": null,
  "number_plate": null,
  "number_plate_confidence": null
}
```

---

## 10. Key Configuration Reference (`config.py`)

| Parameter | Default | Purpose |
| :--- | :--- | :--- |
| `min_readable_vehicle_area` | `2500.0` | Bounding box pixel area cutoff for readable zone. |
| `alpr_frame_interval` | `3` | Run FastALPR every 3 frames per tracked vehicle. |
| `alpr_crop_min_width` | `608` | Upscale smaller vehicle crops to 608px before plate detection. |
| `min_track_frames` | `6` | Minimum frames vehicle must be tracked before emitting. |
| `max_disappeared_frames` | `45` | Frames before an unobserved track is emitted as ended. |
| `min_known_chars` | `3` | Minimum non-`'?'` characters required to retain a plate read. |
| `fuzzy_plate_max_distance` | `2` | Max Levenshtein edit distance for duplicate plate deduplication. |
| `duplicate_plate_window_sec` | `60.0` | Suppression window in seconds for duplicate vehicle events. |
| `box_smoothing_factor` | `0.0` | Set to 0.0 to guarantee zero lag on snapshot crops. |
| `hud_box_smoothing_factor` | `0.65` | Smoothing factor applied only to HUD bounding box drawing. |
| `draw_plate_boxes` | `True` | Draw bright green plate bounding boxes in annotated video. |

---

## 11. Debugging & Verification

- **`--debug-reads` Flag**: Prints live per-frame OCR text, per-character confidences, track stitching decisions, and box diagnostic comparisons:
  ```bash
  python run.py --source stream_2.mp4 --plate-format GENERAL --debug-reads --no-preview
  ```
- **Auditing `plate_reads.jsonl`**: Every line records raw text, per-character confidence array, bounding box dimensions, and crop file path:
  ```json
  {"event_id": "A01-261122111202-00", "track_id": 5, "frame_idx": 9, "raw_text": "DL7CD5017", "per_character_confidences": [0.9998, 0.9998, 0.9998, 0.9998, 0.9998, 0.9998, 0.9998, 0.9998, 0.9998], "plate_w": 248, "plate_h": 84, "crop_file_name": "crop_A01-261122111202-00_f9_2.png"}
  ```

---

## 12. Measured Benchmark Results

> [!NOTE]
> All metrics below reflect **actual measured session runs** on a Google Colab NVIDIA T4 GPU using Indian highway test footage (`stream_2.mp4`, 1920x1080 @ 29.97 FPS, 320 frames, 10.68 seconds) before the A1 raw-box change. Speed after subsequent modifications has not been re-benchmarked. This represents a single short test clip, not a guarantee of accuracy across all environments.

- **Events Emitted**: 5 total events
- **Plates Successfully Read**: 4 plates (`DL7CD5017`, `DL3CBJ1384`, `DL2CAT4762`, `HR26CQ6869`) + 1 edge-cut vehicle (no plate)
- **Duplicate Events**: 0
- **Pipeline Wall-Clock Runtime**: 22.91 seconds
- **Effective Pipeline Throughput**: 13.97 FPS

### Component Profiling Breakdown

| Pipeline Stage | Accounted Time | Share of Processing |
| :--- | :--- | :--- |
| **YOLO Vehicle Detector** | 10.96 s | **51.2%** |
| **Video Encoding (H.264)** | 4.77 s | **22.3%** |
| **HUD Annotation & Drawing** | 1.57 s | **7.3%** |
| **ByteTrack Vehicle Tracker**| 1.41 s | **6.6%** |
| **FastALPR (Detector + OCR)** | 1.40 s | **6.5%** |
| **Event Handling & I/O** | 0.93 s | **4.3%** |
| **Frame Ingestion** | 0.38 s | **1.8%** |

---

## 13. Known Limitations

- **System Clock Timestamps**: Event `time` and `date` are derived from the system clock at processing time, not from the video stream's internal PTS or burn-in clock.
- **Camera Resolution & Angles**: Accurate plate recognition requires $\ge 100$ px horizontal plate resolution. Steep camera angles, heavy motion blur, or inadequate lighting degrade recognition.
- **Global OCR Weights**: The underlying `cct-s-v2-global-model` is trained on international plates and has not been fine-tuned on Pakistani or local provincial plates.
- **Dual Inference Pass**: To prevent Kalman box inflation, the pipeline temporarily runs `model.predict()` in parallel with `model.track()`. Single-pass batching is required for multi-camera scale.
- **Stage 1 Scope**: This repository handles vehicle detection, tracking, snapshots, and ALPR only. Make, model, and colour classification (Stage 2) are not included.

---

## 14. Roadmap

1. **Video Clock Timestamps**: Extract PTS/DTS from video containers for historical batch processing.
2. **24-Hour Batch Runner**: Continuous directory watcher and batch scheduler for unattended processing.
3. **Single-Pass Detection + Tracking**: Unified inference worker with TensorRT / ONNX Runtime to support 40 cameras at 10 FPS on a single server.
4. **Stage 2 Interface Integration**: Stream vehicle snapshots directly into make/model/colour classification backends via the existing `Stage2Worker` queue.
5. **Domain Fine-Tuning**: Retrain `fast-plate-ocr` on collected crops from `plate_crops/` and annotations from `plate_reads.jsonl`.

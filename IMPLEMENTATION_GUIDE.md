# RF-DETR Polyp Detection - Implementation Guide

A complete guide to replicate this GPU-accelerated real-time object detection application with smooth tracking.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Key Components](#key-components)
4. [Dependencies](#dependencies)
5. [Tracking Implementation](#tracking-implementation)
6. [Tunable Parameters](#tunable-parameters)
7. [Cursor AI Prompts for Replication](#cursor-ai-prompts-for-replication)

---

## Project Overview

This application performs real-time polyp detection on colonoscopy videos using:
- **RF-DETR** (Real-time Focused DETR) model exported to ONNX
- **ONNX Runtime** with GPU acceleration (CUDA/TensorRT)
- **DearPyGui** for GPU-accelerated GUI
- **Custom IoU-based tracker** with temporal smoothing

### Features
- Video playback with Play/Pause/Stop/Seek controls
- Real-time object detection with bounding boxes
- Smooth tracking using IoU matching + exponential moving average
- Video recording with detection overlays (MP4 format)
- Adjustable confidence threshold

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         App (Main Controller)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐ │
│  │ VideoPlayer  │   │ Inference    │   │    PolyTracker       │ │
│  │              │   │ Engine       │   │                      │ │
│  │ - Load video │   │              │   │ - IoU matching       │ │
│  │ - Read frame │──▶│ - Preprocess │──▶│ - NMS filtering      │ │
│  │ - Seek       │   │ - ONNX infer │   │ - Temporal smoothing │ │
│  │ - FPS control│   │ - Postprocess│   │ - Track management   │ │
│  └──────────────┘   └──────────────┘   └──────────────────────┘ │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────────┐│
│  │                      DearPyGui UI                            ││
│  │  - Video texture display                                     ││
│  │  - Playback controls                                         ││
│  │  - Recording controls                                        ││
│  │  - Confidence threshold slider                               ││
│  └──────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Components

### 1. InferenceEngine
Handles ONNX model loading and inference with GPU acceleration.

```python
class InferenceEngine:
    def __init__(self, model_path):
        # Pre-allocate input buffer for efficiency
        self.input_buffer = np.zeros((1, 3, 512, 512), dtype=np.float32)
    
    def load(self):
        # Priority: TensorRT > CUDA > CPU
        providers = [
            ('TensorrtExecutionProvider', {...}),
            ('CUDAExecutionProvider', {...}),
            'CPUExecutionProvider'
        ]
        self.session = ort.InferenceSession(model_path, providers=providers)
    
    def preprocess(self, frame):
        # Resize, BGR→RGB, normalize with ImageNet stats
        # HWC → CHW format
    
    def postprocess(self, pred_boxes, pred_logits, width, height, threshold):
        # Apply sigmoid to logits
        # Filter by threshold
        # Convert cxcywh → xyxy
        # Return list of (x1, y1, x2, y2, confidence)
    
    def predict(self, frame, threshold):
        # Full pipeline: preprocess → infer → postprocess
```

### 2. PolyTracker (Custom Tracker)
IoU-based tracker with temporal smoothing - better than ByteTrack for stationary objects.

```python
class PolyTracker:
    def __init__(self, frame_rate):
        self.active_tracks = {}  # track_id → {box, conf, age, hits, history}
        self.iou_threshold = 0.15   # Matching threshold
        self.max_age = frame_rate * 0.2  # Track persistence (0.2 sec)
        self.min_hits = 4           # Required detections before showing
        self.history_size = 8       # Smoothing buffer
    
    def _compute_iou(self, box1, box2):
        # Standard IoU calculation
    
    def _apply_nms(self, detections, threshold):
        # Non-Maximum Suppression
    
    def _smooth_box(self, history):
        # Exponential moving average (alpha=0.3)
    
    def update(self, raw_detections, frame_shape):
        # 1. Apply NMS to input
        # 2. Match detections to existing tracks (IoU)
        # 3. Update matched tracks, create new for unmatched
        # 4. Age and expire old tracks
        # 5. Build results (only tracks with min_hits)
        # 6. Apply NMS to output
        # 7. Return smoothed boxes
```

### 3. VideoPlayer
OpenCV-based video file handling.

```python
class VideoPlayer:
    def load(self, filepath):
        self.cap = cv2.VideoCapture(filepath)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        # ...
    
    def read_frame(self):
        return self.cap.read()
    
    def seek(self, frame_number):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
```

### 4. App (Main Controller)
Orchestrates all components and manages UI.

```python
class App:
    def __init__(self):
        self.engine = InferenceEngine(MODEL_PATH)
        self.video = VideoPlayer()
        self.tracker = None  # Initialized when video loads
    
    def process_and_display_frame(self, frame):
        # 1. Run inference
        raw_detections = self.engine.predict(frame, threshold)
        
        # 2. Apply tracking
        tracked = self.tracker.update(raw_detections, frame_shape)
        
        # 3. Draw boxes
        frame_with_boxes = self.draw_detections(frame, tracked)
        
        # 4. Record if active
        if self.is_recording:
            self.video_writer.write(frame_with_boxes)
        
        # 5. Display in GUI
        self.update_texture(frame_with_boxes)
```

---

## Dependencies

```txt
# requirements.txt
numpy>=1.24.0
opencv-python>=4.8.0
dearpygui>=2.0.0
PyQt6>=6.5.0
supervision>=0.19.0

# ONNX Runtime - install separately for GPU support
# For standard GPUs: pip install onnxruntime-gpu
# For RTX 50-series (sm_120): use custom build from HuggingFace
```

---

## Tracking Implementation

### Why Not ByteTrack?
ByteTrack is designed for **moving objects** (people, cars). For **stationary objects** like polyps:
- Creates new track IDs when detection flickers
- Doesn't expose "lost track" positions for display
- Kalman filter predictions cause drift

### Custom IoU-Based Tracker
Better suited for medical imaging where objects are relatively stationary:

1. **IoU Matching**: New detections matched to existing tracks if overlap ≥ threshold
2. **NMS Filtering**: Applied to both input and output to remove duplicates
3. **Exponential Smoothing**: Weighted average of recent positions
4. **Track Persistence**: Keeps showing tracks briefly after detection loss
5. **Min Hits Filter**: Only shows tracks with consistent detections

### Smoothing Algorithm
```python
def _smooth_box(self, history):
    alpha = 0.3  # Smoothing factor (lower = smoother, higher = responsive)
    smoothed = list(history[0])
    for box in history[1:]:
        for i in range(4):
            smoothed[i] = alpha * box[i] + (1 - alpha) * smoothed[i]
    return tuple(int(x) for x in smoothed)
```

---

## Tunable Parameters

### Tracker Settings (in `PolyTracker.__init__`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `iou_threshold` | 0.15 | Min IoU to match detection to existing track |
| `max_age` | 0.2 sec | How long to keep track without new detection |
| `min_hits` | 4 | Detections needed before displaying track |
| `history_size` | 8 | Number of frames for smoothing buffer |
| `alpha` | 0.3 | Smoothing factor (in `_smooth_box`) |

### NMS Settings (in `update` method)

| Parameter | Default | Description |
|-----------|---------|-------------|
| Input NMS threshold | 0.25 | Suppress overlapping raw detections |
| Output NMS threshold | 0.2 | Suppress overlapping tracked boxes |

### Trade-offs

| Want | Adjust |
|------|--------|
| Smoother boxes | Lower `alpha`, increase `history_size` |
| Faster response | Higher `alpha`, decrease `history_size` |
| Fewer false positives | Increase `min_hits` |
| Longer persistence | Increase `max_age` |
| Less duplicate boxes | Lower NMS thresholds |

---

## Cursor AI Prompts for Replication

### Prompt 1: Basic ONNX Inference Application

```
Create a GPU-accelerated video inference application with the following specifications:

1. **Framework**: Use DearPyGui for the GUI (GPU-accelerated) and PyQt6 for native file dialogs.

2. **Inference Engine**:
   - Load an ONNX model with ONNX Runtime
   - Support TensorRT > CUDA > CPU provider priority
   - Pre-allocate input buffers for efficiency
   - Input: 512x512 RGB image, ImageNet normalized
   - Output: pred_boxes (cxcywh normalized), pred_logits (need sigmoid)

3. **Video Player**:
   - OpenCV-based video loading
   - Play/Pause/Stop controls
   - Seek with scrubber
   - FPS display

4. **UI Layout**:
   - Video display area (1280x720)
   - Playback controls row
   - Confidence threshold slider
   - Status bar

5. **Best Practices**:
   - Use virtual environment
   - Create requirements.txt with versions
   - Add .gitignore for Python projects

Model path: checkpoint_best_total.onnx (RF-DETR Small, 512x512, 2 classes)
```

### Prompt 2: Add Smooth Tracking (Modular)

```
Add a modular object tracking system to smooth detections and reduce flickering.

Create a `PolyTracker` class with:

1. **IoU-based matching** instead of ByteTrack (better for stationary objects)
2. **Non-Maximum Suppression** on both input and output
3. **Exponential moving average smoothing** on box coordinates
4. **Track management**: age, hits, history buffer

Key methods:
- `__init__(frame_rate)`: Initialize with configurable parameters
- `reset()`: Clear all tracks (call on video load/seek)
- `_compute_iou(box1, box2)`: Standard IoU calculation
- `_apply_nms(detections, threshold)`: Remove overlapping boxes
- `_smooth_box(history)`: Exponential moving average
- `update(raw_detections, frame_shape)`: Main tracking logic

Parameters to expose:
- iou_threshold (default 0.15)
- max_age (default 0.2 * frame_rate)
- min_hits (default 4)
- history_size (default 8)
- alpha for smoothing (default 0.3)
- NMS thresholds for input (0.25) and output (0.2)

Integration:
- Initialize tracker when video loads (with video's FPS)
- Reset tracker on seek/stop
- Call tracker.update() in process_and_display_frame()
```

### Prompt 3: Add Recording Feature

```
Add video recording functionality to capture clips with detection overlays.

Requirements:
1. **Record button** that toggles recording on/off
2. **Visual feedback**: Button turns red when recording, show "● REC" indicator with timer
3. **Output format**: MP4 with mp4v codec (YouTube/Facebook compatible)
4. **Save location**: ~/Videos/CADe_Recordings/ folder
5. **Filename**: polyp_detection_YYYYMMDD_HHMMSS.mp4

Recording logic:
- Write frames BEFORE resizing for display (original resolution)
- Include bounding box overlays in recording
- Stop recording automatically when video ends
- Show file info after saving (duration, size, frame count)

UI changes:
- Add Record button between Stop and scrubber
- Add recording indicator text (red color)
- Create red theme for active recording button
```

### Prompt 4: Complete Modular Implementation

```
Create a modular real-time object detection application for medical imaging (colonoscopy polyp detection).

Structure the code with these separate classes:

1. **InferenceEngine** (inference.py or in main.py):
   - ONNX Runtime with GPU acceleration
   - Preprocessing: resize, normalize (ImageNet), HWC→CHW
   - Postprocessing: sigmoid, threshold filter, cxcywh→xyxy conversion

2. **PolyTracker** (tracker.py or in main.py):
   - IoU-based track matching (NOT ByteTrack)
   - NMS on input AND output
   - Exponential moving average smoothing
   - Track lifecycle: create, update, age, expire
   - Configurable: iou_threshold, max_age, min_hits, history_size, alpha

3. **VideoPlayer** (video.py or in main.py):
   - OpenCV video capture
   - Load, read, seek, reset, release
   - Time/duration formatting

4. **App** (main.py):
   - DearPyGui UI setup
   - Component orchestration
   - Recording functionality
   - Main loop

Key implementation details:
- Tracker is initialized with video's actual FPS
- Tracker reset on: video load, seek, stop
- Draw only bounding boxes (no labels for clean output)
- Pre-allocate buffers where possible for performance

Dependencies: numpy, opencv-python, dearpygui, PyQt6, supervision, onnxruntime-gpu
```

---

## File Structure

```
CADe/
├── main.py                 # Main application (all classes)
├── checkpoint_best_total.onnx  # ONNX model file
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── IMPLEMENTATION_GUIDE.md # This file
├── .gitignore             # Git ignore rules
└── .venv/                 # Virtual environment (not in git)
```

---

## Quick Start

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
pip install onnxruntime-gpu  # Or custom build for RTX 50-series

# Run application
python main.py
```

---

## Notes for Future Projects

1. **For stationary objects** (medical imaging, security cameras): Use IoU-based tracker, NOT ByteTrack/DeepSORT

2. **For moving objects** (people, vehicles): ByteTrack/DeepSORT with Kalman filter is better

3. **Smoothing trade-off**: Lower alpha = smoother but laggy; higher alpha = responsive but jittery

4. **NMS is critical**: Apply to both raw detections AND tracked output to prevent duplicates

5. **GPU memory**: Pre-allocate input buffers, reuse numpy arrays where possible

6. **Recording**: Always record at original resolution, resize only for display

---

*Generated from CADe Polyp Detection Project - January 2026*

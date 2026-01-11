"""
RF-DETR Video Inference Application
GPU-accelerated polyp detection using ONNX Runtime and DearPyGui
"""

import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import dearpygui.dearpygui as dpg
import numpy as np
import onnxruntime as ort
import supervision as sv
from PyQt6.QtWidgets import QApplication, QFileDialog

# Create QApplication instance for native file dialogs (must be created before any Qt widgets)
qt_app = QApplication(sys.argv)


# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Model constants
MODEL_INPUT_SIZE = 512
MODEL_PATH = Path(__file__).parent / "checkpoint_best_total.onnx"


class PolyTracker:
    """
    Simple temporal smoothing tracker for polyp detection.
    
    Uses IoU-based matching and temporal persistence instead of ByteTrack.
    Better suited for relatively stationary objects like polyps.
    """
    
    def __init__(self, frame_rate: float = 30.0):
        """
        Initialize the tracker.
        
        Args:
            frame_rate: Video frame rate
        """
        self.frame_rate = frame_rate
        
        # Detection history for temporal smoothing
        # Each entry: (x1, y1, x2, y2, confidence, track_id, frames_since_seen)
        self.active_tracks = {}  # track_id -> {'box': (x1,y1,x2,y2), 'conf': float, 'age': int, 'history': deque}
        
        self.next_track_id = 1
        self.iou_threshold = 0.15  # Even more aggressive matching
        self.max_age = int(frame_rate * 0.2)  # Keep tracks for only 0.2 seconds (6 frames at 30fps)
        self.min_hits = 4  # Require 4 detections before showing
        self.history_size = 8  # Frames of history for smoothing
    
    def reset(self):
        """Reset the tracker state."""
        self.active_tracks.clear()
        self.next_track_id = 1
    
    def _compute_iou(self, box1: tuple, box2: tuple) -> float:
        """Compute Intersection over Union between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _apply_nms(self, detections: list, nms_threshold: float = 0.5) -> list:
        """
        Apply Non-Maximum Suppression to remove overlapping detections.
        
        Args:
            detections: List of (x1, y1, x2, y2, confidence) tuples
            nms_threshold: IoU threshold for suppression
            
        Returns:
            Filtered list of detections
        """
        if len(detections) == 0:
            return []
        
        # Sort by confidence (highest first)
        sorted_dets = sorted(detections, key=lambda x: x[4], reverse=True)
        
        keep = []
        while sorted_dets:
            # Keep the highest confidence detection
            best = sorted_dets.pop(0)
            keep.append(best)
            
            # Remove detections that overlap too much with the best one
            best_box = (best[0], best[1], best[2], best[3])
            remaining = []
            for det in sorted_dets:
                det_box = (det[0], det[1], det[2], det[3])
                if self._compute_iou(best_box, det_box) < nms_threshold:
                    remaining.append(det)
            sorted_dets = remaining
        
        return keep
    
    def _smooth_box(self, history: deque) -> tuple:
        """Apply exponential moving average to box coordinates."""
        if len(history) == 0:
            return None
        if len(history) == 1:
            return history[0]
        
        # Exponential moving average - recent frames weighted more
        alpha = 0.3  # Smoothing factor (higher = more responsive, lower = smoother)
        
        # Start with oldest
        smoothed = list(history[0])
        for i in range(1, len(history)):
            box = history[i]
            for j in range(4):
                smoothed[j] = alpha * box[j] + (1 - alpha) * smoothed[j]
        
        return (int(smoothed[0]), int(smoothed[1]), int(smoothed[2]), int(smoothed[3]))
    
    def update(self, raw_detections: list, frame_shape: tuple, debug: bool = False) -> list:
        """
        Update tracker with new detections and return smoothed tracked objects.
        
        Args:
            raw_detections: List of (x1, y1, x2, y2, confidence) tuples from detector
            frame_shape: (height, width) of the frame
            debug: Print debug information
            
        Returns:
            List of (x1, y1, x2, y2, confidence, track_id) tuples with smoothed boxes
        """
        # Apply NMS to remove duplicate/overlapping detections (aggressive)
        filtered_detections = self._apply_nms(raw_detections, nms_threshold=0.25)
        
        if debug and len(raw_detections) != len(filtered_detections):
            print(f"[DEBUG] NMS: {len(raw_detections)} -> {len(filtered_detections)} detections")
        
        # Increment age for all existing tracks
        for track_id in self.active_tracks:
            self.active_tracks[track_id]['age'] += 1
        
        # Match detections to existing tracks using IoU
        matched_tracks = set()
        unmatched_detections = []
        
        for det in filtered_detections:
            det_box = (det[0], det[1], det[2], det[3])
            det_conf = det[4]
            
            best_iou = 0
            best_track_id = None
            
            for track_id, track in self.active_tracks.items():
                if track_id in matched_tracks:
                    continue
                iou = self._compute_iou(det_box, track['box'])
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_track_id = track_id
            
            if best_track_id is not None:
                # Update existing track
                matched_tracks.add(best_track_id)
                track = self.active_tracks[best_track_id]
                track['box'] = det_box
                track['conf'] = det_conf
                track['age'] = 0
                track['hits'] = track.get('hits', 0) + 1
                track['history'].append(det_box)
            else:
                unmatched_detections.append(det)
        
        # Create new tracks for unmatched detections
        for det in unmatched_detections:
            det_box = (det[0], det[1], det[2], det[3])
            det_conf = det[4]
            
            self.active_tracks[self.next_track_id] = {
                'box': det_box,
                'conf': det_conf,
                'age': 0,
                'hits': 1,
                'history': deque([det_box], maxlen=self.history_size)
            }
            self.next_track_id += 1
        
        # Remove old tracks
        expired = [tid for tid, t in self.active_tracks.items() if t['age'] > self.max_age]
        for tid in expired:
            del self.active_tracks[tid]
        
        # Build results - only show tracks with enough hits
        results = []
        for track_id, track in self.active_tracks.items():
            if track['hits'] >= self.min_hits:
                # Use smoothed box
                smoothed_box = self._smooth_box(track['history'])
                if smoothed_box:
                    # Fade confidence if track is aging (not seen recently)
                    conf = track['conf']
                    if track['age'] > 0:
                        fade_factor = max(0.5, 1.0 - (track['age'] / self.max_age))
                        conf = conf * fade_factor
                    
                    results.append((*smoothed_box, conf, track_id))
        
        if debug and len(raw_detections) > 0:
            print(f"[DEBUG] Raw: {len(raw_detections)}, Active tracks: {len(self.active_tracks)}, Showing: {len(results)}")
        
        # Apply NMS to output to remove overlapping tracked boxes
        if len(results) > 1:
            # Convert to format for NMS (x1, y1, x2, y2, conf)
            results_for_nms = [(r[0], r[1], r[2], r[3], r[4]) for r in results]
            track_ids = [r[5] for r in results]
            
            filtered = self._apply_nms(results_for_nms, nms_threshold=0.2)
            
            # Rebuild results with track IDs
            filtered_set = set((f[0], f[1], f[2], f[3]) for f in filtered)
            results = [r for r in results if (r[0], r[1], r[2], r[3]) in filtered_set]
        
        return results


class InferenceEngine:
    """GPU-accelerated ONNX inference engine for RF-DETR model."""

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.provider = None

        # Pre-allocated input buffer
        self.input_buffer = np.zeros((1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), dtype=np.float32)

    def load(self) -> bool:
        """Load the ONNX model with GPU acceleration."""
        if not self.model_path.exists():
            print(f"Model not found: {self.model_path}")
            return False

        # Configure providers with GPU priority
        providers = []

        # Try TensorRT first (fastest)
        try:
            providers.append(('TensorrtExecutionProvider', {
                'device_id': 0,
                'trt_fp16_enable': True,
            }))
        except Exception:
            pass

        # Then CUDA
        providers.append(('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kSameAsRequested',
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
        }))

        # CPU fallback
        providers.append('CPUExecutionProvider')

        try:
            self.session = ort.InferenceSession(str(self.model_path), providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.provider = self.session.get_providers()[0]
            print(f"Model loaded with provider: {self.provider}")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for model input."""
        # Resize to model input size
        resized = cv2.resize(frame, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)

        # BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0

        # Apply ImageNet normalization
        normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD

        # HWC to CHW and add batch dimension
        chw = normalized.transpose(2, 0, 1)
        self.input_buffer[0] = chw

        return self.input_buffer

    def postprocess(self, pred_boxes: np.ndarray, pred_logits: np.ndarray,
                    frame_width: int, frame_height: int, threshold: float) -> list:
        """
        Postprocess model outputs to get detections.

        Args:
            pred_boxes: Bounding boxes in cxcywh format (normalized 0-1)
            pred_logits: Class logits (need sigmoid)
            frame_width: Original frame width
            frame_height: Original frame height
            threshold: Confidence threshold

        Returns:
            List of (x1, y1, x2, y2, confidence) tuples
        """
        # Apply sigmoid to get probabilities
        probs = 1 / (1 + np.exp(-pred_logits))

        # Get polyp class probabilities (class 1)
        # Shape: [batch, num_queries, num_classes] -> polyp is class 1
        polyp_probs = probs[0, :, 1]  # [num_queries]

        # Filter by threshold
        mask = polyp_probs > threshold

        if not np.any(mask):
            return []

        # Get filtered boxes and scores
        filtered_boxes = pred_boxes[0, mask]  # [N, 4] in cxcywh
        filtered_scores = polyp_probs[mask]  # [N]

        detections = []
        for box, score in zip(filtered_boxes, filtered_scores):
            cx, cy, w, h = box

            # Convert cxcywh to xyxy
            x1 = (cx - w / 2) * frame_width
            y1 = (cy - h / 2) * frame_height
            x2 = (cx + w / 2) * frame_width
            y2 = (cy + h / 2) * frame_height

            # Clamp to frame bounds
            x1 = max(0, min(frame_width, x1))
            y1 = max(0, min(frame_height, y1))
            x2 = max(0, min(frame_width, x2))
            y2 = max(0, min(frame_height, y2))

            detections.append((int(x1), int(y1), int(x2), int(y2), float(score)))

        return detections

    def predict(self, frame: np.ndarray, threshold: float = 0.5) -> list:
        """Run inference on a frame and return detections."""
        if self.session is None:
            return []

        h, w = frame.shape[:2]

        # Preprocess
        input_tensor = self.preprocess(frame)

        # Run inference
        outputs = self.session.run(None, {self.input_name: input_tensor})

        # Model outputs: pred_boxes, pred_logits
        pred_boxes = outputs[0]
        pred_logits = outputs[1]

        # Postprocess
        detections = self.postprocess(pred_boxes, pred_logits, w, h, threshold)

        return detections


class VideoPlayer:
    """Video file loading and playback."""

    def __init__(self):
        self.cap = None
        self.frame_count = 0
        self.fps = 30.0
        self.width = 0
        self.height = 0
        self.current_frame = 0
        self.is_playing = False
        self.filepath = None

    def load(self, filepath: str) -> bool:
        """Load a video file."""
        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(filepath)

        if not self.cap.isOpened():
            print(f"Failed to open video: {filepath}")
            return False

        self.filepath = filepath
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_frame = 0

        print(f"Video loaded: {self.width}x{self.height}, {self.frame_count} frames, {self.fps:.1f} FPS")
        return True

    def read_frame(self) -> tuple:
        """Read the next frame. Returns (success, frame)."""
        if self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        if ret:
            self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        return ret, frame

    def seek(self, frame_number: int):
        """Seek to a specific frame."""
        if self.cap is None:
            return
        frame_number = max(0, min(self.frame_count - 1, frame_number))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        self.current_frame = frame_number

    def reset(self):
        """Reset to the beginning."""
        self.seek(0)
        self.is_playing = False

    def get_duration_str(self) -> str:
        """Get video duration as string."""
        if self.frame_count == 0:
            return "00:00"
        total_seconds = self.frame_count / self.fps
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

    def get_current_time_str(self) -> str:
        """Get current position as string."""
        if self.frame_count == 0:
            return "00:00"
        current_seconds = self.current_frame / self.fps
        minutes = int(current_seconds // 60)
        seconds = int(current_seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

    def release(self):
        """Release video resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class App:
    """Main application class."""

    def __init__(self):
        self.engine = InferenceEngine(MODEL_PATH)
        self.video = VideoPlayer()
        self.tracker = None  # Initialized when video loads with correct FPS
        self.last_frame = None
        self.last_detections = []
        self.fps_counter = 0.0
        self.frame_times = []
        self.display_width = 1280
        self.display_height = 720
        
        # Recording state
        self.is_recording = False
        self.video_writer = None
        self.recording_path = None
        self.recording_start_time = None
        self.frames_recorded = 0

    def setup_ui(self):
        """Set up the DearPyGui interface."""
        dpg.create_context()
        dpg.create_viewport(title="RF-DETR Polyp Detection", width=1400, height=900)

        # Create texture for video display
        with dpg.texture_registry():
            # Initialize with black frame
            default_data = np.zeros((self.display_height, self.display_width, 3), dtype=np.float32).flatten()
            dpg.add_raw_texture(
                width=self.display_width,
                height=self.display_height,
                default_value=default_data,
                format=dpg.mvFormat_Float_rgb,
                tag="video_texture"
            )

        # Apply theme
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 4)
                dpg.add_theme_color(dpg.mvThemeCol_Button, (66, 150, 250))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (86, 170, 255))
        dpg.bind_theme(global_theme)
        
        # Red theme for recording button when active
        with dpg.theme(tag="recording_theme"):
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (200, 50, 50))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (230, 70, 70))

        # Main window
        with dpg.window(label="Main", tag="main_window"):
            # Menu bar
            with dpg.menu_bar():
                with dpg.menu(label="File"):
                    dpg.add_menu_item(label="Open Video...", callback=self.open_video_dialog)
                    dpg.add_separator()
                    dpg.add_menu_item(label="Exit", callback=dpg.stop_dearpygui)

            # Video display
            dpg.add_image("video_texture", tag="video_display")

            dpg.add_spacer(height=10)

            # Playback controls
            with dpg.group(horizontal=True):
                dpg.add_button(label="Play", callback=self.play, width=80)
                dpg.add_button(label="Pause", callback=self.pause, width=80)
                dpg.add_button(label="Stop", callback=self.stop, width=80)
                dpg.add_spacer(width=20)
                dpg.add_button(label="Record", callback=self.toggle_recording, width=80, tag="record_button")
                dpg.add_text("", tag="recording_indicator", color=(255, 50, 50))
                dpg.add_spacer(width=20)
                dpg.add_slider_int(
                    label="",
                    tag="frame_slider",
                    min_value=0,
                    max_value=1000,
                    width=500,
                    callback=self.on_seek
                )
                dpg.add_spacer(width=10)
                dpg.add_text("00:00 / 00:00", tag="time_display")

            dpg.add_spacer(height=10)

            # Settings row
            with dpg.group(horizontal=True):
                dpg.add_text("Confidence Threshold:")
                dpg.add_slider_float(
                    label="",
                    tag="threshold_slider",
                    default_value=0.5,
                    min_value=0.0,
                    max_value=1.0,
                    width=200
                )
                dpg.add_spacer(width=50)
                dpg.add_text("FPS: 0.0", tag="fps_display")

            dpg.add_spacer(height=10)

            # Status bar
            dpg.add_text("Status: Loading model...", tag="status_bar")

        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)

    def open_video_dialog(self):
        """Open native Windows file dialog using PyQt6."""
        filepath, _ = QFileDialog.getOpenFileName(
            None,
            "Select Video File",
            str(Path.home() / "Videos"),
            "Video Files (*.mp4 *.avi *.mkv *.mov);;MP4 Files (*.mp4);;AVI Files (*.avi);;All Files (*.*)"
        )

        if filepath:
            if self.video.load(filepath):
                # Initialize tracker with video's actual frame rate
                self.tracker = PolyTracker(frame_rate=self.video.fps)
                
                dpg.configure_item("frame_slider", max_value=self.video.frame_count - 1)
                dpg.set_value("status_bar", f"Loaded: {Path(filepath).name} ({self.video.width}x{self.video.height})")
                self.update_time_display()
                # Display the first frame
                ret, frame = self.video.read_frame()
                if ret:
                    self.process_and_display_frame(frame)
                    self.video.seek(0)  # Reset to beginning after showing first frame
            else:
                dpg.set_value("status_bar", f"Failed to load: {filepath}")

    def play(self):
        """Start playback."""
        self.video.is_playing = True
        dpg.set_value("status_bar", "Playing...")

    def pause(self):
        """Pause playback."""
        self.video.is_playing = False
        dpg.set_value("status_bar", "Paused")

    def stop(self):
        """Stop and reset playback."""
        # Stop recording if active
        if self.is_recording:
            self.stop_recording()
        
        # Reset tracker to clear stale tracks
        if self.tracker is not None:
            self.tracker.reset()
        
        self.video.reset()
        dpg.set_value("frame_slider", 0)
        self.update_time_display()
        dpg.set_value("status_bar", "Stopped")

    def toggle_recording(self):
        """Toggle video recording on/off."""
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        """Start recording video with detections."""
        if self.video.cap is None:
            dpg.set_value("status_bar", "Load a video first before recording!")
            return
        
        # Create output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save to Videos folder
        videos_folder = Path.home() / "Videos" / "CADe_Recordings"
        videos_folder.mkdir(parents=True, exist_ok=True)
        
        self.recording_path = videos_folder / f"polyp_detection_{timestamp}.mp4"
        
        # Use H.264 codec for best compatibility with YouTube/Facebook
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Record at original video resolution for best quality
        self.video_writer = cv2.VideoWriter(
            str(self.recording_path),
            fourcc,
            self.video.fps,
            (self.video.width, self.video.height)
        )
        
        if not self.video_writer.isOpened():
            dpg.set_value("status_bar", "Failed to start recording!")
            self.video_writer = None
            return
        
        self.is_recording = True
        self.recording_start_time = time.time()
        self.frames_recorded = 0
        
        # Update UI
        dpg.configure_item("record_button", label="Stop Rec")
        dpg.bind_item_theme("record_button", "recording_theme")
        dpg.set_value("recording_indicator", "● REC")
        dpg.set_value("status_bar", f"Recording to: {self.recording_path.name}")

    def stop_recording(self):
        """Stop recording and save the video."""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        
        self.is_recording = False
        
        # Calculate recording duration
        if self.recording_start_time:
            duration = time.time() - self.recording_start_time
            duration_str = f"{int(duration // 60):02d}:{int(duration % 60):02d}"
        else:
            duration_str = "00:00"
        
        # Update UI
        dpg.configure_item("record_button", label="Record")
        dpg.bind_item_theme("record_button", "")
        dpg.set_value("recording_indicator", "")
        
        if self.recording_path and self.recording_path.exists():
            file_size_mb = self.recording_path.stat().st_size / (1024 * 1024)
            dpg.set_value("status_bar", 
                f"Saved: {self.recording_path.name} ({duration_str}, {file_size_mb:.1f} MB, {self.frames_recorded} frames)")
            print(f"Recording saved to: {self.recording_path}")
        else:
            dpg.set_value("status_bar", "Recording stopped")
        
        self.recording_path = None
        self.recording_start_time = None
        self.frames_recorded = 0

    def on_seek(self, sender, app_data):
        """Handle scrubber seeking."""
        self.video.seek(app_data)
        self.update_time_display()
        
        # Reset tracker on seek to avoid stale track associations
        if self.tracker is not None:
            self.tracker.reset()
        
        # Read and display the frame at this position
        ret, frame = self.video.read_frame()
        if ret:
            self.process_and_display_frame(frame)

    def update_time_display(self):
        """Update the time display."""
        current = self.video.get_current_time_str()
        total = self.video.get_duration_str()
        dpg.set_value("time_display", f"{current} / {total}")

    def draw_detections(self, frame: np.ndarray, detections: list, with_track_id: bool = True) -> np.ndarray:
        """
        Draw bounding boxes on frame.
        
        Args:
            frame: The video frame to draw on
            detections: List of tuples. If with_track_id=True: (x1, y1, x2, y2, conf, track_id)
                       Otherwise: (x1, y1, x2, y2, conf)
            with_track_id: Whether detections include track IDs
        """
        for detection in detections:
            if with_track_id and len(detection) >= 6:
                x1, y1, x2, y2, conf, track_id = detection[:6]
            else:
                x1, y1, x2, y2, conf = detection[:5]
            
            # Draw box only - no label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return frame

    def process_and_display_frame(self, frame: np.ndarray):
        """Process a frame through inference, tracking, and display it."""
        start_time = time.perf_counter()

        # Get threshold
        threshold = dpg.get_value("threshold_slider")

        # Run inference to get raw detections
        raw_detections = self.engine.predict(frame, threshold)
        
        # Apply tracking with smoothing if tracker is initialized
        if self.tracker is not None:
            frame_shape = (frame.shape[0], frame.shape[1])
            # Enable debug every 30 frames (once per second at 30fps)
            debug_this_frame = (self.video.current_frame % 30 == 0) and len(raw_detections) > 0
            tracked_detections = self.tracker.update(raw_detections, frame_shape, debug=debug_this_frame)
            # Draw tracked detections (with track IDs)
            frame_with_detections = self.draw_detections(frame.copy(), tracked_detections, with_track_id=True)
            detections = tracked_detections
        else:
            # Fallback to raw detections if tracker not initialized
            frame_with_detections = self.draw_detections(frame.copy(), raw_detections, with_track_id=False)
            detections = raw_detections
        
        # Write to recording at original resolution (before resizing for display)
        if self.is_recording and self.video_writer is not None:
            self.video_writer.write(frame_with_detections)
            self.frames_recorded += 1
            
            # Update recording indicator with time
            if self.recording_start_time:
                rec_duration = time.time() - self.recording_start_time
                rec_time_str = f"{int(rec_duration // 60):02d}:{int(rec_duration % 60):02d}"
                dpg.set_value("recording_indicator", f"● REC {rec_time_str}")

        # Resize for display
        display_frame = cv2.resize(frame_with_detections, (self.display_width, self.display_height))

        # Convert BGR to RGB and normalize for DearPyGui
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        frame_data = np.ascontiguousarray(rgb_frame.astype(np.float32) / 255.0).flatten()

        # Update texture
        dpg.set_value("video_texture", frame_data)

        # Update FPS
        elapsed = time.perf_counter() - start_time
        self.frame_times.append(elapsed)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        avg_time = sum(self.frame_times) / len(self.frame_times)
        self.fps_counter = 1.0 / avg_time if avg_time > 0 else 0
        dpg.set_value("fps_display", f"FPS: {self.fps_counter:.1f}")

        self.last_detections = detections

    def update(self):
        """Main update loop - called every frame."""
        if self.video.is_playing and self.video.cap is not None:
            ret, frame = self.video.read_frame()

            if ret:
                self.process_and_display_frame(frame)
                dpg.set_value("frame_slider", self.video.current_frame)
                self.update_time_display()
            else:
                # End of video
                self.video.is_playing = False
                # Stop recording if active
                if self.is_recording:
                    self.stop_recording()
                else:
                    dpg.set_value("status_bar", "Playback complete")

    def run(self):
        """Run the application."""
        self.setup_ui()

        # Load model
        if self.engine.load():
            dpg.set_value("status_bar", f"Model loaded ({self.engine.provider}). Open a video to begin.")
        else:
            dpg.set_value("status_bar", "Failed to load model. Check console for errors.")

        # Main loop
        while dpg.is_dearpygui_running():
            self.update()
            dpg.render_dearpygui_frame()

        # Cleanup
        if self.is_recording:
            self.stop_recording()
        self.video.release()
        dpg.destroy_context()


if __name__ == "__main__":
    app = App()
    app.run()

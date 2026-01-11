"""
RF-DETR Video Inference Application
GPU-accelerated polyp detection using ONNX Runtime and DearPyGui
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import dearpygui.dearpygui as dpg
import numpy as np
import onnxruntime as ort
from PyQt6.QtWidgets import QApplication, QFileDialog

# Create QApplication instance for native file dialogs (must be created before any Qt widgets)
qt_app = QApplication(sys.argv)


# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Model constants
MODEL_INPUT_SIZE = 512
MODEL_PATH = Path(__file__).parent / "checkpoint_best_total.onnx"


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
        # Read and display the frame at this position
        ret, frame = self.video.read_frame()
        if ret:
            self.process_and_display_frame(frame)

    def update_time_display(self):
        """Update the time display."""
        current = self.video.get_current_time_str()
        total = self.video.get_duration_str()
        dpg.set_value("time_display", f"{current} / {total}")

    def draw_detections(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """Draw bounding boxes and confidence scores on frame."""
        for x1, y1, x2, y2, conf in detections:
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label background
            label = f"Polyp: {conf:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), (0, 255, 0), -1)

            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return frame

    def process_and_display_frame(self, frame: np.ndarray):
        """Process a frame through inference and display it."""
        start_time = time.perf_counter()

        # Get threshold
        threshold = dpg.get_value("threshold_slider")

        # Run inference
        detections = self.engine.predict(frame, threshold)

        # Draw detections on a copy for display
        frame_with_detections = self.draw_detections(frame.copy(), detections)
        
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

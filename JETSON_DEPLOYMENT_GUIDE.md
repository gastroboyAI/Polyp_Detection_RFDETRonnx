# RF-DETR Polyp Detection - Jetson Orin Nano Deployment Guide

Deploy the polyp detection pipeline on NVIDIA Jetson Orin Nano for edge inference.

---

## Table of Contents
1. [Hardware Requirements](#hardware-requirements)
2. [JetPack Setup](#jetpack-setup)
3. [Environment Setup](#environment-setup)
4. [Dependencies Installation](#dependencies-installation)
5. [Model Optimization with TensorRT](#model-optimization-with-tensorrt)
6. [Code Modifications for Jetson](#code-modifications-for-jetson)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting](#troubleshooting)

---

## Hardware Requirements

### Jetson Orin Nano Specs
| Spec | Value |
|------|-------|
| GPU | 1024-core NVIDIA Ampere with 32 Tensor Cores |
| CPU | 6-core Arm Cortex-A78AE |
| RAM | 8GB LPDDR5 |
| Storage | microSD or NVMe SSD (recommended) |
| Power | 7W - 15W modes |

### Recommended Accessories
- NVMe SSD (faster than microSD)
- USB camera or HDMI capture card for live inference
- Active cooling fan
- 5V 4A power supply (for 15W mode)

---

## JetPack Setup

### 1. Flash JetPack 6.0+ (Required for Orin Nano)

```bash
# Use NVIDIA SDK Manager on a Ubuntu host PC
# Or download pre-flashed SD card image from:
# https://developer.nvidia.com/embedded/jetpack

# Verify JetPack version after boot
cat /etc/nv_tegra_release
```

### 2. Update System

```bash
sudo apt update && sudo apt upgrade -y
sudo reboot
```

### 3. Set Power Mode (15W for best performance)

```bash
# Check current power mode
sudo nvpmodel -q

# Set to 15W mode (MAXN)
sudo nvpmodel -m 0

# Maximize clocks
sudo jetson_clocks
```

### 4. Verify CUDA Installation

```bash
# CUDA is pre-installed with JetPack
nvcc --version
# Should show CUDA 12.x

# Verify GPU
python3 -c "import torch; print(torch.cuda.is_available())"
```

---

## Environment Setup

### 1. Create Virtual Environment

```bash
# Install venv if not present
sudo apt install python3-venv python3-pip -y

# Create virtual environment
python3 -m venv ~/cade_env
source ~/cade_env/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 2. Create Project Directory

```bash
mkdir ~/CADe
cd ~/CADe
```

---

## Dependencies Installation

### 1. System Dependencies

```bash
sudo apt install -y \
    libopencv-dev \
    python3-opencv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libfontconfig1
```

### 2. Python Packages

```bash
# NumPy (use system numpy for compatibility)
pip install numpy

# OpenCV with CUDA support (use Jetson's pre-built)
# Don't pip install opencv-python, use system version
sudo apt install python3-opencv

# PyQt5 (PyQt6 not available on ARM64)
pip install PyQt5

# Note: You'll need to modify code to use PyQt5 instead of PyQt6
```

### 3. ONNX Runtime for Jetson (CRITICAL)

```bash
# DO NOT use pip install onnxruntime-gpu
# Use NVIDIA's Jetson-specific wheel

# For JetPack 6.0 / Python 3.10:
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# OR download from NVIDIA directly:
# https://elinux.org/Jetson_Zoo#ONNX_Runtime

# Verify
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Should show: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

### 4. DearPyGui for ARM64

```bash
# DearPyGui may need to be built from source on ARM64
# Try pip first:
pip install dearpygui

# If that fails, build from source:
sudo apt install cmake
git clone https://github.com/hoffstadt/DearPyGui.git
cd DearPyGui
pip install .
```

### Alternative: Use Tkinter if DearPyGui fails

```bash
# Tkinter is pre-installed with Python
# Would require rewriting the UI (simpler but less GPU-accelerated)
```

---

## Model Optimization with TensorRT

### Why TensorRT on Jetson?
- 2-5x faster inference than ONNX Runtime alone
- Optimized for Jetson's GPU architecture
- Lower latency, lower power consumption

### 1. Convert ONNX to TensorRT Engine

```bash
# Use trtexec (included with JetPack)
/usr/src/tensorrt/bin/trtexec \
    --onnx=checkpoint_best_total.onnx \
    --saveEngine=checkpoint_best_total.trt \
    --fp16 \
    --workspace=2048 \
    --verbose

# For INT8 quantization (even faster, slight accuracy loss):
/usr/src/tensorrt/bin/trtexec \
    --onnx=checkpoint_best_total.onnx \
    --saveEngine=checkpoint_best_total_int8.trt \
    --int8 \
    --workspace=2048
```

### 2. Use TensorRT Engine in Code

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

class TensorRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()
    
    def _allocate_buffers(self):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
        
        return inputs, outputs, bindings, stream
    
    def infer(self, input_data):
        # Copy input to device
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # Run inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # Copy output to host
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        self.stream.synchronize()
        
        return [out['host'] for out in self.outputs]
```

---

## Code Modifications for Jetson

### 1. Replace PyQt6 with PyQt5

```python
# Change this:
from PyQt6.QtWidgets import QApplication, QFileDialog

# To this:
from PyQt5.QtWidgets import QApplication, QFileDialog
```

### 2. Optimize Memory Usage (8GB RAM limit)

```python
# In InferenceEngine.__init__:
# Use smaller input buffer if needed
self.input_buffer = np.zeros((1, 3, 512, 512), dtype=np.float16)  # FP16

# Reduce display resolution
self.display_width = 960   # Instead of 1280
self.display_height = 540  # Instead of 720

# Reduce tracking history
self.history_size = 5  # Instead of 8
```

### 3. Add TensorRT Provider Priority

```python
def load(self) -> bool:
    providers = []
    
    # TensorRT first (best on Jetson)
    providers.append(('TensorrtExecutionProvider', {
        'device_id': 0,
        'trt_fp16_enable': True,
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': '/tmp/trt_cache'
    }))
    
    # CUDA fallback
    providers.append(('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kSameAsRequested',
    }))
    
    providers.append('CPUExecutionProvider')
    
    self.session = ort.InferenceSession(str(self.model_path), providers=providers)
```

### 4. Video Capture for USB Camera

```python
# For live camera instead of video files:
def open_camera(self, camera_id=0):
    # Use GStreamer pipeline for better performance on Jetson
    gst_pipeline = (
        f"v4l2src device=/dev/video{camera_id} ! "
        "video/x-raw, width=1280, height=720, framerate=30/1 ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink drop=1"
    )
    self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
```

---

## Performance Optimization

### 1. Set Jetson to Maximum Performance

```bash
# Create startup script
sudo nano /etc/rc.local

# Add these lines:
#!/bin/bash
nvpmodel -m 0
jetson_clocks
exit 0

# Make executable
sudo chmod +x /etc/rc.local
```

### 2. Monitor Performance

```bash
# Install jtop (like htop for Jetson)
sudo pip3 install jetson-stats
sudo systemctl restart jtop.service

# Run monitor
jtop
```

### 3. Expected Performance

| Configuration | FPS (512x512 input) |
|---------------|---------------------|
| ONNX + CUDA | ~15-20 FPS |
| ONNX + TensorRT FP16 | ~25-35 FPS |
| TensorRT Engine FP16 | ~35-45 FPS |
| TensorRT Engine INT8 | ~45-60 FPS |

### 4. Reduce Inference Latency

```python
# Skip frames if processing is slow
class App:
    def __init__(self):
        self.frame_skip = 2  # Process every 2nd frame
        self.frame_count = 0
    
    def update(self):
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0:
            return  # Skip this frame
        
        # Process frame...
```

---

## Complete requirements_jetson.txt

```txt
numpy>=1.24.0
# opencv-python  # Use system OpenCV instead
PyQt5>=5.15.0
dearpygui>=2.0.0
supervision>=0.19.0
# onnxruntime-gpu  # Use Jetson-specific wheel
pycuda>=2022.1
```

---

## Troubleshooting

### Issue: "No module named 'cv2'"
```bash
# Use system OpenCV
sudo apt install python3-opencv
# Add to virtual environment
echo "/usr/lib/python3/dist-packages" > ~/cade_env/lib/python3.10/site-packages/cv2.pth
```

### Issue: CUDA out of memory
```bash
# Reduce batch size to 1 (already set)
# Use FP16 instead of FP32
# Reduce display resolution
# Close other GPU applications
```

### Issue: Low FPS
```bash
# Verify power mode
sudo nvpmodel -q  # Should be MAXN (0)
sudo jetson_clocks

# Use TensorRT instead of ONNX
# Enable frame skipping
```

### Issue: DearPyGui crashes
```bash
# Try with software rendering
export LIBGL_ALWAYS_SOFTWARE=1
python main.py

# Or use alternative UI framework (Tkinter)
```

### Issue: USB camera not detected
```bash
# Check camera
ls /dev/video*

# Install v4l-utils
sudo apt install v4l-utils
v4l2-ctl --list-devices
```

---

## Quick Start Script

```bash
#!/bin/bash
# setup_jetson.sh

echo "Setting up CADe on Jetson Orin Nano..."

# Set performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Create environment
python3 -m venv ~/cade_env
source ~/cade_env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install numpy PyQt5 supervision dearpygui

# Link system OpenCV
echo "/usr/lib/python3/dist-packages" > ~/cade_env/lib/python3.10/site-packages/cv2.pth

# Install ONNX Runtime for Jetson
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

echo "Setup complete! Activate with: source ~/cade_env/bin/activate"
```

---

## File Structure on Jetson

```
~/CADe/
├── main.py                      # Modified for Jetson
├── checkpoint_best_total.onnx   # Original ONNX model
├── checkpoint_best_total.trt    # TensorRT engine (generated)
├── requirements_jetson.txt      # Jetson-specific deps
└── setup_jetson.sh             # Setup script
```

---

## Cursor AI Prompt for Jetson Deployment

If you need to modify the Windows code for Jetson, use this prompt:

```
Modify this Windows desktop application for Jetson Orin Nano deployment:

1. Replace PyQt6 with PyQt5 (ARM64 compatible)
2. Add GStreamer pipeline support for USB cameras
3. Optimize for 8GB RAM:
   - Use FP16 input buffers
   - Reduce display resolution to 960x540
   - Reduce tracking history_size to 5
4. Add TensorRT provider with caching:
   - trt_fp16_enable: True
   - trt_engine_cache_enable: True
   - trt_engine_cache_path: /tmp/trt_cache
5. Add frame skipping option for low FPS scenarios
6. Keep all tracking logic intact (IoU matching, NMS, smoothing)
```

---

*Jetson Orin Nano Deployment Guide - January 2026*

# RF-DETR Polyp Detection

A GPU-accelerated real-time polyp detection application using RF-DETR (Real-time Focused DETR) with ONNX Runtime and a modern DearPyGui interface.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Features

- 🎥 **Video Playback**: Load and play MP4, AVI, MKV, and MOV video files
- 🔍 **Real-time Detection**: GPU-accelerated polyp detection using RF-DETR
- ⚡ **Multi-Provider Support**: Automatic selection of TensorRT, CUDA, or CPU inference
- 🎛️ **Adjustable Threshold**: Fine-tune detection confidence in real-time
- 📊 **FPS Counter**: Monitor inference performance
- 🖥️ **Modern UI**: Clean interface built with DearPyGui

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support (recommended)
- Windows 10/11

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/CADeclaude.git
cd CADeclaude
```

2. Create a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the ONNX model:
   - Place `checkpoint_best_total.onnx` in the project root directory
   - Model specifications: RF-DETR Small, 512x512 input resolution, 2 classes (background + polyp)

## Usage

Run the application:
```bash
python main.py
```

### Controls

- **File → Open Video**: Load a video file for analysis
- **Play/Pause/Stop**: Control video playback
- **Record**: Record video clips with bounding boxes visible (saved as MP4)
- **Scrubber**: Seek to any position in the video
- **Confidence Threshold**: Adjust detection sensitivity (0.0 - 1.0)

### Recording Clips for Social Media

1. Load a video and navigate to the section you want to record
2. Click the **Record** button (turns red when recording)
3. Click **Play** to start playback while recording
4. Click **Stop Rec** when done
5. Your clip is saved to `Videos/CADe_Recordings/` folder as MP4

The recorded video includes:
- Original video resolution for best quality
- Green bounding boxes around detected polyps
- Confidence scores displayed
- Ready to upload to YouTube, Facebook, or other platforms

## Model Architecture

- **Model**: RF-DETR Small
- **Input Size**: 512 × 512 pixels
- **Classes**: 2 (background, polyp)
- **Output**: Bounding boxes with confidence scores

## Tech Stack

- **Deep Learning**: ONNX Runtime with CUDA/TensorRT acceleration
- **Computer Vision**: OpenCV
- **GUI Framework**: DearPyGui
- **File Dialogs**: PyQt6

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- RF-DETR architecture for real-time object detection
- Trained on colonoscopy polyp detection datasets

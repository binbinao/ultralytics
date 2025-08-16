# QWEN.md - Project Context for YOLO-Series-ONNXRuntime-Rust

## Project Overview

This repository is a Rust demo showcasing key Ultralytics YOLO series tasks (Classification, Segmentation, Detection, Pose Estimation, and Oriented Bounding Box detection) using ONNXRuntime. It supports various YOLO models (v5 through v11) and leverages the `usls` crate for efficient inference in Rust.

### Key Features:
- **Model Compatibility**: Supports YOLOv5 to YOLOv11, YOLO-World, and RT-DETR.
- **Task Coverage**: Includes Classification, Segmentation, Detection, Pose, and OBB tasks.
- **Precision Flexibility**: Works with FP16 and FP32 ONNX models.
- **Execution Providers**: Supports CPU, CUDA, CoreML, and TensorRT.
- **Dynamic Input Shapes**: Adjusts to variable batch, width, and height dimensions.
- **Data Loading**: Handles images, folders, videos, and real-time streams.
- **Visualization**: Provides real-time display and video export capabilities.

## Building and Running

### Setup Instructions:
1. **ONNXRuntime Linking**:
   - **Manual Linking**: Download the ONNX Runtime library and set `ORT_DYLIB_PATH`.
   - **Automatic Download**: Use `cargo run -r --example yolo --features auto`.

2. **Optional Dependencies**:
   - Install CUDA, CuDNN, and TensorRT for GPU acceleration.
   - Install `ffmpeg` for video support.

### Running Examples:
```bash
# Run a custom model (e.g., YOLOv8 detection)
cargo run -r -- --task detect --ver v8 --nc 6 --model path/to/your/model.onnx

# Classify examples
cargo run -r -- --task classify --ver v5 --scale s --width 224 --height 224 --nc 1000

# Detect examples
cargo run -r -- --task detect --ver v8 --scale n

# Pose examples
cargo run -r -- --task pose --ver v8 --scale n

# Segment examples
cargo run -r -- --task segment --ver v8 --scale n

# OBB examples
cargo run -r -- --ver v8 --task obb --scale n --width 1024 --height 1024 --source images/dota.png
```

Use `cargo run -- --help` for all available options.

## Development Conventions

### Code Structure:
- **`Cargo.toml`**: Defines project metadata and dependencies.
- **`src/main.rs`**: Entry point for the application, handling CLI arguments and model execution.
- **`README.md`**: Comprehensive documentation for setup, usage, and features.

### Key Dependencies:
- `usls`: For YOLO model inference.
- `clap`: For CLI argument parsing.
- `tracing`: For performance profiling.

### Testing:
- The project does not include explicit test files. Testing is primarily done by running the examples with various configurations.

### Contribution Guidelines:
- Contributions are welcome. Follow the [Ultralytics Contribution Guide](https://docs.ultralytics.com/help/contributing/).

## Usage Notes
- Ensure the ONNX Runtime library is correctly linked.
- For GPU acceleration, install CUDA and TensorRT as needed.
- Use `ffmpeg` for video-related functionalities.

---

This file provides a concise yet comprehensive overview of the project for future interactions. Let me know if you'd like any modifications or additional details!
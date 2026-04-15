# Edge AI-based Face Detection and Robotic Arm Control System


## Competition
> Andes Awards 2024 — Application Track  
- [Final Slides](https://docs.google.com/presentation/d/1jy3ECFNxDfVmHjAojg8dyEjT1ntfuXvriRUWY-4u-hQ/edit?usp=sharing)

---

## Overview

This project deploys a Face Detection Deep Neural Network (DNN) model on a RISC-V edge device (Tinker V). The system captures real-time images via camera or image file, runs on-device inference, and sends control signals to a robotic arm — achieving real-time Human-Computer Interaction (HCI) without relying on cloud computing.

---

## Hardware Specifications

| Item | Specification |
|------|--------------|
| Platform | Tinker V (Single-Board Computer) |
| CPU | RISC-V AndesCore™ AX45MP Single Core @ 1.0 GHz |
| ISA Extension | Packed SIMD (RVP) |
| RAM | 1 GB |
| Storage | 16 GB eMMC + microSD slot |
| Camera Interface | micro USB (external USB camera) |
| Control Interface | GPIO 20-pin (SPI / UART / I²C) |
| Network Interface | RJ45 (wired Ethernet) |
| OS | Linux (Yocto image) |
| Robotic Arm | Taiwan IoT 6-DOF Robotic Arm |

---

## Model Selection

### UltraFace (Ultra-Light-Fast-Generic-Face-Detector)

PoseNet was originally planned but required ~5 minutes per inference on Tinker V's limited CPU, making it impractical. After testing several lightweight alternatives, UltraFace was selected.

**Model benchmarks on Tinker V:**

| Model | Size (int8) | Inference Time (s) |
|-------|-------------|-------------------|
| yolov8n-pose_int8 | 3.22 MB | 243.6 |
| version-RFB-320_float32 (UltraFace) | 1.13 MB | 33.00 |
| version-slim-320_without_postprocessing_float32 (UltraFace) | 1.00 MB | 27.9 |
| **version-slim-320_without_postprocessing_int8 (UltraFace)** ✅ | **372 KB** | **26.7** |
| semgcn_int8 | 300 KB | 0.335 |

> **Final model**: `version-slim-320_without_postprocessing_int8.tflite`  
> **Input**: 240×320×3 normalized image  
> **Output**: 4420 bounding boxes + confidence scores (4420×4 offsets, 4420×2 scores)


---

## System Pipeline

```
Real World
    │
    ▼
USB Camera / Image File
    │  (micro USB / direct file read)
    ▼
Tinker V
    ├── Preprocessing
    │       OpenCV resize → cvtColor (BGR→RGB) → normalize → int8 quantization
    │       generate_anchors() → 4420 anchor boxes
    │
    ├── UltraFace TFLite Inference via TFLM
    │       ~26.7 seconds / frame → 1.5s (~18× speedup) w/ -O3 and RVP
    │
    ├── Postprocessing
    │       Dequantization → decode_regression() → confidence threshold
    │       → nms_boxes() (NMS) → cv::rectangle() → output_frame_bbox.png
    │
    └── GPIO Signal Output → Robotic Arm 
            box_count == 0  →  signal 1
            box_count  > 0  →  signal 0
```

---

## Setup & Build

### 1. Rebuild Yocto Image with Required Packages

Edit `./meta-asus-renesas/conf/machine/rzfive-tinker-v.conf` and append the following to `CORE_IMAGE_EXTRA_INSTALL`:

```
opencv
libopencv-core-dev
libopencv-imgproc-dev
libopencv-imgcodecs-dev
libopencv-videoio-dev
libopencv-objdetect-dev
libopencv-ml-dev
cmake
ffmpeg
python3
python3-pip
curl
openssh
vim
kernel-modules
packagegroup-core-buildessential
libgpiod
libgpiod-dev
libgpiod-tools
```

Then build the image:

```bash
./build.sh
```

### 2. Model Quantization & Conversion

```bash
# Convert ONNX (float32) → TFLite (int8) using onnx2tf
onnx2tf -i version-slim-320_without_postprocessing.onnx -oiqt

# Convert TFLite model → C raw buffer for embedding
xxd -i version-slim-320_without_postprocessing_int8.tflite > model.cc
(echo "#include \"model.h\""; echo "alignas(8)"; cat model.cc) > model.cpp
sed -i -E 's/(unsigned\s.*\s).*(_len|\[\])/const \1g_model\2/g' model.cpp
```

### 3. Build TFLM Library on Tinker V

This project uses a forked version of [generic-tflmicro](https://github.com/iwatake2222/generic-tflmicro), updated to the latest TFLM (September 2024).

> ⚠️ The latest TFLM requires **C++17**. Add the following to `CMakeLists.txt`:
> ```cmake
> set(CXXFLAGS "-std=c++17")
> ```

```bash
mkdir build && cd build
cmake ..
make -j4
# First build takes approximately 18 hours on Tinker V
```

After the first build, only `main_functions.cpp` needs to be modified for subsequent rebuilds — the TFLM static library does not need to be recompiled.

### 4. Verify Camera Connection

```bash
# Check if camera is detected
lsusb
ls /dev/video*

# List supported formats and resolutions
v4l2-ctl --device /dev/video0 --list-formats-ext

# Capture a single frame
v4l2-ctl --device /dev/video0 \
  --set-fmt-video=width=640,height=480,pixelformat=YUYV \
  --stream-mmap=3 --stream-to=frame.yuyv --stream-count=1

# Convert to PNG
ffmpeg -f rawvideo -pix_fmt yuyv422 -s 640x480 \
  -i frame.yuyv -f image2 -vframes 1 output.png
```

> Alternatively, the project integrates `cv::VideoCapture` from OpenCV to handle camera capture directly within the main program, eliminating the need for separate terminal commands.

---

## Usage

```bash
# Run inference on an image file
./tflm_project/tflm <image_path>

# Capture one frame from camera and run inference
./tflm_project/tflm
```

---

## Implementation Details

### Preprocessing

1. `cv::resize()` — scale image to 240×320
2. `cv::cvtColor()` — convert BGR to RGB
3. `cv::normalize()` + `cv::convertTo()` — normalize to float32 in [-1.0, 1.0]
4. `generate_anchors()` — generate 4420 anchor boxes covering the entire image
5. **int8 Quantization**:
   ```
   quantized_int8 = (float32_value / input_scale) + input_zero_point
   ```

### Inference (TFLM API)

1. Register required operators via `tflite::MicroMutableOpResolver`
2. Load model with `tflite::GetModel()`
3. Configure Arena memory size
4. Initialize `tflite::MicroInterpreter` with model, arena, and op resolver
5. Copy quantized input array to TFLM tensor via `memcpy()`
6. Run inference: `tflite::MicroInterpreter::invoke()`

### Postprocessing

1. **Dequantization**:
   ```
   float32_value = (int8_value - output_zero_point) × output_scale
   ```
2. `decode_regression()` — convert bounding box offsets to absolute coordinates
3. Filter boxes by confidence threshold
4. `nms_boxes()` — custom C++ implementation of Non-Maximum Suppression (NMS)
   - Sort by confidence score (descending)
   - Compute IoU via `get_iou_value()`
   - Remove overlapping boxes exceeding the IoU threshold
5. `cv::rectangle()` — draw bounding boxes and save `output_frame_bbox.png`

### GPIO Signal Output

| Condition | GPIO Signal |
|-----------|-----------|
| `box_count == 0` (no face detected) | `1` |
| `box_count > 0` (face detected) | `0` |

### Optimization with RVP

- See [muriscvnn-rvp-conv](https://github.com/pepe996/generic-tflmicro/tree/muriscvnn-rvp-conv) branch

---

## Results

| Metric | Result |
|--------|--------|
| Test dataset | WIDER FACE face detection dataset |
| Inference time | ~27.88 seconds / frame (1.5 seconds (~18× speedup) w/ -O3 and RVP)|
| Model size | 372 KB (int8) |
| Arena memory used | 1,289,256 bytes |
| Bounding boxes output (group photo test) | 26 boxes detected |
| GPIO signal output | Verified (Pin P15_0) |

![image](https://hackmd.io/_uploads/B1Ipft6R0.png)

![image](https://hackmd.io/_uploads/rkuAMFpA0.png)


Both image file input and live camera capture modes successfully detect faces and draw bounding boxes.

---

## References

| # | Link |
|---|------|
| [1] | [Tinker V Wiki](https://github.com/TinkerBoard/TinkerBoard/wiki/Tinker-V) |
| [2] | [UltraFace GitHub](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) |
| [3] | [TFLite Pose Estimation](https://github.com/joonb14/TFLitePoseEstimation) |
| [4] | [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers) |
| [5] | [YOLOv8-pose](https://github.com/ultralytics/ultralytics/issues/1915) |
| [6] | [SemGCN](https://github.com/garyzhao/SemGCN) |
| [7] | [Taiwan IoT 6-DOF Robotic Arm](https://www.taiwaniot.com.tw/product/6%E8%BB%B8%E6%A9%9F%E6%A2%B0%E6%89%8B%E8%87%82-%E9%96%8B%E7%99%BC%E5%A5%97%E4%BB%B6%E7%B5%84-6dof-robot-arm/) |
| [8] | [Tinker V USB Camera Fix](https://github.com/TinkerBoard/TinkerBoard/issues/7) |
| [9] | [ONNX](https://onnx.ai/) |
| [10] | [onnx2tf](https://github.com/PINTO0309/onnx2tf) |
| [11] | [generic-tflmicro](https://github.com/iwatake2222/generic-tflmicro) |
| [12] | [TFLM New Platform Support](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/docs/new_platform_support.md) |
| [13] | [Forked generic-tflmicro (updated TFLM)](https://github.com/pepe996/generic-tflmicro) |
| [14] | [NMS Algorithm Explained](https://chih-sheng-huang821.medium.com/%E6%A9%9F%E5%99%A8-%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92-%E7%89%A9%E4%BB%B6%E5%81%B5%E6%B8%AC-non-maximum-suppression-nms-aa70c45adffa) |
| [15] | [WIDER FACE Dataset](http://shuoyang1213.me/WIDERFACE/) |
| [16] | [muriscv-nn](https://github.com/tum-ei-eda/muriscv-nn) |
| [17] | [Andes Development Kit](https://github.com/andestech/Andes-Development-Kit/releases) |
| [18] | [Tinker V Static Linking Solution](https://github.com/TinkerBoard/TinkerBoard/issues/15#issuecomment-2320946218) |


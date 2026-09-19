# Native C++ TensorRT deployment

This page documents the PerceptionRT native C++ TensorRT FP16 deployment pipeline. The v0.8.0 release extends the validated v0.7.0 raw-tensor runtime with direct image preprocessing and native prediction postprocessing.

## Scope

The native deployment stage adds a standalone C++17 executable that:

- decodes images with OpenCV and applies deterministic 320 × 640 center-crop preprocessing;
- performs BGR-to-RGB conversion, ImageNet normalization and HWC-to-CHW FP16 conversion;
- deserializes and validates the TensorRT FP16 engine;
- executes inference using reusable CUDA buffers and a dedicated stream;
- writes the three raw FP16 outputs;
- performs native semantic, metric-depth and uncertainty postprocessing;
- writes semantic, depth and uncertainty PNG visualizations.

The executable does not import Python, PyTorch, NumPy or ONNX Runtime. Python
is used only by the validation and benchmark orchestration tools.

## Validated environment

| Component | Version or configuration |
|---|---|
| Operating system | Ubuntu 24.04 x86_64 |
| Compiler | GCC 13.3.0 |
| CMake | 3.28.3 |
| C++ standard | C++17 |
| OpenCV | 4.6.0 |
| TensorRT | 11.2.1.2 |
| CUDA runtime | 13.0 |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU |
| Compute capability | 8.6 |
| GPU memory | 6 GB |

TensorRT engines are tied to their TensorRT version, GPU architecture and
build environment. The generated engine is intentionally excluded from Git.

## Native dependencies

The Python TensorRT wheel contains runtime libraries but does not provide the
complete C++ development headers. The native build uses these pinned NVIDIA
packages:

| Package | Version | Purpose |
|---|---|---|
| `libnvinfer-headers-dev` | 11.2.1.2-1+cuda13.3 | TensorRT C++ headers |
| `cuda-cudart-dev-13-0` | 13.0.88-1 | CUDA Runtime API headers |
| `cuda-crt-13-0` | 13.0.88-1 | CUDA internal runtime headers |

`scripts/build_native_cpp.sh` downloads and extracts only these header packages
under `.venv/`. No root access, system package installation, complete CUDA
toolkit or `nvcc` compiler is required. The executable links against the
TensorRT and CUDA shared libraries already installed by the pinned Python
deployment extras.

This bootstrap path currently targets Ubuntu 24.04 x86_64. Other Linux
distributions can use the same CMake project by supplying compatible header and
library paths directly.

## Build

Install the existing deployment dependencies:

```bash
python -m pip install -e ".[dev,export,tensorrt]"
```

Configure, compile and run CTest:

```bash
bash scripts/build_native_cpp.sh
```

The generated executable is:

```text
build/native/perception_rt_native
```

Build files are excluded from Git.

## Engine contract

The executable accepts only the validated static FP16 contract:

| Tensor | Mode | Shape | Type |
|---|---|---|---|
| `image` | Input | `1 × 3 × 320 × 640` | FP16 |
| `semantic_logits` | Output | `1 × 15 × 320 × 640` | FP16 |
| `log_depth` | Output | `1 × 1 × 320 × 640` | FP16 |
| `depth_log_scale` | Output | `1 × 1 × 320 × 640` | FP16 |

Startup fails if a tensor name, order, mode, shape or type differs. This avoids
silently executing an incompatible engine.

## Image and tensor input

The runtime supports two mutually exclusive input modes.

Direct image inference:

```bash
build/native/perception_rt_native \\
    --engine outputs/tensorrt/perception_rt_mit_b2_fp16.engine \\
    --image datasets/vkitti2/raw/Scene02/fog/frames/rgb/Camera_0/rgb_00134.jpg \\
    --output-dir outputs/native_cpp/predictions
```

The image path performs OpenCV decode, a deterministic `320 × 640` center crop, BGR-to-RGB conversion, scaling by `1/255`, ImageNet normalization, HWC-to-CHW conversion and FP32-to-FP16 conversion. On the deterministic development image all `614,400` FP16 input values matched the Python reference exactly.

The original raw input path remains available:

```bash
build/native/perception_rt_native \\
    --engine outputs/tensorrt/perception_rt_mit_b2_fp16.engine \\
    --input outputs/native_cpp/image.fp16.bin \\
    --output-dir outputs/native_cpp/predictions
```

Raw input must contain exactly `1,228,800` bytes. `--image` and `--input` cannot be used together.

The runtime always writes the raw FP16 tensors `semantic_logits.fp16.bin`, `log_depth.fp16.bin` and `depth_log_scale.fp16.bin`. With `--image` it additionally writes `semantic.png`, `depth.png` and `uncertainty.png`.

Semantic output uses argmax over 15 classes. Metric depth is decoded as `clamp(exp(log_depth), 0.001, 200.0)`. Uncertainty is decoded as `exp(clamp(depth_log_scale, -6, 6))` and represents a learned scale in log-depth space, not direct ± metres uncertainty.

The integrated `--image` path produced bit-for-bit identical TensorRT outputs to the validated raw `--input` path, and the integrated PNG visualizations matched the separately validated development probes bit-for-bit.

## Native parity validation

Run:

```bash
python -m perception_rt.validate_native_tensorrt
```

The validator uses the same deterministic held-out Scene18 indices as the ONNX
and TensorRT deployment stages. For each sample it converts the normalized input
to contiguous FP16, runs the same engine through both the Python TensorRT runtime
and the standalone C++ executable, then compares every raw output.

| Index | Variation | Frame | Semantic argmax agreement |
|---:|---|---:|---:|
| 0 | `15-deg-left` | 0 | 1.00000000 |
| 847 | `30-deg-left` | 169 | 1.00000000 |
| 1695 | `fog` | 0 | 1.00000000 |
| 2542 | `overcast` | 169 | 1.00000000 |
| 3389 | `sunset` | 338 | 1.00000000 |

| Output | Maximum absolute error | Maximum mean absolute error | Minimum exact fraction |
|---|---:|---:|---:|
| `semantic_logits` | 0 | 0 | 100% |
| `log_depth` | 0 | 0 | 100% |
| `depth_log_scale` | 0 | 0 | 100% |

Native C++–Python TensorRT FP16 parity: **passed**.

This test isolates language-binding and buffer-management correctness. The
earlier v0.5.0 validation separately compares the FP16 TensorRT engine against
the FP32 PyTorch checkpoint.

## Native benchmark

Run:

```bash
python -m perception_rt.benchmark_native_tensorrt
```

Protocol:

- deterministic Scene18 sample index 0;
- static batch size one;
- 30 warmup iterations;
- 100 measured iterations;
- dedicated non-default CUDA stream;
- synchronization after every measured inference;
- input and output buffers remain device-resident during timing;
- engine loading and tensor transfers are excluded.

| Backend | Mean latency | P95 latency | Throughput |
|---|---:|---:|---:|
| Native C++ TensorRT FP16 | 6.603 ms | 6.646 ms | 151.45 FPS |

The measurement represents synchronous model execution, not complete camera-to-
prediction application latency. It is hardware- and power-state-specific.

## Verification

The release candidate passed:

- 128 Python tests;
- two CTests for the native command-line interface;
- native engine deserialization and four-tensor contract validation;
- finite-output smoke inference;
- five-sample bit-exact native parity;
- a 30-warmup, 100-iteration native benchmark.

## Limitations

- Only the static batch-one FP16 engine is supported by the C++ executable.
- Direct image preprocessing uses a fixed `320 × 640` center crop; smaller images are rejected rather than resized.
- Depth and uncertainty PNGs are visualization products; raw tensors remain the numerical outputs.
- OpenCV C++ development files are required.
- TensorRT engines must be rebuilt for the target environment.
- The measured result is specific to the RTX 3060 Laptop GPU and its runtime
  state.
- The build bootstrap currently targets Ubuntu 24.04 x86_64.
- No custom CUDA kernels or `nvcc` compilation are included yet.
- KITTI-360 transfer evaluation and ROS 2 integration remain future stages.

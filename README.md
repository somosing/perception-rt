# PerceptionRT

Reliability-aware 3D vision for autonomous robots.

PerceptionRT is a deployment-oriented computer-vision project combining:

- Semantic segmentation
- Monocular metric-depth estimation
- Learned depth uncertainty
- Reliability-aware semantic bird's-eye-view occupancy
- PyTorch training
- ONNX and TensorRT inference
- Native C++ and CUDA processing
- ROS 2 integration
- Reproducible testing and benchmarking

## Objective

The project investigates whether a monocular perception network can jointly predict
semantics, metric depth, and a useful estimate of its own depth error—and whether the
complete pipeline can be deployed efficiently on NVIDIA hardware.

## Current status

### v0.6.0 release candidate — Selective TensorRT INT8 evaluation

- [x] Leakage-safe Scene06 post-training calibration
- [x] Explicit Q/DQ ONNX quantization
- [x] Accuracy-guided selection of 13 encoder spatial-reduction convolutions
- [x] Strongly typed TensorRT engine with FP32 external I/O
- [x] Frozen parity gates evaluated once on held-out Scene18 samples
- [x] Reproducible five-backend benchmark
- [x] 116 passing tests

The selective INT8 engine passed the frozen held-out gates with `0.99957031`
minimum semantic agreement and at least `99.4072%` numerical-tolerance
coverage for every output. It achieved `15.841 ms` mean latency and
`63.13 FPS` on the RTX 3060 Laptop GPU: `1.046×` faster than TensorRT FP32,
but substantially slower than TensorRT FP16. FP16 therefore remains the
recommended deployment configuration. See
[`docs/tensorrt_int8.md`](docs/tensorrt_int8.md) for the selection study,
protocol, results and limitations.

### v0.5.0 — TensorRT FP16 optimization

- [x] Native FP16 ONNX export from the epoch-16 checkpoint
- [x] Strongly typed static FP16 TensorRT engine
- [x] Precision-aware native CUDA runtime
- [x] Strict numerical tolerances with an explicit 99% pixel-coverage gate
- [x] Held-out Scene18 FP32 PyTorch–FP16 TensorRT parity
- [x] Reproducible four-backend FP32/FP16 benchmark
- [x] 107 passing tests

The FP16 engine passed parity on five deterministic held-out Scene18
samples with `0.99965332` minimum semantic agreement and at least `99.2310%`
numerical-tolerance coverage for every output. On the validated RTX 3060
Laptop GPU it achieved `6.846 ms` mean latency and `146.06 FPS`: `2.462×`
faster than TensorRT FP32. See
[`docs/tensorrt_fp16.md`](docs/tensorrt_fp16.md) for the complete protocol,
results and limitations.

### v0.4.0 — TensorRT FP32 deployment baseline

- [x] Strongly typed static FP32 TensorRT engine builder
- [x] Validated four-tensor deployment contract
- [x] Native TensorRT CUDA inference
- [x] Reusable device-resident output buffers
- [x] Dedicated non-default CUDA stream
- [x] Held-out Scene18 PyTorch–TensorRT parity validation
- [x] Reproducible PyTorch, ONNX Runtime and TensorRT benchmark
- [x] 92 passing tests

The TensorRT engine passed direct PyTorch parity on five deterministic
held-out Scene18 samples. On an RTX 3060 Laptop GPU it achieved `16.579 ms`
mean latency and `60.32 FPS`, a `1.404×` speedup over PyTorch and `1.603×`
over ONNX Runtime CUDA. See
[`docs/tensorrt_inference.md`](docs/tensorrt_inference.md) for the complete
protocol, results and limitations.

### v0.3.0 — ONNX deployment baseline

- [x] Static batch-one FP32 ONNX export
- [x] Stable three-output deployment contract
- [x] Export from the validation-selected epoch-16 checkpoint
- [x] ONNX graph validation
- [x] ONNX Runtime CUDA inference
- [x] Held-out Scene18 numerical parity validation
- [x] Absolute, relative and semantic-agreement parity gates
- [x] 62 passing tests

The exported model passed PyTorch–ONNX Runtime CUDA parity on five
deterministic held-out Scene18 samples. See
[`docs/onnx_export.md`](docs/onnx_export.md) for the complete contract,
results, tolerances and limitations.

### v0.2.0 — Multitask perception baseline

- [x] Training-ready PyTorch dataset
- [x] Shared MiT-B2 encoder with semantic, metric-depth and uncertainty decoders
- [x] Class-weighted semantic and uncertainty-aware depth losses
- [x] Mixed-precision training with gradient accumulation and checkpointing
- [x] Validation-based best-checkpoint selection
- [x] Held-out Scene18 evaluation
- [x] Qualitative prediction visualization
- [x] 52 passing tests

The official `best.pt` checkpoint was selected on Scene06 validation data and
evaluated on the untouched Scene18 test split. See
[`docs/vkitti2_results.md`](docs/vkitti2_results.md) for the complete protocol,
results and limitations.

## Dataset strategy

Initial development uses Virtual KITTI 2 because it provides synchronized RGB images,
dense metric depth, semantic labels, and camera information.

Real-world evaluation is planned with KITTI-360 to measure synthetic-to-real transfer
on German driving scenes.

Dataset files are not distributed with this repository. Download and use them under
their respective licences.

## Virtual KITTI 2 setup

Download the following Virtual KITTI 2 archives from the official dataset page:

- `vkitti_2.0.3_rgb.tar`
- `vkitti_2.0.3_depth.tar`
- `vkitti_2.0.3_classSegmentation.tar`
- `vkitti_2.0.3_textgt.tar.gz`

Extract them so that the dataset root has the following structure:

```text
datasets/vkitti2/raw/
├── Scene01/
├── Scene02/
├── Scene06/
├── Scene18/
└── Scene20/

```

Each scene contains variations such as `clone`, `fog`, `morning`, `rain`,
`sunset`, and viewpoint changes. RGB, depth and semantic files are aligned using
scene, variation, camera and frame identity.

The repository intentionally ignores `datasets/` and `outputs/`.

## Installation

Create and activate a Python 3.12 virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Verify the installation:

```bash
ruff check .
pytest
```

## Dataset visualization

Generate a visualization containing RGB, metric depth, depth validity and semantic
ground truth:

```bash
python -m perception_rt.visualize \
    --index 0 \
    --output outputs/vkitti2_sample_00000.png
```

The 99th-percentile depth limit affects only the visualization. It does not modify
the decoded metric-depth target.

## Dataset audit

Run a small smoke audit first:

```bash
python -m perception_rt.audit \
    --limit 100 \
    --progress-every 20 \
    --output outputs/vkitti2_audit_smoke.json
```

Run the complete audit:

```bash
python -m perception_rt.audit \
    --progress-every 500 \
    --output outputs/vkitti2_audit_full.json
```

The complete audit validated all 21,260 left-camera samples:

- Resolution: `375 × 1242`
- Mean valid-depth fraction: `84.32%`
- Valid-depth fraction range: `64.15%–99.21%`
- Observed valid-depth range: `1.08–655.34 m`
- Unknown semantic colours: none
- Spatial shape mismatches: none
- Unreadable aligned samples: none

Generated audit reports remain local and are not committed to Git.

## Leakage-safe splits

Splits are separated by physical scene:

| Split | Scenes | Samples |
|---|---|---:|
| Training | Scene01, Scene02, Scene20 | 15,170 |
| Validation | Scene06 | 2,700 |
| Test | Scene18 | 3,390 |

All variations of an underlying scene and frame remain in the same split. This
prevents synthetic weather or viewpoint variants from leaking between training and
evaluation.
## Multitask baseline

PerceptionRT uses an ImageNet-pretrained MiT-B2 encoder with separate multi-scale
decoders for 15-class semantic segmentation, metric log-depth and learned
log-depth uncertainty.

The validation-selected `best.pt` checkpoint from epoch 16 produced the following
results on all 3,390 held-out Scene18 samples:

| Metric | Result |
|---|---:|
| Semantic mIoU | 0.7526 |
| Depth AbsRel | 0.1876 |
| Depth RMSE | 17.51 m |
| Depth delta1 | 0.6604 |
| Uncertainty/error Pearson correlation | 0.2501 |

Train the baseline:

    python -m perception_rt.train \
        --config configs/train_vkitti2.yaml

Evaluate the official checkpoint:

    python -m perception_rt.evaluate \
        --config configs/train_vkitti2.yaml \
        --checkpoints outputs/training/vkitti2_multitask/best.pt \
        --output outputs/evaluation/vkitti2_test.json

Generate held-out qualitative predictions:

    python -m perception_rt.qualitative \
        --config configs/train_vkitti2.yaml \
        --checkpoint outputs/training/vkitti2_multitask/best.pt

Detailed methodology, checkpoint-selection rules and limitations are documented
in [`docs/vkitti2_results.md`](docs/vkitti2_results.md).

## ONNX export and parity validation

Install the pinned export dependencies:

~~~bash
python -m pip install -e ".[dev,export]"
~~~

Export the official checkpoint:

~~~bash
python -m perception_rt.export_onnx
~~~

Run FP32 parity validation on five deterministic held-out Scene18 samples:

~~~bash
python -m perception_rt.validate_onnx
~~~

The validator compares PyTorch CUDA and ONNX Runtime CUDA using identical
normalized inputs.

| Validation item | Result |
|---|---:|
| Samples | 5 |
| Minimum semantic argmax agreement | 0.99993164 |
| Maximum semantic-logit absolute error | 0.0178061 |
| Maximum log-depth absolute error | 0.0134659 |
| Maximum post-processed depth absolute error | 0.100380 m |
| Overall parity | Passed |

The complete contract, measurements, thresholds and limitations are documented
in [`docs/onnx_export.md`](docs/onnx_export.md).

## TensorRT FP32 deployment

Install the pinned deployment dependencies:

~~~bash
python -m pip install -e ".[dev,export,tensorrt]"
~~~

Build the static FP32 engine from the validated ONNX model:

~~~bash
python -m perception_rt.build_tensorrt
~~~

Run held-out PyTorch–TensorRT parity validation:

~~~bash
python -m perception_rt.validate_tensorrt
~~~

Run the three-backend device-resident benchmark:

~~~bash
python -m perception_rt.benchmark_inference
~~~

The generated engine is approximately `134.60 MiB`. It uses a static
batch-one `1 × 3 × 320 × 640` FP32 input and produces semantic logits,
metric log-depth and uncertainty log-scale.

| Backend | Mean latency | P95 latency | Throughput |
|---|---:|---:|---:|
| PyTorch FP32 | 23.279 ms | 23.735 ms | 42.96 FPS |
| ONNX Runtime CUDA FP32 | 26.575 ms | 26.967 ms | 37.63 FPS |
| TensorRT FP32 | 16.579 ms | 17.087 ms | 60.32 FPS |

TensorRT speedup was `1.404×` over PyTorch and `1.603×` over ONNX Runtime
CUDA. Model loading, preprocessing and host-device transfer were excluded.

Detailed engine construction, parity criteria, benchmark methodology and
hardware limitations are documented in
[`docs/tensorrt_inference.md`](docs/tensorrt_inference.md).

## TensorRT FP16 optimization

Export the checkpoint as an explicit FP16 ONNX graph:

~~~bash
python -m perception_rt.export_onnx --precision fp16
~~~

Build the strongly typed FP16 TensorRT engine:

~~~bash
python -m perception_rt.build_tensorrt --precision fp16
~~~

Validate the FP16 engine against the FP32 PyTorch checkpoint:

~~~bash
python -m perception_rt.validate_tensorrt --precision fp16
~~~

Benchmark all current inference backends:

~~~bash
python -m perception_rt.benchmark_inference
~~~

| Backend | Mean latency | P95 latency | Throughput |
|---|---:|---:|---:|
| PyTorch FP32 | 23.659 ms | 24.230 ms | 42.27 FPS |
| ONNX Runtime CUDA FP32 | 26.879 ms | 27.293 ms | 37.20 FPS |
| TensorRT FP32 | 16.859 ms | 17.305 ms | 59.32 FPS |
| TensorRT FP16 | 6.846 ms | 6.918 ms | 146.06 FPS |

TensorRT FP16 was `2.462×` faster than TensorRT FP32, `3.456×` faster
than PyTorch FP32 and `3.926×` faster than ONNX Runtime CUDA FP32.
The FP16 ONNX model is `52.71 MiB`; the generated engine is `85.00 MiB`.

Detailed precision design, parity criteria, benchmark methodology and
limitations are documented in
[`docs/tensorrt_fp16.md`](docs/tensorrt_fp16.md).

## Selective TensorRT INT8 evaluation

Create the calibrated Q/DQ ONNX model using only Scene06 validation data:

~~~bash
python -m perception_rt.quantize_onnx
~~~

Build and validate the selective INT8 TensorRT engine:

~~~bash
python -m perception_rt.build_tensorrt --precision int8
python -m perception_rt.validate_tensorrt --precision int8
~~~

Run the five-backend device-resident benchmark:

~~~bash
python -m perception_rt.benchmark_inference
~~~

Only the 13 encoder attention spatial-reduction convolutions are quantized.
The engine retains FP32 input and output tensors; explicit Q/DQ nodes control
internal INT8 execution. The INT8 ONNX model is `88.50 MiB`, and the generated
engine is `122.69 MiB`.

| Backend | Mean latency | P95 latency | Throughput |
|---|---:|---:|---:|
| PyTorch FP32 | 23.661 ms | 24.380 ms | 42.26 FPS |
| ONNX Runtime CUDA FP32 | 26.577 ms | 27.070 ms | 37.63 FPS |
| TensorRT FP32 | 16.571 ms | 16.944 ms | 60.35 FPS |
| TensorRT FP16 | 6.727 ms | 6.819 ms | 148.66 FPS |
| TensorRT selective INT8 | 15.841 ms | 16.045 ms | 63.13 FPS |

Selective INT8 was `1.046×` faster than TensorRT FP32 but approximately
`2.354×` slower than TensorRT FP16. This result is retained because it
demonstrates leakage-safe calibration, sensitivity analysis, explicit-Q/DQ
deployment and evidence-based rejection of a weaker optimization path.

Detailed calibration, selection, parity, benchmark methodology and limitations
are documented in [`docs/tensorrt_int8.md`](docs/tensorrt_int8.md).

## Development environment

The initial implementation was developed and validated with:

- Ubuntu 24.04
- NVIDIA GeForce RTX 3060 Laptop GPU
- 6 GB VRAM
- Python 3.12
- CUDA-enabled PyTorch
- ROS 2 Jazzy

## Dataset licences

Virtual KITTI 2 is owned by Naver Corporation and is available for non-commercial use
under the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 licence:

- https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/

KITTI-360 is distributed under the Creative Commons
Attribution-NonCommercial-ShareAlike 3.0 licence and requires registration:

- https://www.cvlibs.net/datasets/kitti-360/

Users are responsible for reviewing and complying with the official dataset terms.
This repository does not redistribute either dataset.

## Source-code licence

PerceptionRT source code is licensed under the Apache License 2.0. See
[`LICENSE`](LICENSE).
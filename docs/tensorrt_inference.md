# TensorRT FP32 Inference and Benchmarking

This page documents the PerceptionRT v0.4.0 TensorRT FP32 deployment
baseline, direct parity against the original PyTorch checkpoint, and
device-resident inference performance.

## Scope

The goal is to establish a validated native TensorRT runtime before adding
reduced-precision optimization. This stage covers:

- Static FP32 TensorRT engine construction
- Native CUDA inference through the TensorRT Python API
- Direct PyTorch–TensorRT numerical parity
- Reproducible device-resident latency and throughput measurement

This backend-equivalence evaluation does not replace the model-accuracy
results in [`vkitti2_results.md`](vkitti2_results.md).

## Validated environment

- Ubuntu 24.04
- NVIDIA GeForce RTX 3060 Laptop GPU
- Compute capability 8.6
- 6 GB VRAM
- NVIDIA driver 595.84
- Python 3.12
- PyTorch 2.13.0 with CUDA 13.0
- ONNX Runtime GPU 1.29.0
- TensorRT 11.2.1.2

The pinned TensorRT package is `tensorrt-cu13==11.2.1.2`.

## Deployment contract

The TensorRT engine retains the validated ONNX contract.

| Tensor | Mode | Shape | Type |
|---|---|---:|---|
| `image` | Input | `1 × 3 × 320 × 640` | FP32 |
| `semantic_logits` | Output | `1 × 15 × 320 × 640` | FP32 |
| `log_depth` | Output | `1 × 1 × 320 × 640` | FP32 |
| `depth_log_scale` | Output | `1 × 1 × 320 × 640` | FP32 |

The engine is strongly typed, static batch-one, and approximately
`134.60 MiB`.

Post-processing remains outside the engine:

- Semantic prediction: `argmax(semantic_logits)`
- Metric depth: `exp(log_depth)`, clamped to `0.001–200 m`
- Uncertainty: `exp(clamp(depth_log_scale, -6, 6))`

## Installation

Install the development, ONNX and TensorRT extras:

~~~bash
python -m pip install -e ".[dev,export,tensorrt]"
~~~

## Engine construction

First export the official epoch-16 checkpoint:

~~~bash
python -m perception_rt.export_onnx
~~~

Build the TensorRT engine:

~~~bash
python -m perception_rt.build_tensorrt
~~~

Default paths:

- ONNX model: `outputs/onnx/perception_rt_mit_b2_fp32.onnx`
- TensorRT engine: `outputs/tensorrt/perception_rt_mit_b2_fp32.engine`

The builder:

1. Parses the ONNX graph with TensorRT.
2. Validates every tensor name, shape and datatype.
3. Creates a strongly typed network.
4. Uses a `1024 MiB` workspace limit.
5. Serializes the engine only after a successful build.

The measured build time was approximately `37.92 seconds` on the validated
system. Build time is informational and is not an inference benchmark.

## Native runtime

`TensorRTRunner` deserializes the engine and validates the complete runtime
contract before inference.

The runtime:

- Accepts a contiguous FP32 CUDA tensor.
- Binds memory through PyTorch tensor device pointers.
- Reuses preallocated output tensors.
- Executes through `execute_async_v3`.
- Uses a dedicated non-default CUDA stream.
- Synchronizes before returning outputs.

No separate `cuda-python` dependency is required for this implementation.

Returned output buffers are reused by the next inference call. Consumers
must clone or copy values that need to persist across calls.

## Held-out parity protocol

TensorRT was compared directly with the validation-selected PyTorch
checkpoint from epoch 16.

Five deterministic centre-cropped Scene18 samples were evaluated:

| Index | Variation | Frame | Semantic agreement |
|---:|---|---:|---:|
| 0 | `15-deg-left` | 0 | 0.99994141 |
| 847 | `30-deg-left` | 169 | 0.99996582 |
| 1695 | `fog` | 0 | 0.99999023 |
| 2542 | `overcast` | 169 | 0.99998047 |
| 3389 | `sunset` | 338 | 0.99999023 |

The validator compares all three raw outputs, post-processed depth,
post-processed uncertainty, and semantic argmax predictions.

## Acceptance thresholds

Raw outputs and post-processed depth use:

`absolute_error <= 0.02 + 0.01 × abs(PyTorch reference)`

Semantic argmax agreement must be at least `0.999`.

Uncertainty is predicted in log space and exponentiated during
post-processing. Its linear-space relative tolerance is therefore derived
from the raw absolute log-scale tolerance:

`exp(0.02) - 1 = 0.02020134`

Post-processed uncertainty consequently uses an absolute tolerance of
`0.02` and a relative tolerance of approximately `2.0201%`. This criterion
is mathematically derived from the fixed log-space gate rather than fitted
to the observed TensorRT results.

## Parity results

All five samples passed every acceptance gate.

| Output | Maximum absolute error | Maximum mean absolute error | Within tolerance |
|---|---:|---:|---:|
| `semantic_logits` | 0.0178661 | 0.00170676 | 100% |
| `log_depth` | 0.0180016 | 0.000237618 | 100% |
| `depth_log_scale` | 0.0201530 | 0.000287344 | 100% |
| Post-processed depth | 0.108421 m | 0.00245566 m | 100% |
| Post-processed uncertainty | 4.73819 | 0.00231592 | 100% |

Minimum semantic argmax agreement was `0.99994141`.

Overall FP32 PyTorch–TensorRT parity: **passed**.

The larger maximum uncertainty difference occurs at high uncertainty values
after exponentiation. The raw `depth_log_scale` output remained within its
fixed FP32 acceptance gate.

## Benchmark methodology

The benchmark uses:

- One held-out `Scene18/15-deg-left` frame
- Input shape `1 × 3 × 320 × 640`
- FP32 execution for all backends
- Device-resident input and output buffers
- 30 warm-up iterations
- 100 measured iterations
- Synchronous wall-clock latency
- CUDA synchronization after every inference

Model loading, engine deserialization, dataset preprocessing and
host-device transfers are excluded.

ONNX Runtime uses CUDA I/O binding. TensorRT uses reusable native CUDA
buffers. PyTorch uses ordinary eager FP32 inference.

## Benchmark results

| Backend | Mean | Median | P95 | Throughput |
|---|---:|---:|---:|---:|
| PyTorch FP32 | 23.279 ms | — | 23.735 ms | 42.96 FPS |
| ONNX Runtime CUDA FP32 | 26.575 ms | — | 26.967 ms | 37.63 FPS |
| TensorRT FP32 | 16.579 ms | — | 17.087 ms | 60.32 FPS |

TensorRT achieved:

- `1.404×` mean-latency speedup over PyTorch FP32
- `1.603×` mean-latency speedup over ONNX Runtime CUDA FP32

These are measurements from the validated laptop GPU, not universal
performance guarantees.

## Reproduction

Run parity validation:

~~~bash
python -m perception_rt.validate_tensorrt
~~~

This writes `outputs/tensorrt/parity.json` and exits non-zero when any gate
fails.

Run the benchmark:

~~~bash
python -m perception_rt.benchmark_inference
~~~

This writes `outputs/tensorrt/benchmark.json`.

Models, engines, checkpoints and generated reports remain under `outputs/`
and are intentionally excluded from Git.

## Limitations

- The engine is specific to the TensorRT version, GPU architecture and
  platform on which it is built.
- Users should build the engine locally rather than treat it as a portable
  model artifact.
- Parity covers five samples rather than all 3,390 Scene18 samples.
- Benchmarking covers one machine and one held-out input.
- GPU clocks, power limits and competing system workloads were not locked.
- The benchmark measures synchronous deployment-oriented APIs rather than
  isolated kernel execution.
- PyTorch allocates model outputs while ONNX Runtime and TensorRT reuse
  bound output buffers.
- The engine supports only static batch-one FP32 inference.
- FP16 and INT8 optimization remain future stages.
- Native C++/CUDA and ROS 2 deployment remain future stages.

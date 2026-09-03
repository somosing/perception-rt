# TensorRT FP16 Optimization

This page documents the PerceptionRT v0.5.0 TensorRT FP16 deployment
baseline. It extends the validated v0.4.0 FP32 runtime with explicit FP16
ONNX export, strongly typed engine construction, native inference,
held-out parity validation and reproducible benchmarking.

## Scope

This stage covers:

- Native FP16 export from the validation-selected epoch-16 checkpoint
- Static, strongly typed TensorRT FP16 engine construction
- FP16 input and output buffer support in `TensorRTRunner`
- Direct comparison against the original FP32 PyTorch checkpoint
- Explicit numerical-coverage and semantic-agreement gates
- Four-backend device-resident benchmarking

This is a deployment-equivalence study. The original model-quality results
remain documented in [`vkitti2_results.md`](vkitti2_results.md).

## Validated environment

- Ubuntu 24.04
- NVIDIA GeForce RTX 3060 Laptop GPU
- Compute capability 8.6
- 6 GB VRAM
- NVIDIA driver 595.84
- Python 3.12
- PyTorch 2.13.0 with CUDA 13.0
- ONNX 1.22.0
- ONNX Runtime GPU 1.29.0
- TensorRT 11.2.1.2

## Explicit-precision design

TensorRT 11.2 strongly typed networks do not expose the former FP16
builder flag. Precision is therefore encoded directly in the ONNX graph.

The official checkpoint is converted to FP16 during export. The generated
graph has FP16 model parameters, input and outputs. TensorRT preserves
those explicit types when constructing the strongly typed engine.

No ONNX conversion package is required.

## Deployment contract

| Tensor | Mode | Shape | Type |
|---|---|---:|---|
| `image` | Input | `1 × 3 × 320 × 640` | FP16 |
| `semantic_logits` | Output | `1 × 15 × 320 × 640` | FP16 |
| `log_depth` | Output | `1 × 1 × 320 × 640` | FP16 |
| `depth_log_scale` | Output | `1 × 1 × 320 × 640` | FP16 |

The engine remains static batch-one. Post-processing is unchanged:

- Semantic prediction: `argmax(semantic_logits)`
- Metric depth: `exp(log_depth)`, clamped to `0.001–200 m`
- Uncertainty: `exp(clamp(depth_log_scale, -6, 6))`

## Artifact sizes

| Artifact | FP32 | FP16 | Reduction |
|---|---:|---:|---:|
| ONNX model | 103.00 MiB | 52.71 MiB | 48.83% |
| TensorRT engine | 134.60 MiB | 85.00 MiB | 36.85% |

Models and engines remain under `outputs/` and are excluded from Git.

## Reproduction

Install the pinned dependencies:

~~~bash
python -m pip install -e ".[dev,export,tensorrt]"
~~~

Export the FP16 ONNX model:

~~~bash
python -m perception_rt.export_onnx --precision fp16
~~~

Build the FP16 TensorRT engine:

~~~bash
python -m perception_rt.build_tensorrt --precision fp16
~~~

Run held-out parity validation:

~~~bash
python -m perception_rt.validate_tensorrt --precision fp16
~~~

Run the four-backend benchmark:

~~~bash
python -m perception_rt.benchmark_inference
~~~

Default FP16 paths:

- ONNX: `outputs/onnx/perception_rt_mit_b2_fp16.onnx`
- Engine: `outputs/tensorrt/perception_rt_mit_b2_fp16.engine`
- Parity report: `outputs/tensorrt/parity_fp16.json`
- Benchmark report: `outputs/tensorrt/benchmark_fp16.json`

## Held-out parity protocol

The FP16 TensorRT engine was compared directly with the FP32 PyTorch
checkpoint from epoch 16 on five deterministic centre-cropped Scene18
samples.

| Index | Variation | Frame | Semantic agreement |
|---:|---|---:|---:|
| 0 | `15-deg-left` | 0 | 0.99965332 |
| 847 | `30-deg-left` | 169 | 0.99968750 |
| 1695 | `fog` | 0 | 0.99994141 |
| 2542 | `overcast` | 169 | 0.99991699 |
| 3389 | `sunset` | 338 | 0.99989258 |

The comparison includes all three raw outputs, post-processed depth,
post-processed uncertainty and semantic argmax predictions.

## Acceptance policy

Every raw output and post-processed depth value uses the original FP32
pointwise tolerance:

`absolute_error <= 0.02 + 0.01 × abs(FP32 PyTorch reference)`

Post-processed uncertainty retains its mathematically derived relative
tolerance:

`exp(0.02) - 1 = 0.02020134`

Semantic argmax agreement must be at least `0.999`.

FP32 validation requires 100% of values to satisfy the pointwise
tolerance. FP16 validation requires at least 99% for every output on every
sample. This explicit coverage gate preserves the original error bounds
while acknowledging sparse reduced-precision outliers. It does not claim
bitwise or full-value equivalence.

## Parity results

| Output | Maximum absolute error | Maximum mean absolute error | Minimum coverage |
|---|---:|---:|---:|
| `semantic_logits` | 0.152422 | 0.011304 | 99.8484% |
| `log_depth` | 0.204821 | 0.003736 | 99.2310% |
| `depth_log_scale` | 0.207245 | 0.004662 | 99.6172% |
| Post-processed depth | 1.530449 | 0.047477 | 99.5469% |
| Post-processed uncertainty | 35.663788 | 0.023709 | 99.5215% |

Minimum semantic argmax agreement was
`0.99965332`.

Every output exceeded the 99% coverage requirement on every sample.

Overall FP32 PyTorch–FP16 TensorRT parity: **passed**.

The larger maximum post-processed uncertainty error occurs at high
uncertainty values because exponentiation magnifies log-scale differences.
The maximum mean absolute error remains `0.023709`.

## Benchmark methodology

The benchmark uses:

- One held-out `Scene18/15-deg-left` frame
- Static batch-one `1 × 3 × 320 × 640` inputs
- 30 warm-up iterations
- 100 measured iterations
- Device-resident input and output buffers
- Synchronous wall-clock latency
- CUDA synchronization after every inference
- Fixed backend order recorded in the JSON report

Model loading, engine deserialization, preprocessing, host-device transfer
and the one-time FP32-to-FP16 input conversion are excluded.

## Benchmark results

| Backend | Mean latency | P95 latency | Throughput |
|---|---:|---:|---:|
| PyTorch FP32 | 23.659 ms | 24.230 ms | 42.27 FPS |
| ONNX Runtime CUDA FP32 | 26.879 ms | 27.293 ms | 37.20 FPS |
| TensorRT FP32 | 16.859 ms | 17.305 ms | 59.32 FPS |
| TensorRT FP16 | 6.846 ms | 6.918 ms | 146.06 FPS |

TensorRT FP16 achieved:

- `2.462×` speedup over TensorRT FP32
- `3.456×` speedup over PyTorch FP32
- `3.926×` speedup over ONNX Runtime CUDA FP32

These measurements describe the validated laptop GPU and are not
universal performance guarantees.

## Runtime behavior

`TensorRTRunner` determines the required PyTorch dtype from the serialized
engine contract. FP32 engines require `torch.float32`; FP16 engines require
`torch.float16`.

Both paths:

- Validate tensor names, shapes, types and I/O modes
- Bind PyTorch CUDA memory through tensor device pointers
- Reuse preallocated output buffers
- Execute through `execute_async_v3`
- Use a dedicated non-default CUDA stream
- Synchronize before returning

Returned output buffers are reused by the next inference call. Consumers
must clone values that need to persist.

## Limitations

- The serialized engine is specific to its TensorRT version, GPU
  architecture and platform and must be built locally.
- Parity covers five samples, not the complete 3,390-sample Scene18 split.
- FP16 parity uses a 99% per-output coverage gate rather than requiring
  every value to satisfy the FP32 pointwise tolerance.
- Benchmark results cover one GPU and one held-out input.
- GPU clocks, power limits and background workloads were not locked.
- Input conversion and host-device transfer are excluded.
- FP16 values have less numerical range and precision than FP32.
- INT8 calibration and deployment remain future work.
- Native C++/CUDA and ROS 2 integration remain future stages.

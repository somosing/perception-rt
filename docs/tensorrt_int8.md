# TensorRT selective INT8 evaluation

This page documents the PerceptionRT v0.6.0 release-candidate study of
post-training INT8 quantization. The result is deliberately reported as an
evaluated optimization path: selective INT8 passed the frozen numerical-parity
criteria and slightly improved FP32 TensorRT latency, but it did not outperform
the existing FP16 engine.

## Outcome

The production policy quantizes only the 13 spatial-reduction convolutions in
the MiT-B2 encoder attention blocks. All other operators remain at their
original precision. The TensorRT engine exposes the unchanged deployment
contract:

| Tensor | Role | Shape | External dtype |
|---|---|---:|---|
| `image` | Input | `1 × 3 × 320 × 640` | FP32 |
| `semantic_logits` | Output | `1 × 15 × 320 × 640` | FP32 |
| `log_depth` | Output | `1 × 1 × 320 × 640` | FP32 |
| `depth_log_scale` | Output | `1 × 1 × 320 × 640` | FP32 |

Explicit ONNX QuantizeLinear and DequantizeLinear nodes define the internal
INT8 regions. The TensorRT builder does not use a legacy implicit calibrator or
an INT8 builder flag.

## Artifacts

Generated deployment artifacts remain local and are ignored by Git:

| Artifact | Path | Size |
|---|---|---:|
| Preprocessed FP32 ONNX | `outputs/onnx/perception_rt_mit_b2_fp32_preprocessed.onnx` | approximately 101 MiB |
| Selective INT8 Q/DQ ONNX | `outputs/onnx/perception_rt_mit_b2_int8.onnx` | 88.50 MiB |
| Selective INT8 TensorRT engine | `outputs/tensorrt/perception_rt_mit_b2_int8.engine` | 122.69 MiB |
| Held-out parity report | `outputs/tensorrt/parity_int8.json` | generated locally |
| Five-backend benchmark | `outputs/tensorrt/benchmark_int8.json` | generated locally |

The Q/DQ graph contains 26 `QuantizeLinear` and 39
`DequantizeLinear` nodes. TensorRT engines are hardware- and
software-specific, so the serialized engine is not distributed as a release
asset.

## Leakage-safe calibration

Calibration uses only the Scene06 validation split. Scene18 remains held out
until the quantization policy and acceptance gates are frozen.

The calibration procedure uses:

- 20 deterministic, evenly spaced Scene06 samples
- two samples from each of the ten synthetic variations
- batch-one normalized FP32 images
- ONNX Runtime MinMax calibration
- signed symmetric INT8 activations and weights
- per-channel weight quantization
- explicit Q/DQ representation
- unquantized bias tensors

The selected variations are `15-deg-left`, `15-deg-right`,
`30-deg-left`, `30-deg-right`, `clone`, `fog`,
`morning`, `overcast`, `rain`, and `sunset`.

## Sensitivity-driven scope selection

Broad post-training quantization caused unacceptable degradation, especially
in fog and in the depth outputs. Quantization was therefore narrowed by layer
family using Scene06 only.

| Candidate scope | Minimum semantic agreement | Decision |
|---|---:|---|
| Encoder Conv and Gemm | 0.926626 | Rejected |
| Patch-embedding convolutions | 0.952720 | Rejected |
| Depthwise MLP convolutions | 0.994155 | Rejected |
| Attention spatial-reduction convolutions | 0.999795 | Selected |

The selected spatial-reduction scope also retained at least 98.1587% Scene06
coverage under the original pointwise tolerances. Increasing broad calibration
from 20 to 100 images did not recover acceptable accuracy, so calibration size
was not treated as the primary failure.

A hybrid FP16 graph with the same selective INT8 scope was also rejected. It
measured 12.540 ms, compared with roughly 6.8 ms for the pure FP16 engine.
The extra conversions and limited quantized scope outweighed the benefit.

## Frozen acceptance criteria

The following gates were selected from Scene06 results before the one-time
Scene18 evaluation:

| Gate | Threshold |
|---|---:|
| Minimum semantic argmax agreement | 0.9995 |
| Minimum pointwise-tolerance coverage for every raw output | 98% |
| Minimum pointwise-tolerance coverage for every post-processed output | 98% |

The underlying numerical comparison uses an absolute tolerance of `0.02`,
a relative tolerance of `0.01`, and a separate uncertainty relative tolerance
equivalent to `expm1(0.02)`.

Passing does not mean every element lies within tolerance. It means every
reported output meets the frozen 98% coverage gate and every sample meets the
semantic-agreement gate.

## Held-out Scene18 parity

The epoch-16 validation-selected PyTorch checkpoint and the production INT8
engine were compared on the same five deterministic Scene18 samples used by
the earlier deployment stages.

| Index | Variation | Frame | Semantic agreement | Passed |
|---:|---|---:|---:|:---:|
| 0 | `15-deg-left` | 0 | 0.99957520 | Yes |
| 847 | `30-deg-left` | 169 | 0.99957031 | Yes |
| 1695 | `fog` | 0 | 0.99988770 | Yes |
| 2542 | `overcast` | 169 | 0.99992676 | Yes |
| 3389 | `sunset` | 338 | 0.99990234 | Yes |

Minimum semantic argmax agreement was `0.99957031`.

### Numerical summary

| Output | Minimum coverage | Maximum absolute error | Maximum mean absolute error |
|---|---:|---:|---:|
| `semantic_logits` | 99.7994% | 0.144810 | 0.00904663 |
| `log_depth` | 99.7368% | 0.107719 | 0.00333806 |
| `depth_log_scale` | 99.7173% | 0.131084 | 0.00388940 |
| Post-processed depth | 99.5347% | 0.515152 m | 0.0230077 m |
| Post-processed uncertainty scale | 99.4072% | 33.8899 | 0.0203103 |

Overall held-out selective-INT8 parity: **passed**.

The maximum uncertainty-scale error is amplified by exponentiating
`depth_log_scale`. The corresponding minimum coverage remained above the
frozen 98% gate, and the maximum mean absolute error was `0.0203103`.

## Benchmark methodology

The benchmark uses:

- NVIDIA GeForce RTX 3060 Laptop GPU, compute capability 8.6
- PyTorch 2.13.0 with CUDA 13.0
- ONNX Runtime GPU 1.29.0
- TensorRT 11.2.1.2
- static batch-one `1 × 3 × 320 × 640` input
- sample index 0 from held-out Scene18
- 30 warm-up iterations
- 100 measured iterations
- device-resident input and output buffers
- synchronous wall-clock timing with CUDA synchronization after every run
- model loading, preprocessing, and host-device transfers excluded

### Latency results

| Backend | Mean latency | P95 latency | Throughput |
|---|---:|---:|---:|
| PyTorch FP32 | 23.661 ms | 24.380 ms | 42.26 FPS |
| ONNX Runtime CUDA FP32 | 26.577 ms | 27.070 ms | 37.63 FPS |
| TensorRT FP32 | 16.571 ms | 16.944 ms | 60.35 FPS |
| TensorRT FP16 | 6.727 ms | 6.819 ms | 148.66 FPS |
| TensorRT selective INT8 | 15.841 ms | 16.045 ms | 63.13 FPS |

Selective INT8 achieved:

- `1.046×` speedup over TensorRT FP32
- approximately `1.494×` speedup over PyTorch FP32
- approximately `1.678×` speedup over ONNX Runtime CUDA FP32
- only `0.425×` the FP16 speed, or approximately `2.354×` higher latency

The result shows that sparse Q/DQ INT8 placement is not enough to beat the
well-optimized FP16 path on this GPU. FP16 remains the recommended deployment
configuration.

## Reproduction

Install all development and deployment dependencies:

~~~bash
python -m pip install -e ".[dev,export,tensorrt]"
~~~

Create the selective INT8 ONNX graph:

~~~bash
python -m perception_rt.quantize_onnx
~~~

Build the production TensorRT engine:

~~~bash
python -m perception_rt.build_tensorrt --precision int8
~~~

Validate against the held-out Scene18 samples:

~~~bash
python -m perception_rt.validate_tensorrt --precision int8
~~~

Run the five-backend benchmark:

~~~bash
python -m perception_rt.benchmark_inference
~~~

Generated models, engines, calibration intermediates, and reports stay under
`outputs/` and are intentionally excluded from version control.

## Limitations

- Calibration uses only 20 synthetic Scene06 images.
- Parity covers five deterministic Scene18 samples, not the full test split.
- Quantization scope was selected for this model, checkpoint, and input shape.
- The benchmark uses one laptop GPU and a device-resident batch-one workload.
- End-to-end preprocessing, transfers, visualization, and application latency
  are excluded.
- TensorRT engines must be rebuilt for the target GPU and software stack.
- No real-world KITTI-360 transfer evaluation has been completed.
- The selective INT8 engine is not the fastest configuration; FP16 is.

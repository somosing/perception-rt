# ONNX Export and Runtime Parity

This page documents the PerceptionRT v0.3.0 FP32 ONNX deployment baseline
and its validation against the original PyTorch checkpoint.

## Scope

The purpose is to verify that ONNX export does not materially change model
outputs before TensorRT optimization begins. This backend-equivalence test
does not replace the model-accuracy evaluation in
[`vkitti2_results.md`](vkitti2_results.md).

## Reference checkpoint

The exported model uses `best.pt` from epoch 16, selected using Scene06
validation loss. Checkpoints, ONNX models and generated reports are not
committed to Git.

## Validated environment

- Ubuntu 24.04
- NVIDIA GeForce RTX 3060 Laptop GPU
- PyTorch 2.13.0 with CUDA 13.0
- ONNX 1.22.0
- ONNX Script 0.7.1
- ONNX Runtime GPU 1.29.0
- CUDA execution provider with CPU fallback registered

## Deployment contract

The exporter creates a static batch-one FP32 graph using ONNX opset 18.

| Tensor | Shape | Type | Meaning |
|---|---:|---|---|
| `image` | `1 × 3 × 320 × 640` | FP32 | ImageNet-normalized RGB |
| `semantic_logits` | `1 × 15 × 320 × 640` | FP32 | Raw semantic logits |
| `log_depth` | `1 × 1 × 320 × 640` | FP32 | Raw metric log-depth |
| `depth_log_scale` | `1 × 1 × 320 × 640` | FP32 | Raw uncertainty log-scale |

Post-processing remains outside the graph:

- Semantic prediction: `argmax(semantic_logits)`
- Metric depth: `exp(log_depth)`, clamped to `0.001–200 m`
- Uncertainty: `exp(clamp(depth_log_scale, -6, 6))`

The generated model is approximately 103 MiB and stores its weights inside
one ONNX file.

## Export

Install the pinned dependencies:

~~~bash
python -m pip install -e ".[dev,export]"
~~~

Export the validation-selected checkpoint:

~~~bash
python -m perception_rt.export_onnx
~~~

Default paths:

- Configuration: `configs/train_vkitti2.yaml`
- Checkpoint: `outputs/training/vkitti2_multitask/best.pt`
- Model: `outputs/onnx/perception_rt_mit_b2_fp32.onnx`

The exporter checks the generated graph with the ONNX checker.

## CUDA smoke validation

ONNX Runtime loaded the graph with these providers:

1. `CUDAExecutionProvider`
2. `CPUExecutionProvider`

A zero-valued input produced finite outputs with the expected names, types
and shapes.

## Held-out parity protocol

Five deterministic centre-cropped samples were selected across Scene18.

| Index | Variation | Frame | Semantic agreement |
|---:|---|---:|---:|
| 0 | `15-deg-left` | 0 | 0.99993164 |
| 847 | `30-deg-left` | 169 | 0.99996094 |
| 1695 | `fog` | 0 | 0.99999512 |
| 2542 | `overcast` | 169 | 0.99999023 |
| 3389 | `sunset` | 338 | 0.99999023 |

Each input is processed once by FP32 PyTorch CUDA and once by FP32 ONNX
Runtime CUDA. The validator compares all three raw outputs, post-processed
depth, post-processed uncertainty and semantic argmax predictions.

## Acceptance thresholds

Every numerical value must satisfy:

`absolute_error <= 0.02 + 0.01 × abs(PyTorch reference)`

Semantic argmax agreement must be at least `0.999`.

These thresholds measure backend equivalence, not prediction accuracy. They
are fixed for future FP32 ONNX exports. TensorRT precision modes will use
separately documented criteria.

## Results

All values satisfied the numerical gate and all samples exceeded the semantic
agreement threshold.

| Output | Maximum absolute error | Maximum mean absolute error | Within tolerance |
|---|---:|---:|---:|
| `semantic_logits` | 0.0178061 | 0.00179873 | 100% |
| `log_depth` | 0.0134659 | 0.000236070 | 100% |
| `depth_log_scale` | 0.0156231 | 0.000301362 | 100% |
| Post-processed depth | 0.100380 m | 0.00261906 m | 100% |
| Post-processed uncertainty | 3.56454 | 0.00183911 | 100% |

Minimum semantic argmax agreement was `0.99993164`.

Overall FP32 PyTorch–ONNX Runtime CUDA parity: **passed**.

The larger maximum uncertainty difference appears after exponentiating the
log-scale output. Its raw maximum error was `0.0156231`, its maximum mean
error was `0.000301362`, and all values passed the combined tolerance.

## Reproduction

Run:

~~~bash
python -m perception_rt.validate_onnx
~~~

The command writes `outputs/onnx/parity.json` and exits non-zero if numerical
parity fails, semantic agreement is too low, or CUDA is unavailable.

## Limitations

- Parity covers five samples rather than all 3,390 Scene18 samples.
- Inputs are deterministic `320 × 640` centre crops.
- The graph supports static batch-one inference only.
- This release validates FP32 ONNX Runtime CUDA only.
- Latency, throughput, memory use and warm-up are not yet benchmarked.
- TensorRT FP32, FP16 and INT8 remain future stages.
- Native C++/CUDA and ROS 2 deployment remain future stages.

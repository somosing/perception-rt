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

### v0.1.0 — Dataset foundation

- [x] Python project environment
- [x] CUDA-enabled PyTorch verification
- [x] Unit-test and lint configuration
- [x] Virtual KITTI 2 metadata parsing
- [x] RGB, metric-depth, and semantic-label decoding
- [x] Aligned multimodal sample indexing and loading
- [x] Leakage-safe scene-based dataset splits
- [x] Multimodal sample visualization
- [x] Dataset-wide integrity and statistics audit

Model development has not started yet. This release establishes a tested and
reproducible dataset foundation.

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
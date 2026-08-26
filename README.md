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

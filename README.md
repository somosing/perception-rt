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

## Dataset strategy

Initial development uses Virtual KITTI 2 because it provides synchronized RGB images,
dense metric depth, semantic labels, and camera calibration.

Real-world evaluation will later use KITTI-360 to measure synthetic-to-real transfer
on German driving scenes.

Dataset files are not distributed with this repository. They must be downloaded
separately under their respective licences.

## Current status

`v0.1.0` — dataset ingestion and validation:

- [x] Python project environment
- [x] CUDA-enabled PyTorch verification
- [x] Initial unit-test and lint configuration
- [ ] Virtual KITTI 2 dataset validation
- [ ] RGB, metric-depth, and semantic-label decoding
- [ ] Dataset visualization
- [ ] Dataset statistics and sequence-based splits

## Development environment

- Ubuntu 24.04
- NVIDIA GeForce RTX 3060 Laptop GPU
- 6 GB VRAM
- Python 3.12
- PyTorch with CUDA
- ROS 2 Jazzy

## License

The source-code licence will be finalized before the first public release.

Virtual KITTI 2 and KITTI-360 are external datasets with their own licensing terms.
No dataset files are included in this repository.

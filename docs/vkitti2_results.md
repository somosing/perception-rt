# Virtual KITTI 2 multitask results

This page documents the PerceptionRT v0.2.0 training baseline and its evaluation on the held-out Virtual KITTI 2 test scene.

## Evaluation protocol

The dataset is separated by physical scene to prevent weather and viewpoint variations of the same scene from leaking across splits.

| Split | Scenes | Samples |
|---|---|---:|
| Training | Scene01, Scene02, Scene20 | 15,170 |
| Validation | Scene06 | 2,700 |
| Test | Scene18 | 3,390 |

Only Camera 0 samples are used. Training uses random aligned crops, while validation and test evaluation use deterministic centre crops of `320 × 640` pixels. Valid metric depth is limited to `200 m`.

The official checkpoint is `best.pt` from epoch 16. It was selected using the lowest total validation loss on Scene06. The final `latest.pt` checkpoint from epoch 30 is retained only as a diagnostic checkpoint and was not selected using Scene18.

## Model

The network contains:

- A shared ImageNet-pretrained MiT-B2 encoder
- A 15-class semantic-segmentation decoder
- A metric log-depth decoder
- A heteroscedastic log-depth uncertainty decoder
- Multi-scale feature fusion in every prediction decoder

The uncertainty output represents a learned scale in log-depth space. It is not a calibrated probability that a depth prediction is correct.

## Training configuration

| Setting | Value |
|---|---:|
| Epochs | 30 |
| Crop size | 320 × 640 |
| Batch size | 4 |
| Gradient accumulation | 2 steps |
| Effective batch size | 8 |
| Optimizer | AdamW |
| Initial learning rate | 0.0002 |
| Weight decay | 0.01 |
| Warmup | 500 optimizer steps |
| Schedule | Cosine decay |
| Minimum LR ratio | 0.05 |
| Mixed precision | FP16 AMP |
| Maximum depth | 200 m |
| Random seed | 42 |

The training objective combines class-weighted semantic cross-entropy, heteroscedastic Laplace negative log-likelihood in log-depth space, and a masked log-depth gradient loss. Their weights are `1.0`, `1.0`, and `0.5`, respectively.

## Held-out Scene18 results

The validation-selected `best.pt` checkpoint produced:

| Metric | Result |
|---|---:|
| Semantic mIoU | 0.7526 |
| Depth AbsRel | 0.1876 |
| Depth RMSE | 17.51 m |
| Depth delta1 | 0.6604 |
| Uncertainty/error Pearson correlation | 0.2501 |

The positive uncertainty/error correlation shows that the learned uncertainty contains information about prediction error. Its modest magnitude also shows that uncertainty calibration remains an open problem.

The correlation is computed between predicted uncertainty scale and absolute log-depth error over valid test pixels.

## Reproduction

Train the baseline:

    python -m perception_rt.train \
        --config configs/train_vkitti2.yaml

Evaluate the official checkpoint:

    python -m perception_rt.evaluate \
        --config configs/train_vkitti2.yaml \
        --checkpoints outputs/training/vkitti2_multitask/best.pt \
        --output outputs/evaluation/vkitti2_test.json

Generate qualitative predictions:

    python -m perception_rt.qualitative \
        --config configs/train_vkitti2.yaml \
        --checkpoint outputs/training/vkitti2_multitask/best.pt

Evaluation reports, checkpoints, datasets, and generated figures remain local and are not committed to Git.

## Qualitative inspection

The visualization tool generates eight-panel comparisons containing:

- RGB input
- Semantic ground truth and prediction
- Semantic disagreement
- Metric-depth ground truth and prediction
- Absolute metric-depth error
- Predicted log-depth uncertainty scale

These figures are diagnostic outputs. The quantitative Scene18 evaluation above remains the official baseline.

## Limitations

- Training and evaluation currently use synthetic Virtual KITTI 2 data.
- Scene18 is held out by physical scene, but it belongs to the same synthetic dataset.
- Evaluation uses deterministic centre crops rather than complete-resolution images.
- Long-range errors have a strong influence on metric RMSE.
- The uncertainty signal is useful but weakly correlated with actual error.
- KITTI-360 transfer evaluation has not yet been completed.
- ONNX, TensorRT, native C++/CUDA, and ROS 2 deployment remain future release stages.

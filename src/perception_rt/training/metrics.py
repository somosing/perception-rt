"""Streaming semantic, depth and uncertainty metrics."""

import math

import torch
from torch import Tensor


class DenseMetricAccumulator:
    """Accumulate dense prediction metrics across many batches."""

    def __init__(
        self,
        number_of_classes: int,
        *,
        maximum_depth_m: float,
    ) -> None:
        if number_of_classes <= 1:
            raise ValueError("At least two classes are required")

        if maximum_depth_m <= 0.0:
            raise ValueError("Maximum depth must be positive")

        self.number_of_classes = number_of_classes
        self.maximum_depth_m = maximum_depth_m

        self.confusion = torch.zeros(
            number_of_classes,
            number_of_classes,
            dtype=torch.int64,
        )

        self.depth_count = 0
        self.absolute_relative_sum = 0.0
        self.squared_error_sum = 0.0
        self.delta1_count = 0

        self.correlation_count = 0
        self.uncertainty_sum = 0.0
        self.error_sum = 0.0
        self.uncertainty_squared_sum = 0.0
        self.error_squared_sum = 0.0
        self.uncertainty_error_sum = 0.0

    def update(
        self,
        semantic_logits: Tensor,
        predicted_log_depth: Tensor,
        predicted_log_scale: Tensor,
        semantic_target: Tensor,
        target_depth_m: Tensor,
        depth_valid: Tensor,
    ) -> None:
        """Accumulate metrics from one prediction batch."""
        semantic_prediction = semantic_logits.argmax(dim=1)
        semantic_target = semantic_target.to(torch.int64)

        semantic_valid = (semantic_target >= 0) & (semantic_target < self.number_of_classes)
        encoded_pairs = (
            self.number_of_classes * semantic_target[semantic_valid]
            + semantic_prediction[semantic_valid]
        )
        batch_confusion = torch.bincount(
            encoded_pairs,
            minlength=(self.number_of_classes * self.number_of_classes),
        ).reshape(
            self.number_of_classes,
            self.number_of_classes,
        )
        self.confusion += batch_confusion.cpu()

        valid = depth_valid.to(torch.bool)

        if not torch.any(valid):
            return

        target = target_depth_m[valid].to(torch.float64)
        predicted_log = predicted_log_depth[valid].to(torch.float64)
        predicted = torch.exp(predicted_log).clamp(
            min=1e-3,
            max=self.maximum_depth_m,
        )

        absolute_error = torch.abs(predicted - target)
        ratio = torch.maximum(
            predicted / target,
            target / predicted,
        )

        count = int(target.numel())
        self.depth_count += count
        self.absolute_relative_sum += float((absolute_error / target).sum())
        self.squared_error_sum += float(absolute_error.square().sum())
        self.delta1_count += int(torch.count_nonzero(ratio < 1.25))

        target_log = torch.log(target)
        log_depth_error = torch.abs(predicted_log - target_log)
        uncertainty = torch.exp(predicted_log_scale[valid].to(torch.float64).clamp(-6.0, 6.0))

        self.correlation_count += count
        self.uncertainty_sum += float(uncertainty.sum())
        self.error_sum += float(log_depth_error.sum())
        self.uncertainty_squared_sum += float(uncertainty.square().sum())
        self.error_squared_sum += float(log_depth_error.square().sum())
        self.uncertainty_error_sum += float((uncertainty * log_depth_error).sum())

    def semantic_iou(self) -> Tensor:
        """Return per-class IoU, with NaN for absent target classes."""
        confusion = self.confusion.to(torch.float64)
        true_positive = confusion.diag()
        target_count = confusion.sum(dim=1)
        prediction_count = confusion.sum(dim=0)
        union = target_count + prediction_count - true_positive

        iou = torch.full(
            (self.number_of_classes,),
            float("nan"),
            dtype=torch.float64,
        )
        observed = target_count > 0
        iou[observed] = true_positive[observed] / union[observed]
        return iou

    def uncertainty_error_pearson(self) -> float:
        """Return Pearson correlation between uncertainty and error."""
        count = self.correlation_count

        if count < 2:
            return 0.0

        numerator = count * self.uncertainty_error_sum - self.uncertainty_sum * self.error_sum
        uncertainty_term = count * self.uncertainty_squared_sum - self.uncertainty_sum**2
        error_term = count * self.error_squared_sum - self.error_sum**2
        denominator = math.sqrt(max(uncertainty_term, 0.0) * max(error_term, 0.0))

        if denominator == 0.0:
            return 0.0

        return numerator / denominator

    def compute(self) -> dict[str, float]:
        """Compute aggregate metrics."""
        if self.depth_count == 0:
            raise ValueError("No valid depth pixels were accumulated")

        iou = self.semantic_iou()
        observed_iou = iou[torch.isfinite(iou)]

        return {
            "mean_iou": float(observed_iou.mean()),
            "depth_abs_rel": (self.absolute_relative_sum / self.depth_count),
            "depth_rmse_m": math.sqrt(self.squared_error_sum / self.depth_count),
            "depth_delta1": (self.delta1_count / self.depth_count),
            "uncertainty_error_pearson": (self.uncertainty_error_pearson()),
        }

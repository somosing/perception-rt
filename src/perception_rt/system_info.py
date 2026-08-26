"""Inspect the PyTorch and CUDA environment used by PerceptionRT."""

import torch
import torchvision


def collect_system_info() -> dict[str, object]:
    """Return relevant PyTorch and GPU information."""
    info: dict[str, object] = {
        "pytorch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        device = torch.cuda.get_device_properties(0)
        info.update(
            {
                "gpu": device.name,
                "compute_capability": f"{device.major}.{device.minor}",
                "vram_gib": round(device.total_memory / 1024**3, 2),
            }
        )

    return info


def main() -> None:
    """Print the environment report and execute a small GPU operation."""
    for key, value in collect_system_info().items():
        print(f"{key}: {value}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    tensor = torch.randn(1024, 1024, device="cuda")
    result = tensor @ tensor
    torch.cuda.synchronize()

    print(f"GPU test: shape={tuple(result.shape)}, device={result.device}")


if __name__ == "__main__":
    main()

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from perception_rt.training.engine import (
    append_history,
    restore_checkpoint,
    save_checkpoint,
    seed_everything,
)


def test_seed_everything_reproduces_random_values() -> None:
    seed_everything(42)
    first = (
        random.random(),
        float(np.random.rand()),
        float(torch.rand(())),
    )

    seed_everything(42)
    second = (
        random.random(),
        float(np.random.rand()),
        float(torch.rand(())),
    )

    assert first == second


def test_checkpoint_restores_model_and_training_state(
    tmp_path: Path,
) -> None:
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    expected_weight = model.weight.detach().clone()
    path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        epoch=2,
        global_step=17,
        best_validation_loss=1.25,
    )

    with torch.no_grad():
        model.weight.add_(10.0)

    next_epoch, global_step, best_loss = restore_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(model.weight, expected_weight)
    assert next_epoch == 3
    assert global_step == 17
    assert best_loss == 1.25


def test_append_history_writes_header_once(
    tmp_path: Path,
) -> None:
    path = tmp_path / "history.csv"
    losses = {
        "total": 3.0,
        "semantic": 1.0,
        "depth_nll": 1.5,
        "depth_gradient": 0.5,
    }

    append_history(
        path,
        epoch=1,
        global_step=10,
        split="train",
        losses=losses,
    )
    append_history(
        path,
        epoch=1,
        global_step=10,
        split="validation",
        losses=losses,
    )

    lines = path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 3
    assert lines[0].startswith("epoch,global_step,split")
    assert lines[1].startswith("1,10,train")
    assert lines[2].startswith("1,10,validation")

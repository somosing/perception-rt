import random
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from perception_rt.training.engine import (
    append_history,
    build_warmup_cosine_scheduler,
    optimizer_step_count,
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
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    optimizer.step()
    scheduler.step()
    expected_scheduler_epoch = scheduler.last_epoch
    expected_weight = model.weight.detach().clone()
    path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=2,
        global_step=17,
        best_validation_loss=1.25,
    )

    with torch.no_grad():
        model.weight.add_(10.0)

    optimizer.step()
    scheduler.step()

    next_epoch, global_step, best_loss = restore_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(model.weight, expected_weight)
    assert scheduler.last_epoch == expected_scheduler_epoch
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


def test_optimizer_step_count_includes_partial_window() -> None:
    assert (
        optimizer_step_count(
            batches_per_epoch=9,
            epochs=3,
            accumulation_steps=4,
            maximum_steps=None,
        )
        == 9
    )

    assert (
        optimizer_step_count(
            batches_per_epoch=9,
            epochs=3,
            accumulation_steps=4,
            maximum_steps=5,
        )
        == 5
    )


def test_warmup_cosine_scheduler_reaches_minimum_rate() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_steps=6,
        warmup_steps=2,
        minimum_learning_rate_ratio=0.1,
    )

    learning_rates = [optimizer.param_groups[0]["lr"]]

    for _ in range(5):
        optimizer.step()
        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]["lr"])

    assert learning_rates[0] == pytest.approx(0.5)
    assert learning_rates[1] == pytest.approx(1.0)
    assert learning_rates[-1] == pytest.approx(0.1)
    assert learning_rates[2] < learning_rates[1]
    assert all(left >= right for left, right in pairwise(learning_rates[1:]))

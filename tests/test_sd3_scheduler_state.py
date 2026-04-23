from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from src.sd35_task_aware_vae.sd3 import finetune


class FakeScheduler:
    def __init__(self, num_train_timesteps: int = 1000) -> None:
        self.config = SimpleNamespace(num_train_timesteps=num_train_timesteps)
        self.timesteps = None
        self.sigmas = None

    def set_timesteps(self, num_timesteps: int, device=None) -> None:
        device = device or torch.device("cpu")
        self.timesteps = torch.arange(int(num_timesteps) - 1, -1, -1, device=device)
        self.sigmas = torch.linspace(1.0, 0.0, int(num_timesteps), device=device)


class FakeImage:
    def save(self, path) -> None:
        Path(path).write_bytes(b"fake")


class FakePipe:
    def __init__(self) -> None:
        self.scheduler = FakeScheduler()
        self.scheduler.set_timesteps(1000)
        self._execution_device = torch.device("cpu")

    def __call__(self, *args, num_inference_steps: int, **kwargs):
        self.scheduler.set_timesteps(num_inference_steps)
        return SimpleNamespace(images=[FakeImage()])


def test_training_timestep_sampling_recovers_from_preview_scheduler_state() -> None:
    scheduler = FakeScheduler()
    scheduler.set_timesteps(40)

    timesteps = finetune._sample_training_timesteps(
        scheduler,
        batch_size=8,
        device=torch.device("cpu"),
        diffusion_cfg={"weighting_scheme": "uniform"},
    )

    assert len(scheduler.timesteps) == 1000
    assert timesteps.shape == (8,)
    assert int(timesteps.max()) <= 999
    assert int(timesteps.min()) >= 0


def test_preview_generation_does_not_mutate_pipeline_scheduler(tmp_path) -> None:
    pipe = FakePipe()
    out_path = tmp_path / "preview.png"

    result = finetune._maybe_generate_text2img_preview(
        pipe,
        {
            "enabled": True,
            "kind": "text2img",
            "num_inference_steps": 40,
            "prompt": "ultra-widefield fundus photograph",
        },
        out_path,
    )

    assert result == out_path
    assert out_path.exists()
    assert len(pipe.scheduler.timesteps) == 1000

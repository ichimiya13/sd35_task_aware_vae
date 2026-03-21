from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class WandbSession:
    enabled: bool
    run: Any | None = None
    module: Any | None = None
    log_interval_steps: int = 0

    def log(self, payload: dict[str, Any], *, step: int | None = None) -> None:
        if not self.enabled or self.run is None:
            return
        try:
            if step is None:
                self.run.log(payload)
            else:
                self.run.log(payload, step=step)
        except Exception:
            pass

    def set_summary(self, key: str, value: Any) -> None:
        if not self.enabled or self.run is None:
            return
        try:
            self.run.summary[key] = value
        except Exception:
            pass

    def finish(self) -> None:
        if not self.enabled or self.run is None:
            return
        try:
            self.run.finish()
        except Exception:
            pass



def init_wandb_session(
    cfg: dict[str, Any],
    *,
    out_dir: str | Path,
    experiment_name: str,
    default_project: str,
    extra_config: dict[str, Any] | None = None,
    enabled: bool = True,
) -> WandbSession:
    wandb_cfg = cfg.get("wandb", {}) or {}
    if not enabled or not bool(wandb_cfg.get("enabled", False)):
        return WandbSession(enabled=False)

    try:
        import wandb
    except Exception as e:  # pragma: no cover - optional dependency
        print(f"[wandb] disabled because wandb import failed: {e}", flush=True)
        return WandbSession(enabled=False)

    mode = str(wandb_cfg.get("mode", "online"))
    if mode.lower() in {"disabled", "off", "false", "0"}:
        return WandbSession(enabled=False)

    config_payload = dict(cfg)
    if extra_config:
        config_payload.update(extra_config)

    init_kwargs = dict(
        project=wandb_cfg.get("project", default_project),
        entity=wandb_cfg.get("entity", None),
        name=wandb_cfg.get("name", experiment_name),
        group=wandb_cfg.get("group", None),
        tags=wandb_cfg.get("tags", None),
        notes=wandb_cfg.get("notes", None),
        dir=str(wandb_cfg.get("dir", out_dir)),
        config=config_payload,
        mode=mode,
        save_code=bool(wandb_cfg.get("save_code", False)),
    )
    init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}
    try:
        run = wandb.init(**init_kwargs)
    except Exception as e:  # pragma: no cover - runtime dependent
        print(f"[wandb] disabled because wandb.init failed: {e}", flush=True)
        return WandbSession(enabled=False)

    session = WandbSession(
        enabled=True,
        run=run,
        module=wandb,
        log_interval_steps=int(wandb_cfg.get("log_interval_steps", 0)),
    )
    session.set_summary("experiment_name", experiment_name)
    session.set_summary("output_dir", str(out_dir))
    return session



def maybe_build_wandb_image(session: WandbSession, image: Any, *, caption: str | None = None):
    if not session.enabled or session.module is None:
        return None
    try:
        return session.module.Image(image, caption=caption)
    except Exception:
        return None

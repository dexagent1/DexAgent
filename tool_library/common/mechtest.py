"""Shared plumbing for the reconstruction mechanism tests.

A mechanism test puts a constructed asset on trial with FORCES ONLY (nothing
is ever posed) and turns the measurement into a verdict. Every test in
`tool_library/verifiers/` that ends in `_test` uses the three helpers here:

* `resolve_built(built, rebuild)` accepts either the dict a constructor
  returned or the path of its `<name>_params.json` / `<name>_built.json` and
  gives back the built dict, so a test can be run long after the build.
* `run_headless(model, data, control, seconds)` steps the simulation without
  a renderer, calling `control(t, model, data)` once per step, and raises
  the moment MuJoCo goes unstable (a reset state looks finite again a few
  steps later, so the warning counter is what is checked).
* `report(ok, verdict, measured, **extra)` is the one result shape.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


def resolve_built(built: dict | str | Path, rebuild: Callable[..., dict] | None = None,
                  out_dir: str | Path | None = None) -> dict:
    """Return the built dict for `built`, which may be the dict itself, a
    `<name>_built.json`, or a `<name>_params.json` (then `rebuild` re-runs the
    constructor into `out_dir`, default: beside the json)."""
    if isinstance(built, dict):
        return built
    p = Path(built)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.name.endswith("_built.json"):
        return json.loads(p.read_text())
    if p.suffix == ".json":
        if rebuild is None:
            raise ValueError(f"{p} is a params file but no rebuild() was given")
        return rebuild(p, out_dir) if out_dir is not None else rebuild(p)
    raise ValueError(f"cannot resolve a built dict from {p}")


def run_headless(model, data, control: Callable[[float, Any, Any], Any] | None,
                 seconds: float, on_tick: Callable[[float, Any, Any], None] | None = None,
                 tick_every: float = 0.05) -> str:
    """Step `seconds` of sim time. `control(t, model, data)` runs every step
    (it should clear and re-apply its forces); `on_tick` runs every
    `tick_every` seconds of sim time for logging. Returns the last label."""
    import mujoco
    dt = float(model.opt.timestep)
    n = int(round(float(seconds) / dt))
    label = ""
    next_tick = 0.0
    for i in range(n):
        if control is not None:
            label = control(data.time, model, data) or label
        t = float(data.time)                     # MuJoCo's reset on a bad acceleration sets the time back to 0
        mujoco.mj_step(model, data)
        if int(data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number):
            raise RuntimeError(
                f"simulation went unstable at t={t:.3f}s "
                f"(mjWARN_BADQACC at dof {int(data.warning[mujoco.mjtWarning.mjWARN_BADQACC].lastinfo)}) — MuJoCo reset the state")
        if on_tick is not None and data.time >= next_tick:
            on_tick(data.time, model, data)
            next_tick += float(tick_every)
    import numpy as np
    if not np.isfinite(data.qpos).all():
        raise RuntimeError(f"simulation went non-finite at t={data.time:.3f}s")
    return str(label or "")


def jsonable(x):
    """Round floats / convert numpy scalars so a measurement dict dumps."""
    import numpy as np
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return round(float(x), 4)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def report(ok: bool, verdict: str, measured: dict | None = None,
           out_json: str | Path | None = None, **extra) -> dict:
    out = {"ok": bool(ok), "verdict": str(verdict),
           "measured": jsonable(measured or {})}
    out.update(jsonable(extra))
    if out_json:
        Path(out_json).write_text(json.dumps(out, indent=1))
    return out

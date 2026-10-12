"""Use every recorded integrator state for final collision audits."""
from pathlib import Path

import numpy as np


def collision_recording(trajectory):
    if not isinstance(trajectory, (str, Path)):
        return trajectory
    with np.load(str(trajectory), allow_pickle=False) as z:
        if "substep_qpos" not in z.files:
            return trajectory  # proposal diagnostics; the final recording gate rejects missing substeps
        required = {"initial_qpos", "substep_time_s"}
        if not required <= set(z.files):
            raise ValueError("incomplete collision recording: initial state or substep timestamps missing")
        times = np.r_[float(z["initial_time_s"]) if "initial_time_s" in z.files else 0.,
                      np.asarray(z["substep_time_s"], float)]
        Q = np.concatenate([np.asarray(z["initial_qpos"], float)[None], np.asarray(z["substep_qpos"], float)], axis=0)
        if len(times) != len(Q) or len(Q) < 2 or not np.isfinite(Q).all() or np.any(np.diff(times) <= 0):
            raise ValueError("collision recording has invalid states or timestamps")
        out = {"qpos": Q, "fps": 1. / float(np.median(np.diff(times))), "time_s": times,
               "recording_grid": "physics_substeps",
               "labels": [f"both: physical substep {i}" for i in range(len(Q))]}
        if "eq_active_start" in z.files and "substep_eq_active" in z.files:
            active = np.concatenate([z["eq_active_start"][None], z["substep_eq_active"]], axis=0)
            if len(active) != len(Q):
                raise ValueError("collision recording has incomplete mechanism states")
            out["eq_active"] = active
        return out

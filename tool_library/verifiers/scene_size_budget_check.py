"""Scene size budget check: does the model's memory arena hold what its contacts and constraints need?"""
from __future__ import annotations

TOOL = {
    "name": "scene_size_budget_check",
    "title": "Scene size budget check",
    "type": "verifier",
    "category": "Scene Compile Checks",
    "summary": "Measures the peak arena memory (contacts, constraint rows, solver workspace) the scene uses at its keyframe, during a "
               "short passive simulation and on the frames of a trajectory, and fails when it exceeds `headroom` of the arena the "
               "model will run with, or when contacts or constraints overflow.",
    "description": (
        "MuJoCo allocates contacts, constraint rows and solver workspace from one arena whose size is fixed at compile time "
        "(<size memory=...>, about 18 MB by default for the bench). A contact-dense state (a seated lid with a strap over its "
        "catch and a hand pressing on it, a finger buried among many small geoms) overflows it: mj_stackAlloc fails and the "
        "simulation stops, which is why the hand scenes compile with 256 MB. The check compiles the scene as its consumer "
        "runs it (with `arena_mb` when given) and evaluates the keyframe, a passive simulation of `seconds` (robot joints held, servos holding the "
        "keyframe pose) and, with `trajectory`, every `stride`-th recorded frame, and records MuJoCo's peak arena use and "
        "the largest numbers of contacts and constraint rows (with `execution_model=True` it measures the paired model the "
        "Stage 3 executor builds, whose hand-object pairs make contact states much denser). An arena exhausted on the way "
        "(MuJoCo raises), a peak above `headroom` of the arena, a contact/constraint overflow warning, or nconmax/njmax caps that the measured counts exceed fail, naming the "
        "<size memory> to set."
    ),
    "inputs": [
        {"name": "scene", "type": "str | dict", "description": "MJCF bench scene, or a hand-scene JSON path / dict."},
        {"name": "trajectory", "type": "str | None", "description": "Episode npz with qpos [T, nq] whose frames are evaluated."},
        {"name": "arena_mb", "type": "float | None", "description": "Arena the consumer compiles the scene with; default the scene's own."},
        {"name": "execution_model", "type": "bool", "description": "Measure the executor's paired model (hand-object contact pairs). Default False."},
        {"name": "seconds", "type": "float", "description": "Passive simulation from the keyframe (0 skips it). Default 0.5."},
        {"name": "stride", "type": "int", "description": "Evaluate every stride-th trajectory frame. Default 10."},
        {"name": "headroom", "type": "float", "description": "Largest fraction of the arena the peak may use. Default 0.8."},
        {"name": "robot_prefix", "type": "str", "description": "Name prefix of robot bodies. Default 'robot_'."},
    ],
    "outputs": [
        {"name": "report", "type": "dict",
         "description": "ok; issues; arena_mb (the budget); peak_mb; peak_frac; peak_at; max_contacts; max_efc; nconmax; njmax; "
                        "warnings; suggested_memory_mb."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["compile", "memory", "arena", "contacts", "mj_stackAlloc", "budget"],
    "status": "tested",
    "entrypoint": "scene_size_budget_check",
    "cli": "python -m tool_library.verifiers.scene_size_budget_check scene.xml [trajectory.npz]",
}

import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

from tool_library.common import compile_checks as C

def _build(scene, execution_model, arena_mb):
    """(the model as its consumer runs it, kind, initial MjData or None). Peak use is measured inside the real arena: MuJoCo
    records the high-water mark (maxuse_arena) and raises on exhaustion, so no larger measuring copy is needed."""
    if isinstance(scene, dict) or (isinstance(scene, (str, Path)) and str(scene).lower().endswith(".json")):
        spec = scene if isinstance(scene, dict) else json.loads(Path(scene).read_text())
        from tool_library.common import hand as H
        with C.capture_mj_warnings():
            m, d, rig, ids = H.open_scene(spec)
        return m, "hand_scene", d
    p = C.scene_path(scene)
    if p is None:
        raise ValueError("scene must be an MJCF path or a hand-scene JSON / dict")
    if execution_model:
        if arena_mb is not None:
            raise ValueError("arena_mb cannot be combined with execution_model (the executor fixes its own arena)")
        from tool_library.common import episode as E
        with C.capture_mj_warnings():
            real, _ = E.build_execution_model(str(p))
        return real, "execution_model", None
    model, err = C.compile_scene(p)
    if model is None:
        return None, err, None
    if arena_mb is not None:
        spec = mujoco.MjSpec.from_file(str(p))
        spec.memory = int(float(arena_mb) * 2 ** 20)
        with C.capture_mj_warnings():
            model = spec.compile()
    return model, "mjcf", None


def scene_size_budget_check(scene, trajectory=None, arena_mb=None, execution_model=False, seconds=0.5, stride=10, headroom=0.8,
                            robot_prefix=C.ROBOT_PREFIX):
    if not 0 < float(headroom) <= 1:
        raise ValueError("headroom must be in (0, 1]")
    if int(stride) < 1:
        raise ValueError("stride must be >= 1")
    meas, kind, d0 = _build(scene, execution_model, arena_mb)
    if meas is None:
        return C.compile_failure("fix the error MuJoCo names", kind)
    budget = int(meas.narena)
    data = mujoco.MjData(meas)
    if d0 is not None:
        data.qpos[:] = d0.qpos
    elif meas.nkey:
        mujoco.mj_resetDataKeyframe(meas, data, 0)
    peak = {"bytes": 0, "at": None}
    counts = {"con": 0, "efc": 0}
    msgs = []
    overflow = None
    cur = {"at": "keyframe"}

    def note(where):
        if int(data.maxuse_arena) > peak["bytes"]:
            peak["bytes"], peak["at"] = int(data.maxuse_arena), where
        counts["con"] = max(counts["con"], int(data.ncon))
        counts["efc"] = max(counts["efc"], int(data.nefc))

    with C.capture_mj_warnings(msgs):
        try:
            mujoco.mj_forward(meas, data)
            note("keyframe")
            if seconds and seconds > 0:
                for s in C.servo_actuators(meas):
                    data.ctrl[s["act"]] = data.qpos[s["qadr"]]
                qa, da = C.robot_kinematic_dofs(meas, robot_prefix)
                hold = data.qpos[qa].copy()
                for i in range(int(round(float(seconds) / meas.opt.timestep))):
                    cur["at"] = f"passive t={(i + 1) * meas.opt.timestep:.3f}s"
                    mujoco.mj_step(meas, data)
                    if len(qa):
                        data.qpos[qa] = hold
                        data.qvel[da] = 0.0
                    note(f"passive t={(i + 1) * meas.opt.timestep:.3f}s")
            if trajectory is not None:
                z = np.load(str(trajectory), allow_pickle=True)
                Q = np.asarray(z["qpos"] if "qpos" in z.files else z["Q"], float)
                if Q.shape[1] != meas.nq:
                    raise ValueError(f"trajectory has {Q.shape[1]} qpos columns, the scene has nq={meas.nq}")
                for t in list(range(0, len(Q), int(stride))) + [len(Q) - 1]:
                    cur["at"] = f"frame {t}"
                    data.qpos[:] = Q[t]
                    data.qvel[:] = 0.0
                    mujoco.mj_forward(meas, data)
                    note(f"frame {t}")
        except mujoco.FatalError as exc:
            overflow = str(exc)
            peak["at"] = cur["at"]
    warns = C.warning_counts(data)
    issues = []
    peak_mb, budget_mb = peak["bytes"] / 2 ** 20, budget / 2 ** 20
    frac = peak["bytes"] / budget if budget else float("inf")
    suggest = int(2 ** math.ceil(math.log2(max(peak["bytes"] / headroom / 2 ** 20, 1.0))))
    if overflow:
        issues.append(f"the {budget_mb:.1f} MB arena overflowed at {cur['at']} ({overflow[:120]}): the simulation "
                      f"stops there; set <size memory=\"{max(suggest, 2 * int(budget_mb) + 1)}M\"/> or more, or simplify the colliding geometry")
    elif frac > headroom:
        issues.append(f"peak arena use {peak_mb:.1f} MB ({peak['at']}) is {100 * frac:.0f}% of the {budget_mb:.1f} MB arena the model runs "
                      f"with (> {100 * headroom:.0f}%): a denser contact state stops the simulation with mj_stackAlloc; set "
                      f"<size memory=\"{suggest}M\"/> (or compile with that memory)")
    for w in ("CONTACTFULL", "CNSTRFULL"):
        if warns.get(w):
            issues.append(f"MuJoCo reported {w} ({warns[w]}x): contacts or constraint rows were dropped; raise nconmax/njmax or the memory")
    ncm, njm = int(getattr(meas, "nconmax", -1)), int(getattr(meas, "njmax", -1))
    if ncm > 0 and counts["con"] >= ncm:
        issues.append(f"nconmax={ncm} but {counts['con']} contacts were needed: contacts beyond the cap are dropped silently; remove the cap")
    if njm > 0 and counts["efc"] >= njm:
        issues.append(f"njmax={njm} but {counts['efc']} constraint rows were needed: remove the cap")
    return C.jsonable({"ok": not issues, "compiled": True, "kind": kind, "issues": issues, "arena_mb": budget_mb, "peak_mb": peak_mb,
                       "peak_frac": frac, "peak_at": peak["at"], "max_contacts": counts["con"], "max_efc": counts["efc"],
                       "nconmax": ncm, "njmax": njm, "warnings": warns, "suggested_memory_mb": suggest,
                       "mj_messages": [m.strip()[:160] for m in msgs[:3]]})


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    rep = scene_size_budget_check(argv[0], trajectory=argv[1] if len(argv) > 1 else None)
    print(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

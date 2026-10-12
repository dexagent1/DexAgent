"""Simulation stability check: does the scene simulate passively from its keyframe without blowing up?"""
from __future__ import annotations

TOOL = {
    "name": "simulation_stability_check",
    "title": "Simulation stability check",
    "type": "verifier",
    "category": "Scene Compile Checks",
    "summary": "A passive simulation of a few seconds from keyframe 0 (robot arms held, hand servos holding the keyframe pose) stays "
               "finite, raises no BADQACC/BADQVEL/BADQPOS warning, keeps every object below explosive speeds, and never gains "
               "mechanical energy beyond a small tolerance.",
    "description": (
        "MuJoCo answers an unstable scene by printing 'Nan, Inf or huge value in QACC ... The simulation is unstable' and "
        "silently resetting the state, after which every later measurement describes a different scene. The check steps the "
        "scene for `seconds` from keyframe `key` with the robot's kinematically driven joints (the arms) held at the "
        "keyframe and, by default, every position servo commanded to its joint's keyframe position (`ctrl='hold'`; "
        "`ctrl='key'` uses the keyframe's own ctrl). It records MuJoCo's warning counters, whether the state stays finite, "
        "the largest linear and angular speed of any non-robot body, and the mechanical energy of the passive non-robot "
        "bodies (kinetic + gravitational + joint springs; objects an actuator drives are left out, since actuators do work) "
        "relative to the start. A passive scene can only lose energy, so a gain "
        "above `energy_gain_j` means the solver pumped it in (a joint or weld violated at the start snapping into place, a "
        "contact too stiff for the timestep, an unstable servo). Distinct from settle_stability_check (where objects end "
        "up): this asks whether the physics itself holds together, everywhere and for the whole run."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF file, XML string, or compiled model."},
        {"name": "seconds", "type": "float", "description": "Simulated time. Default 3.0."},
        {"name": "key", "type": "int | str", "description": "Keyframe to start from. Default 0 (qpos0 without keyframes)."},
        {"name": "ctrl", "type": "str", "description": "'hold' (servos hold the keyframe joint positions) or 'key' (the keyframe's ctrl). Default 'hold'."},
        {"name": "hold_robot", "type": "bool", "description": "Hold the robot joints that no actuator drives at the keyframe. Default True."},
        {"name": "max_speed_mps", "type": "float", "description": "Largest linear speed of a non-robot body. Default 5.0."},
        {"name": "max_ang_speed", "type": "float", "description": "Largest angular speed of a non-robot body, rad/s. Default 200."},
        {"name": "energy_gain_j", "type": "float", "description": "Largest mechanical energy the non-robot bodies may gain, joules. Default 0.02."},
        {"name": "robot_prefix", "type": "str", "description": "Name prefix of robot bodies. Default 'robot_'."},
    ],
    "outputs": [
        {"name": "report", "type": "dict",
         "description": "ok; issues; finite; warnings {name: count}; first_warning_s; max_speed_mps {body, value, t}; max_ang_speed; "
                        "energy {start_j, max_gain_j, t_max_gain, end_j, n_passive_bodies}; mj_messages (MuJoCo's own warning text); seconds."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["compile", "physics", "stability", "badqacc", "energy"],
    "status": "tested",
    "entrypoint": "simulation_stability_check",
    "cli": "python -m tool_library.verifiers.simulation_stability_check scene.xml",
}

import json
import sys

import mujoco
import numpy as np

from tool_library.common import compile_checks as C

_BAD = ("BADQACC", "BADQVEL", "BADQPOS")


def _energy(model, data, bodies, dofs, spring_joints):
    """Kinetic + gravitational + joint-spring energy of the given bodies / dofs (J)."""
    v = np.zeros(model.nv)
    v[dofs] = data.qvel[dofs]
    Mv = np.zeros(model.nv)
    mujoco.mj_mulM(model, data, Mv, v)
    ke = 0.5 * float(v @ Mv)
    g = -float(model.opt.gravity[2])
    pe = sum(float(model.body_mass[b]) * g * float(data.xipos[b][2]) for b in bodies)
    for j in spring_joints:
        a = int(model.jnt_qposadr[j])
        pe += 0.5 * float(model.jnt_stiffness[j]) * (float(data.qpos[a]) - float(model.qpos_spring[a])) ** 2
    return ke + pe


def simulation_stability_check(scene, seconds=3.0, key=0, ctrl="hold", hold_robot=True, max_speed_mps=5.0, max_ang_speed=200.0,
                               energy_gain_j=0.02, robot_prefix=C.ROBOT_PREFIX):
    C.require_scene(scene)
    if ctrl not in ("hold", "key"):
        raise ValueError("ctrl must be 'hold' or 'key'")
    if not seconds or float(seconds) <= 0:
        raise ValueError("seconds must be > 0")
    model, err = C.compile_scene(scene)
    if model is None:
        return C.compile_failure("fix the error MuJoCo names", err)
    data = C.key_data(model, key)
    if ctrl == "hold":
        for s in C.servo_actuators(model):
            data.ctrl[s["act"]] = data.qpos[s["qadr"]]
    qa, da = C.robot_kinematic_dofs(model, robot_prefix) if hold_robot else (np.array([], int), np.array([], int))
    q_hold = data.qpos[qa].copy()
    robot = C.robot_body_ids(model, robot_prefix)
    bodies = [b for b in range(1, model.nbody) if b not in robot and C.top_body(model, b) not in robot]
    # actuators do work on what they drive (a servo pulling its joint, a magnet pulling a ring): the energy balance is
    # taken over the passive objects only, speeds over all non-robot bodies
    driven_tops = set()
    for a in range(model.nu):
        t, i = int(model.actuator_trntype[a]), int(model.actuator_trnid[a][0])
        if t in (int(mujoco.mjtTrn.mjTRN_JOINT), int(mujoco.mjtTrn.mjTRN_JOINTINPARENT)):
            driven_tops.add(C.top_body(model, int(model.jnt_bodyid[i])))
        elif t == int(mujoco.mjtTrn.mjTRN_BODY):
            driven_tops.add(C.top_body(model, i))
        elif t == int(mujoco.mjtTrn.mjTRN_SITE):
            driven_tops.add(C.top_body(model, int(model.site_bodyid[i])))
    passive = [b for b in bodies if C.top_body(model, b) not in driven_tops]
    dofs = [d for d in range(model.nv) if int(model.dof_bodyid[d]) in set(passive)]
    springs = [j for j in range(model.njnt) if int(model.jnt_bodyid[j]) in set(passive) and float(model.jnt_stiffness[j]) > 0
               and int(model.jnt_type[j]) in (C.HINGE, C.SLIDE)]
    e0 = _energy(model, data, passive, dofs, springs)
    steps = int(round(float(seconds) / model.opt.timestep))
    msgs = []
    first_warn, finite = None, True
    vmax, wmax, gain = (0.0, None, 0.0), (0.0, None, 0.0), (0.0, 0.0)
    every = max(1, int(round(0.01 / model.opt.timestep)))        # sample speeds and energy every 10 ms
    with C.capture_mj_warnings(msgs):
        for i in range(steps):
            mujoco.mj_step(model, data)
            if len(qa):
                data.qpos[qa] = q_hold
                data.qvel[da] = 0.0
            w = C.warning_counts(data)
            if first_warn is None and any(k in w for k in _BAD):
                first_warn = (i + 1) * model.opt.timestep
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                finite = False
                break
            if i % every == 0 or i == steps - 1:
                vel6 = np.zeros(6)
                for b in bodies:
                    # the body's own velocity (cvel's linear part is taken at its tree's centre of mass)
                    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, int(b), vel6, 0)
                    lin = float(np.linalg.norm(vel6[3:]))
                    ang = float(np.linalg.norm(vel6[:3]))
                    if lin > vmax[0]:
                        vmax = (lin, b, (i + 1) * model.opt.timestep)
                    if ang > wmax[0]:
                        wmax = (ang, b, (i + 1) * model.opt.timestep)
                e = _energy(model, data, passive, dofs, springs)
                if e - e0 > gain[0]:
                    gain = (e - e0, (i + 1) * model.opt.timestep)
    warns = C.warning_counts(data)
    e_end = _energy(model, data, passive, dofs, springs) if finite else None
    issues = []
    bad = {k: v for k, v in warns.items() if k in _BAD}
    if not finite:
        issues.append(f"the state became non-finite at t = {(i + 1) * model.opt.timestep:.3f} s: the simulation diverged; look for an "
                      "equality or joint violated at the keyframe, a contact stiffer than the timestep allows, or a massless moving part")
    if bad:
        txt = next((m for m in msgs if "unstable" in m or "QACC" in m), "")
        issues.append(f"MuJoCo flagged an unstable simulation ({bad}) first at t = {first_warn:.3f} s ({txt.strip()[:120]}): MuJoCo resets "
                      "the state on this warning, so every later measurement is of a different scene; fix the instability (a part "
                      "snapping onto a violated equality, a too-stiff contact or servo, a light body with large damping)")
    if vmax[0] > max_speed_mps:
        issues.append(f"body '{C.body_name(model, vmax[1])}' reaches {vmax[0]:.2f} m/s at t = {vmax[2]:.2f} s in a passive scene "
                      f"(> {max_speed_mps} m/s): it is being thrown, not resting")
    if wmax[0] > max_ang_speed:
        issues.append(f"body '{C.body_name(model, wmax[1])}' spins at {wmax[0]:.0f} rad/s at t = {wmax[2]:.2f} s (> {max_ang_speed}): "
                      "an explosive rotation (often a part twisted onto a violated weld)")
    if gain[0] > energy_gain_j:
        issues.append(f"the non-robot bodies gain {gain[0]:.3f} J of mechanical energy by t = {gain[1]:.2f} s (> {energy_gain_j} J): a "
                      "passive scene can only lose energy, so the solver pumped it in")
    return C.jsonable({"ok": not issues, "compiled": True, "issues": issues, "finite": finite, "warnings": warns,
                       "first_warning_s": first_warn,
                       "max_speed_mps": {"body": C.body_name(model, vmax[1]) if vmax[1] is not None else None, "value": vmax[0], "t": vmax[2]},
                       "max_ang_speed": {"body": C.body_name(model, wmax[1]) if wmax[1] is not None else None, "value": wmax[0], "t": wmax[2]},
                       "energy": {"start_j": e0, "max_gain_j": gain[0], "t_max_gain": gain[1], "end_j": e_end,
                                  "n_passive_bodies": len(passive)},
                       "mj_messages": [m.strip()[:200] for m in msgs[:5]], "seconds": float(seconds), "ctrl": ctrl})


def main(argv=None):
    rep = simulation_stability_check((sys.argv[1:] if argv is None else argv)[0])
    print(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

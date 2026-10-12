"""Actuator mapping check: does every actuator drive what it should, with consistent ranges, in the order the code expects?"""
from __future__ import annotations

TOOL = {
    "name": "actuator_mapping_check",
    "title": "Actuator mapping check",
    "type": "verifier",
    "category": "Scene Compile Checks",
    "summary": "Every actuator drives an existing joint/site/body/tendon with valid ctrl and force ranges (a position servo's ctrlrange "
               "inside its joint's range, no dead actuator), the robot's actuators come first and contiguously (matching the setup's "
               "list when given), and every non-robot actuator is listed with its ctrl slot.",
    "description": (
        "Stage 3 and Stage 4 address the robot through ctrl: ctrl[:56] are the two hands' servos on the bench. An object asset "
        "that brings its own actuators (EgoDex magnetic rings add three adhesion actuators after the robot's 56) changes nu, "
        "and code that sizes or slices ctrl as robot-only breaks; an actuator inserted before or between the robot's shifts "
        "every robot command. The check compiles the scene, assigns each actuator to the robot or not by its target's name "
        "(joint/site/body/tendon carrying the robot prefix), and fails: robot actuators that are not the first block of ctrl; "
        "a robot actuator list that differs in count, names or order from the `reference` scene's (setup/scene.xml) or from "
        "`expected_robot_nu`; ctrllimited with ctrlrange[0] >= ctrlrange[1] and forcelimited with an empty forcerange; a "
        "position servo whose ctrlrange reaches beyond its limited joint's range by more than `range_tol` (it drives the joint "
        "into its stop); an actuator with zero gain and zero bias (it does nothing); and, when `allow_non_robot` is False, any "
        "non-robot actuator. Non-robot actuators are always listed with their ctrl indices and targets."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF file, XML string, or compiled model."},
        {"name": "reference", "type": "str | None", "description": "Scene whose robot actuators (names, order) must be reproduced, e.g. setup/scene.xml."},
        {"name": "expected_robot_nu", "type": "int | None", "description": "Number of robot actuators expected (setup.yaml robot.actuators: 56)."},
        {"name": "allow_non_robot", "type": "bool", "description": "Accept actuators that drive non-robot parts (listed either way). Default True."},
        {"name": "range_tol", "type": "float", "description": "How far a servo's ctrlrange may exceed its joint range (rad or m). Default 0.01."},
        {"name": "robot_prefix", "type": "str", "description": "Name prefix of robot elements. Default 'robot_'."},
    ],
    "outputs": [
        {"name": "report", "type": "dict",
         "description": "ok; issues; nu; robot_nu; robot_ctrl_slice [first, last+1]; non_robot: [{index, name, trntype, target}]; "
                        "actuators: [{index, name, robot, trntype, target, ctrlrange, joint_range, problem}] (problems first); worst."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["compile", "actuator", "ctrl", "servo", "layout"],
    "status": "tested",
    "entrypoint": "actuator_mapping_check",
    "cli": "python -m tool_library.verifiers.actuator_mapping_check scene.xml [reference.xml]",
}

import json
import sys

import mujoco
import numpy as np

from tool_library.common import compile_checks as C

_TRN = {int(v): k.replace("mjTRN_", "").lower() for k, v in mujoco.mjtTrn.__members__.items() if k != "mjTRN_UNDEFINED"}


def _target(model, a):
    t, i = int(model.actuator_trntype[a]), int(model.actuator_trnid[a][0])
    if t in (int(mujoco.mjtTrn.mjTRN_JOINT), int(mujoco.mjtTrn.mjTRN_JOINTINPARENT)):
        return C.joint_name(model, i), C.body_name(model, int(model.jnt_bodyid[i]))
    if t == int(mujoco.mjtTrn.mjTRN_SITE):
        return C.name_of(model, mujoco.mjtObj.mjOBJ_SITE, i, "site"), C.body_name(model, int(model.site_bodyid[i]))
    if t == int(mujoco.mjtTrn.mjTRN_BODY):
        return C.body_name(model, i), C.body_name(model, i)
    if t == int(mujoco.mjtTrn.mjTRN_TENDON):
        nm = C.name_of(model, mujoco.mjtObj.mjOBJ_TENDON, i, "tendon")
        return nm, nm
    return str(i), C.actuator_name(model, a)


def _robot_actuators(model, robot_prefix):
    return [a for a in range(model.nu) if C.is_robot(_target(model, a)[1], robot_prefix)]


def actuator_mapping_check(scene, reference=None, expected_robot_nu=None, allow_non_robot=True, range_tol=0.01, robot_prefix=C.ROBOT_PREFIX):
    C.require_scene(scene)
    model, err = C.compile_scene(scene)
    if model is None:
        return C.compile_failure("an actuator whose joint/site/body does not exist, or an invalid range: fix the actuator named in the "
                                 "message", err, actuators=[])
    servo = {s["act"]: s for s in C.servo_actuators(model)}
    rows, issues = [], []
    robot = set(_robot_actuators(model, robot_prefix))
    for a in range(model.nu):
        tgt, owner = _target(model, a)
        probs = []
        cr = [float(x) for x in model.actuator_ctrlrange[a]]
        fr = [float(x) for x in model.actuator_forcerange[a]]
        if bool(model.actuator_ctrllimited[a]) and cr[0] >= cr[1]:
            probs.append(f"ctrlrange {cr} is empty")
        if bool(model.actuator_forcelimited[a]) and fr[0] >= fr[1]:
            probs.append(f"forcerange {fr} is empty")
        jr = None
        if a in servo:
            j = servo[a]["joint"]
            if bool(model.jnt_limited[j]):
                jr = [float(x) for x in model.jnt_range[j]]
                if bool(model.actuator_ctrllimited[a]) and (cr[0] < jr[0] - range_tol or cr[1] > jr[1] + range_tol):
                    probs.append(f"position servo ctrlrange {np.round(cr, 4).tolist()} reaches beyond its joint's range "
                                 f"{np.round(jr, 4).tolist()}: a command outside the range drives the joint into its stop")
        gain = np.asarray(model.actuator_gainprm[a][:3], float)
        bias = np.asarray(model.actuator_biasprm[a][:3], float)
        if int(model.actuator_gaintype[a]) == int(mujoco.mjtGain.mjGAIN_FIXED) and not gain.any() and \
                (int(model.actuator_biastype[a]) == int(mujoco.mjtBias.mjBIAS_NONE) or not bias.any()):
            probs.append("zero gain and zero bias: the actuator produces no force")
        rows.append({"index": a, "name": C.actuator_name(model, a), "robot": a in robot,
                     "trntype": _TRN.get(int(model.actuator_trntype[a]), str(int(model.actuator_trntype[a]))), "target": tgt,
                     "ctrlrange": cr if bool(model.actuator_ctrllimited[a]) else None, "joint_range": jr, "problem": "; ".join(probs) or None})
        for p in probs:
            issues.append(f"actuator {a} '{C.actuator_name(model, a)}' ({rows[-1]['trntype']} '{tgt}'): {p}")
    r_idx = sorted(robot)
    n_r = len(r_idx)
    if r_idx and r_idx != list(range(n_r)):
        first_foreign = next(i for i in range(model.nu) if i not in robot)
        issues.append(f"the robot's {n_r} actuators are not the first block of ctrl: non-robot actuator {first_foreign} "
                      f"'{C.actuator_name(model, first_foreign)}' sits before robot actuator {r_idx[-1]}, so every robot command "
                      "after it is shifted; attach object assets (and their actuators) after the robot")
    non_robot = [{"index": r["index"], "name": r["name"], "trntype": r["trntype"], "target": r["target"]} for r in rows if not r["robot"]]
    notes = []
    if non_robot:
        notes.append(f"ctrl has {model.nu} entries: the robot's {n_r} then {len(non_robot)} non-robot actuators "
                     f"{[x['name'] for x in non_robot][:5]} at {[x['index'] for x in non_robot][:5]}; code that sizes ctrl as robot-only "
                     f"must slice ctrl[:{n_r}]")
        if not allow_non_robot:
            issues.append(f"{len(non_robot)} actuators drive non-robot parts ({[x['name'] for x in non_robot][:5]} at ctrl "
                          f"{[x['index'] for x in non_robot][:5]}): this consumer expects a robot-only ctrl of {n_r}; slice ctrl[:{n_r}] "
                          "or give the object a mechanism that needs no actuator")
    if expected_robot_nu is not None and n_r != int(expected_robot_nu):
        issues.append(f"the scene has {n_r} robot actuators, {int(expected_robot_nu)} expected: robot commands will not line up")
    ref_rep = None
    if reference is not None:
        rmodel, rerr = C.compile_scene(reference)
        if rmodel is None:
            raise ValueError(f"reference does not compile: {rerr}")
        rnames = [C.actuator_name(rmodel, a) for a in _robot_actuators(rmodel, robot_prefix)]
        snames = [C.actuator_name(model, a) for a in r_idx]
        ref_rep = {"n_reference": len(rnames), "n_scene": len(snames)}
        if rnames != snames:
            diff = next((i for i in range(min(len(rnames), len(snames))) if rnames[i] != snames[i]), min(len(rnames), len(snames)))
            ref_rep["first_difference"] = diff
            issues.append(f"the robot actuators differ from the reference's ({len(snames)} vs {len(rnames)}; first difference at index {diff}: "
                          f"{snames[diff] if diff < len(snames) else None!r} vs {rnames[diff] if diff < len(rnames) else None!r}): a "
                          "trajectory's ctrl recorded on the setup would drive the wrong joints")
    rows.sort(key=lambda r: r["problem"] is None)
    return C.jsonable({"ok": not issues, "compiled": True, "issues": issues, "notes": notes, "nu": int(model.nu), "robot_nu": n_r,
                       "robot_ctrl_slice": [0, n_r] if r_idx == list(range(n_r)) else None, "non_robot": non_robot,
                       "reference": ref_rep, "actuators": rows[:80], "worst": rows[0] if rows and rows[0]["problem"] else None})


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    rep = actuator_mapping_check(argv[0], reference=argv[1] if len(argv) > 1 else None)
    print(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

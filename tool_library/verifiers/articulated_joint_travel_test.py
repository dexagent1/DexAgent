"""Articulated joint travel test: do the joints of a ported or constructed
articulated object actually travel their range under a hand-sized force?

A static check can read a joint's range off the model; this one pulls on
the part. For every slide/hinge joint it applies a force along the joint
axis (or a torque about it) on the child body, at its handle site when the
asset has one, ramps it to `force_N` / `torque_Nm`, and measures how far
the joint went toward its upper limit; then it reverses and measures the
return toward the lower limit. A drawer that reaches 80% of its travel both
ways under 10 N is a drawer; one that moves 3 mm is a decoration, and one
that drags the whole cabinet along has no carcass mass or no base contact.

    from tool_library.verifiers.articulated_joint_travel_test import test_articulated_joint_travel
    r = test_articulated_joint_travel(info)        # info = port_urdf_asset(...) or an MJCF path
"""
from __future__ import annotations

TOOL = {
    "name": "articulated_joint_travel_test",
    "title": "Articulated joint travel test",
    "type": "verifier",
    "category": "Reconstruction Mechanism Tests",
    "summary": "Pulls each slide/hinge joint of an articulated asset through its range with a "
               "capped force on the moving part, both ways, and reports the fraction of travel reached.",
    "description": (
        "Loads the MJCF (or the info sheet port_urdf_asset wrote) and, for each requested "
        "joint (default: every limited slide or hinge), runs two trials from the initial state: "
        "a force of up to force_N along the joint axis in world frame (torque of up to "
        "torque_Nm about it for a hinge) applied with xfrc_applied on the child body, at the "
        "handle site when the info sheet names one, ramped over 0.5 s and held for `seconds`; "
        "then the opposite force for the same time. It records the travel fraction reached "
        "toward each limit, the time to 80% of the range, and the displacement of the root "
        "body named by the info sheet's root_body (a free-based cabinet must not slide more "
        "than root_slip_m; None when no root body is named, so not measured). With hold_root "
        "(default) a free root is welded to the world for the test, as a hand or the table "
        "holds it. Passes when every joint reaches min_fraction of its range in both "
        "directions and the root, when measured, stays put. "
        "Hinges whose range is unlimited are measured against one full turn."
    ),
    "inputs": [
        {"name": "scene", "type": "str | dict",
         "description": "MJCF path, or the info dict/json from port_urdf_asset (gives handle sites and joint names)."},
        {"name": "joints", "type": "list[str] | None", "description": "Joint names to test. Default: every limited slide/hinge."},
        {"name": "force_N", "type": "float", "description": "Cap on the pull for slides. Default 10."},
        {"name": "torque_Nm", "type": "float", "description": "Cap on the torque for hinges. Default 1.0."},
        {"name": "seconds", "type": "float", "description": "Hold time per direction. Default 3."},
        {"name": "min_fraction", "type": "float", "description": "Fraction of the range to reach each way. Default 0.8."},
        {"name": "root_slip_m", "type": "float", "description": "Allowed root displacement (measured only for the info sheet's root_body). Default 0.005."},
        {"name": "out_json", "type": "str | None", "description": "Write the report here."},
        {"name": "hold_root", "type": "bool", "description": "Weld a free root body to the world for the test (a hand or the table holds a drawer unit or a box while its part is moved). Default True."},
    ],
    "outputs": [
        {"name": "ok", "type": "bool", "description": "Every tested joint travelled both ways and the root (when measured) stayed put."},
        {"name": "verdict", "type": "str", "description": "One sentence naming the joints that failed, if any."},
        {"name": "measured", "type": "dict",
         "description": "per joint: type, range, forward {fraction, t80_s, q_end}, back {fraction, t80_s, q_end}, root_slip_m (None when not measured), ok."},
    ],
    "status": "tested",
    "entrypoint": "articulated_joint_travel_test",
    "requirements": ["mujoco", "numpy"],
    "simulators": ["mujoco"],
    "tags": ["mechanism", "articulated", "drawer", "hinge", "urdf", "force-driven"],
    "used_in": ["Place Toy in Drawer"],
    "cli": "python -m tool_library.verifiers.articulated_joint_travel_test <asset.xml | info.json>",
}


def test_articulated_joint_travel(scene, joints=None, force_N: float = 10.0,
                                  torque_Nm: float = 1.0, seconds: float = 3.0,
                                  min_fraction: float = 0.8, root_slip_m: float = 0.005,
                                  out_json=None, hold_root: bool = True) -> dict:
    """`hold_root`: when the asset's root is a free body, weld it to the world for the test (a hand or the
    table holds a drawer unit or a box while its part is moved); otherwise the capped force topples it."""
    import json
    from pathlib import Path
    import mujoco
    import numpy as np
    from tool_library.common.mjcf import load, clear_forces, apply_wrench
    from tool_library.common.mechtest import run_headless, report

    info = None
    if isinstance(scene, dict):
        info, xml = scene, scene["xml"]
    elif str(scene).endswith(".json"):
        info = json.loads(Path(scene).read_text())
        xml = info["xml"]
    else:
        xml = str(scene)
    m = load(xml)
    if hold_root:
        free = [j for j in range(m.njnt) if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)]
        if free:
            spec = mujoco.MjSpec.from_file(str(xml)) if not str(xml).lstrip().startswith("<") else mujoco.MjSpec.from_string(str(xml))
            held = 0
            for b in spec.bodies:
                if any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in b.joints):
                    w = spec.add_equality(); w.type = mujoco.mjtEq.mjEQ_WELD; w.name1 = b.name; w.name2 = "world"
                    w.objtype = mujoco.mjtObj.mjOBJ_BODY; held += 1
                    # anchor at the origin, relpose quat all zero = hold the pose it compiles in (the spec's default
                    # relpose is (0, 1, 0): it yanked the root a metre and the run diverged)
                    w.data = [0.0] * 10 + [1.0]
            if held:
                if not any(g.type == mujoco.mjtGeom.mjGEOM_PLANE for g in spec.worldbody.geoms):
                    spec.worldbody.add_geom(name="test_floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.1], rgba=[0.8, 0.8, 0.8, 1])
                m = spec.compile()
    handles = {}
    if info:
        for link, h in (info.get("handles") or {}).items():
            handles[f"{info['name']}_{link}"] = h["site"]
    root_bid = m.body(info["root_body"]).id if info and "root_body" in info else None

    cands = []
    for j in range(m.njnt):
        jt = int(m.jnt_type[j])
        if jt not in (int(mujoco.mjtJoint.mjJNT_SLIDE), int(mujoco.mjtJoint.mjJNT_HINGE)):
            continue
        name = m.joint(j).name
        if joints is not None and name not in joints:
            continue
        cands.append(j)
    if joints is not None:
        missing = [n for n in joints if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) < 0]
        if missing:
            raise KeyError(f"joints not in the model: {missing}")
    if not cands:
        return report(False, "NO JOINTS — the model has no slide or hinge joint to test", {},
                      out_json=out_json)

    results = {}
    for j in cands:
        name = m.joint(j).name
        is_slide = int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_SLIDE)
        qadr = int(m.jnt_qposadr[j])
        bid = int(m.jnt_bodyid[j])
        if m.jnt_limited[j]:
            lo, hi = float(m.jnt_range[j][0]), float(m.jnt_range[j][1])
        else:
            lo, hi = 0.0, (2 * np.pi if not is_slide else 1.0)
        span = max(hi - lo, 1e-9)
        site = handles.get(m.body(bid).name)
        site_id = m.site(site).id if site else None
        cap = float(force_N if is_slide else torque_Nm)
        per_dir = {}
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        q0 = float(d.qpos[qadr])
        root0 = d.xpos[root_bid].copy() if root_bid is not None else None
        for tag, sign, target in (("forward", +1.0, hi), ("back", -1.0, lo)):
            t_start = d.time
            log = {"t80": None, "best": float(d.qpos[qadr])}

            def control(t, mm, dd, sign=sign, log=log, t_start=t_start):
                clear_forces(dd)
                ramp = min(1.0, (t - t_start) / 0.5)
                R = dd.xmat[bid].reshape(3, 3)
                ax_w = R @ m.jnt_axis[j]
                if is_slide:
                    pt = (dd.site_xpos[site_id] if site_id is not None else dd.xipos[bid])
                    apply_wrench(dd, bid, force=sign * cap * ramp * ax_w, point=pt, com=dd.xipos[bid])
                else:
                    apply_wrench(dd, bid, torque=sign * cap * ramp * ax_w)
                q = float(dd.qpos[qadr])
                frac = (q - lo) / span if sign > 0 else (hi - q) / span
                if log["t80"] is None and frac >= 0.8:
                    log["t80"] = round(t - t_start, 3)
                log["best"] = q if (sign > 0 and q > log["best"]) or (sign < 0 and q < log["best"]) else log["best"]
                return tag

            run_headless(m, d, control, float(seconds))
            q_end = float(d.qpos[qadr])
            frac = (log["best"] - lo) / span if sign > 0 else (hi - log["best"]) / span
            per_dir[tag] = {"fraction": round(float(np.clip(frac, 0.0, 1.5)), 3),
                            "t80_s": log["t80"], "q_end": round(q_end, 5),
                            "q_best": round(float(log["best"]), 5)}
        slip = (float(np.linalg.norm(d.xpos[root_bid] - root0)) if root_bid is not None else None)
        ok_j = (per_dir["forward"]["fraction"] >= float(min_fraction)
                and per_dir["back"]["fraction"] >= float(min_fraction)
                and (slip is None or slip <= float(root_slip_m)))
        results[name] = {"type": "slide" if is_slide else "hinge", "range": [lo, hi],
                         "cap": cap, "applied_at": site or "body com", "q_start": q0,
                         **per_dir, "root_slip_m": None if slip is None else round(slip, 5), "ok": bool(ok_j)}
    bad = [n for n, r in results.items() if not r["ok"]]
    ok = not bad
    unit = lambda r: "N" if r["type"] == "slide" else "N*m"
    slip_txt = lambda s: "not measured" if s is None else f"{1000 * s:.1f} mm"
    slips = [r["root_slip_m"] for r in results.values() if r["root_slip_m"] is not None]
    verdict = (f"JOINTS OK — {len(results)} joint(s) each travelled >= {100*min_fraction:.0f}% of "
               f"their range both ways under "
               + ", ".join(sorted({f'{r["cap"]:g} {unit(r)}' for r in results.values()}))
               + (f"; root slip <= {1000*max(slips):.1f} mm" if slips else "; root slip not measured")
               if ok else
               "JOINTS FAILED — " + "; ".join(
                   f"{n}: forward {results[n]['forward']['fraction']:.2f}, back "
                   f"{results[n]['back']['fraction']:.2f}, root slip {slip_txt(results[n]['root_slip_m'])}"
                   for n in bad))
    return report(ok, verdict, results, out_json=out_json)


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(test_articulated_joint_travel(sys.argv[1]), indent=1))


articulated_joint_travel_test = test_articulated_joint_travel   # the library calls every tool by its own name

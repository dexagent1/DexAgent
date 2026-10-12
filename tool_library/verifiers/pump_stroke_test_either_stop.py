"""Force trial of a sprung push button or pump head: preload holds it, a finger force strokes it, it springs back, repeatedly."""
from __future__ import annotations

TOOL = {
    "name": "pump_stroke_test_either_stop",
    "title": "Pump stroke test (either rest stop)",
    "type": "verifier",
    "category": "Reconstruction Mechanism Tests",
    "summary": "Pushes a sprung slide joint (a spray pump head, a push button, a plunger) along its axis with a capped force: it must rest on its stop, resist a sub-preload push, follow its spring rate, bottom out under a finger force and spring back, for several cycles.",
    "description": (
        "Fork of pump_stroke_test that does not assume the rest stop is the upper limit: the rest stop is the limit the "
        "spring pushes towards (springref beyond it), and the press goes towards the other limit, so it accepts both "
        "joint conventions. Loads the asset (a built dict with `xml` and `joint`, or an MJCF path), removes its free joints so the parent is "
        "held, and finds the slide joint (`joint`, else the built's `joint`, else the only slide joint). From the model it "
        "reads the stroke (the joint range), the stiffness and the preload (stiffness x the distance from springref to the "
        "rest stop). It then applies a force on the moving body along the joint axis, from the rest stop towards the other "
        "stop: 0 N (must rest within rest_tol_frac of the stroke from the rest stop), half the preload (must not move: the "
        "preload is real), preload + half "
        "the spring travel force (displacement gives the measured stiffness, compared within stiffness_tol_pct), the full "
        "press force (must reach full_frac of the stroke) and release (must return to rest), repeated `cycles` times. The "
        "moving body's weight is included in the applied force. Reports every displacement and the verdict; never raises "
        "on a bad mechanism."
    ),
    "inputs": [
        {"name": "built", "type": "dict | path", "description": "Constructor output with 'xml' (and 'joint'), or an MJCF path."},
        {"name": "joint", "type": "str | None", "description": "Slide joint to test. Default: built['joint'] or the only slide joint."},
        {"name": "press_force_N", "type": "float | None", "description": "Full press force. Default preload + 1.5 x stiffness x stroke."},
        {"name": "cycles", "type": "int", "description": "Press/release cycles. Default 3."},
        {"name": "phase_s", "type": "float", "description": "Seconds per phase. Default 0.4."},
        {"name": "rest_tol_frac", "type": "float", "description": "Allowed rest offset as a fraction of the stroke. Default 0.15."},
        {"name": "full_frac", "type": "float", "description": "Fraction of the stroke the full press must reach. Default 0.9."},
        {"name": "stiffness_tol_pct", "type": "float", "description": "Allowed error of the measured spring rate. Default 35."},
        {"name": "out_json", "type": "path | None", "description": "Optional report file."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; joint; stroke_m; stiffness_N_per_m; preload_N; measured_stiffness_N_per_m; displacements per phase and cycle; press_force_N."},
    ],
    "status": "tested",
    "entrypoint": "pump_stroke_test_either_stop",
    "requirements": ["mujoco>=3.1", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["pump", "spray bottle", "button", "plunger", "spring", "slide joint", "mechanism test"],
    "used_in": [],
    "cli": "python -m tool_library.verifiers.pump_stroke_test_either_stop built.json",
}

import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")


def _load_without_free(xml_path):
    import mujoco
    xp = Path(xml_path)
    txt = xp.read_text()
    txt = re.sub(r"<freejoint[^>]*/>", "", txt)
    txt = re.sub(r"<joint[^>]*type=\"free\"[^>]*/>", "", txt)
    fd, tmp = tempfile.mkstemp(suffix=".xml", dir=str(xp.parent))
    os.close(fd)
    try:
        Path(tmp).write_text(txt)
        return mujoco.MjModel.from_xml_path(tmp)
    finally:
        os.unlink(tmp)


def pump_stroke_test_either_stop(built, joint=None, press_force_N=None, cycles=3, phase_s=0.4, rest_tol_frac=0.15, full_frac=0.9,
                     stiffness_tol_pct=35.0, out_json=None):
    import mujoco
    issues = []
    if isinstance(built, (str, Path)) and str(built).endswith(".json"):
        built = json.loads(Path(built).read_text())
    xml = built.get("xml") if isinstance(built, dict) else str(built)
    if not xml or not Path(xml).exists():
        raise ValueError(f"built: no MJCF found ({xml!r}); pass the constructor's output or an xml path")
    jname = joint or (built.get("joint") if isinstance(built, dict) else None)
    rep = {"ok": False, "issues": issues, "xml": str(xml)}
    try:
        m = _load_without_free(xml)
    except Exception as e:
        issues.append(f"the asset does not compile: {str(e)[:200]}")
        return rep
    slides = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE]
    if jname:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            issues.append(f"joint {jname!r} not found; slide joints: {[m.joint(j).name for j in slides]}")
            return rep
    elif len(slides) == 1:
        jid = slides[0]
    else:
        issues.append(f"need exactly one slide joint or a `joint` name; found {len(slides)}: build the moving part on a slide joint")
        return rep
    if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_SLIDE:
        issues.append(f"joint {m.joint(jid).name!r} is not a slide joint")
        return rep
    rep["joint"] = m.joint(jid).name
    if not m.jnt_limited[jid]:
        issues.append("the slide joint is unlimited: give it a range [-stroke, 0] so the stroke has stops")
        return rep
    lo, hi = (float(x) for x in m.jnt_range[jid])
    stroke = hi - lo
    k = float(m.jnt_stiffness[jid])
    qadr = int(m.jnt_qposadr[jid])
    sp = float(m.qpos_spring[qadr])
    if sp >= hi:
        rest_hi = True
    elif sp <= lo:
        rest_hi = False
    else:
        issues.append(f"the spring rest {sp:.4g} lies inside the range [{lo:.4g}, {hi:.4g}]: no preload holds the part on a stop; put springref beyond the rest stop")
        return rep
    preload = k * (sp - hi) if rest_hi else k * (lo - sp)
    rep.update(stroke_m=stroke, stiffness_N_per_m=k, preload_N=preload, rest_stop="upper" if rest_hi else "lower")
    if k <= 0:
        issues.append("the slide joint has no spring (stiffness 0): it would not return after a press; set the spring stiffness")
        return rep
    if stroke <= 0:
        issues.append("the joint range has no travel")
        return rep
    bid = int(m.jnt_bodyid[jid])
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    axis_w = d.xaxis[jid].copy()
    push = -axis_w if rest_hi else axis_w                       # world direction of the press
    weight = -float(m.body_subtreemass[bid]) * float(np.dot(m.opt.gravity, push))  # <0 when gravity helps the press
    full = float(press_force_N) if press_force_N is not None else preload + 1.5 * k * stroke
    nstep = max(1, int(round(float(phase_s) / m.opt.timestep)))

    def run(F):
        # F pushes from the rest stop towards the other stop (along `push`); gravity is already in the sim
        d.xfrc_applied[:] = 0
        d.xfrc_applied[bid, :3] = F * push
        for _ in range(nstep):
            mujoco.mj_step(m, d)
        q = float(d.qpos[qadr])
        if not np.isfinite(q):
            raise FloatingPointError("simulation diverged")
        return (hi - q) if rest_hi else (q - lo)   # displacement from the rest stop, >= 0 into the stroke

    out = {"rest": [], "sub_preload": [], "mid": [], "full": [], "release": []}
    meas_k = []
    f_mid = preload + 0.5 * k * stroke
    try:
        for c in range(int(cycles)):
            out["rest"].append(run(0.0))
            if preload > 0:
                out["sub_preload"].append(run(0.5 * preload))
            out["mid"].append(run(f_mid))
            out["full"].append(run(full))
            out["release"].append(run(0.0))
            dm = out["mid"][-1]
            f_eff = f_mid - weight
            if dm > 1e-6:
                meas_k.append((f_eff - preload) / dm)
    except FloatingPointError as e:
        issues.append(f"{e}: raise the armature or damping of the slide joint")
        rep["displacements_m"] = out
        return rep
    tol = float(rest_tol_frac) * stroke
    if max(out["rest"]) > tol:
        issues.append(f"at rest the part sits {max(out['rest'])*1000:.2f} mm off its rest stop (allowed {tol*1000:.2f}): raise the preload")
    if out["sub_preload"] and max(out["sub_preload"]) > tol:
        issues.append(f"half the preload already moves it {max(out['sub_preload'])*1000:.2f} mm: the preload is not holding")
    if min(out["full"]) < float(full_frac) * stroke:
        issues.append(f"a {full:.2f} N press reaches only {min(out['full'])*1000:.2f} of {stroke*1000:.2f} mm: lower the stiffness/preload or the friction")
    if max(out["release"]) > tol:
        issues.append(f"after release the part stays {max(out['release'])*1000:.2f} mm into its stroke: the spring does not return it")
    mk = float(np.median(meas_k)) if meas_k else None
    if mk is None or abs(mk - k) / k * 100 > float(stiffness_tol_pct):
        issues.append(f"measured spring rate {mk} N/m vs built {k:.1f} N/m (tolerance {stiffness_tol_pct}%)")
    rep.update(ok=not issues, measured_stiffness_N_per_m=mk, press_force_N=full, mid_force_N=f_mid, cycles=int(cycles),
               displacements_m={kk: [round(v, 6) for v in vv] for kk, vv in out.items()},
               verdict=("PUMP OK: rests preloaded, strokes fully under %.1f N, springs back" % full) if not issues else "PUMP FAILED")
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    r = pump_stroke_test_either_stop(sys.argv[1])
    print(json.dumps(r, indent=1))
    sys.exit(0 if r["ok"] else 1)

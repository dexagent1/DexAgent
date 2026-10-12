"""Force trial of a soft pad on a table: a palm press squashes it at its spring rate, friction holds it against a light drag, a strong drag slides it flat without lifting, and it recovers on release."""
from __future__ import annotations

TOOL = {
    "name": "pad_press_slide_test",
    "title": "Pad press-and-slide test",
    "type": "verifier",
    "category": "Reconstruction Mechanism Tests",
    "summary": "Lays a compliant pad (sponge, scrubber, eraser) on a floor, presses its top with a palm-like force, then drags it sideways below and above the friction limit: it must squash at its built stiffness, hold under the light drag, slide flat and in contact under the strong drag, and spring back on release.",
    "description": (
        "Loads the asset (built dict with `xml`, or an MJCF path), adds a floor plane with friction `floor_friction` and "
        "finds the free root body and the compliant slide joints in its subtree (the foam cells). Phases: settle; press "
        "(press_force_N spread evenly over the top bodies: the cells, or the root when there are none) and measure the "
        "mean squash and the whole-pad stiffness F / squash against built['stiffness_N_per_m'] or the sum of the cell "
        "stiffnesses; hold (keep the press, add a horizontal drag from a palm target moving at drag_speed_m_s (spring-damper coupling) capped at hold_frac x mu x normal load: the pad must move "
        "less than max_hold_slip_m); slide (the same servo capped at slide_frac x mu x load, i.e. a palm moving with the pad: the pad must travel at least min_slide_m "
        "while its slab stays within max_lift_m of the floor, tilts less than max_tilt_deg and keeps floor contact (or stays within 0.5 mm of its rest height) in "
        "min_contact_frac of the steps); release (all forces off: the squash must recover to within recover_frac of the "
        "cell travel). A rigid pad without compliant joints fails the squash check. Never raises on a bad pad; reports."
    ),
    "inputs": [
        {"name": "built", "type": "dict | path", "description": "Constructor output with 'xml', or an MJCF path."},
        {"name": "press_force_N", "type": "float", "description": "Palm press. Default 5.0."},
        {"name": "floor_friction", "type": "float", "description": "Floor sliding friction. Default 1.0."},
        {"name": "hold_frac", "type": "float", "description": "Light drag as a fraction of the friction limit. Default 0.4."},
        {"name": "slide_frac", "type": "float", "description": "Strong drag as a fraction of the friction limit. Default 1.6."},
        {"name": "drag_speed_m_s", "type": "float", "description": "Speed the palm drags at; the drag force is servoed to it and capped. Default 0.1."},
        {"name": "phase_s", "type": "float", "description": "Seconds per phase. Default 0.5."},
        {"name": "stiffness_tol_pct", "type": "float", "description": "Allowed stiffness error. Default 40."},
        {"name": "min_compression_m", "type": "float", "description": "Smallest squash that counts as compliant. Default 0.001."},
        {"name": "max_hold_slip_m", "type": "float", "description": "Largest travel under the light drag. Default 0.005."},
        {"name": "min_slide_m", "type": "float", "description": "Smallest travel under the strong drag. Default 0.02."},
        {"name": "max_lift_m", "type": "float", "description": "Largest rise of the slab while sliding. Default 0.003."},
        {"name": "max_tilt_deg", "type": "float", "description": "Largest tilt while sliding. Default 10."},
        {"name": "min_contact_frac", "type": "float", "description": "Fraction of sliding steps with floor contact. Default 0.95."},
        {"name": "recover_frac", "type": "float", "description": "Residual squash allowed after release, fraction of cell travel. Default 0.1."},
        {"name": "out_json", "type": "path | None", "description": "Optional report file."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; compression_m; measured_stiffness_N_per_m; expected_stiffness_N_per_m; hold_slip_m; slide_m; max_lift_m; max_tilt_deg; contact_frac; residual_compression_m; verdict."},
    ],
    "status": "tested",
    "entrypoint": "pad_press_slide_test",
    "requirements": ["mujoco>=3.1", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["sponge", "pad", "soft", "compliance", "friction", "slide", "wipe", "scrub", "mechanism test"],
    "used_in": [],
    "cli": "python -m tool_library.verifiers.pad_press_slide_test built.json",
}

import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")


def _load_with_floor(xml_path, mu):
    import mujoco
    xp = Path(xml_path)
    txt = xp.read_text()
    if "<worldbody>" not in txt:
        raise ValueError("no <worldbody> in the asset")
    floor = f'<worldbody>\n    <geom name="_test_floor" type="plane" size="3 3 0.01" friction="{mu} 0.01 0.001" condim="4"/>'
    txt = txt.replace("<worldbody>", floor, 1)
    fd, tmp = tempfile.mkstemp(suffix=".xml", dir=str(xp.parent)); os.close(fd)
    try:
        Path(tmp).write_text(txt)
        return mujoco.MjModel.from_xml_path(tmp)
    finally:
        os.unlink(tmp)


def _descendants(m, root):
    out = [root]
    for b in range(m.nbody):
        a = b
        while a > 0:
            a = int(m.body_parentid[a])
            if a == root:
                out.append(b); break
    return out


def pad_press_slide_test(built, press_force_N=5.0, floor_friction=1.0, hold_frac=0.4, slide_frac=1.6, phase_s=0.5,
                         stiffness_tol_pct=40.0, drag_speed_m_s=0.1, min_compression_m=0.001, max_hold_slip_m=0.005, min_slide_m=0.02,
                         max_lift_m=0.003, max_tilt_deg=10.0, min_contact_frac=0.95, recover_frac=0.1, out_json=None):
    import mujoco
    if isinstance(built, (str, Path)) and str(built).endswith(".json"):
        built = json.loads(Path(built).read_text())
    xml = built.get("xml") if isinstance(built, dict) else str(built)
    if not xml or not Path(xml).exists():
        raise ValueError(f"built: no MJCF found ({xml!r})")
    if press_force_N <= 0:
        raise ValueError("press_force_N must be > 0")
    issues = []
    rep = {"ok": False, "issues": issues, "xml": str(xml)}
    try:
        m = _load_with_floor(xml, float(floor_friction))
    except Exception as e:
        issues.append(f"the asset does not compile with a floor: {str(e)[:200]}")
        return rep
    frees = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    if not frees:
        issues.append("the pad has no free joint: a pad that is slid over the table must be a free body")
        return rep
    root = int(m.jnt_bodyid[frees[0]])
    sub = _descendants(m, root)
    slides = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE and int(m.jnt_bodyid[j]) in sub]
    tops = [int(m.jnt_bodyid[j]) for j in slides] or [root]
    floor_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "_test_floor")
    pad_geoms = {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in sub}
    mu_pad = max(float(m.geom_friction[g][0]) for g in pad_geoms) if pad_geoms else 1.0
    mu = max(mu_pad, float(floor_friction))
    M = float(m.body_subtreemass[root])
    load = float(press_force_N) + M * 9.81
    expected_k = float(built.get("stiffness_N_per_m")) if isinstance(built, dict) and built.get("stiffness_N_per_m") else float(sum(m.jnt_stiffness[j] for j in slides))
    travel = max((-float(m.jnt_range[j][0]) for j in slides), default=0.0)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    nstep = max(1, int(round(phase_s / m.opt.timestep)))
    qadr = [int(m.jnt_qposadr[j]) for j in slides]
    fqa = int(m.jnt_qposadr[frees[0]])

    def squash():
        return float(np.mean([max(0.0, -d.qpos[a]) for a in qadr])) if qadr else 0.0

    fda = int(m.jnt_dofadr[frees[0]])

    def run(fz, fmax, track=False):
        # the palm: a press plus a horizontal drag towards a target moving at drag_speed_m_s, capped at fmax (a real hand moves
        # with the pad; an open-loop force above the friction limit would accelerate it without bound)
        # palm = a spring-damper to a target moving at drag_speed_m_s, gains bounded by the pad mass so the explicit
        # coupling stays stable at the model timestep
        lifts, tilts, contact = [], [], 0
        xt = float(d.qpos[fqa])
        kp = 0.25 * M / m.opt.timestep ** 2
        kd = math.sqrt(kp * M)
        for _ in range(nstep):
            xt += drag_speed_m_s * m.opt.timestep
            fx = float(np.clip(kp * (xt - float(d.qpos[fqa])) - kd * float(d.qvel[fda]) + kd * drag_speed_m_s, -fmax, fmax)) if fmax > 0 else 0.0
            d.xfrc_applied[:] = 0
            for b in tops:
                d.xfrc_applied[b, 2] = -fz / len(tops)
                d.xfrc_applied[b, 0] = fx / len(tops)
            mujoco.mj_step(m, d)
            if track:
                lifts.append(float(d.qpos[fqa + 2]))
                zax = d.xmat[root].reshape(3, 3)[:, 2]
                tilts.append(math.degrees(math.acos(max(-1.0, min(1.0, float(zax[2]))))))
                hit = False
                for c in range(d.ncon):
                    con = d.contact[c]
                    if (con.geom1 == floor_gid and con.geom2 in pad_geoms) or (con.geom2 == floor_gid and con.geom1 in pad_geoms):
                        hit = True; break
                # a sub-millimetre separation for a step (contact chatter at gram scale) still counts as touching
                contact += hit or (float(d.qpos[fqa + 2]) - z0) < 0.0005
        if not np.all(np.isfinite(d.qpos)):
            raise FloatingPointError("simulation diverged")
        return lifts, tilts, contact

    try:
        run(0.0, 0.0)
        z0 = float(d.qpos[fqa + 2]); x0 = float(d.qpos[fqa])
        run(press_force_N, 0.0)
        comp = squash()
        k_meas = press_force_N / comp if comp > 1e-7 else None
        xh0 = float(d.qpos[fqa])
        run(press_force_N, hold_frac * mu * load)
        hold_slip = abs(float(d.qpos[fqa]) - xh0)
        xs0 = float(d.qpos[fqa])
        lifts, tilts, contact = run(press_force_N, slide_frac * mu * load, track=True)
        slide = float(d.qpos[fqa]) - xs0
        run(0.0, 0.0)
        run(0.0, 0.0)
        resid = squash()
    except FloatingPointError as e:
        issues.append(f"{e}: raise the cell armature or damping")
        return rep
    lift = max(lifts) - z0 if lifts else 0.0
    tilt = max(tilts) if tilts else 0.0
    cfrac = contact / max(1, len(lifts))
    if not slides:
        issues.append("no compliant joints under the pad: it cannot squash under the palm; build it with sprung cells")
    elif comp < min_compression_m:
        issues.append(f"a {press_force_N} N press squashes it only {comp*1000:.2f} mm (< {min_compression_m*1000:.1f}): lower the stiffness")
    if slides and expected_k > 0 and (k_meas is None or abs(k_meas - expected_k) / expected_k * 100 > stiffness_tol_pct):
        issues.append(f"measured stiffness {k_meas} N/m vs expected {expected_k:.0f} N/m: the cells bottom out or the spring is wrong")
    if hold_slip > max_hold_slip_m:
        issues.append(f"a drag of {hold_frac:.0%} of the friction limit slid it {hold_slip*1000:.1f} mm: friction is too low")
    if slide < min_slide_m:
        issues.append(f"a drag of {slide_frac:.0%} of the friction limit moved it only {slide*1000:.1f} mm: it sticks")
    if lift > max_lift_m:
        issues.append(f"the slab rose {lift*1000:.1f} mm while sliding: it hops instead of sliding flat")
    if tilt > max_tilt_deg:
        issues.append(f"the pad tilted {tilt:.1f} deg while sliding: it tips over its front edge")
    if cfrac < min_contact_frac:
        issues.append(f"floor contact held in only {cfrac:.0%} of the sliding steps")
    if slides and resid > recover_frac * travel:
        issues.append(f"after release it stays squashed {resid*1000:.2f} mm: the cells do not spring back")
    rep.update(ok=not issues, compression_m=comp, measured_stiffness_N_per_m=k_meas, expected_stiffness_N_per_m=expected_k,
               friction_mu=mu, normal_load_N=load, hold_slip_m=hold_slip, slide_m=slide, max_lift_m=lift, max_tilt_deg=tilt,
               contact_frac=cfrac, residual_compression_m=resid, n_cells=len(slides), x_start=x0,
               verdict="PAD OK: squashes, grips, slides flat, recovers" if not issues else "PAD FAILED")
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    r = pad_press_slide_test(sys.argv[1])
    print(json.dumps(r, indent=1))
    sys.exit(0 if r["ok"] else 1)

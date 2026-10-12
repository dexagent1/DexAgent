"""Squeeze test for a soft free-bodied object: does it give under a grasp-sized force and spring back?

    from tool_library.verifiers.soft_squeeze_test import soft_squeeze_test
    rep = soft_squeeze_test("assets/plushie/plushie.xml", body="plushie_plush")
    rep["ok"], rep["compression_mm"], rep["issues"]

A rigid-body simulator shows a plush toy, a sponge or a foam block squashing in
the hand only through contact compliance. This test lays the object on a floor,
lets it settle, then presses it down with a ramp of forces (applied at its centre
of mass) and measures how far it sinks into the floor at each force, then releases
it and checks that it comes back. The compression at the reference force must sit
in a band: too little and the object is a rigid shell to the fingers, too much and
fingers would pass through it.
"""
from __future__ import annotations

TOOL = {
    "name": "soft_squeeze_test",
    "title": "Soft squeeze test",
    "type": "verifier",
    "category": "Reconstruction Mechanism Tests",
    "summary": "Presses a soft free body onto a floor with a ramp of forces and checks that it compresses by a grasp-sized amount at a reference force, monotonically, and springs back when released.",
    "description": (
        "Loads the asset MJCF, adds a floor plane (MuJoCo's default contact parameters) when none "
        "is present, settles the object for settle_s, records its rest height, "
        "then applies each downward force of `forces_N` to the body's centre of mass for hold_s "
        "and records the sink (rest z minus loaded z of the body origin, in mm). Passes when the "
        "state stays finite, the sink grows with force (within 0.3 mm), the sink at ref_force_N is "
        "within [min_mm, max_mm], the object does not tunnel below the floor by more than "
        "max_mm at the largest force, and after release it returns to within recover_mm of the "
        "rest height. The measurement is the compliance a pinch would feel, combined with the "
        "floor's own contact parameters (MuJoCo mixes the two geoms' solref). It does not "
        "simulate fabric deformation; it judges contact compliance."
    ),
    "inputs": [
        {"name": "built", "type": "str | dict", "description": "Asset MJCF path, or a constructor result with 'xml'."},
        {"name": "body", "type": "str | None", "description": "The free body to press; default the result's 'body' or the first free body."},
        {"name": "forces_N", "type": "list[float]", "description": "Downward forces, increasing. Default [0.5, 1, 2, 4]."},
        {"name": "ref_force_N", "type": "float", "description": "Grasp-sized reference force; must be one of forces_N. Default 2."},
        {"name": "min_mm", "type": "float", "description": "Least sink at the reference force. Default 2."},
        {"name": "max_mm", "type": "float", "description": "Largest acceptable sink at any force. Default 25."},
        {"name": "recover_mm", "type": "float", "description": "Allowed residual sink after release. Default 1."},
        {"name": "settle_s", "type": "float", "description": "Settling time. Default 1.0."},
        {"name": "hold_s", "type": "float", "description": "Time at each force. Default 0.6."},
        {"name": "out_json", "type": "str | None", "description": "Optional report path."},
    ],
    "outputs": [
        {"name": "ok", "type": "bool", "description": "All checks passed."},
        {"name": "issues", "type": "list[str]", "description": "What failed and which parameter to change."},
        {"name": "compression_mm", "type": "dict", "description": "Sink in mm per force."},
        {"name": "rest_z_m", "type": "float", "description": "Settled body origin height."},
        {"name": "recovered_mm", "type": "float", "description": "Residual sink after release."},
        {"name": "verdict", "type": "str", "description": "One sentence."},
    ],
    "status": "tested",
    "entrypoint": "soft_squeeze_test",
    "requirements": ["numpy", "mujoco>=3.2"],
    "tags": ["soft", "compliance", "squeeze", "plush", "sponge", "grasp", "deformable"],
    "used_in": [],
    "cli": "python -m tool_library.verifiers.soft_squeeze_test asset.xml [body]",
}

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")


def soft_squeeze_test(built, body=None, forces_N=(0.5, 1.0, 2.0, 4.0), ref_force_N=2.0, min_mm=2.0,
                      max_mm=25.0, recover_mm=1.0, settle_s=1.0, hold_s=0.6, out_json=None):
    import mujoco
    import numpy as np
    xml = built.get("xml") if isinstance(built, dict) else built
    if body is None and isinstance(built, dict):
        body = built.get("body")
    if not xml or not Path(xml).exists():
        raise ValueError(f"built: asset xml {xml!r} not found")
    forces = [float(f) for f in forces_N]
    if not forces or any(b <= a for a, b in zip(forces, forces[1:])) or forces[0] <= 0:
        raise ValueError("forces_N must be positive and strictly increasing")
    if float(ref_force_N) not in forces:
        raise ValueError("ref_force_N must be one of forces_N")
    if not (0 <= min_mm < max_mm):
        raise ValueError("need 0 <= min_mm < max_mm")
    issues = []
    spec = mujoco.MjSpec.from_file(str(xml))
    m0 = spec.compile()
    has_plane = any(m0.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE for g in range(m0.ngeom))
    if not has_plane:
        g = spec.worldbody.add_geom()
        g.type = mujoco.mjtGeom.mjGEOM_PLANE; g.size = [1, 1, 0.01]; g.name = "squeeze_floor"
    m = spec.compile(); d = mujoco.MjData(m)
    if body is None:
        for j in range(m.njnt):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                body = m.body(m.jnt_bodyid[j]).name; break
    try:
        bid = m.body(body).id
    except Exception:
        return {"ok": False, "issues": [f"body {body!r} not in the asset; pass body="], "compression_mm": {}}
    ja = m.body_jntadr[bid]
    if ja < 0 or m.jnt_type[ja] != mujoco.mjtJoint.mjJNT_FREE:
        return {"ok": False, "issues": [f"body {body!r} has no free joint; the test needs a free object"], "compression_mm": {}}
    qa = m.jnt_qposadr[ja]

    def run(secs, f):
        d.xfrc_applied[bid, :] = 0; d.xfrc_applied[bid, 2] = -f
        zs = []
        for _ in range(max(1, int(secs / m.opt.timestep))):
            mujoco.mj_step(m, d); zs.append(d.qpos[qa + 2])
        n = max(1, len(zs) // 5)
        return float(np.mean(zs[-n:]))

    z_rest = run(settle_s, 0.0)
    comp = {}
    finite = True
    for f in forces:
        z = run(hold_s, f)
        if not np.all(np.isfinite(d.qpos)):
            finite = False; break
        comp[f"{f:g}"] = round((z_rest - z) * 1000, 3)
    z_back = run(settle_s, 0.0)
    rec = round((z_rest - z_back) * 1000, 3)
    if not finite:
        issues.append("state became non-finite under load: raise soft_timeconst or lower the forces")
    vals = list(comp.values())
    if any(b < a - 0.3 for a, b in zip(vals, vals[1:])):
        issues.append(f"sink is not monotonic in force {comp}: contact is unstable, raise the damping ratio")
    ref = comp.get(f"{float(ref_force_N):g}")
    if ref is None or ref < min_mm:
        issues.append(f"sink at {ref_force_N} N is {ref} mm < {min_mm} mm: the object is too rigid; raise solref time constant or solimp width")
    if vals and max(vals) > max_mm:
        issues.append(f"sink reaches {max(vals)} mm > {max_mm} mm: fingers would pass through; lower the time constant or the solimp width")
    if abs(rec) > recover_mm:
        issues.append(f"after release the object stays {rec} mm away from rest: it does not spring back")
    ok = not issues
    rep = {"ok": ok, "issues": issues, "body": body, "rest_z_m": round(z_rest, 5), "compression_mm": comp,
           "recovered_mm": rec, "ref_force_N": float(ref_force_N),
           "verdict": (f"SOFT OK: sinks {ref} mm at {ref_force_N} N and springs back" if ok else "SOFT FAIL: " + issues[0])}
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True); Path(out_json).write_text(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    a = sys.argv[1:]
    print(json.dumps(soft_squeeze_test(a[0], a[1] if len(a) > 1 else None), indent=1))

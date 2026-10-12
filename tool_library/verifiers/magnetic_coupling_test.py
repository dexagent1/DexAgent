"""Magnetic coupling test: rings that must cling to each other rim to rim, hang, survive a pull apart as a chain and rejoin.

Force-and-kinematics trials on up to three copies of magnetic ring assets (adhesion-actuated magnets), held in a
vertical plane like a person holds fidget rings: two rings are driven kinematically (as by two hands), the third
moves only under gravity, contact and its magnets.
"""
from __future__ import annotations

TOOL = {
    "name": "magnetic_coupling_test",
    "title": "Magnetic ring coupling test",
    "type": "verifier",
    "category": "Reconstruction Mechanism Tests",
    "summary": "Hangs one magnetic ring from another, ramps a pull until it separates, then holds two rings like two hands and pulls them apart and back together: the third ring must stay attached to both throughout.",
    "description": (
        "Loads one to three ring asset MJCFs (one path is copied three times; every name is prefixed per copy), "
        "strips floors, and builds a gravity-only test world with the ring axes horizontal (along world y) so the "
        "rings move in the vertical x-z plane. A ring's magnet is found as the body targeted by a body-transmission "
        "(adhesion) actuator; its geom's radius sets the touching centre distance D = 2 r. Held rings are driven "
        "kinematically every step (pose and velocity set), the free ring is simulated. Trials: (1) hang: ring B "
        "hangs below held ring A for hold_s; it must stay attached and not sag; (2) pull-off: a downward force on B "
        "ramps to pull_max_N; the force at which B separates must lie in [min_pull_N, max_pull_N] (default min = "
        "3 x B's weight); (3) chain: A and C held touching with B hanging (or sitting) against both, then A and C "
        "move apart to chain_frac x D centre separation over move_s: B must still touch both while A and C no "
        "longer touch; (4) rejoin: A and C move back together: B must still touch both. 'Touching' is a centre "
        "distance at most D + tol_m. Returns ok, issues (what to change), and every measurement."
    ),
    "inputs": [
        {"name": "rings", "type": "list[path]", "description": "One to three ring asset MJCF paths (one path is used for all three rings)."},
        {"name": "hold_s", "type": "float", "description": "Hang duration in seconds. Default 2.0"},
        {"name": "pull_max_N", "type": "float", "description": "Peak of the pull-off force ramp in newtons. Default 20.0"},
        {"name": "pull_ramp_s", "type": "float", "description": "Duration of the pull-off ramp. Default 2.0"},
        {"name": "min_pull_N", "type": "float", "description": "Smallest acceptable separation force, or None for 3 x the hanging ring's weight. Default None"},
        {"name": "max_pull_N", "type": "float", "description": "Largest acceptable separation force (a hand must be able to pull rings apart). Default 15.0"},
        {"name": "chain_frac", "type": "float", "description": "Centre separation of the two held rings at the end of the pull, as a multiple of D (< 2). Default 1.95"},
        {"name": "move_s", "type": "float", "description": "Duration of each pull-apart / rejoin motion. Default 1.5"},
        {"name": "third_start", "type": "str", "description": "'below' (B hangs under the held pair) or 'above' (B sits on top). Default below"},
        {"name": "tol_m", "type": "float", "description": "Contact tolerance on the centre distance. Default 0.0015"},
        {"name": "timestep", "type": "float", "description": "Simulation timestep; use the scene's own. Default 0.002"},
        {"name": "integrator", "type": "str", "description": "MuJoCo integrator of the test world; use the scene's own. Default implicitfast"},
        {"name": "impratio", "type": "float", "description": "Friction-to-normal impedance ratio of the test world; use the scene's own. Default 5.0"},
        {"name": "out_json", "type": "path", "description": "Optional report path."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; verdict; D_m; weight_N; magnet_force_N per ring; hang {ok, d_AB_m, sag_m}; pull_off {ok, separation_force_N, separated}; chain {ok, d_AB_m, d_CB_m, d_AC_m}; rejoin {ok, d_AB_m, d_CB_m, d_AC_m}."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["magnet", "magnetic", "ring", "adhesion", "chain", "pull apart", "mechanism test", "verifier"],
    "status": "tested",
    "entrypoint": "magnetic_coupling_test",
    "cli": "python -m tool_library.verifiers.magnetic_coupling_test ring.xml",
}

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

_REF_ATTRS = ("name", "body", "joint", "geom", "site", "body1", "body2", "joint1", "joint2", "geom1", "geom2",
              "tendon", "mesh", "material", "texture", "class", "childclass", "target", "objname", "cranksite",
              "slidersite", "refsite", "actuator", "hfield", "skin")
_PARAMS = {"hold_s": 2.0, "pull_max_N": 20.0, "pull_ramp_s": 2.0, "min_pull_N": None, "max_pull_N": 15.0,
           "chain_frac": 1.95, "move_s": 1.5, "third_start": "below", "tol_m": 0.0015, "timestep": 0.002,
           "integrator": "implicitfast", "impratio": 5.0, "out_json": None}


def _prefixed(path, prefix):
    root = ET.parse(path).getroot()
    base = Path(path).resolve().parent
    comp = root.find("compiler")
    meshdir = base / (comp.get("meshdir") or "") if comp is not None else base
    texdir = base / (comp.get("texturedir") or "") if comp is not None else base
    names = {el.get("name") for el in root.iter() if el.get("name")}
    names |= {el.get("class") for el in root.iter("default") if el.get("class")}
    for el in root.iter():
        for a in _REF_ATTRS:
            v = el.get(a)
            if v is not None and (a == "name" or v in names):
                el.set(a, prefix + v)
        f = el.get("file")
        if f and not Path(f).is_absolute():
            el.set("file", str(((texdir if el.tag == "texture" else meshdir) / f).resolve()))
    return root


def _build_world(paths, timestep, integrator="implicitfast", impratio=1.0):
    out = ET.Element("mujoco", {"model": "magnetic_coupling_test"})
    secs = {}
    for tag in ("compiler", "option", "default", "asset", "worldbody", "actuator", "equality", "contact", "tendon"):
        secs[tag] = None
    for i, p in enumerate(paths):
        r = _prefixed(p, f"r{i}_")
        for child in r:
            if child.tag == "compiler":
                if secs["compiler"] is None:
                    c = ET.SubElement(out, "compiler", {k: v for k, v in child.attrib.items()
                                                        if k not in ("meshdir", "texturedir")})
                    secs["compiler"] = c
                continue
            if child.tag in ("option", "size", "keyframe", "visual", "statistic", "custom", "extension"):
                continue
            if child.tag == "default":
                if secs["default"] is None:
                    secs["default"] = ET.SubElement(out, "default")
                    for d in child:
                        secs["default"].append(d)
                else:
                    for d in child:
                        if d.tag == "default":
                            secs["default"].append(d)
                continue
            if secs.get(child.tag) is None:
                secs[child.tag] = ET.SubElement(out, child.tag)
            for sub in child:
                if child.tag == "worldbody" and sub.tag == "geom" and sub.get("type") == "plane":
                    continue
                secs[child.tag].append(sub)
    opt = ET.Element("option", {"timestep": f"{timestep:g}", "gravity": "0 0 -9.81", "integrator": str(integrator),
                                "impratio": f"{impratio:g}"})
    out.insert(1 if secs["compiler"] is not None else 0, opt)
    return ET.tostring(out, encoding="unicode")


def magnetic_coupling_test(rings, **params):
    import mujoco
    unknown = sorted(set(params) - set(_PARAMS))
    if unknown:
        raise ValueError(f"magnetic_coupling_test: unknown parameter(s) {unknown}; allowed: {sorted(_PARAMS)}")
    P = dict(_PARAMS)
    P.update(params)
    if isinstance(rings, (str, Path)):
        rings = [rings]
    rings = [str(r) for r in rings]
    if not 1 <= len(rings) <= 3:
        raise ValueError("magnetic_coupling_test: rings must list one to three asset paths")
    for r in rings:
        if not Path(r).exists():
            raise ValueError(f"magnetic_coupling_test: ring asset not found: {r}")
    if P["third_start"] not in ("below", "above"):
        raise ValueError("magnetic_coupling_test: third_start must be 'below' or 'above'")
    if not 1.0 < float(P["chain_frac"]) < 2.0:
        raise ValueError("magnetic_coupling_test: chain_frac must be between 1 and 2")
    paths = (rings * 3)[:3] if len(rings) == 1 else (rings + [rings[-1]] * 3)[:3]
    issues = []
    report = {"ok": False, "issues": issues, "rings": paths}
    try:
        m = mujoco.MjModel.from_xml_string(_build_world(paths, float(P["timestep"]), P["integrator"],
                                                        float(P["impratio"])))
    except Exception as e:  # report, do not raise, on a bad asset
        issues.append(f"the ring assets do not compile together: {e}")
        report["verdict"] = "NOT TESTED"
        return _finish(report, P)
    d = mujoco.MjData(m)
    info = []
    for i in range(3):
        pre = f"r{i}_"
        root = None
        for b in range(1, m.nbody):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            if nm.startswith(pre) and m.body_jntnum[b] > 0 and m.jnt_type[m.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_FREE:
                root = b
                break
        mag_bodies, force = [], 0.0
        for a in range(m.nu):
            an = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ""
            if an.startswith(pre) and m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_BODY:
                mag_bodies.append(int(m.actuator_trnid[a, 0]))
                force += float(m.actuator_biasprm[a, 0]) + float(m.actuator_gainprm[a, 0]) * float(m.actuator_ctrlrange[a, 1])
        if root is None:
            issues.append(f"ring asset {paths[i]} has no free root body: the ring must be a free body")
            report["verdict"] = "NOT TESTED"
            return _finish(report, P)
        if not mag_bodies:
            issues.append(f"ring asset {paths[i]} has no body-transmission (adhesion) actuator: build it with a magnet "
                          f"(construct_magnetic_ring with magnet_force_N > 0)")
            report["verdict"] = "NOT TESTED"
            return _finish(report, P)
        geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in mag_bodies]
        g0 = geoms[0]
        r_mag = float(m.geom_size[g0, 0]) if m.geom_type[g0] == mujoco.mjtGeom.mjGEOM_CYLINDER else float(m.geom_rbound[g0])
        qadr, dadr = int(m.jnt_qposadr[m.body_jntadr[root]]), int(m.jnt_dofadr[m.body_jntadr[root]])
        info.append(dict(root=root, geom=g0, r=r_mag, qadr=qadr, dadr=dadr, force=force,
                         mass=float(m.body_subtreemass[root])))
    # orientation that puts each ring's magnet axis along world y, and the centre offset at that orientation
    for it in info:
        d.qpos[:] = m.qpos0
        d.qpos[it["qadr"]:it["qadr"] + 3] = 0.0
        d.qpos[it["qadr"] + 3:it["qadr"] + 7] = [1, 0, 0, 0]
        mujoco.mj_kinematics(m, d)
        axis_b = d.geom_xmat[it["geom"]].reshape(3, 3)[:, 2].copy()
        q1 = np.zeros(4)
        mujoco.mju_quatZ2Vec(q1, axis_b)
        q2 = np.zeros(4)
        mujoco.mju_quatZ2Vec(q2, np.array([0.0, 1.0, 0.0]))
        q1c = np.array([q1[0], -q1[1], -q1[2], -q1[3]])
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, q2, q1c)
        d.qpos[it["qadr"] + 3:it["qadr"] + 7] = q
        mujoco.mj_kinematics(m, d)
        it["quat"] = q.copy()
        it["offset"] = d.geom_xpos[it["geom"]].copy()
    D = info[0]["r"] + info[1]["r"]
    Dac = info[0]["r"] + info[2]["r"]
    tol = float(P["tol_m"])
    rng = float(max(m.geom_margin[it["geom"]] for it in info)) or 0.002
    wB = info[1]["mass"] * 9.81
    report.update(D_m=D, weight_N=wB, magnet_force_N=[it["force"] for it in info])
    dt = m.opt.timestep
    Z0 = 0.5

    def place(i, c, v=None):
        it = info[i]
        d.qpos[it["qadr"]:it["qadr"] + 3] = np.asarray(c) - it["offset"]
        d.qpos[it["qadr"] + 3:it["qadr"] + 7] = it["quat"]
        d.qvel[it["dadr"]:it["dadr"] + 6] = 0.0
        if v is not None:
            d.qvel[it["dadr"]:it["dadr"] + 3] = v

    def centre(i):
        return d.geom_xpos[info[i]["geom"]].copy()

    def pose_info(i):
        c = centre(i)
        ax = d.geom_xmat[info[i]["geom"]].reshape(3, 3)[:, 2]
        return {"B_axial_offset_m": float(c[1]), "B_tilt_deg": float(np.degrees(np.arccos(min(1.0, abs(ax[1])))))}

    def dist(i, j):
        return float(np.linalg.norm(centre(i) - centre(j)))

    def run(seconds, held, force_fn=None, stop_fn=None):
        n = int(round(seconds / dt))
        for k in range(n):
            t = k * dt
            for i, fn in held.items():
                c0, c1 = fn(t), fn(t + dt)
                place(i, c0, (c1 - c0) / dt)
            d.xfrc_applied[:] = 0.0
            if force_fn is not None:
                d.xfrc_applied[info[1]["root"], :3] = force_fn(t)
            mujoco.mj_step(m, d)
            if stop_fn is not None:
                mujoco.mj_kinematics(m, d)
                if stop_fn(t):
                    return t
        mujoco.mj_kinematics(m, d)
        return None

    def reset():
        mujoco.mj_resetData(m, d)
        d.ctrl[:] = 0.0

    park = lambda x: (lambda t: np.array([x, 0.0, Z0]))
    gap0 = 0.5 * rng
    # (1) hang
    reset()
    cA = np.array([0.0, 0.0, Z0])
    place(0, cA)
    place(1, cA - [0, 0, D + gap0])
    place(2, [1.0, 0.0, Z0])
    mujoco.mj_forward(m, d)
    run(float(P["hold_s"]), {0: park(0.0), 2: park(1.0)})
    dAB = dist(0, 1)
    sag = float((cA[2] - D) - centre(1)[2])
    hang_ok = dAB <= D + tol and sag <= tol
    report["hang"] = {"ok": bool(hang_ok), "d_AB_m": dAB, "sag_m": sag}
    if not hang_ok:
        issues.append(f"hang: ring B fell or sagged off ring A (centre distance {dAB:.4f} m vs touching {D:.4f} m, "
                      f"sag {sag:.4f} m): raise magnet_force_N or magnet_range_m")
    # (2) pull-off ramp
    pull = {"ok": False, "separation_force_N": None, "separated": False}
    if hang_ok:
        fmax, ramp = float(P["pull_max_N"]), float(P["pull_ramp_s"])
        t_sep = run(ramp, {0: park(0.0), 2: park(1.0)},
                    force_fn=lambda t: np.array([0.0, 0.0, -fmax * min(t / ramp, 1.0)]),
                    stop_fn=lambda t: dist(0, 1) > D + max(4 * rng, 2 * tol))
        lo = float(P["min_pull_N"]) if P["min_pull_N"] is not None else 3.0 * wB
        hi = float(P["max_pull_N"])
        if t_sep is None:
            pull.update(separated=False, ok=False)
            issues.append(f"pull-off: ring B did not separate under {fmax:.1f} N: lower magnet_force_N so that a hand "
                          f"can pull the rings apart")
        else:
            f = fmax * min(t_sep / ramp, 1.0)
            pull.update(separated=True, separation_force_N=f, ok=bool(lo <= f <= hi), min_N=lo, max_N=hi)
            if f < lo:
                issues.append(f"pull-off: ring B separated at {f:.2f} N, below {lo:.2f} N: raise magnet_force_N")
            elif f > hi:
                issues.append(f"pull-off: ring B separated only at {f:.2f} N, above {hi:.2f} N: lower magnet_force_N")
    report["pull_off"] = pull
    # (3) chain and (4) rejoin
    reset()
    s0 = Dac * 1.0005
    s1 = float(P["chain_frac"]) * D
    move = float(P["move_s"])
    sign = -1.0 if P["third_start"] == "below" else 1.0
    hb = math.sqrt(max(D ** 2 - (s0 / 2) ** 2, 0.0)) + gap0
    place(0, [-s0 / 2, 0, Z0])
    place(2, [s0 / 2, 0, Z0])
    place(1, [0, 0, Z0 + sign * hb])
    mujoco.mj_forward(m, d)
    smooth = lambda u: 0.0 if u <= 0 else (1.0 if u >= 1 else u * u * (3 - 2 * u))

    def sep_path(sa, sb, t0):
        return lambda i: (lambda t: np.array([(-1 if i == 0 else 1) * 0.5 * (sa + (sb - sa) * smooth((t - t0) / move)), 0.0, Z0]))

    run(0.5, {0: sep_path(s0, s0, 0)(0), 2: sep_path(s0, s0, 0)(2)})
    run(move + 0.5, {0: sep_path(s0, s1, 0)(0), 2: sep_path(s0, s1, 0)(2)})
    ch = {"d_AB_m": dist(0, 1), "d_CB_m": dist(2, 1), "d_AC_m": dist(0, 2), "B_z_m": float(centre(1)[2] - Z0), **pose_info(1)}
    ch["ok"] = bool(ch["d_AB_m"] <= D + tol and ch["d_CB_m"] <= D + tol and ch["d_AC_m"] >= Dac + rng)
    report["chain"] = ch
    if not ch["ok"]:
        issues.append(f"chain: after pulling A and C apart to {s1:.4f} m, B touches A: {ch['d_AB_m'] <= D + tol}, "
                      f"touches C: {ch['d_CB_m'] <= D + tol}, A-C separated: {ch['d_AC_m'] >= Dac + rng}; "
                      f"raise magnet_force_N or friction if B fell, lower chain_frac if A and C never separated")
    run(move + 0.5, {0: sep_path(s1, s0, 0)(0), 2: sep_path(s1, s0, 0)(2)})
    rj = {"d_AB_m": dist(0, 1), "d_CB_m": dist(2, 1), "d_AC_m": dist(0, 2), "B_z_m": float(centre(1)[2] - Z0), **pose_info(1)}
    rj["ok"] = bool(rj["d_AB_m"] <= D + tol and rj["d_CB_m"] <= D + tol)
    report["rejoin"] = rj
    if not rj["ok"]:
        issues.append("rejoin: after pushing A and C back together B no longer touches both: raise magnet_force_N")
    report["ok"] = bool(hang_ok and pull["ok"] and ch["ok"] and rj["ok"])
    report["verdict"] = ("MAGNETIC COUPLING OK" if report["ok"] else "MAGNETIC COUPLING FAILED") + (
        f": hang sag {sag * 1000:.1f} mm, pull-off {pull['separation_force_N']} N, chain d_AB {ch['d_AB_m']:.4f} "
        f"d_CB {ch['d_CB_m']:.4f} d_AC {ch['d_AC_m']:.4f} (D {D:.4f} m)")
    return _finish(report, P)


def _finish(report, P):
    if P.get("out_json"):
        Path(P["out_json"]).parent.mkdir(parents=True, exist_ok=True)
        Path(P["out_json"]).write_text(json.dumps(report, indent=1, default=float))
    return report


if __name__ == "__main__":
    import sys as _sys
    print(json.dumps(magnetic_coupling_test(_sys.argv[1:]), indent=1, default=float))

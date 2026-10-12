"""Plug-port retention test: a plug must slide into a receptacle cavity, latch (or be held by friction) there, come out under a hand-sized pull, and be blocked when misaligned."""
from __future__ import annotations

TOOL = {'name': 'plug_port_retention_test',
 'title': 'Plug-port insertion and retention test',
 'type': 'verifier',
 'category': 'Reconstruction Mechanism Tests',
 'summary': 'Pushes a standard plug into a built receptacle with a capped force, then loads it outward, '
            'ramps a pull-out force and tries a misaligned push, in the physics Stage 3 episodes run in and with the '
            'connector latch the executor applies: the plug must seat, stay seated, come out at about the latch\'s '
            'release force and be blocked when off the slot.',
 'description': 'Loads the asset XML of a device with a declared port (a build dict with xml, body and '
                'port: {width, height, depth, mouth_local, axis_local}), holds the device kinematically at '
                'the origin (as a hand would) and adds a free plug: a box tip sized to the port minus a '
                'clearance on a larger housing that stops at the edge face. By default the trials run in the '
                'episode\'s physics (common.episode.EXECUTION_PHYSICS: timestep 0.002, implicitfast, elliptic cones, '
                'noslip 5, impratio 5) and the test plug carries the latch Stage 3 gives a connector '
                '(add_connector_latch): an inactive weld between the device and the plug that the executor\'s own '
                'rule code (common.episode.latch_begin / latch_step, the code servo_rollout and Stage 4\'s physics '
                'replay run) switches on when the plug tip has bottomed out within seat_m of the port floor (and is no '
                'longer pressed on it) and off when the '
                'pull-out load exceeds release_N (a real USB-C plug is held by its latch springs, and a gram-scale '
                'friction fit holds or not depending on the contact model). latch=False tests a pure friction fit '
                '(with timestep 0.0002, noslip_iterations 0, Euler, pyramidal cones and impratio 1 it is the regime '
                'the port\'s grip was first tuned in), and the report says which retention and physics were tested. '
                'The device is held weightless and very heavy, re-posed every step (its gravcomp is set before '
                'compiling: set after, MuJoCo ignores it and the device fell at g inside every step, a load the plug '
                'felt; that is what made the friction fit look thrown out at the episode\'s timestep). '
                'Four force-only trials with gravity on: (1) insert: an axial push of insert_force_N for insert_s '
                'from 3 mm outside the mouth must reach seat_frac of the cavity depth (and engage the latch); (2) '
                'hold: with the push removed and an outward load of hold_load_N plus gravity along the pull '
                'direction (port facing down, the worst case), the tip must still be min_held_depth_m deep after '
                'hold_s (creep beyond max_slip_m is reported as a warning in hold_creep_warning and in the verdict, '
                'with the depths at the start and end of the hold, the creep speed and the time until the plug would '
                'leave the port); (3) extract: an outward force ramped to max_extract_N must pull the plug out (not '
                'a permanent weld) at no less than min_extract_N; with the latch it comes out where the latch lets '
                'go, about release_N; (4) blocked: a plug offset sideways by one port width must not enter deeper '
                'than 0.5 mm nor latch. Assumes the device is a rigid body with the port as a cavity in its '
                'collision geometry.',
 'inputs': [
  {'name': 'built', 'type': 'dict|path', 'description': 'Build dict (or its JSON path) with xml, body and port {width, height, depth, mouth_local, axis_local}.'},
  {'name': 'latch', 'type': 'bool', 'description': 'Give the test plug the connector latch (the executor rule: seat_m, release_N) as Stage 3 does; False tests a pure friction fit. Default True'},
  {'name': 'seat_m', 'type': 'float', 'description': 'Latch: tip-to-floor distance that engages it, m. Default 0.0015'},
  {'name': 'release_N', 'type': 'float', 'description': 'Latch: pull-out load at which it lets go, N. Default 15.0'},
  {'name': 'tip_clearance_m', 'type': 'float', 'description': 'Gap between plug tip and cavity on each side. Default 0.0001'},
  {'name': 'housing_scale', 'type': 'float', 'description': 'Housing cross-section relative to the port (must be > 1 so it stops at the edge). Default 1.5'},
  {'name': 'housing_len', 'type': 'float', 'description': 'Housing length in metres. Default 0.013'},
  {'name': 'plug_mass', 'type': 'float', 'description': 'Plug mass in kg. Default 0.006'},
  {'name': 'insert_force_N', 'type': 'float', 'description': 'Axial push cap during insertion (a USB-C plug takes 5-20 N by hand). Default 8.0'},
  {'name': 'insert_s', 'type': 'float', 'description': 'Insertion duration in seconds. Default 0.6'},
  {'name': 'seat_frac', 'type': 'float', 'description': 'Fraction of the cavity depth the tip must reach. Default 0.85'},
  {'name': 'min_extract_N', 'type': 'float', 'description': 'The static pull-out force must be at least this (the hold). Default 1.0'},
  {'name': 'min_held_depth_m', 'type': 'float', 'description': 'After the hold the tip must still be at least this deep in the cavity. Default 0.001'},
  {'name': 'hold_load_N', 'type': 'float', 'description': 'Outward load during the hold (cord weight). Default 0.3'},
  {'name': 'hold_s', 'type': 'float', 'description': 'Hold duration in seconds. Default 1.0'},
  {'name': 'max_slip_m', 'type': 'float', 'description': 'Slip during the hold above this is reported as soft-contact creep (a warning in the verdict, not a failure). Default 0.0015'},
  {'name': 'max_extract_N', 'type': 'float', 'description': 'Peak of the pull-out ramp; the plug must come out below it. Default 25.0'},
  {'name': 'push_speed', 'type': 'float', 'description': 'The push is a velocity servo along the axis toward this speed (m/s), capped at insert_force_N, like a hand that moves the plug at a steady pace. Default 0.4'},
  {'name': 'contact_timeconst', 'type': 'float', 'description': 'Contact time constant of the plug geoms (they take priority over the device geoms). MuJoCo soft contacts scale with mass, so a gram-scale plug needs a short one or it sinks through the cavity floor; clamped to >= 2 timesteps. Default 0.0005'},
  {'name': 'timestep', 'type': 'float', 'description': 'Simulation timestep. Default 0.002 (the episode\'s)'},
  {'name': 'noslip_iterations', 'type': 'int', 'description': 'MuJoCo noslip iterations for the trials (without them soft friction lets a gram-scale plug creep out at a few mm/s under a steady load). Default 5 (the episode\'s)'},
  {'name': 'integrator', 'type': 'str', 'description': "MuJoCo integrator: 'implicitfast' (the bench's), 'Euler', 'implicit' or 'RK4'. Default implicitfast"},
  {'name': 'cone', 'type': 'str', 'description': "Friction cone: 'elliptic' (the episode's) or 'pyramidal'. Default elliptic"},
  {'name': 'impratio', 'type': 'float', 'description': 'Friction-to-normal impedance ratio. Default 5 (the episode\'s)'},
  {'name': 'out_json', 'type': 'path', 'description': 'Optional report path.'}],
 'outputs': [
  {'name': 'ok', 'type': 'bool', 'description': 'All four trials passed.'},
  {'name': 'issues', 'type': 'list[str]', 'description': 'What failed and which parameter to change.'},
  {'name': 'retention', 'type': 'str', 'description': "'latch' (the executor's latch rule on the test plug) or 'friction' (a pure friction fit): what was tested."},
  {'name': 'physics', 'type': 'dict', 'description': 'timestep, integrator, cone, noslip_iterations, impratio the trials ran with, and episode_physics (whether they are the episode\'s).'},
  {'name': 'latch_report', 'type': 'dict | None', 'description': 'With the latch: seat_m, release_N, engaged_in_insert, engage_time_s and engage_depth_m (trial 1), engaged_through_hold, release_force_N (the applied pull when it let go, trial 3), release_pull_N (the load the latch carried then), port_push_out_N (their difference: what the squeezed port pushes the plug out with), engaged_in_blocked.'},
  {'name': 'insert_depth_m', 'type': 'float', 'description': 'Tip depth reached in trial 1.'},
  {'name': 'hold_start_depth_m', 'type': 'float', 'description': 'Tip depth when the hold starts (seated by the push).'},
  {'name': 'hold_end_depth_m', 'type': 'float', 'description': 'Tip depth when the hold ends (what min_held_depth_m is checked on).'},
  {'name': 'hold_slip_m', 'type': 'float', 'description': 'Outward slip in trial 2 = hold_start_depth_m - hold_end_depth_m.'},
  {'name': 'hold_creep_m_per_s', 'type': 'float', 'description': 'Outward creep speed over the second half of the hold.'},
  {'name': 'hold_seconds_to_release', 'type': 'float | None', 'description': 'At that speed, seconds until the tip leaves the port (None without creep).'},
  {'name': 'hold_creep_warning', 'type': 'str', 'description': 'Set when the plug crept more than max_slip_m during the hold without falling out; the verdict says so too.'},
  {'name': 'extract_force_N', 'type': 'float', 'description': 'Force at release in trial 3 (None when it never came out).'},
  {'name': 'blocked_depth_m', 'type': 'float', 'description': 'Depth reached by the misaligned plug in trial 4.'},
  {'name': 'verdict', 'type': 'str', 'description': 'One sentence naming the retention and physics tested.'}],
 'requirements': ['mujoco>=3.2', 'numpy'],
 'simulators': ['MuJoCo'],
 'tags': ['plug', 'port', 'socket', 'receptacle', 'usb-c', 'insert', 'friction fit', 'latch', 'retention', 'connector'],
 'status': 'tested',
 'entrypoint': 'plug_port_retention_test',
 'cli': 'python -m tool_library.verifiers.plug_port_retention_test built.json'}

import json
from pathlib import Path

import numpy as np

from tool_library.common import episode as E

_PHYS = E.EXECUTION_PHYSICS
_DEF = dict(latch=True, seat_m=E.LATCH_SEAT_M, release_N=E.LATCH_RELEASE_N,
            tip_clearance_m=0.0001, housing_scale=1.5, housing_len=0.013, plug_mass=0.006,
            insert_force_N=8.0, min_extract_N=1.0, min_held_depth_m=0.001, insert_s=0.6, seat_frac=0.85, hold_load_N=0.3, hold_s=1.0,
            max_slip_m=0.0015, max_extract_N=25.0, push_speed=0.4, contact_timeconst=0.0005,
            timestep=_PHYS["timestep"], noslip_iterations=_PHYS["noslip_iterations"], integrator=_PHYS["integrator"],
            cone=_PHYS["cone"], impratio=_PHYS["impratio"], out_json=None)
_INTEGRATORS = {"euler": "mjINT_EULER", "implicit": "mjINT_IMPLICIT", "implicitfast": "mjINT_IMPLICITFAST", "rk4": "mjINT_RK4"}
_LATCH = "_plug_latch"


def _lat_dir(axis):
    lat = np.cross(axis, [0, 0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0, 0])
    return lat / np.linalg.norm(lat)


def _model(b, prm):
    import mujoco
    spec = mujoco.MjSpec.from_file(str(b["xml"]))
    spec.option.timestep = float(prm["timestep"])
    integ = _INTEGRATORS.get(str(prm["integrator"]).lower())
    if integ is None:
        raise ValueError(f"integrator must be one of {sorted(_INTEGRATORS)}")
    spec.option.integrator = getattr(mujoco.mjtIntegrator, integ)
    if str(prm["cone"]).lower() not in ("elliptic", "pyramidal"):
        raise ValueError("cone must be 'elliptic' or 'pyramidal'")
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC if str(prm["cone"]).lower() == "elliptic" else mujoco.mjtCone.mjCONE_PYRAMIDAL
    spec.option.impratio = float(prm["impratio"])
    spec.option.noslip_iterations = int(prm["noslip_iterations"])
    port = b["port"]
    pw, ph, pd = float(port["width"]), float(port["height"]), float(port["depth"])
    c = float(prm["tip_clearance_m"])
    tl = pd + 0.001
    hs, hl = float(prm["housing_scale"]), float(prm["housing_len"])
    body = spec.worldbody.add_body(name="_plug", pos=[0, 0, 0])
    body.add_freejoint(name="_plug_free")
    # plug frame: +z along the insertion axis, x across the port width, y through its height; origin = the tip's front
    m = float(prm["plug_mass"])
    tv = (pw - 2 * c) * (ph - 2 * c) * tl
    hv = (pw * hs) * (ph * hs) * hl
    for nm, size, pos, vol in (("_plug_tip", [pw / 2 - c, ph / 2 - c, tl / 2], [0, 0, -tl / 2], tv),
                               ("_plug_housing", [pw * hs / 2, ph * hs / 2, hl / 2], [0, 0, -tl - hl / 2], hv)):
        g = body.add_geom(name=nm, type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=pos)
        g.mass = m * vol / (tv + hv)
        g.friction = [1.0, 0.01, 0.001]
        g.condim = 4
        g.solref = [max(float(prm['contact_timeconst']), 2.0 * float(prm['timestep'])), 1.0]
        g.priority = 1
    # the latch's two sites, framed alike (x across the width, z along the insertion axis) so they coincide when seated
    body.add_site(name="_plug_tip_site", pos=[0, 0, 0], size=[0.001, 0, 0])
    axis = np.asarray(port["axis_local"], float); axis /= np.linalg.norm(axis)
    lat = _lat_dir(axis)
    R = np.column_stack([lat, np.cross(axis, lat), axis])
    qf = np.zeros(4); mujoco.mju_mat2Quat(qf, R.reshape(-1))
    dev = spec.body(str(b["body"]))
    if dev is None:
        raise ValueError(f"the asset has no body {b['body']!r}")
    # the device is held weightless: gravcomp must be set before compiling (a model compiled without any gravcomp body
    # skips it, and the device then fell at g inside every step between the re-poses, a load the plug felt)
    dev.gravcomp = 1.0
    floor = np.asarray(port["mouth_local"], float) + pd * axis
    dev.add_site(name="_port_floor_site", pos=floor.tolist(), quat=qf.tolist(), size=[0.001, 0, 0])
    if prm["latch"]:
        # the weld add_connector_latch writes: device -> plug, inactive, held by the rollout's latch rule
        eq = spec.add_equality(name=_LATCH)
        eq.type = mujoco.mjtEq.mjEQ_WELD; eq.objtype = mujoco.mjtObj.mjOBJ_BODY
        eq.name1, eq.name2 = str(b["body"]), "_plug"
        eq.active = False
        eq.solref = list(E.LATCH_WELD_SOLREF); eq.solimp = list(E.LATCH_WELD_SOLIMP)
    model = spec.compile()
    if prm["latch"]:
        e = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, _LATCH)
        model.eq_data[e] = E.latch_seated_eq_data(model, "_plug_tip_site", "_port_floor_site")
    return model, tl


def _run(b, prm, trial):
    import mujoco
    port = b["port"]
    model, tl = _model(b, prm)
    d = mujoco.MjData(model)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(b["body"]))
    pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "_plug")
    dev_jnt = model.body_jntadr[bid]
    # the device is held: weightless (gravcomp, set in _model) and so heavy that the plug's reactions cannot move it
    # between the per-step re-poses (otherwise the plug creeps along with the device)
    model.body_mass[bid] *= 1e4
    model.body_inertia[bid] *= 1e4
    mouth = np.asarray(port["mouth_local"], float)
    axis = np.asarray(port["axis_local"], float); axis /= np.linalg.norm(axis)
    # device pose: identity, except the hold trial turns the port to face down (axis -> +z)
    if trial in ("hold", "extract"):
        qd = np.zeros(4); mujoco.mju_quatZ2Vec(qd, axis)  # rotation taking z to axis
        qinv = np.array([qd[0], -qd[1], -qd[2], -qd[3]])   # takes axis to z: port axis points up into the device
    else:
        qinv = np.array([1.0, 0, 0, 0])
    Rd = np.zeros(9); mujoco.mju_quat2Mat(Rd, qinv); Rd = Rd.reshape(3, 3)
    dev_pos = np.array([0, 0, 0.3])
    lat_dir = _lat_dir(axis)
    lateral = float(port["width"]) if trial == "blocked" else 0.0
    mouth_w = dev_pos + Rd @ (mouth + lateral * lat_dir)
    axis_w = Rd @ axis
    start = -0.003 if trial in ("insert", "blocked") else float(port["depth"]) * 0.98 - 0.0002
    qp = np.zeros(4)
    # plug local z = axis; local x = across the width (lateral dir)
    xw = Rd @ lat_dir
    zw = axis_w
    yw = np.cross(zw, xw)
    mujoco.mju_mat2Quat(qp, np.column_stack([xw, yw, zw]).reshape(-1))

    is_free = dev_jnt >= 0 and int(model.jnt_type[dev_jnt]) == int(mujoco.mjtJoint.mjJNT_FREE)

    def hold_device():
        if is_free:
            a = model.jnt_qposadr[dev_jnt]; v = model.jnt_dofadr[dev_jnt]
            d.qpos[a:a + 3] = dev_pos; d.qpos[a + 3:a + 7] = qinv; d.qvel[v:v + 6] = 0
    pj = model.jnt_qposadr[model.body_jntadr[pid]]
    pv = model.jnt_dofadr[model.body_jntadr[pid]]
    gain = 0.5 * float(prm["plug_mass"]) / model.opt.timestep

    mp = float(model.body_subtreemass[pid])
    Ip = float(np.max(model.body_inertia[pid]))
    w = 200.0
    kp, kd, kr, kdr = mp * w * w, 2 * mp * w, Ip * w * w, 2 * Ip * w

    def push():
        # a hand moving the plug: steady axial speed (force-capped), compliant lateral and
        # rotational alignment to the line through the (possibly offset) mouth, weight carried
        v = float(np.dot(d.qvel[pv:pv + 3], axis_w))
        f = float(np.clip(gain * (float(prm["push_speed"]) - v), -float(prm["insert_force_N"]), float(prm["insert_force_N"])))
        e = d.qpos[pj:pj + 3] - mouth_w
        e = e - np.dot(e, axis_w) * axis_w
        vl = d.qvel[pv:pv + 3] - np.dot(d.qvel[pv:pv + 3], axis_w) * axis_w
        F = f * axis_w - kp * e - kd * vl - mp * model.opt.gravity
        qc = d.qpos[pj + 3:pj + 7].copy()
        qe = np.zeros(4); qcc = np.array([qc[0], -qc[1], -qc[2], -qc[3]])
        mujoco.mju_mulQuat(qe, qp, qcc)
        rv = np.zeros(3); mujoco.mju_quat2Vel(rv, qe, 1.0)
        Rc = np.zeros(9); mujoco.mju_quat2Mat(Rc, qc)
        om = Rc.reshape(3, 3) @ d.qvel[pv + 3:pv + 6]
        d.xfrc_applied[pid, 3:] = kr * rv - kdr * om
        return F
    hold_device()
    if not is_free:
        raise ValueError("plug_port_retention_test: the device body must be a free body in its asset (free=True)")
    d.qpos[pj:pj + 3] = mouth_w + start * axis_w
    d.qpos[pj + 3:pj + 7] = qp
    mujoco.mj_forward(model, d)

    # the executor's latch rule, run exactly as servo_rollout runs it: latch_begin once, latch_step after every step
    live = E.latch_rules_live(model, [{"kind": "latch", "eq": _LATCH, "site_a": "_plug_tip_site", "site_b": "_port_floor_site",
                                       "seat_m": float(prm["seat_m"]), "release_N": float(prm["release_N"])}]) if prm["latch"] else []
    inactive, events = set(), []
    if live:
        E.latch_begin(model, d, live, inactive, None, events, 0)
    n_step = [0]

    def depth():
        return float(np.dot(d.xpos[pid] - mouth_w, axis_w))

    def step(force):
        hold_device()
        d.xfrc_applied[pid, :3] = force
        mujoco.mj_step(model, d)
        n_step[0] += 1
        if live:
            return E.latch_step(model, d, live, inactive, events, n_step[0])
        return []

    def engaged():
        return bool(live) and live[0]["state"] == "engaged"

    def engage_info():
        ev = next((x for x in events if x["event"] == "engage"), None)
        return None if ev is None else {"time_s": round(ev["frame"] * model.opt.timestep, 4), "gap_mm": ev["gap_mm"]}

    dt = model.opt.timestep
    if trial in ("insert", "blocked"):
        eng_depth = None
        for _ in range(int(float(prm["insert_s"]) / dt)):
            new = step(push())
            if any(x["event"] == "engage" for x in new) and eng_depth is None:
                eng_depth = depth()
        return {"depth": depth(), "finite": bool(np.all(np.isfinite(d.qpos))), "engaged": engaged(), "engage": engage_info(),
                "engage_depth": eng_depth, "events": events}
    # hold and extract: seat with a short push first
    for _ in range(int(0.3 / dt)):
        step(push())
    d0 = depth(); seated_latched = engaged()
    d.qvel[pv:pv + 6] = 0; d.xfrc_applied[pid, 3:] = 0
    if trial == "hold":
        n_hold = int(float(prm["hold_s"]) / dt)
        d_half = d0
        for i in range(n_hold):
            step(-float(prm["hold_load_N"]) * axis_w)
            if i == n_hold // 2:
                d_half = depth()
        # the creep rate over the second half of the hold, when any rebound from the push has died out
        rate = (d_half - depth()) / max((n_hold - 1 - n_hold // 2) * dt, 1e-9)
        return {"depth0": d0, "depth1": depth(), "slip": d0 - depth(), "creep_m_per_s": rate,
                "finite": bool(np.all(np.isfinite(d.qpos))), "latched_at_start": seated_latched, "latched_at_end": engaged(),
                "events": events}
    # extract
    T = 1.5; n = int(T / dt); fout = None; frel = None; prel = None
    for i in range(n):
        f = float(prm["max_extract_N"]) * (i + 1) / n
        new = step(-f * axis_w)
        rel = next((x for x in new if x["event"] == "release"), None)
        if rel is not None and frel is None:
            frel, prel = f, rel["pull_n"]
        if depth() < -0.001 and fout is None:
            fout = f
            break
    return {"depth0": d0, "extract_force": fout, "finite": bool(np.all(np.isfinite(d.qpos))), "latched_at_start": seated_latched,
            "release_force": frel, "release_pull": prel, "events": events}


def plug_port_retention_test(built, **params):
    unknown = sorted(set(params) - set(_DEF))
    if unknown:
        raise ValueError(f"plug_port_retention_test: unknown parameter(s) {unknown}")
    prm = {**_DEF, **params}
    prm["latch"] = bool(prm["latch"])
    if isinstance(built, (str, Path)):
        built = json.loads(Path(built).read_text())
    issues = []
    physics = {k: prm[k] for k in ("timestep", "integrator", "cone", "noslip_iterations", "impratio")}
    physics["episode_physics"] = all(str(physics[k]).lower() == str(_PHYS[k]).lower() if isinstance(_PHYS[k], str)
                                     else abs(float(physics[k]) - float(_PHYS[k])) < 1e-12 for k in _PHYS)
    retention = "latch" if prm["latch"] else "friction"
    rep = {"ok": False, "issues": issues, "retention": retention, "physics": physics}
    try:
        for k in ("xml", "body", "port"):
            if k not in built:
                raise ValueError(f"built lacks '{k}'")
        if float(prm["housing_scale"]) <= 1.0:
            raise ValueError("housing_scale must be > 1 so the housing stops at the edge")
        pd = float(built["port"]["depth"])
        ins = _run(built, prm, "insert")
        hold = _run(built, prm, "hold")
        ext = _run(built, prm, "extract")
        blk = _run(built, prm, "blocked")
    except Exception as e:  # report, never raise on a bad asset
        issues.append(f"could not run the trials: {type(e).__name__}: {e}")
        rep["verdict"] = "NOT RUN"
        return rep
    tl_needed = float(prm["seat_frac"]) * pd
    rate = max(0.0, float(hold["creep_m_per_s"]))
    rep.update(insert_depth_m=round(ins["depth"], 5), required_depth_m=round(tl_needed, 5),
               hold_start_depth_m=round(hold["depth0"], 5), hold_end_depth_m=round(hold["depth1"], 5),
               hold_slip_m=round(hold["slip"], 5), hold_creep_m_per_s=round(rate, 5),
               hold_seconds_to_release=(round(hold["depth1"] / rate, 2) if rate > 1e-6 and hold["depth1"] > 0 else None),
               extract_force_N=None if ext["extract_force"] is None else round(ext["extract_force"], 2),
               blocked_depth_m=round(blk["depth"], 5))
    lr = None
    if prm["latch"]:
        lr = {"seat_m": float(prm["seat_m"]), "release_N": float(prm["release_N"]),
              "engaged_in_insert": ins["engaged"], "engage_time_s": (ins["engage"] or {}).get("time_s"),
              "engage_depth_m": None if ins["engage_depth"] is None else round(ins["engage_depth"], 5),
              "engaged_through_hold": bool(hold["latched_at_start"] and hold["latched_at_end"]),
              "release_force_N": None if ext["release_force"] is None else round(ext["release_force"], 2),
              "release_pull_N": ext["release_pull"], "engaged_in_blocked": blk["engaged"],
              # what the port itself pushes the seated plug out with (squeezed lead-in pads): the latch carries it on top
              # of the pull, so the plug comes out that much below release_N
              "port_push_out_N": (round(float(ext["release_pull"]) - float(ext["release_force"]), 2)
                                  if ext["release_force"] is not None and ext["release_pull"] is not None else None)}
    rep["latch_report"] = lr
    if not all(t["finite"] for t in (ins, hold, ext, blk)):
        issues.append("simulation went non-finite; lower timestep or port_grip_m")
    if ins["depth"] < tl_needed:
        issues.append(f"plug reached {ins['depth']*1000:.2f} mm of the {pd*1000:.2f} mm cavity (< {prm['seat_frac']:.0%}): "
                      "the port is too tight or blocked; reduce port_grip_m or widen port_w/port_h")
    if ins["depth"] > pd + 0.001:
        issues.append("plug went deeper than the cavity floor: the cavity has no floor (check port_depth / slab geometry)")
    if lr is not None and ins["depth"] >= tl_needed and not ins["engaged"]:
        issues.append(f"plug reached {ins['depth']*1000:.2f} mm but the latch did not engage (its tip never came within seat_m = "
                      f"{float(prm['seat_m'])*1000:.1f} mm of the port floor): the cavity is deeper than the plug reaches; check port_depth or seat_m")
    if hold["depth1"] < float(prm["min_held_depth_m"]):
        thrown = " (thrown clear of the port)" if hold["depth1"] < -0.01 else ""
        if prm["latch"]:
            issues.append(f"plug fell out under {prm['hold_load_N']} N + gravity with the port facing down (depth {hold['depth1']*1000:.2f} mm{thrown}) "
                          f"although latched: {'the latch let go' if hold['latched_at_start'] else 'it never latched while seated'}; check seat_m and release_N")
        else:
            issues.append(f"plug fell out under {prm['hold_load_N']} N + gravity with the port facing down (depth {hold['depth1']*1000:.2f} mm{thrown}): "
                          "no friction hold; raise port_grip_m or port_friction, or give the connector its latch (add_connector_latch) "
                          "and test with latch=True")
    creep = hold["slip"] > float(prm["max_slip_m"])
    rep["hold_creep_warning"] = (
        f"plug crept from {hold['depth0']*1000:.2f} to {hold['depth1']*1000:.2f} mm deep in {prm['hold_s']} s "
        f"({rate*1000:.2f} mm/s at the end, out in about {rep['hold_seconds_to_release']} s more) while held: MuJoCo "
        "soft-friction creep at gram scale; a scene that must keep it seated longer should latch it once seated "
        "(add_connector_latch) or use noslip_iterations" if creep else None)
    if ext["extract_force"] is not None and ext["extract_force"] < float(prm["min_extract_N"]):
        issues.append(f"pull-out force {ext['extract_force']:.2f} N < {prm['min_extract_N']} N: the fit is loose; raise port_grip_m")
    if ext["extract_force"] is None:
        issues.append(f"plug did not come out under {prm['max_extract_N']} N: "
                      + (f"the latch did not let go (release_N {float(prm['release_N']):.1f} N); lower release_N or raise max_extract_N"
                         if prm["latch"] else "the fit is a weld; reduce port_grip_m"))
    if blk["depth"] > 0.0005:
        issues.append(f"misaligned plug entered {blk['depth']*1000:.2f} mm: the edge face does not block it (cavity too wide or face missing)")
    if lr is not None and blk["engaged"]:
        issues.append("the misaligned plug latched: the latch engaged off the slot; lower seat_m")
    rep["ok"] = not issues
    phys = (f"dt {float(prm['timestep']):g} s, {prm['integrator']}, {prm['cone']} cones, noslip {int(prm['noslip_iterations'])}"
            + (", the episode's physics" if physics["episode_physics"] else ""))
    if prm["latch"]:
        held = (f"latches {1000 * (lr['engage_depth_m'] or 0.0):.2f} mm deep and stays seated under {float(prm['hold_load_N']):.2f} N + gravity "
                f"(slip {1000 * hold['slip']:.3f} mm in {float(prm['hold_s']):.1f} s)")
        rel = f" (the latch lets go at {lr['release_force_N']} N, release_N {float(prm['release_N']):g})" if lr["release_force_N"] is not None else ""
        rep["verdict"] = (f"PORT OK (latch, {phys}): plug seats {1000 * ins['depth']:.2f} mm, {held}, pulls out at {ext['extract_force']:.1f} N"
                          f"{rel} and is blocked off-axis" if rep["ok"] else f"PORT FAILED (latch, {phys}): " + "; ".join(issues))
    else:
        held = ("holds by friction" if not creep else
                "stays in for the %.1f s hold but creeps out at %.1f mm/s" % (float(prm["hold_s"]), rate * 1000))
        rep["verdict"] = (f"PORT OK (friction fit, no latch, {phys}): plug seats, {held}, pulls out at {ext['extract_force']:.1f} N and is "
                          "blocked off-axis" if rep["ok"] else f"PORT FAILED (friction fit, no latch, {phys}): " + "; ".join(issues))
    if prm["out_json"]:
        Path(prm["out_json"]).parent.mkdir(parents=True, exist_ok=True)
        Path(prm["out_json"]).write_text(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    import sys
    print(json.dumps(plug_port_retention_test(sys.argv[1]), indent=1))

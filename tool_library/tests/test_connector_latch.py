"""The executor's connector latch on the real bench (setup/scene.xml) at its 0.002 s timestep: a construct_phone port and a
USB-C-sized test plug on a slide joint pushed and pulled by a constant-force drive (the hands hold their home pose).

  - add_connector_latch writes the inactive weld (seated relpose: the tip site's frame on the port floor site's frame)
    and returns the rule and the scene_recon fields; latch_rules_from_recon / Stage 4's physics_rules / the executor's
    plan loader read it back from the plug's entry;
  - servo_rollout: the plug pushed in bottoms out without latching while the push presses it, latches where it sits
    once let go (eq_active 0 -> 1 recorded, eq_active_start as the scene starts it), holds under a 2 N and a 12 N
    outward load in the next segments (latched handed over; without the hand-over it re-engages on the first step where
    it sits; handed over as pulled out it stays off), a 25 N pull frees it above release_N (recorded 1 -> 0), and pushed
    in again and let go it latches again;
  - a plug that stops 3 mm short of the floor never engages, pressed or let go;
  - the recorded episode passes equality_constraint_check, and Stage 4's Replay.replay of the same segments (same rule
    code, latches carried) ends where the executor's rollouts did.

    python tool_library/tests/test_connector_latch.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from tool_library.common import episode as E  # noqa: E402
from tool_library.common import rigid_robot as RR  # noqa: E402
from tool_library.common import stage4 as S4  # noqa: E402
from tool_library.skills.add_connector_latch import add_connector_latch  # noqa: E402
from tool_library.skills.construct_phone import construct_phone  # noqa: E402
from tool_library.skills.execute_subgoal_plan import _load_plan, latch_rules_from_recon  # noqa: E402
from tool_library.skills.stage_on_bench import stage_on_bench  # noqa: E402
from tool_library.verifiers.equality_constraint_check import equality_constraint_check  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="dexagent_latch_", dir=os.environ.get("TMPDIR")))
FAILS = 0
PX, PY = 0.10, 0.45                 # phone centre on the bench (rig metres), port edge toward -y
STANDOFF = 0.003                    # the plug tip starts this far outside the port mouth


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def plug_xml(path, stop_m):
    """A 6 g USB-C-sized plug (tip 8.2 x 2.4 mm, larger housing) on a slide joint along +y, the phone port's insertion axis,
    whose travel ends stop_m past the start (a stiff limit: a gram-scale body runs centimetres through MuJoCo's default
    one); a constant-force drive (affine bias, set per segment) pushes or pulls it. Its geoms take priority with a short
    contact time constant, as plug_port_retention_test's plug does.
    Its tip site is framed like construct_phone's port sites (x thin side, y wide side, z insertion)."""
    tl, hl = 0.0078, 0.013
    path.write_text(f"""<mujoco model="plug">
  <option timestep="0.002"/>
  <worldbody>
    <body name="plug">
      <joint name="slide" type="slide" axis="0 1 0" limited="true" range="-0.03 {stop_m:.6f}" solreflimit="0.004 1"/>
      <geom name="tip" type="box" size="0.0041 {tl / 2:.6f} 0.0012" pos="0 {-tl / 2:.6f} 0" mass="0.002" friction="1 0.01 0.001" condim="4" solref="0.004 1" priority="1"/>
      <geom name="housing" type="box" size="0.0063 {hl / 2:.6f} 0.0019" pos="0 {-tl - hl / 2:.6f} 0" mass="0.004" friction="1 0.01 0.001" condim="4" solref="0.004 1" priority="1"/>
      <site name="tip" pos="0 0 0" xyaxes="0 0 1 1 0 0" size="0.001"/>
    </body>
  </worldbody>
  <actuator><general name="drive" joint="slide" gainprm="0" biastype="affine" biasprm="0 0 -5"/></actuator>
</mujoco>
""")
    return path


def bench(ws, stop_m):
    ws.mkdir(parents=True, exist_ok=True)
    ph = construct_phone(str(ws / "phone"), free=False, timestep=0.002)
    L, T = ph["dims"]["length"], ph["dims"]["thickness"]
    pl = plug_xml(ws / "plug.xml", stop_m)
    stage_on_bench(str(ROOT / "setup" / "scene.xml"),
                   [{"name": "phone", "xml": ph["xml"], "x": PX, "y": PY},
                    {"name": "plug", "xml": str(pl), "x": PX, "y": PY - L / 2 - STANDOFF, "z": T / 2}],
                   str(ws / "scene.xml"), rig={"origin": [0, 0, 0], "R": np.eye(3).tolist()})
    return ph


def tip_gap(model, data):
    return float(np.linalg.norm(data.site_xpos[model.site("plug_tip").id] - data.site_xpos[model.site("phone_phone_port_floor").id]))


def drive(model, force_n):
    model.actuator_biasprm[model.actuator("plug_drive").id, 0] = float(force_n)


def end_gap(model, q):
    d = mujoco.MjData(model); d.qpos[:] = q; mujoco.mj_kinematics(model, d)
    return tip_gap(model, d)


def main():
    # ---------------- 1. the skill writes the latch; the readers pick it up
    ws = TMP / "full"
    ph = bench(ws, stop_m=0.012)
    res = add_connector_latch(str(ws / "scene.xml"), "plug_plug", "plug_tip", "phone_phone", "phone_phone_port_floor")
    rule, lf = res["rule"], res["scene_fields"]["latch"]
    m0 = mujoco.MjModel.from_xml_path(str(ws / "scene.xml"))
    e = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_EQUALITY, res["equality"])
    check("add_connector_latch writes an inactive body-body weld with the seated relpose", e >= 0 and not m0.eq_active0[e]
          and int(m0.eq_type[e]) == int(mujoco.mjtEq.mjEQ_WELD) and res["check"]["quat_norm"] == 1.0 and res["check"]["torquescale"] == 1.0
          and rule == {"kind": "latch", "eq": "latch_plug_plug", "site_a": "plug_tip", "site_b": "phone_phone_port_floor",
                       "seat_m": E.LATCH_SEAT_M, "release_N": E.LATCH_RELEASE_N}, json.dumps(res["check"]))
    # the seated relpose holds exactly where the tip sits on the floor (the slide travel to the floor is standoff + depth)
    d0 = mujoco.MjData(m0); mujoco.mj_resetDataKeyframe(m0, d0, 0)
    d0.qpos[m0.joint("plug_slide").qposadr[0]] = STANDOFF + ph["port"]["depth"]
    d0.eq_active[e] = 1; mujoco.mj_forward(m0, d0)
    rows = [i for i in range(d0.ne) if d0.efc_id[i] == e]
    check("the written relpose is the seated pose (weld residual zero with the tip on the floor)", tip_gap(m0, d0) < 1e-6
          and float(np.abs(d0.efc_pos[rows]).max()) < 1e-6, f"gap {tip_gap(m0, d0):.2e} m, residual {float(np.abs(d0.efc_pos[rows]).max()):.2e}")
    eqc = equality_constraint_check(str(ws / "scene.xml"))
    check("equality_constraint_check accepts the scene with the inactive latch", eqc["ok"], str(eqc["issues"])[:160])
    for bad, kw in (("a site on another body", dict(plug_site="phone_phone_port_floor")), ("an unknown body", dict(plug_body="nope")),
                    ("active while the plug is out of its port", dict(active=True))):
        try:
            add_connector_latch(str(ws / "scene.xml"), **{**dict(plug_body="plug_plug", plug_site="plug_tip", port_body="phone_phone",
                                                                   port_site="phone_phone_port_floor", out_xml=str(ws / "scene_bad.xml")), **kw})
            check(f"add_connector_latch refuses {bad}", False)
        except ValueError as ex:
            check(f"add_connector_latch refuses {bad}", True, str(ex)[:90])
    man = {"objects": [{"id": "phone", "type": "rigid", "root_body": "phone_phone"},
                       {"id": "cable", "type": "deformable", "root_body": "plug_plug", "latch": lf}]}
    (ws / "scene_recon.json").write_text(json.dumps(man))
    rr = latch_rules_from_recon(ws / "scene_recon.json")
    s4 = S4.physics_rules(ws / "scene_recon.json")
    _, plan_rules = _load_plan({"scene_recon_json": str(ws / "scene_recon.json"), "subgoals": []})
    two = dict(man, objects=[dict(man["objects"][1], latch=[lf, dict(lf, eq="latch_b", site_a="b_tip")])])
    (ws / "recon_two.json").write_text(json.dumps(two))
    check("latch_rules_from_recon, Stage 4's physics_rules and the plan loader read the entry's latch",
          rr == [rule] and [r for r in s4 if r.get("kind") == "latch"] == [rule] and [r for r in plan_rules if r.get("kind") == "latch"] == [rule]
          and len(latch_rules_from_recon(ws / "recon_two.json")) == 2, json.dumps(rr))

    # ---------------- 2. the rule in servo_rollout, segment by segment as execute_subgoal_plan runs them
    # The drive is a dedicated mechanism-test actuator. Apply its force to the
    # actual runtime model, never to the floating proposal after its source spec
    # was registered. Production episodes keep model constants immutable.
    runtime = RR.compile_rigid(ws / "scene.xml")
    model = runtime.model
    runtime_scene = ws / "scene_runtime.xml"
    runtime_scene.write_text(runtime.xml)
    check("the execution model runs at the bench's 0.002 s with elliptic cones and noslip", abs(model.opt.timestep - 0.002) < 1e-12
          and model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC and model.opt.noslip_iterations >= 5)
    names = E.equality_names(model)
    col = names.index(rule["eq"])
    d = mujoco.MjData(model); mujoco.mj_resetDataKeyframe(model, d, 0)
    q0 = d.qpos.copy()
    rules = [rule]
    segs = []

    def seg(force, q_start, T, latched=None, eq_inactive=(), keep=True):
        drive(model, force)
        r = E.servo_rollout(model, np.tile(q_start, (T, 1)), 30.0, qpos_start=q_start, rules=rules, latched=latched, eq_inactive=eq_inactive)
        if keep:
            segs.append(r)
        return r

    def events(r, kind=None):
        return [x for x in r["latch_events"] if kind is None or x["event"] == kind]

    r1 = seg(3.0, q0, 15)
    check("pushed in at 3 N it bottoms out but does not latch while the push presses it on the floor",
          not r1["latch_events"] and not r1["eq_active"][:, col].any() and end_gap(model, r1["qpos"][-1]) < E.LATCH_SEAT_M,
          f"tip {1000 * end_gap(model, r1['qpos'][-1]):.3f} mm from the floor")
    r1b = seg(0.0, r1["qpos"][-1], 10, latched=r1["latched"], eq_inactive=r1["eq_inactive"])
    eng = events(r1b, "engage")
    t_on = int(np.argmax(r1b["eq_active"][:, col])) if r1b["eq_active"][:, col].any() else None
    check("let go, it latches where it sits (within the first frame)", len(eng) == 1 and eng[0]["frame"] <= 1
          and eng[0]["gap_mm"] <= 1000 * E.LATCH_SEAT_M and rule["eq"] in r1b["latched"] and end_gap(model, r1b["qpos"][-1]) < E.LATCH_SEAT_M,
          f"engaged at frame {eng[0]['frame'] if eng else None}, {eng[0]['gap_mm'] if eng else None} mm from the floor")
    check("eq_active records it: inactive at the episode's start (as the scene starts it), on from the engaging frame",
          r1["eq_active_start"][col] == 0 and t_on is not None and t_on == eng[0]["frame"] and bool(r1b["eq_active"][t_on:, col].all())
          and not r1b["eq_active"][:t_on, col].any(), f"on from frame {t_on}")
    g1 = end_gap(model, r1b["qpos"][-1])
    r2 = seg(-2.0, r1b["qpos"][-1], 30, latched=r1b["latched"], eq_inactive=r1b["eq_inactive"])
    check("handed over (latched), it holds a 2 N outward load for 1 s with no new event", not r2["latch_events"] and bool(r2["eq_active"][:, col].all())
          and abs(end_gap(model, r2["qpos"][-1]) - g1) < 5e-5, f"moved {1e6 * abs(end_gap(model, r2['qpos'][-1]) - g1):.1f} um")
    r2b = seg(-12.0, r2["qpos"][-1], 15, latched=r2["latched"], eq_inactive=r2["eq_inactive"])
    check("and a 12 N pull, under release_N", not r2b["latch_events"] and bool(r2b["eq_active"][:, col].all()) and rule["eq"] in r2b["latched"]
          and abs(end_gap(model, r2b["qpos"][-1]) - g1) < 1e-4, f"moved {1e6 * abs(end_gap(model, r2b['qpos'][-1]) - g1):.1f} um")
    r2c = seg(-12.0, r2b["qpos"][-1], 10, keep=False)
    ev2c = r2c["latch_events"]
    check("without the hand-over it re-engages on the first step where it sits (no snap)", len(ev2c) == 1 and ev2c[0]["frame"] == 0
          and ev2c[0]["event"] == "engage" and bool(r2c["eq_active"][:, col].all())
          and abs(end_gap(model, r2c["qpos"][-1]) - end_gap(model, r2b["qpos"][-1])) < 5e-5, f"gap at re-engage {ev2c[0]['gap_mm'] if ev2c else None} mm")
    r2d = seg(0.0, r2b["qpos"][-1], 10, eq_inactive=[rule["eq"]], keep=False)
    check("a latch handed over as pulled out (eq_inactive) stays off while its plug is still in the port", not r2d["latch_events"]
          and not r2d["eq_active"][:, col].any() and rule["eq"] in r2d["eq_inactive"] and r2d["latched"] == {})
    r3 = seg(-25.0, r2b["qpos"][-1], 15, latched=r2b["latched"], eq_inactive=r2b["eq_inactive"])
    rel = events(r3, "release")
    t_off = int(np.argmin(r3["eq_active"][:, col])) if not r3["eq_active"][:, col].all() else None
    check("a 25 N pull frees it above release_N, recorded 1 -> 0; out of the port it is armed again",
          len(rel) == 1 and rel[0]["pull_n"] > E.LATCH_RELEASE_N and t_off == rel[0]["frame"] and not r3["eq_active"][t_off:, col].any()
          and r3["latched"] == {} and rule["eq"] not in r3["eq_inactive"] and end_gap(model, r3["qpos"][-1]) > 0.005,
          f"released at frame {t_off} with {rel[0]['pull_n'] if rel else None} N, tip {1000 * end_gap(model, r3['qpos'][-1]):.1f} mm out")
    r4 = seg(3.0, r3["qpos"][-1], 30, latched=r3["latched"], eq_inactive=r3["eq_inactive"])
    r4b = seg(0.0, r4["qpos"][-1], 10, latched=r4["latched"], eq_inactive=r4["eq_inactive"])
    check("pushed in again and let go, it latches again", not r4["latch_events"] and [x["event"] for x in r4b["latch_events"]] == ["engage"]
          and rule["eq"] in r4b["latched"] and end_gap(model, r4b["qpos"][-1]) < E.LATCH_SEAT_M)

    # the episode as execute_subgoal_plan writes it: eq_active concatenated, eq_active_start from the first segment
    traj = TMP / "episode.npz"
    Q = np.vstack([r["qpos"] for r in segs]); EQ = np.vstack([r["eq_active"] for r in segs])
    E.save_episode(traj, Q, ["seg"] * len(Q), 30.0, eq_active=EQ, eq_active_start=segs[0]["eq_active_start"], eq_names=np.array(names))
    ec = equality_constraint_check(str(runtime_scene), trajectory=str(traj), stride=1)
    on = np.flatnonzero(np.diff(np.concatenate([[segs[0]["eq_active_start"][col]], EQ[:, col]]).astype(int)))
    check("the recorded episode is consistent: equality_constraint_check passes; the latch switches exactly three times",
          ec["ok"] and len(on) == 3, f"switches at frames {on.tolist()}; {str(ec['issues'])[:160]}")

    # ---------------- 3. Stage 4's physics replay runs the same segments with the same rule code
    sys.path.insert(0, str(ROOT / "stage4_robot_data_generation"))
    from stage4.physics import Replay
    from tool_library.common import hand as H
    forces, bounds = [3.0, 0.0, -5.0], [(0, 14), (15, 24), (25, 44)]
    seq, frames, q = [], [], q0.copy()
    velocity, full_state, mechanism = np.zeros(model.nv), None, None
    for f, (a, b) in zip(forces, bounds):
        prev = seq[-1] if seq else {"latched": {}, "eq_inactive": []}
        drive(model, f)
        prefix = int(a > 0)
        result = E.servo_rollout(model, np.tile(q0, (b - a + 1 + prefix, 1)), 30.0,
            qpos_start=q, qvel_start=velocity, state_start=full_state, mechanism_state_start=mechanism,
            rules=rules, latched=prev["latched"], eq_inactive=prev["eq_inactive"])
        seq.append(result); frames.append(result["qpos"][prefix:])
        q, velocity = result["qpos"][-1], result["qvel"][-1]
        full_state, mechanism = result["final_state"], result["mechanism_final"]
    Qs = np.vstack(frames)
    rp = Replay.__new__(Replay)
    rp.E, rp.H, rp.model, rp.parts = E, H, model, E.robot_parts(model)
    rp.segments, rp.rules, rp.releases = [(a, b, {}) for a, b in bounds], rules, []
    rp.ep = type("Ep", (), {"fps": 30.0, "Q": Qs})()
    rp.initial_state = {k: seq[0][k] for k in ("initial_qpos", "initial_qvel", "eq_active_start", "initial_state", "eq_data_start")}
    rp.initial_state["mechanism_initial"] = seq[0]["mechanism_initial"]
    rp.last_scene_path, rp.scene_recon = runtime_scene, None
    import hashlib
    rp.source_scene_sha256 = hashlib.sha256(runtime_scene.read_bytes()).hexdigest()
    # This dedicated drive fixture mutates a test actuator between segments;
    # it tests state/latched-mechanism carry, not publishable scene provenance.
    # Real scene/hash/audit roundtrips are in Stage4 test_runtime_replay.py.
    rp.model_for = lambda _overrides: model
    drive(model, forces[0])

    def next_drive(m, Qr, diag, last):
        k = [b for _, b in bounds].index(last)
        if k + 1 < len(forces):
            drive(m, forces[k + 1])
        return False
    _, Qr, _, unstable, _ = rp.replay(Qs, np.tile(q0, (len(Qs), 1)), after_segment=next_drive)
    ps = model.joint("plug_slide").qposadr[0]
    check("Stage 4 preserves full state and latch mechanism across segments", unstable is None
          and abs(float(Qr[-1, ps] - Qs[-1, ps])) < 2e-5 and end_gap(model, Qr[-1]) < E.LATCH_SEAT_M and rule["eq"] in seq[-1]["latched"],
          f"plug {1e6 * abs(float(Qr[-1, ps] - Qs[-1, ps])):.2f} um apart at the end, tip {1000 * end_gap(model, Qr[-1]):.3f} mm from the floor")

    # ---------------- 4. a plug that stops short never latches
    ws2 = TMP / "short"
    bench(ws2, stop_m=STANDOFF + 0.0068 - 0.003)          # the travel ends 3 mm before the floor
    add_connector_latch(str(ws2 / "scene.xml"), "plug_plug", "plug_tip", "phone_phone", "phone_phone_port_floor")
    m2 = RR.compile_rigid(ws2 / "scene.xml").model
    d2 = mujoco.MjData(m2); mujoco.mj_resetDataKeyframe(m2, d2, 0)
    drive(m2, 3.0)
    r5 = E.servo_rollout(m2, np.tile(d2.qpos.copy(), (20, 1)), 30.0, qpos_start=d2.qpos.copy(), rules=rules)
    drive(m2, 0.0)
    r5b = E.servo_rollout(m2, np.tile(d2.qpos.copy(), (10, 1)), 30.0, qpos_start=r5["qpos"][-1], rules=rules)
    r5 = {k: (np.vstack([r5[k], r5b[k]]) if k in ("qpos", "eq_active") else r5[k] + r5b[k] if k == "latch_events" else r5b[k])
          for k in ("qpos", "eq_active", "latch_events", "latched")}
    c2 = E.equality_names(m2).index(rule["eq"])
    check("a plug that stops 3 mm short of the floor never engages, pressed or let go", not r5["latch_events"] and not r5["eq_active"][:, c2].any()
          and r5["latched"] == {} and end_gap(m2, r5["qpos"][-1]) > 0.002, f"tip {1000 * end_gap(m2, r5['qpos'][-1]):.2f} mm from the floor")

    print("ALL PASS" if not FAILS else f"{FAILS} FAILED")
    return FAILS


if __name__ == "__main__":
    sys.exit(1 if main() else 0)

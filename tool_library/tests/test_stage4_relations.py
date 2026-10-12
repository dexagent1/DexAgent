"""Relational goals in Stage 4 and how write_stage4_config infers them.

A synthetic free hand carries box A onto box B, then box C onto A (stage4_robot_data_generation/tests/synthetic.py).
Stage 4 must move the whole tower with B, resolve the relations in order, refuse cycles, and reject a layout whose
object would jump because its config says it rests on B while its hand put it down relative to the world. The
inference side must read each object's carrier and release from that object's own phases (a later let-go anchored
to A is not A's release) and find rests_on / inside from the plan, the release anchor and the final contacts.

    python tool_library/tests/test_stage4_relations.py
"""
import copy
import json
import os
import sys
import tempfile
from unittest.mock import patch
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "stage4_robot_data_generation" / "tests"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import synthetic  # noqa: E402
from tool_library.common import stage4 as S4  # noqa: E402
from tool_library.skills.write_stage4_config import write_stage4_config as _write_stage4_config  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="dexagent_s4rel_", dir=os.environ.get("TMPDIR")))
FAILS = 0
config = S4.stage4_module("config")
randomize = S4.stage4_module("randomize")


def write_stage4_config(scene_recon_json, episode_dir, *args, **kwargs):
    # These free-hand fixtures test relational inference only. The actual
    # runtime boundary is independently tested by test_runtime_replay.py.
    # They must never be treated as publishable physics episodes.
    scene = Path(scene_recon_json).parent / "scene.xml"
    trajectory = Path(episode_dir) / "trajectory.npz"
    with np.load(trajectory) as z:
        arrays = {key: z[key].copy() for key in z.files}
    arrays["reference"] = arrays["qpos"].copy()
    fixture = (scene, trajectory, mujoco.MjModel.from_xml_path(str(scene)), arrays)
    with patch.object(S4, "load_runtime_episode", return_value=fixture), patch.object(S4, "load_runtime_pair", return_value=fixture):
        return _write_stage4_config(scene_recon_json, episode_dir, *args, **kwargs)


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def episode(task, objects=None, segments=None):
    cfg_p, labels, phases = synthetic.stack_task(TMP / task, objects=objects, segments=segments)
    return randomize.Episode(config.load_config(cfg_p)), labels, phases


def end_pos(ep, Q, body):
    d = mujoco.MjData(ep.model)
    d.qpos[:] = Q[-1]
    mujoco.mj_kinematics(ep.model, d)
    return d.xpos[ep.model.body(body).id].copy()


def expected_follow(ep, layout, body, base="B"):
    """Where `body` must end when it rests (directly or through a stack) on `base`: base's start displacement
    applied to body's recorded end position."""
    dx, dy, dyaw = layout["objects"][base]
    p0 = ep.ref[("obj", base)][0][0]
    R = randomize.K.rot_z(np.radians(dyaw))
    return R @ (end_pos(ep, ep.Q, body) - p0) + p0 + np.array([dx, dy, 0.0])


def main():
    # 1. a two-level tower follows its randomized base, in dependency order
    ep, labels, phases = episode("tower")
    errs, oks = [], []
    for s in range(3):
        lay = ep.sample_layout(np.random.default_rng([0, s]))
        Q, _, rep = ep.adapt(lay)
        oks.append(rep["ok"])
        errs += [np.linalg.norm(end_pos(ep, Q, b) - expected_follow(ep, lay, b)) for b in ("A", "C")]
    check("tower follows its base (A on B, C on A)", all(oks) and max(errs) < 1e-6 and ep.order.index("B") < ep.order.index("A") < ep.order.index("C"),
          f"ok {oks}, worst end error {max(errs):.2e} m, order {ep.order}")

    # 2. without rests_on the release anchor of the carrier gives the same relation
    objs = json.loads((TMP / "tower" / "stage4.json").read_text())["randomize"]["objects"]
    plain = copy.deepcopy(objs)
    for k in ("A", "C"):
        plain[k].pop("rests_on")
    ep2, _, _ = episode("tower_anchor", objects=plain)
    lay = ep2.sample_layout(np.random.default_rng([0, 1]))
    Q2, _, rep2 = ep2.adapt(lay)
    e2 = max(np.linalg.norm(end_pos(ep2, Q2, b) - expected_follow(ep2, lay, b)) for b in ("A", "C"))
    check("release anchor alone stacks the same way", rep2["ok"] and e2 < 1e-6 and ep2.objects["C"]["rest"] == "A", f"err {e2:.2e}")

    # 3. a zero layout reproduces the trajectory exactly
    zero = {"objects": {n: [0.0, 0.0, 0.0] for n in ep.objects}, "robot": {}}
    Qz, _, _ = ep.adapt(zero)
    check("zero layout is the verified trajectory", np.array_equal(Qz, ep.Q))

    # 4. a cycle is refused with the cycle named
    cyc = copy.deepcopy(objs)
    cyc["A"]["rests_on"], cyc["C"]["rests_on"] = "C", "A"
    try:
        episode("cycle", objects=cyc)
        check("cycle refused", False, "no error")
    except ValueError as e:
        check("cycle refused", "cycle" in str(e) and "'A'" in str(e) and "'C'" in str(e), str(e)[:120])

    # 5. rests_on on an object that is never carried is refused (it has no release)
    bad = copy.deepcopy(objs)
    bad["B"]["rests_on"] = "A"
    try:
        episode("no_release", objects=bad)
        check("rests_on without a release refused", False, "no error")
    except ValueError as e:
        check("rests_on without a release refused", "release" in str(e), str(e)[:100])

    # 6. the config says A rests on B but the hand puts it down relative to the world: Stage 4 warns, and a layout that
    #    moves B makes A jump at its release (object_steps rejects it)
    segs = synthetic.stack_segments()
    segs["right"][2] = {"phases": ["right: carry A #2"], "blend": ["A", "world"]}
    segs["right"][3] = {"phases": ["right: set down A #2", "right: let go of A #2", "right: retreat #2"], "anchor": "world"}
    segs["right"][4] = {"phases": ["right: reach for C #3"], "blend": ["world", "C"]}
    ep3, _, _ = episode("mismatch", segments=segs)
    lay = ep3.sample_layout(np.random.default_rng([0, 0]))
    _, _, rep3 = ep3.adapt(lay)
    check("rests_on against a world-anchored set-down is caught", bool(ep3.warnings) and not rep3["object_steps"]["ok"],
          f"warnings {len(ep3.warnings)}, A step {rep3['object_steps'].get('A')}")

    # 7. carrier and release are read from the object's own phases: C's let-go is anchored to A, it is not A's release
    moved = {"A": 0.3, "B": 0.0, "C": 0.3}
    roles, _ = S4.object_roles(phases, ["A", "B", "C"], moved)
    old_style = [p["label"] for p in phases if p.get("anchor") == "A" and "let go" in p["label"]]
    check("object_roles: each object's own let-go", roles.get("A") == {"carrier": "right", "release": "right: let go of A #2"}
          and roles.get("C") == {"carrier": "right", "release": "right: let go of C #4"} and "B" not in roles
          and old_style[-1] == "right: let go of C #4", str(roles))

    # 8. rests_on from the plan, from the release anchor, and from the final contacts
    model = ep.model
    plan_sg = [{"motion": "place", "object": "A", "plan": {"motion": "place", "object": "A", "on": "B"}},
               {"motion": "place", "object": "C", "plan": {"motion": "place", "object": "C", "on": "A"}}]
    r_plan = S4.infer_relations(model, plan_sg, phases, roles, ["A", "B", "C"], q_end=ep.Q[-1])
    r_anchor = S4.infer_relations(model, [], phases, roles, ["A", "B", "C"], q_end=None)
    bare = [{k: v for k, v in p.items() if k != "anchor"} for p in phases]
    r_contact = S4.infer_relations(model, [], bare, roles, ["A", "B", "C"], q_end=ep.Q[-1])
    want = {"A": "B", "C": "A"}
    got = [{k: v.get("rests_on") for k, v in r.items()} for r in (r_plan, r_anchor, r_contact)]
    srcs = [{v["source"] for v in r.values()} for r in (r_plan, r_anchor, r_contact)]
    check("infer_relations: plan / release anchor / final contacts", all(g == want for g in got)
          and srcs == [{"plan"}, {"release anchor"}, {"final contacts"}], f"{got} {srcs}")
    none = S4.infer_relations(model, [], bare, roles, ["A", "B", "C"], q_end=ep.Q[0])
    check("no relation when nothing holds the object up but the floor", none == {}, str(none))

    # 9. inside a container vs on a plate
    bowl = ("<body name='bowl' pos='0.5 0.2 0'><freejoint/>"
            "<geom type='box' size='0.05 0.05 0.004' pos='0 0 0.004'/><geom type='box' size='0.004 0.05 0.03' pos='0.046 0 0.03'/>"
            "<geom type='box' size='0.004 0.05 0.03' pos='-0.046 0 0.03'/><geom type='box' size='0.05 0.004 0.03' pos='0 0.046 0.03'/>"
            "<geom type='box' size='0.05 0.004 0.03' pos='0 -0.046 0.03'/></body>"
            "<body name='cup' pos='0.5 0.2 0.0275'><freejoint/><geom type='box' size='0.02 0.02 0.02'/></body>"
            "<body name='plate' pos='0.7 0.2 0.005'><freejoint/><geom type='box' size='0.05 0.05 0.005'/></body>"
            "<body name='puck' pos='0.7 0.2 0.0195'><freejoint/><geom type='box' size='0.02 0.02 0.01'/></body>")
    synthetic.stack_task(TMP / "bowl", extra_xml=bowl)
    mb = mujoco.MjModel.from_xml_path(str(TMP / "bowl" / "scene.xml"))
    db = mujoco.MjData(mb)
    mujoco.mj_resetData(mb, db)
    sup_in = S4.resting_support(mb, db.qpos, "cup", ["bowl", "plate"])
    sup_on = S4.resting_support(mb, db.qpos, "puck", ["bowl", "plate"])
    check("resting_support: inside a bowl, on a plate", sup_in == ("bowl", "inside") and sup_on == ("plate", "rests_on"), f"{sup_in} {sup_on}")

    # 10. objects close together at their start move as one group; far apart ones do not
    groups, close = S4.interaction_groups(mb, db.qpos[None].repeat(3, 0), ["bowl", "cup", "plate", "puck", "A", "B"], clearance_m=0.01)
    gs = sorted(sorted(g) for g in groups)
    check("interaction_groups: touching objects grouped", gs == [["bowl", "cup"], ["plate", "puck"]], str(gs))
    lim = [S4.default_ranges(c)["dx"][1] for c in (0.002, 0.02, 0.2)]
    check("default_ranges: half the clearance, 5 mm .. 3 cm", lim == [0.005, 0.01, 0.03], str(lim))

    # 11. write_stage4_config inference on a synthetic Stage 3 folder (runtime boundary mocked)
    ws = TMP / "stage2"
    ws.mkdir()
    (ws / "scene.xml").write_text((TMP / "tower" / "scene.xml").read_text())
    (ws / "camera.json").write_text(json.dumps({"width": 64, "height": 48, "fovy_deg": 45.0, **{k: synthetic.CAMERA[k] for k in ("pos", "quat_wxyz")}}))
    (ws / "scene_recon.json").write_text(json.dumps({"scene_xml": "scene.xml", "camera_json": "camera.json", "objects": [
        {"id": n, "type": "rigid", "root_body": n, "asset_xml": f"assets/{n}/{n}.xml"} for n in ("A", "B", "C")]}))
    epd = TMP / "stage3" / "episode"
    epd.mkdir(parents=True)
    (epd / "phases.json").write_text(json.dumps(phases))
    np.savez_compressed(epd / "trajectory.npz", qpos=ep.Q, fps=30.0, labels=np.array(labels))
    np.savez_compressed(epd / "trajectory_hands.npz", qpos=ep.Q, fps=30.0, labels=np.array(labels), reference=ep.Q)
    (TMP / "stage3" / "trajectory_manifest.json").write_text(json.dumps({"subgoals": [
        {"index": 2, "frames": [0, 120], "motion": "place", "side": "right", "object": "A", "plan": plan_sg[0]["plan"], "expect": {"object": "A", "released": True}},
        {"index": 4, "frames": [121, len(labels) - 1], "motion": "place", "side": "right", "object": "C", "plan": plan_sg[1]["plan"], "expect": {"object": "C"}}]}))
    (TMP / "human.mp4").write_bytes(b"x")
    out = write_stage4_config(str(ws / "scene_recon.json"), str(epd), str(TMP / "task"), human_video=str(TMP / "human.mp4"))
    o = out["objects"]
    cfg = json.loads(Path(out["config"]).read_text())
    ep4 = randomize.Episode(config.load_config(out["config"]))
    lay = ep4.sample_layout(np.random.default_rng([0, 0]))
    _, _, rep4 = ep4.adapt(lay)
    check("write_stage4_config: carriers, releases, relations, manifest, reference",
          o["A"].get("release") == "right: let go of A #2" and o["C"].get("release") == "right: let go of C #4"
          and o["A"].get("rests_on") == "B" and o["C"].get("rests_on") == "A" and "carrier" not in o["B"]
          and cfg.get("manifest") == "trajectory_manifest.json" and out["reference"] and out["n_subgoals_with_expectations"] == 2
          and cfg["checks"]["physics_replay"] is True and rep4["ok"],
          f"{ {k: {kk: v for kk, v in d.items() if kk in ('carrier', 'release', 'rests_on')} for k, d in o.items()} } draw ok {rep4['ok']}")
    # a manifest left over from an earlier execution (its windows run past this episode) is not used
    man_p = TMP / "stage3" / "trajectory_manifest.json"
    stale = json.loads(man_p.read_text())
    stale["subgoals"][1]["frames"] = [121, len(labels) + 500]
    man_p.write_text(json.dumps(stale))
    out2 = write_stage4_config(str(ws / "scene_recon.json"), str(epd), str(TMP / "task_stale"), human_video=str(TMP / "human.mp4"))
    check("write_stage4_config ignores a stale manifest", out2["n_subgoals_with_expectations"] == 0
          and any("does not match this episode" in w for w in out2["warnings"]), str(out2["warnings"])[:120])

    # the transport of an object that rests on a moving body is anchored to that body, across subgoals, when the
    # planner anchored it to the world (a world target that happened to lie on the support)
    from tool_library.skills.write_stage4_config import _anchor_to_support
    worldly = [dict(p, anchor=None, subgoal=k) if p["label"].endswith("#2") and "A" in p["objects"] else dict(p)
               for k, p in enumerate(phases)]
    fixed = {p["label"]: p["anchor"] for p in _anchor_to_support(worldly, {"A": {"carrier": "right", "release": "right: let go of A #2", "rests_on": "B"}})}
    check("world-anchored transport re-anchored to the support", all(fixed[f"right: {v} #2"] == "B" for v in ("carry A", "set down A", "let go of A", "retreat"))
          and fixed["right: lift A #1"] == "A" and fixed["right: return home #5"] is None, str({k: v for k, v in fixed.items() if "#2" in k or "lift A" in k}))

    from tool_library.skills.write_stage4_config import _segments
    # a hand that lets go of something it held still (no anchor on the let-go) stays with that object until it
    # has backed off: blending to the world with the fingers still on a moved box drives them into it
    held = [{"label": "right: reach to hold box #1", "hand": "right", "objects": ["box"], "anchor": "box"},
            {"label": "right: hold box #1", "hand": "right", "objects": ["box"], "anchor": "box"},
            {"label": "right: let go of box #2", "hand": "right", "objects": ["box"], "anchor": None},
            {"label": "right: retreat #2", "hand": "right", "objects": ["box"], "anchor": None},
            {"label": "right: return home #3", "hand": "right", "objects": [], "anchor": None}]
    sg_h, _ = _segments(held, ["right"], ["box"])
    check("let-go of a held object stays with it, home blends to the world",
          sg_h["right"] == [{"phases": ["right: reach to hold box #1"], "blend": ["world", "box"]},
                            {"phases": ["right: hold box #1", "right: let go of box #2", "right: retreat #2"], "anchor": "box"},
                            {"phases": ["right: return home #3"], "blend": ["box", "world"]}], str(sg_h["right"]))

    # a label that recurs in two frame ranges (a retreat after a let-go at a world spot, and again after a press on
    # the object) gets one segment per range, each with its own anchor, so the hand does not jump between them
    rec = [{"label": "right: let go of jar #1", "hand": "right", "objects": ["jar"], "anchor": None, "frames": [[0, 9]]},
           {"label": "right: retreat #1", "hand": "right", "objects": ["jar"], "anchor": None, "frames": [[10, 19], [40, 49]]},
           {"label": "right: press jar #1", "hand": "right", "objects": ["jar"], "anchor": "jar", "frames": [[20, 29]]},
           {"label": "right: let go of jar lid #1", "hand": "right", "objects": ["jar"], "anchor": None, "frames": [[30, 39]]},
           {"label": "right: return home #2", "hand": "right", "objects": [], "anchor": None, "frames": [[50, 59]]}]
    sg_r, _ = _segments(rec, ["right"], ["jar"])
    check("a recurring label gets one segment per frame range", sg_r["right"] == [
        {"phases": ["right: let go of jar #1"], "anchor": "world"}, {"frames": [10, 19], "anchor": "world"},
        {"phases": ["right: press jar #1"], "blend": ["world", "jar"]}, {"phases": ["right: let go of jar lid #1"], "anchor": "jar"},
        {"frames": [40, 49], "anchor": "jar"}, {"phases": ["right: return home #2"], "blend": ["jar", "world"]}], str(sg_r["right"]))

    # two sides walking the same timed phases keep separate segment lists (a shared list grew on both sides)
    timed = [dict(p) for p in phases]
    for p in timed:
        p["frames"] = [[min(i for i, l in enumerate(labels) if l == p["label"]), max(i for i, l in enumerate(labels) if l == p["label"])]]
    sg_t, _ = _segments(timed, ["left", "right"], ["A", "B", "C"])
    check("timed phases: one segment list per side", sg_t["right"] == synthetic.stack_segments()["right"], str(sg_t["right"][:2]))

    # a part of an object (a latch on a lid) anchors the hand to the object it hangs from, whatever its name
    parts = [{"label": "right: reach for latch #1", "hand": "right", "objects": ["latch"], "anchor": "box_latch_x"},
             {"label": "right: close latch #1", "hand": "right", "objects": ["latch"], "anchor": "box_latch_x"}]
    sg_p, _ = _segments(parts, ["right"], ["box_body", "box_lid"], {"box_latch_x": "box_lid"})
    check("a part anchors the hand to its root object", sg_p["right"][0] == {"phases": ["right: reach for latch #1"], "blend": ["world", "box_lid"]}
          and sg_p["right"][1]["anchor"] == "box_lid", str(sg_p["right"]))

    # a "deformable" that is one free body (a plush toy) is randomized and carried; a rope (bodies on joints) is not
    from tool_library.skills.write_stage4_config import _single_rigid_body
    mr = mujoco.MjModel.from_xml_string("<mujoco><worldbody><body name='plush'><freejoint/><geom size='0.03'/></body>"
                                        "<body name='rope'><freejoint/><geom size='0.01'/><body pos='0.02 0 0'><joint type='ball'/>"
                                        "<geom size='0.01'/></body></body></worldbody></mujoco>")
    check("single rigid body vs rope", _single_rigid_body(mr, "plush") and not _single_rigid_body(mr, "rope"))

    # 12. simultaneous two-hand phases ('left: X | right: Y'), listed once per hand: each hand keeps its own anchor
    #     even where both change anchor in the same phase, and roles are read from each hand's part of the label
    from tool_library.skills.write_stage4_config import _segments
    two = []
    for lab, la, ra in (("left: reach for A | right: reach for C #1", "A", "C"), ("left: lift A | right: lift C #1", "A", "C"),
                        ("left: carry A | right: carry C #2", "B", "B"), ("left: let go of A | right: let go of C #2", "B", "B"),
                        ("left: return home | right: return home #3", None, None)):
        two += [{"label": lab, "hand": "left", "anchor": la, "subgoal": 1}, {"label": lab, "hand": "right", "anchor": ra, "subgoal": 1}]
    segs2, _ = _segments(two, ["left", "right"], ["A", "B", "C"])

    def at(side, lab):
        hits = [e for e in segs2[side] if lab in e["phases"]]
        return hits[0] if len(hits) == 1 else hits

    lab_c = "left: carry A | right: carry C #2"
    ok12 = (at("left", lab_c) == {"phases": [lab_c], "blend": ["A", "B"]} and at("right", lab_c) == {"phases": [lab_c], "blend": ["C", "B"]}
            and at("left", "left: reach for A | right: reach for C #1")["blend"] == ["world", "A"]
            and at("right", "left: reach for A | right: reach for C #1")["blend"] == ["world", "C"])
    roles2, _ = S4.object_roles([dict(p, objects=None) for p in two], ["A", "C"], {"A": 0.3, "C": 0.3})
    check("two-hand phases: own anchors per hand, roles per hand", ok12 and roles2 == {
        "A": {"carrier": "left", "release": "left: let go of A | right: let go of C #2"},
        "C": {"carrier": "right", "release": "left: let go of A | right: let go of C #2"}}, f"{at('right', lab_c)} {roles2}")
    # 13. a part set down right beside another randomized part gets next_to it (write_stage4_config, from the set-down's
    #     clearance), and Stage 4 keeps it there when the neighbour moves
    cfg_b, labels_b, phases_b = synthetic.beside_task(TMP / "beside", next_to=False)
    epb = randomize.Episode(config.load_config(cfg_b))
    ws = TMP / "stage2_beside"
    ws.mkdir()
    (ws / "scene.xml").write_text((TMP / "beside" / "scene.xml").read_text())
    (ws / "camera.json").write_text(json.dumps({"width": 64, "height": 48, "fovy_deg": 45.0, **{k: synthetic.CAMERA[k] for k in ("pos", "quat_wxyz")}}))
    (ws / "scene_recon.json").write_text(json.dumps({"scene_xml": "scene.xml", "camera_json": "camera.json", "objects": [
        {"id": n, "type": "rigid", "root_body": n, "asset_xml": f"assets/{n}/{n}.xml"} for n in ("A", "B")]}))
    epd = TMP / "stage3_beside" / "episode"
    epd.mkdir(parents=True)
    (epd / "phases.json").write_text(json.dumps(phases_b))
    np.savez_compressed(epd / "trajectory.npz", qpos=epb.Q, fps=30.0, labels=np.array(labels_b))
    outb = write_stage4_config(str(ws / "scene_recon.json"), str(epd), str(TMP / "task_beside"), human_video=str(TMP / "human.mp4"),
                               group_clearance_m=0.01)
    ob = outb["objects"]
    epb2 = randomize.Episode(config.load_config(outb["config"]))
    lay = epb2.sample_layout(np.random.default_rng([0, 0]))
    _, _, repb = epb2.adapt(lay)
    check("write_stage4_config: a part set down 5 mm beside another gets next_to it", ob["A"].get("next_to") == "B"
          and (outb["relations"].get("A") or {}).get("gap_m", 1.0) < 0.01 and "next_to" not in ob["B"] and repb["ok"],
          f"A {ob['A']} relations {outb['relations']} draw {repb['failed']}")
    # 14. bases fixed to the world (with or without moving parts) are found; free objects are not
    cabxml = ("<body name='cab' pos='0.45 0.50 0.05'><geom type='box' size='0.05 0.05 0.05'/>"
              "<body name='door' pos='0.05 0 0'><joint name='hinge' type='hinge' axis='0 0 1'/><geom type='box' size='0.004 0.05 0.05'/></body></body>")
    synthetic.stack_task(TMP / "cab", extra_xml=cabxml)
    mc = mujoco.MjModel.from_xml_path(str(TMP / "cab" / "scene.xml"))
    check("fixed_bases: the cabinet, not the free boxes", S4.fixed_bases(mc, ["cab", "A", "B", "door"]) == ["cab"], str(S4.fixed_bases(mc, ["cab", "A", "B", "door"])))
    print(f"{'ALL PASS' if FAILS == 0 else f'{FAILS} FAILED'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()

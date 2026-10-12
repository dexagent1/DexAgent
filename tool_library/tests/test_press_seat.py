"""press_seat: a press-fit part pushed home by the WRIST. A fingertip press moved a perched stud brick 0 mm in the lego
run (the finger servos' torque cap gives a few newtons at the pad, under the stud clutch). Cases with construct_stud_brick
bricks on the setup bench (the lego run's brick: 3 cm pitch, a 0.8 mm clutch margin, which leaves a dropped brick
perched 1.7 mm on the stud caps): a flat hand pressing with the 40 N wrist cap seats the brick (`seated`); the same
press capped at 1 N leaves it perched and the outcome says so; the press force is never planned above the 40 N cap; the
cap applies during the press only; the planner refuses a press without a seat target, from a hand that still holds
something, and an over-long travel. About a minute.

    MUJOCO_GL=egl python tool_library/tests/test_press_seat.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from tool_library.common import hand as H  # noqa: E402
from tool_library.skills.construct_stud_brick import construct_stud_brick  # noqa: E402
from tool_library.skills.execute_subgoal_plan import execute_subgoal_plan  # noqa: E402
from tool_library.skills.flat_hand_pose import flat_hand_pose  # noqa: E402
from tool_library.skills.hand_scene_from_recon import hand_scene_from_recon  # noqa: E402
from tool_library.skills.plan_subgoal_keyframes import plan_subgoal_keyframes  # noqa: E402
from tool_library.skills.rollout_keyframes_physics import _force_schedule  # noqa: E402
from tool_library.skills.stage_on_bench import stage_on_bench  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="dexagent_pressseat_", dir=os.environ.get("TMPDIR")))
FAILS = 0
BRICK = dict(pitch=0.030, body_height=0.026, stud_radius=0.0082, stud_height=0.0065, clutch_margin=0.0008, clearance=0.0006,
             lead_in_frac=0.6, cap_friction=0.05, mass=0.02)
BX, BY = 0.16, 0.50


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def main():
    H_, HS = BRICK["body_height"], BRICK["stud_height"]
    ws = TMP / "stage2"; ws.mkdir()
    lo = construct_stud_brick(str(TMP / "assets" / "lo"), name="lo", **BRICK)
    up = construct_stud_brick(str(TMP / "assets" / "up"), name="up", **BRICK)
    # the upper brick released just above the lower one's stud tops: it settles perched on the caps
    stage_on_bench(str(ROOT / "setup" / "scene.xml"), assets=[{"name": "lo", "xml": lo["xml"], "x": BX, "y": BY},
                                                              {"name": "up", "xml": up["xml"], "x": BX, "y": BY, "z": H_ + HS + 0.0005}],
                   out_xml=str(ws / "scene.xml"), rig={"origin": [0, 0, 0], "R": np.eye(3).tolist()})
    scene = str(ws / "scene.xml")
    man = {"scene_xml": "scene.xml", "objects": [{"id": n, "type": "rigid", "asset_xml": b["xml"], "root_body": f"{n}_{n}", "bodies": [f"{n}_{n}"]}
                                                 for n, b in (("lo", lo), ("up", up))]}
    (ws / "scene_recon.json").write_text(json.dumps(man))
    hs = hand_scene_from_recon(str(ws / "scene_recon.json"), "right", "up", out_json=str(TMP / "hs.json"))
    # straight fingers on the stud tops, pads down, the palm tilted up 25 deg (the arm holds it there)
    flat_hand_pose(hs, [BX, BY, 2 * H_ + HS], [0, 0, 1], [0, 1, 0], contact="pad", pitch_deg=25.0, out_json=str(TMP / "pose.json"))
    entry = {"index": 1, "motion": "press_seat", "side": "right", "object": "up_up", "pose": str(TMP / "pose.json"), "hand_scene": str(TMP / "hs.json"),
             "seat_on": "lo_lo", "seat_offset_m": lo["seat_offset_m"]}
    # 1. seated by the wrist
    s = execute_subgoal_plan(scene, {"subgoals": [entry]}, str(TMP / "press40"))
    r = json.loads((TMP / "press40" / "outcomes.json").read_text())[0]
    mm = r.get("measured", {})
    check("a flat hand pressing with the wrist (40 N cap) seats a perched stud brick", s["ok"] and r["checks"].get("seated") is True
          and mm.get("seat_travel_mm", 0) > 1.0
          and abs(mm.get("seat_remaining_mm", 9)) <= 1000 * r["expect"]["seated"]["tol_m"],
          f"travel {mm.get('seat_travel_mm')} mm, remaining {mm.get('seat_remaining_mm')} mm, press force {mm.get('peak_press_force_n')} N, {r.get('failed')}")
    check("the brick started perched on the stud caps", r["expect"]["seated"]["depth_m"] > 0.001, f"seat depth {r['expect']['seated']['depth_m']} m")
    check("the press force stays within the 40 N wrist cap", mm.get("peak_press_force_n", 99) <= H.WRIST_FORCE_N + 0.5, f"{mm.get('peak_press_force_n')} N")
    # 2. the same press capped at 1 N: the clutch holds, the outcome says the brick is not seated
    s1 = execute_subgoal_plan(scene, {"subgoals": [dict(entry, force_n=1.0)]}, str(TMP / "press1"))
    r1 = json.loads((TMP / "press1" / "outcomes.json").read_text())[0]
    m1 = r1.get("measured", {})
    check("a 1 N press leaves the brick perched and the outcome fails `seated` (only that)", not s1["ok"] and r1["failed"] == ["seated"]
          and m1.get("seat_remaining_mm", 0) > 0.5 and "short of its seat" in m1.get("hint", ""),
          f"remaining {m1.get('seat_remaining_mm')} mm, failed {r1.get('failed')}")
    # 3. never above the cap; the cap only during the press
    pl = plan_subgoal_keyframes(scene, dict(entry, force_n=80.0), auto_via=False)
    cap = pl["rollout"]["wrist_force_n"]["right"]
    check("a press asked for 80 N is planned at the 40 N wrist cap", pl["expect"]["press"]["force_n"] == H.WRIST_FORCE_N
          and pl["expect"]["press"]["force_clamped_from_n"] == 80.0 and max(cap["cap"]) <= H.WRIST_FORCE_N, json.dumps(pl["expect"]["press"]))
    pl1 = plan_subgoal_keyframes(scene, dict(entry, force_n=5.0), auto_via=False)
    labs = []
    for k in pl1["keyframes"]:
        labs += [k["label"]] * max(1, int(k.get("hold", 1)) + int(k.get("transit", 0)))
    F = _force_schedule(pl1["rollout"]["wrist_force_n"], labs, len(labs), H.WRIST_FORCE_N)["right"]
    on = [i for i, l in enumerate(labs) if l == "right: press up_up down"]
    off = [i for i, l in enumerate(labs) if l != "right: press up_up down"]
    check("the 5 N cap holds on the press frames only (z axis), 40 N elsewhere", on and np.allclose(F[on, 2], 5.0) and np.allclose(F[on, :2], 40.0)
          and np.allclose(F[off], 40.0), f"{len(on)} press frames")
    # 4. what the planner refuses
    errs = {}
    for name, sg, st in (("no seat target", {k: v for k, v in entry.items() if k not in ("seat_on", "seat_offset_m")}, None),
                         ("a hand still holding", entry, {"held": {"right": "lo_lo"}, "grip": {}}),
                         ("an over-long travel", dict(entry, travel_m=0.2), None)):
        try:
            plan_subgoal_keyframes(scene, sg, state=st, auto_via=False)
            errs[name] = None
        except ValueError as e:
            errs[name] = str(e)
    check("press_seat refuses a missing seat target, a holding hand and a travel over 5 cm", all(errs.values()), str({k: (v or "")[:50] for k, v in errs.items()}))
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}  ({TMP})")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())

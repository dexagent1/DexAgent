TOOL = {
    "name": "seat_grasp_for_lift",
    "title": "Seat a grasp for lifting",
    "type": "skill",
    "category": "Grasp & Poses",
    "summary": "Search the small seat a pose solved exactly at contact needs to lift: the wrist moved 0-4 mm deeper along "
               "its approach, with the clamp depth and pre-opening, judged by the approach-close-lift rollout in the hand "
               "scene and by the hand's visible overlap with the part and the table.",
    "description": (
        "Contact solvers (pose_from_points, generate_flat_pinches, handle_pinch_pose) put the pads exactly on the "
        "surface. The bench arm holds the hand where it is commanded (the hand scenes carry the hand's weight the same "
        "way), so a pinch whose pads only touch may close without wedging the part, and a seat a few millimetres deeper "
        "along the approach is worth trying before the pose is given up. With realistic pad contact (2 cm twisting, 1 mm "
        "rolling patches) a seat rarely decides it (a tripod on a thin pen lying on a table can lift at no seat, the "
        "closing thumb rolling the pen out), so every configuration is reported with why it failed. For each seat "
        "depth (shallowest first) and each clamp depth / pre-opening of the ladder, the hand is placed at the seated "
        "pose, its static overlap is measured (collision and visible meshes, pads included: the visible overlap with "
        "the part and the table must stay within max_visible_mm), and the shared stress rollout runs approach, close "
        "and lift. The first configuration that lifts wins: its pose is returned with `pos` moved, `seat_m`, "
        "`press_depth_rad` and `open_frac` written, ready for the executor. Every tried configuration is reported. "
        "Assumes the pose's approach vector points away from the part (the grasp tools' convention)."
    ),
    "inputs": [
        {"name": "scene", "type": "dict | str", "description": "Hand scene (hand_scene_from_recon output) holding the part as it is now."},
        {"name": "pose", "type": "dict | str", "description": "Hand pose {pos, euler, finger_targets, grip_fingers, approach, ...} or its json."},
        {"name": "seats_mm", "type": "list[float]", "description": "Seat depths along the approach, shallowest first. Default [0, 1, 2, 3, 4]."},
        {"name": "press_depths", "type": "list[float] | None", "description": "Clamp depths (rad past touch). Default [the pose's own or 0.12, 0.25]."},
        {"name": "open_fracs", "type": "list[float] | None", "description": "Pre-openings. Default [the pose's own or 0.2]."},
        {"name": "lift_heights_m", "type": "list[float]", "description": "Commanded lifts. Default [0.04, 0.08]."},
        {"name": "max_visible_mm", "type": "float", "description": "Largest visible overlap of the seated hand with the part or the table. Default 4.0."},
        {"name": "out_json", "type": "str | None", "description": "Write the seated pose here."},
    ],
    "outputs": [
        {"name": "result", "type": "dict", "description": "ok, pose (seated; None when nothing lifts), seat_m, press_depth_rad, open_frac, lift_mm [per commanded height], tried [{seat_mm, press_depth_rad, open_frac, visible_mm, ok, first_fail, lift_mm}]."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["grasp", "pinch", "seat", "lift", "thin-object", "pen", "stage3"],
    "used_in": ["Drawing and Wiping"],
    "status": "tested",
    "entrypoint": "seat_grasp_for_lift",
}

import json
from pathlib import Path

import mujoco
import numpy as np

from tool_library.common import hand as H


def seat_grasp_for_lift(scene, pose, seats_mm=(0.0, 1.0, 2.0, 3.0, 4.0), press_depths=None, open_fracs=None,
                        lift_heights_m=(0.04, 0.08), max_visible_mm=4.0, out_json=None):
    spec = H.load_scene_spec(scene)
    if isinstance(pose, (str, Path)):
        pose = json.loads(Path(pose).read_text())
    if not pose.get("finger_targets"):
        raise ValueError("pose has no finger_targets: give a pose from a grasp tool")
    model, data0, rig, ids = H.open_scene(spec)
    bid = H.grasp_body(model, spec, ids)
    d = mujoco.MjData(model); d.qpos[:] = data0.qpos
    H.apply_pose(model, d, rig, pose)
    axis = H.approach_axis(model, d, rig, pose)                # away from the part
    presses = list(press_depths) if press_depths else [float(pose.get("press_depth_rad") or 0.12), 0.25]
    opens = list(open_fracs) if open_fracs else [float(pose["open_frac"]) if pose.get("open_frac") is not None else 0.2]
    presses = list(dict.fromkeys(round(float(x), 4) for x in presses)); opens = list(dict.fromkeys(round(float(x), 4) for x in opens))
    pos0 = np.asarray(pose["pos"], float)
    tried, best = [], None
    for s_mm in seats_mm:
        seated = dict(pose); seated["pos"] = (pos0 - axis * float(s_mm) / 1000.0).tolist()
        dd = mujoco.MjData(model); dd.qpos[:] = data0.qpos; H.apply_pose(model, dd, rig, seated)
        cl = H.clearance(model, dd, rig, ids, ignore_tips=())
        vis = max(float(cl.get("visual_object_mm", cl["object_mm"])), float(cl.get("visual_floor_mm", cl["floor_mm"])))
        for pdr in presses:
            for of in opens:
                rec = {"seat_mm": float(s_mm), "press_depth_rad": pdr, "open_frac": of, "visible_mm": round(vis, 2)}
                if vis > float(max_visible_mm):
                    rec.update(ok=False, first_fail="visible_overlap", lift_mm=None); tried.append(rec); continue
                gg = dict(seated, press_depth_rad=pdr, open_frac=of)
                r = H.stress_rollout(model, data0, rig, ids, gg, bid, phases=("approach", "close", "lift"), lift_heights=tuple(lift_heights_m))
                rec.update(ok=bool(r["ok"]), first_fail=r["first_fail"], lift_mm=[x["achieved_mm"] for x in r["lifts"]])
                tried.append(rec)
                if r["ok"]:
                    best = (gg, rec)
                    break
            if best:
                break
        if best:
            break
    out = {"ok": best is not None, "pose": None, "tried": tried}
    if best:
        gg, rec = best
        gg = dict(gg, seat_m=round(rec["seat_mm"] / 1000.0, 4))
        out.update(pose=gg, seat_m=gg["seat_m"], press_depth_rad=rec["press_depth_rad"], open_frac=rec["open_frac"], lift_mm=rec["lift_mm"])
        if out_json:
            Path(out_json).parent.mkdir(parents=True, exist_ok=True)
            Path(out_json).write_text(json.dumps(gg, indent=1, default=float)); out["out_json"] = str(out_json)
    return out

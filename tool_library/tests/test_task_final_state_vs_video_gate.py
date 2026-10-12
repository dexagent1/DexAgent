"""task_final_state_vs_video_gate: the final state matching the video's last-frame masks, one object left 8 cm off, a
per-frame (moving) camera whose last pose is the right one, a robot hand over an object (ignored), a folder of masks with
a hands occluder, excluded objects, misuse."""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from tool_library.verifiers.task_final_state_vs_video_gate import task_final_state_vs_video_gate  # noqa: E402

FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f" {detail}"))
    FAILS += 0 if ok else 1


CAM_POS, CAM_QUAT, FOVY = [0.0, -0.45, 0.45], [0.9239, 0.3827, 0.0, 0.0], 45.0
XML = f"""<mujoco><worldbody>
<camera name="ego" pos="{CAM_POS[0]} {CAM_POS[1]} {CAM_POS[2]}" quat="{CAM_QUAT[0]} {CAM_QUAT[1]} {CAM_QUAT[2]} {CAM_QUAT[3]}" fovy="{FOVY}"/>
<body name="table" pos="0 0 -0.02"><geom type="box" size="0.4 0.4 0.02" rgba="0.3 0.3 0.6 1"/></body>
<body name="cup" pos="-0.08 0 0.03"><freejoint/><geom type="cylinder" size="0.03 0.03" rgba="0.2 0.4 0.9 1" mass="0.05"/></body>
<body name="bowl" pos="0.1 0.05 0.02"><freejoint/><geom type="box" size="0.05 0.05 0.02" rgba="0.2 0.8 0.3 1" mass="0.1"/></body>
<body name="robot_left_pad" pos="0.3 0.3 0.3"><freejoint/><geom type="box" size="0.03 0.03 0.005" rgba="0.8 0.8 0.8 1" mass="0.01"/></body>
</worldbody></mujoco>"""
W, H = 320, 240


def masks_at(model, q):
    data = mujoco.MjData(model)
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=H, width=W) as r:
        r.enable_segmentation_rendering()
        r.update_scene(data, camera="ego")
        seg = r.render()[..., 0]
    out = {}
    for nm in ("cup", "bowl"):
        g = [i for i in range(model.ngeom) if model.geom_bodyid[i] == model.body(nm).id]
        out[nm] = np.isin(seg, g)
    return out


def main():
    tmp = Path(tempfile.mkdtemp())
    scene = tmp / "scene.xml"
    scene.write_text(XML)
    m = mujoco.MjModel.from_xml_path(str(scene))
    q0 = m.qpos0.copy()
    cup, pad = (int(m.jnt_qposadr[m.body(b).jntadr[0]]) for b in ("cup", "robot_left_pad"))
    final = q0.copy()
    final[cup:cup + 3] = [0.1, 0.05, 0.07]             # the demonstration leaves the cup on the bowl
    video = masks_at(m, final)
    paths = {}
    for k, v in video.items():
        paths[k] = str(tmp / f"{k}.png")
        Image.fromarray(v.astype(np.uint8) * 255).save(paths[k])

    def save(name, last):
        Q = np.vstack([q0, last])
        f = tmp / f"{name}.npz"
        np.savez(f, qpos=Q, fps=30.0)
        return str(f)

    same = save("same", final)
    r1 = task_final_state_vs_video_gate(str(scene), same, paths, "ego")
    off = final.copy()
    off[cup:cup + 3] = [0.02, 0.05, 0.03]
    r2 = task_final_state_vs_video_gate(str(scene), save("off", off), paths, "ego")
    check("the final state matching the video passes; a cup left 8 cm off its spot fails, naming it",
          r1["ok"] and min(r1["iou"].values()) > 0.95 and (not r2["ok"]) and r2["failures"] == ["cup"], f"{r1['iou']} {r2['iou']}")
    moving = [{"frame": 0, "pos": [0.3, -0.4, 0.5], "quat_wxyz": [0.9, 0.3, 0.2, 0.1], "fovy_deg": 60.0, "width": W, "height": H},
              {"frame": 120, "pos": CAM_POS, "quat_wxyz": CAM_QUAT, "fovy_deg": FOVY, "width": W, "height": H}]
    r3 = task_final_state_vs_video_gate(str(scene), same, paths, moving)
    r4 = task_final_state_vs_video_gate(str(scene), same, paths, moving, video_frame=0)
    check("a per-frame camera: its last pose matches, its first pose (a different view) does not", r3["ok"] and not r4["ok"],
          f"{r3['iou']} {r4['iou']}")
    hover = final.copy()
    hover[pad:pad + 3] = [0.1, 0.05, 0.12]
    hv = save("hover", hover)
    r5 = task_final_state_vs_video_gate(str(scene), hv, paths, "ego")
    r6 = task_final_state_vs_video_gate(str(scene), hv, paths, "ego", ignore_robot=False)
    check("a robot hand parked over the cup is ignored, and lowers the IoU when not ignored",
          r5["ok"] and r5["iou"]["cup"] > r6["iou"]["cup"] + 0.05, f"{r5['iou']} {r6['iou']}")
    folder = tmp / "masks"
    folder.mkdir()
    for k in ("cup", "bowl"):
        Image.open(paths[k]).save(folder / f"{k}.png")
    hands = np.zeros((H, W), np.uint8)
    hands[:, : W // 2] = 255
    Image.fromarray(hands).save(folder / "hands.png")
    r7 = task_final_state_vs_video_gate(str(scene), save("off2", off), str(folder), "ego", exclude=["cup"])
    check("a folder of masks (hands.png taken as the occluder) with an excluded deviation passes", r7["ok"] and "cup" not in r7["iou"], str(r7))
    raised = []
    for args, kw in (((paths, "nocam"), {}), (({"ghost": paths["cup"]}, "ego"), {}), ((paths, "ego"), {"min_iou": 0.9, "warn_iou": 0.5}),
                     ((paths, dict(moving[1], width=W, height=W)), {})):
        try:
            task_final_state_vs_video_gate(str(scene), same, *args, **kw)
        except ValueError as e:
            raised.append(str(e).split(":")[0])
    check("an unknown camera, an object with no body, inverted thresholds and a camera of another aspect raise",
          raised == ["camera", "masks", "min_iou / warn_iou", "masks"], str(raised))
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())

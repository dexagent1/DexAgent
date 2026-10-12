"""hand.press_targets: a positive depth flexes the grip fingers past their closed targets and keeps a tool's squeezed
targets where they go deeper; a negative depth spreads the grip fingers (toward extension, the rings of scissors pushed
apart) and the squeezed targets, which flex, do not undo it; the fingers outside the grip never move. On the setup's
right hand, two finger sets."""
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tool_library.common import hand as H  # noqa: E402

FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def main():
    model, data0, rig, _ids = H.open_scene({"hand_xml": str(ROOT / "setup" / "robot" / "sharpa_right.xml"), "side": "right", "objects": [], "floor": False})
    flex = {k: v for k, v in rig.finger_range.items() if H.joint_role(k) == "flex"}
    mid = {k: 0.5 * (lo + hi) for k, (lo, hi) in flex.items()}
    toward_flex = lambda k: 1.0 if flex[k][1] >= flex[k][0] else -1.0      # noqa: E731
    for grip in (["thumb", "index"], ["index", "middle", "ring"]):
        g = [k for k in mid if H.finger_of_joint(k) in grip]
        others = [k for k in mid if H.finger_of_joint(k) not in grip]
        squeezed = {k: mid[k] + 0.3 * toward_flex(k) for k in g}              # a tool's deeper cinch
        pr = H.press_targets(rig, mid, grip, depth_rad=0.12, squeezed=squeezed)
        check(f"{'+'.join(grip)}: a press flexes the grip fingers and keeps the deeper squeezed targets",
              all(abs(pr[k] - squeezed[k]) < 1e-9 for k in g) and all(pr[k] == mid[k] for k in others))
        sp = H.press_targets(rig, mid, grip, depth_rad=-0.15, squeezed=squeezed)
        check(f"{'+'.join(grip)}: a negative depth spreads the grip fingers; the squeezed targets do not undo it",
              all((sp[k] - mid[k]) * toward_flex(k) < -0.1 for k in g) and all(sp[k] == mid[k] for k in others),
              str({k: round(sp[k] - mid[k], 3) for k in g[:3]}))
    print("ALL PASS" if not FAILS else f"{FAILS} FAILED")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())

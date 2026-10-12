"""Shared inputs of the grasp robustness verifier tests: hand scenes built from setup/robot/sharpa_<side>.xml and the
primitive objects in fixtures/grasp_robust, and grasp poses solved on them under the library's hand-object contact
(hand.HAND_PAIR_FRICTION): a three-finger pinch of a box through its centre of mass with the right hand (thumb against
index and middle), a five-finger top grasp of a cylinder with the left hand, and a deliberately weak two-pad pinch of the
same box (thumb against index only), which twists, drops and slips where the three-finger pinch holds."""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FX = ROOT / "tool_library" / "tests" / "fixtures" / "grasp_robust"

BOX_XML = """<mujoco model="{name}"><worldbody>
  <body name="{name}" pos="0 0 0.025"><freejoint/>
    <geom name="{name}_body" type="box" size="0.025 0.02 0.025" mass="{mass}" friction="{mu} 0.005 0.0001" rgba="0.8 0.45 0.2 1"/>
  </body></worldbody></mujoco>"""


def hand_xml(side):
    return str(ROOT / "setup" / "robot" / f"sharpa_{side}.xml")


def scene(obj="box", side="right", objects=None, **extra):
    """A hand-scene dict: the fixture object `obj` at the origin (or the given objects list), grasped last."""
    objs = objects if objects is not None else [{"name": obj, "xml": str(FX / f"{obj}.xml"), "pos": [0.0, 0.0, 0.0]}]
    out = {"hand_xml": hand_xml(side), "side": side, "objects": objs, "floor": True}
    out.update(extra)
    return out


def box_variant(tmp, name="box", mass=0.12, mu=1.0, pos=(0.0, 0.0, 0.0), side="right"):
    """The fixture box with another mass or friction, written to tmp (same size, so the fixture pinch still fits)."""
    p = Path(tmp) / f"{name}.xml"
    p.write_text(BOX_XML.format(name=name, mass=mass, mu=mu))
    return scene(side=side, objects=[{"name": name, "xml": str(p), "pos": list(pos)}])


def pose(name):
    return json.loads((FX / f"{name}.json").read_text())


BOX_PINCH3 = "box_right_pinch3"     # the sound right-hand grasp
BOX_PINCH = "box_right_pinch"       # the weak two-pad pinch (negative cases)
CYL_TOP = "cyl_left_top"


class Report:
    """[PASS]/[FAIL] lines and the exit code."""

    def __init__(self):
        self.failed = 0

    def check(self, name, cond, info=""):
        if cond:
            print(f"[PASS] {name}")
        else:
            self.failed += 1
            print(f"[FAIL] {name} {info}")
        return bool(cond)

    def raises(self, name, fn, exc=ValueError):
        try:
            fn()
        except exc as e:
            print(f"[PASS] {name} ({type(e).__name__}: {str(e)[:80]})")
            return True
        except Exception as e:                         # the wrong error
            self.failed += 1
            print(f"[FAIL] {name} raised {type(e).__name__}: {e}")
            return False
        self.failed += 1
        print(f"[FAIL] {name} did not raise")
        return False

    def exit(self):
        return 1 if self.failed else 0

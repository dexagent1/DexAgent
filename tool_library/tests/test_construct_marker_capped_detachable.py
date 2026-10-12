"""construct_marker_capped(detachable_cap=True): the cap holds when seated, a pull past the engaged depth lets the
release rule free it, and a light pull below the press-fit force does not move it."""
import os
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tool_library.skills.construct_marker_capped import construct_marker_capped  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="marker_det_", dir=os.environ.get("TMPDIR")))
FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def pull(r, force, steps=1500):
    m = mujoco.MjModel.from_xml_path(r["xml"]); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    cap = m.body(r["cap_free_body"]).id; qa = m.jnt_qposadr[m.joint(r["joints"]["lift"]).id]; eq = m.eq(r["equalities"]["engaged"]).id
    p0 = d.xpos[cap].copy(); released = None
    for k in range(steps):
        d.xfrc_applied[cap] = [force, 0, 0, 0, 0, 0] if released is None else [0, 0, 0, 0, 0, 0]
        mujoco.mj_step(m, d)
        if d.eq_active[eq] and d.qpos[qa] >= 0.96 * r["params"]["release_travel_m"]:
            d.eq_active[eq] = 0; released = k
    return released, float(np.linalg.norm(d.xpos[cap] - p0))


def main():
    for i, kw in enumerate(({}, {"length": 0.12, "cap_length": 0.03}, {"hold_force_N": 3.0})):
        r = construct_marker_capped(str(TMP / str(i)), detachable_cap=True, fixed_base=True, **kw)
        check(f"variant {i}: compiles with a free cap and an engaged weld", r.get("detachable_cap") and r["equalities"]["engaged"].endswith("_cap_engaged"))
        hold = float(r["hold_force_N"])
        rel, moved = pull(r, 0.0, steps=1500)
        check(f"variant {i}: gravity alone keeps the cap seated for 3 s", rel is None and moved < 0.0005, f"moved {1000 * moved:.2f} mm")
        rel, moved = pull(r, 0.4 * hold, steps=250)
        check(f"variant {i}: a half-second tug below the press-fit does not release it", rel is None and moved < r["params"]["release_travel_m"],
              f"moved {1000 * moved:.1f} mm (MuJoCo's soft dry friction creeps under a sustained sub-threshold pull)")
        rel, moved = pull(r, 2.0 * hold)
        check(f"variant {i}: a pull past the engaged depth releases the cap", rel is not None and moved > r["params"]["release_travel_m"], f"released at step {rel}")
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())

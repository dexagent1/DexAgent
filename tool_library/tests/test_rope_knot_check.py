"""rope_knot_check: an open overhand (cut trefoil) is a trefoil, an open U and a straight rope are the unknot, a cable
laid as an overhand in its own MuJoCo scene reads as a trefoil from a trajectory frame (and the straight frame does
not), and a rope through the table is reported."""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from tool_library.common import cable as C  # noqa: E402
from tool_library.common import knot as K  # noqa: E402
from tool_library.skills.construct_rope import construct_rope  # noqa: E402
from tool_library.verifiers.rope_knot_check import rope_knot_check  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="knot_", dir=os.environ.get("TMPDIR")))
FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def open_overhand(scale=0.1, lift=0.0):
    Q = K._param_curve("trefoil", n=400) * (scale / 0.1)
    k = int(np.argmax(np.linalg.norm(Q[:, :2] - Q[:, :2].mean(0), axis=1)))
    Q = np.roll(Q, -k, axis=0)[4:-4]
    Q[:, 2] += lift
    return Q


def main():
    r = rope_knot_check(points=open_overhand(lift=0.2).tolist())
    check("open overhand is a trefoil", r["ok"] and r["type"] == "trefoil", r["type"])
    u = np.linspace(0, 1, 60)
    U = np.stack([np.cos(np.pi * u) * 0.3, np.sin(np.pi * u) * 0.39, 0 * u + 0.014], 1)
    r = rope_knot_check(points=U.tolist())
    check("open U is not a knot", (not r["ok"]) and r["type"] == "unknot", r["issues"][0][:80] if r["issues"] else "")
    r = rope_knot_check(points=U.tolist(), expect="unknot")
    check("open U passes expect=unknot", r["ok"], r["type"])
    deep = open_overhand(lift=0.0)                # z swings +-0.1 around the table top
    r = rope_knot_check(points=deep.tolist())
    check("a rope through the table is reported", any("under the table" in s for s in r["issues"]), str(r["issues"])[:120])

    # a real cable: built, laid straight (frame 0) and as an overhand (frame 1), read through scene + trajectory
    built = construct_rope(str(TMP / "rope"), name="cord", length=1.2, segments=60, radius=0.004)
    model = mujoco.MjModel.from_xml_path(built["xml"])
    data = mujoco.MjData(model)
    Q = []
    C.set_centreline(model, data, built, np.array([[0.0, 0.0, 0.3], [1.2, 0.0, 0.3]]))
    Q.append(data.qpos.copy())
    C.set_centreline(model, data, built, open_overhand(scale=0.06, lift=0.3))
    Q.append(data.qpos.copy())
    np.savez(TMP / "traj.npz", qpos=np.asarray(Q), fps=30.0)
    mujoco.mj_kinematics(model, data)
    r = rope_knot_check(scene=built["xml"], trajectory=str(TMP / "traj.npz"), frame=1)
    check("cable laid as an overhand reads as a trefoil", r["ok"] and r["type"] == "trefoil", f"{r['type']} {r['frames'][0]}")
    r = rope_knot_check(scene=built["xml"], trajectory=str(TMP / "traj.npz"), frame=1, also_frames=[0])
    check("the straight frame fails the knot", (not r["ok"]) and r["frames"][1]["type"] == "unknot", str([f["type"] for f in r["frames"]]))
    # the same verdict through subgoal_outcome_check's knot expectation (the tie subgoal of a plan)
    from tool_library.verifiers.subgoal_outcome_check import subgoal_outcome_check
    o1 = subgoal_outcome_check(scene=built["xml"], trajectory=str(TMP / "traj.npz"), expect={"object": "cord", "knot": True}, frames=[0, 1])
    o0 = subgoal_outcome_check(scene=built["xml"], trajectory=str(TMP / "traj.npz"), expect={"object": "cordB_5", "knot": True}, frames=[0, 0])
    check("subgoal_outcome_check expect.knot", o1["ok"] and not o0["ok"], f"{o1.get('measured', {}).get('knot')} / {o0.get('failed')}")
    try:
        rope_knot_check(scene=built["xml"], trajectory=str(TMP / "traj.npz"), rope="nothere")
        check("unknown rope prefix raises", False)
    except ValueError as e:
        check("unknown rope prefix raises", "no cable bodies" in str(e))
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())

"""Smoke test for construct_cuttable_rope on three variations (a thin cord cut near one end, a looped cord with a
self-crossing cut at the point nearest a world point, a thick capped rope on an arc): the asset compiles, both halves
are laid on the measured centreline and meet exactly at the cut point, end caps sit on the real ends only, crossings
are lifted, the rope stages onto the setup bench through stage_on_bench at the origin with the scene names that
scene_fields reports, it settles in place, the join holds under a pull while active and lets the ends part once the
equality is switched off.

    MUJOCO_GL=egl python tool_library/tests/test_construct_cuttable_rope.py
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from tool_library.common import cutrope as CR  # noqa: E402
from tool_library.common.outcome import join_end_gap  # noqa: E402
from tool_library.skills.construct_cuttable_rope import construct_cuttable_rope  # noqa: E402
from tool_library.skills.stage_on_bench import stage_on_bench  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="dexagent_cutrope_", dir=os.environ.get("TMPDIR")))
FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def centrelines():
    # 1. a thin cord lying roughly straight across the table, gently bent
    x = np.linspace(-0.20, 0.15, 12)
    line = np.stack([x, 0.40 + 0.02 * np.sin(8 * x), np.full_like(x, 0.003)], 1)
    # 2. a cord with a teardrop loop that crosses itself (as a rope dropped on a cloth lies), then a tail
    u = np.linspace(0.0, 2 * np.pi * 0.85, 40)
    loop = np.stack([-0.05 + 0.06 * np.sin(u), 0.55 - 0.06 + 0.06 * np.cos(u) + 0.10 * u / (2 * np.pi), np.zeros_like(u)], 1)
    tail = np.stack([np.linspace(loop[-1, 0], 0.10, 8)[1:], np.linspace(loop[-1, 1], 0.30, 8)[1:], np.zeros(7)], 1)
    looped = np.concatenate([np.stack([np.linspace(0.06, loop[0, 0], 6)[:-1], np.full(5, 0.49), np.zeros(5)], 1), loop, tail])
    # 3. a thick rope on a half circle
    a = np.linspace(0.0, np.pi, 25)
    arc = np.stack([0.15 * np.cos(a), 0.30 + 0.15 * np.sin(a), np.full_like(a, 0.008)], 1)
    return [("thin cord, cut at 30 %", dict(centreline=line.tolist(), cut_frac=0.3, radius=0.003, segments=30), None),
            ("looped cord, cut nearest a world point", dict(centreline={"points": looped.tolist()}, radius=0.0041, segments=56, z=0.0041),
             [0.10, 0.36, 0.02]),
            ("thick capped rope on an arc, cut at 70 %", dict(centreline=arc.tolist(), cut_frac=0.7, radius=0.008, segments=24,
                                                              cap_length=0.03, cap_rgba=[0.8, 0.7, 0.2, 1.0]), None)]


def main():
    for k, (tag, params, cut_point) in enumerate(centrelines()):
        out = TMP / f"v{k}"
        kw = dict(params)
        built = construct_cuttable_rope(str(out / "rope"), cut_point=cut_point, scene_name=f"cord{k}", preview=(k == 1), **kw)
        P = CR.load_centreline(params["centreline"])
        if params.get("z") is not None:
            P[:, 2] = params["z"]
        check(f"[{tag}] builds two halves joined at the cut", built["join_gap_mm_at_build"] < 0.01
              and min(h["n_capsules"] for h in built["halves"].values()) >= 3,
              f"gap {built['join_gap_mm_at_build']} mm, capsules {[h['n_capsules'] for h in built['halves'].values()]}, length {built['length_m']} m")
        m = mujoco.MjModel.from_xml_path(built["xml"]); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        first = d.xpos[m.body(built["halves"]["a"]["first"]).id]
        hb_last = m.body(built["halves"]["b"]["last"]).id
        g_last = max((g for g in range(m.ngeom) if m.geom_bodyid[g] == hb_last), key=lambda g: g)
        end_b = d.xpos[hb_last] + d.xmat[hb_last].reshape(3, 3) @ np.array([2 * m.geom_size[g_last][1], 0, 0])
        e0 = float(np.linalg.norm(first[:2] - P[0, :2])); e1 = float(np.linalg.norm(end_b[:2] - P[-1, :2]))
        check(f"[{tag}] laid on the measured centreline (ends in place)", e0 < 0.002 and e1 < 0.004, f"start {1000 * e0:.2f} mm, end {1000 * e1:.2f} mm")
        if cut_point is not None:
            s, dist = CR.nearest_s(CR.plan_rope(dict(CR.DEFAULTS, **kw))["P"], cut_point)
            check(f"[{tag}] cut at the rope point nearest the given point", abs(s - built["cut"]["s_m"]) < 0.003,
                  f"s {built['cut']['s_m']} m (nearest {s:.4f}), {1000 * dist:.0f} mm from the point")
        else:
            check(f"[{tag}] cut at the requested arc-length fraction", abs(built["cut"]["frac"] - params["cut_frac"]) < 0.01, str(built["cut"]))
        if k == 1:
            check(f"[{tag}] self-crossing lifted", len(built["crossings"]) >= 1 and bool(built.get("stills")), f"{built['crossings']}")
        if k == 2:
            cap = np.array(params["cap_rgba"])
            capped = [m.body(int(m.geom_bodyid[g])).name for g in range(m.ngeom) if np.allclose(m.geom_rgba[g], cap, atol=1e-3)]
            near_join = {built["halves"]["a"]["last"], built["halves"]["b"]["first"]}
            check(f"[{tag}] end caps on the real ends only", len(capped) >= 2 and not (set(capped) & near_join)
                  and built["halves"]["a"]["first"] in capped and built["halves"]["b"]["last"] in capped, f"{capped}")
        # --- staged on the bench like construct_rope, at the origin
        sf = built["scene_fields"]
        stage_on_bench(str(ROOT / "setup" / "scene.xml"), [sf["stage_on_bench"]], str(out / "scene.xml"),
                       rig={"origin": [0, 0, 0], "R": np.eye(3).tolist()})
        M = mujoco.MjModel.from_xml_path(str(out / "scene.xml")); D = mujoco.MjData(M)
        mujoco.mj_resetDataKeyframe(M, D, 0); mujoco.mj_forward(M, D)
        j = sf["cut_joins"][0]
        names_ok = all(mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_BODY, b) >= 0 for b in sf["bodies"]) and \
            mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_EQUALITY, j["eq"]) >= 0 and mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_BODY, j["body"]) >= 0
        gap0 = join_end_gap(M, D, j["eq"])
        check(f"[{tag}] staged: scene_fields names exist, join continuous in the keyframe", names_ok and gap0 < 1e-4,
              f"join gap {1000 * gap0:.3f} mm, eq {j['eq']}")
        rope_g = [g for g in range(M.ngeom) if M.body(int(M.geom_bodyid[g])).name in set(sf["bodies"])]
        worst = min([D.contact[i].dist for i in range(D.ncon) if D.contact[i].geom1 in rope_g or D.contact[i].geom2 in rope_g] + [0.0])
        check(f"[{tag}] no rope interpenetration in the keyframe", worst > -0.001, f"deepest {1000 * worst:.2f} mm")
        ids = [M.body(b).id for b in sf["bodies"]]
        x0 = D.xpos[ids].copy()
        for _ in range(int(0.5 / M.opt.timestep)):
            mujoco.mj_step(M, D)
        drift = float(np.abs(D.xpos[ids] - x0).max())
        check(f"[{tag}] settles where it was laid", drift < 0.006 and int(D.warning[mujoco.mjtWarning.mjWARN_BADQACC].number) == 0,
              f"drift {1000 * drift:.2f} mm")
        # --- the join holds while active and parts once it is switched off: pull the second half's far end away
        # (a lift of the far end at twice the second half's weight: off the table, so friction does not hide anything)
        bl = M.body(sf["halves"]["b"]["last"]).id
        w_b = 9.81 * sum(float(M.body_mass[M.body(b).id]) for b in sf["halves"]["b"]["bodies"])
        pull = np.array([0.0, 0.0, 3.0 * w_b])
        for _ in range(int(0.4 / M.opt.timestep)):
            D.xfrc_applied[bl, :3] = pull
            mujoco.mj_step(M, D)
        held = join_end_gap(M, D, j["eq"])
        D.eq_active[M.equality(j["eq"]).id] = 0
        for _ in range(int(0.3 / M.opt.timestep)):
            D.xfrc_applied[bl, :3] = pull
            mujoco.mj_step(M, D)
        parted = join_end_gap(M, D, j["eq"])
        # whole = the join stays within one rope radius under three times the half's weight (the CONNECT is a soft
        # constraint on gram-scale capsules); parted = the ends a few millimetres apart once it is off
        check(f"[{tag}] whole under a pull, parts when the join is switched off",
              held < float(params["radius"]) and parted > 0.005 and np.isfinite(D.qpos).all(),
              f"pull {pull[2]:.2f} N: gap while whole {1000 * held:.2f} mm, after the cut {1000 * parted:.1f} mm")
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}  ({TMP})")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())

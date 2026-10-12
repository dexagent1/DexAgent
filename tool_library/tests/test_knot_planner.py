"""Rope over-moves for a knot, without contact physics (common/knotplan.py, common/ropemoves.py, knot.crossing_order):

1. the crossing order of an open overhand is 'a+ b- c+ a- b+ c-' (interleaved, alternating) and the fast determinant
   screen agrees with knot.check;
2. the follow-the-leader predictor: an end carried across a straight rope makes one loop with the carried rope on top
   (rope_knot_check reads the predicted diagram as the unknot), a move that would drag a pinned link or close a loop
   tighter than a thick rope can is refused, a sideways carry leaves the far end of a long U where it lay (pure
   follow-the-leader drags it), a tail pulled along itself through a crossing keeps its order there;
3. the overhand as three over-moves (overhand_search) on a U lying on the table: loop, tail over it, loop over the
   tail; every intermediate prediction is unknotted, the last is a trefoil that rope_knot_check confirms;
4. the executor's `lay` and `drag` on a bench cable (construct_rope staged on the setup bench), planned only: the
   phases (reach, lift, carry, set down, let go with the wrist still, retreat), the link carried clear of the strand
   it passes over, and the expectations (target in x-y, released, at rest, the crossing order); the executed crossing
   order is matched to the planned one with extra curls dropped, never the planned crossings.

    MUJOCO_GL=egl python tool_library/tests/test_knot_planner.py      # about two minutes
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from tool_library.common import knot as K  # noqa: E402
from tool_library.common import knotplan as KP  # noqa: E402
from tool_library.common import ropemoves as RM  # noqa: E402
from tool_library.verifiers.rope_knot_check import rope_knot_check  # noqa: E402

R = 0.016


def resample(P, n):
    P = np.asarray(P, float)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, P[:, i]) for i in range(P.shape[1])], 1)


def u_points(L, w=0.40, y0=0.30):
    """A U lying on the table, both ends toward the operator (an arch), the first end on the left."""
    h = (L - np.pi * w / 2) / 2
    u = np.linspace(np.pi, 0, 40)
    arc = np.c_[w / 2 * np.cos(u), y0 + h + w / 2 * np.sin(u)]
    return np.vstack([[[-w / 2, y0]], arc, [[w / 2, y0]]])


def rope_of(P2, seg=0.036):
    L = float(np.linalg.norm(np.diff(P2, axis=0), axis=1).sum())
    P = resample(P2, int(round(L / seg)) + 1)
    return KP.Rope.from_points(np.c_[P, np.full(len(P), R)], R)


def near(rp, xy):
    c = np.array([rp.centre(k) for k in range(rp.n - 1)])
    return int(np.argmin(np.linalg.norm(c - np.asarray(xy), axis=1)))


class CrossingOrder(unittest.TestCase):
    def test_open_overhand_order_and_screen(self):
        Q = K._param_curve("trefoil", n=400)
        k = int(np.argmax(np.linalg.norm(Q[:, :2] - Q[:, :2].mean(0), axis=1)))
        Q = np.roll(Q, -k, axis=0)[4:-4]
        od = K.crossing_order(Q)
        self.assertEqual(od["letters"], "abcabc")
        self.assertTrue(od["alternating"] and od["interleaved"], od["code"])
        got = {kd: K.alexander_det(K._param_curve(kd)) for kd in ("trefoil", "figure8", "circle", "kink")}
        self.assertEqual(got, {"trefoil": 3, "figure8": 5, "circle": 1, "kink": 1})
        self.assertEqual(K.quick_type(Q), "trefoil")
        self.assertEqual(K.check(Q)["type"], "trefoil")

    def test_executed_order_against_planned(self):
        P = RM.parse_code
        self.assertTrue(RM.matches_order(*P("a- a+"), *P("a- a+")))
        self.assertTrue(RM.matches_order(*P("a- b+ b- a+"), *P("a- a+")))          # an extra curl of the executed rope
        self.assertTrue(RM.matches_order(*P("a+ b+ b- c- a- c+"), *P("a+ b- a- b+")))
        self.assertFalse(RM.matches_order(*P(""), *P("a- a+")))                    # the planned loop is never dropped
        self.assertFalse(RM.matches_order(*P("a+ a-"), *P("a- a+")))               # the loop laid the other way round


class Predictor(unittest.TestCase):
    def test_end_over_straight_rope_is_one_loop_on_top(self):
        rp = rope_of(np.array([[-0.5, 0.45], [0.5, 0.45]]))
        k = rp.n - 2
        # (a 15 cm loop: allowed here, refused at the default min_loop_m, see test_tight_loop_refused)
        nr, info = KP.lay(rp, k, [[0.25, 0.72], [0.1, 0.30]], 0.05, min_loop_m=0.1)
        self.assertIsNotNone(nr, info)
        od = nr.order()
        self.assertEqual((od["letters"], od["over"]), ("aa", "-+"))       # the carried (later) rope on top
        v = rope_knot_check(points=nr.polyline().tolist(), expect="unknot")
        self.assertTrue(v["ok"], v["issues"])
        nr2, why = KP.lay(rp, k, [[0.25, 0.72], [0.1, 0.30]], 0.05, pins=(k - 4,), min_loop_m=0.1)
        self.assertIsNone(nr2)
        self.assertIn("pinned", why)

    def test_tight_loop_refused(self):
        rp = rope_of(np.array([[-0.5, 0.45], [0.5, 0.45]]))
        k = rp.n - 2
        # the end swung back over the rope: the loop it closes is 15 cm round, tighter than a 3 cm rope lies
        nr, why = KP.lay(rp, k, [[0.25, 0.72], [0.1, 0.30]], 0.05)
        self.assertIsNone(nr)
        self.assertIn("loop", why)

    def test_far_end_stays_and_tail_keeps_its_order(self):
        U = rope_of(u_points(1.5))
        kk = near(U, (0.20, 0.36))
        tgt = [U.centre(kk) + np.array([-0.46, 0.06])]
        pr, info = KP.lay(U, kk, tgt, 0.05)
        ft, _ = KP.lay(U, kk, tgt, 0.05, model="ftl", step_m=0.006)
        self.assertIsNotNone(pr, info)
        far_p = float(np.linalg.norm(pr.xy[0] - U.xy[0]))
        far_f = float(np.linalg.norm(ft.xy[0] - U.xy[0])) if ft is not None else 1.0
        self.assertLess(far_p, 0.03)
        self.assertGreater(far_f, 0.1)
        self.assertEqual(pr.order()["n_crossings"], 1)
        # the tail lying UNDER the crossing (the first end) dragged out along itself: the crossing slides along it and
        # stays an under-crossing (a pull-through keeps its order)
        self.assertEqual(pr.order()["over"], "-+")
        d = pr.xy[0] - pr.xy[1]
        nr, why = KP.lay(pr, 0, [pr.centre(0) + 0.06 * d / np.linalg.norm(d)], RM.DRAG_LIFT_M)
        self.assertIsNotNone(nr, why)
        self.assertEqual(nr.order()["code"], pr.order()["code"])
        self.assertNotEqual({(c["i"], c["j"]) for c in nr.crossings()}, {(c["i"], c["j"]) for c in pr.crossings()})


class OverhandSearch(unittest.TestCase):
    def test_three_over_moves_tie_a_trefoil(self):
        U = rope_of(u_points(1.5))
        box = (-0.42, 0.42, 0.24, 0.70)
        inb = lambda p: box[0] <= p[0] <= box[1] and box[2] <= p[1] <= box[3]
        plans, st = KP.overhand_search(U, lambda s_, r_, k: [{"flip": 0}] if inb(r_.centre(k)) else [],
                                       lambda s_, r_, k, p_, path: all(inb(q) for q in path), bounds=box, sides=("left",),
                                       max_results=1, time_budget_s=600)
        self.assertTrue(plans, st)
        states = KP.replay(U, plans[0]["moves"], 0.05)
        types = [s_.knot_type() for s_ in states]
        self.assertEqual(types[-1], "trefoil", [s_.order()["code"] for s_ in states])
        self.assertNotIn("trefoil", types[:-1])
        self.assertEqual(states[0].order()["n_crossings"], 1)
        self.assertEqual(states[1].order()["letters"], "abab")
        v = rope_knot_check(points=states[-1].polyline().tolist())
        self.assertTrue(v["ok"], v["issues"])


class LayOnBench(unittest.TestCase):
    """The executor's lay / drag, planned on a construct_rope cable staged on the setup bench (no physics)."""

    @classmethod
    def setUpClass(cls):
        from tool_library.skills.construct_rope import construct_rope
        from tool_library.skills.stage_on_bench import stage_on_bench
        cls.tmp = Path(tempfile.mkdtemp(prefix="dexagent_knot_bench_", dir=os.environ.get("TMPDIR")))
        pts = u_points(1.2)
        L = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        b = construct_rope(str(cls.tmp / "assets/rope"), name="rope", centreline=pts.tolist(), radius=R, segments=int(round(L / 0.037)) + 1,
                           grips=False, include_floor=False, friction=[1.0, 0.02, 0.001], density=700)
        cls.scene = str(cls.tmp / "scene.xml")
        stage_on_bench(str(ROOT / "setup" / "scene.xml"), assets=[{"name": "rope", "xml": b["xml"], "x": 0.0, "y": 0.0}], out_xml=cls.scene,
                       rig={"origin": [0, 0, 0], "R": np.eye(3).tolist()})

    def _plan(self, motion, lift_m=None):
        import mujoco
        from tool_library.skills.plan_subgoal_keyframes import plan_subgoal_keyframes
        from tool_library.common import subgoal as SG
        from tool_library.verifiers.rope_knot_check import _cable_bodies
        self.assertIn(motion, SG.MOTIONS)
        m = mujoco.MjModel.from_xml_path(self.scene)
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        pre, bodies = _cable_bodies(m, None)
        k = 3                                               # a link of the left leg, laid over the right leg
        name = m.body(bodies[k]).name
        a, c = d.xpos[bodies[k]].copy(), d.xpos[bodies[k + 1]].copy()
        ax = (c - a) / np.linalg.norm(c - a)
        side = np.cross([0, 0, 1.0], ax)
        pose = {"side": "right", "pos": (a + 0.5 * (c - a) + np.array([0, 0, 0.12])).tolist(), "euler": [np.pi, 0.0, float(np.arctan2(ax[1], ax[0]))],
                "finger_targets": {}, "grip_fingers": ["thumb", "index"], "approach": [0.0, 0.0, 1.0],
                "points": {"thumb": (a + 0.5 * (c - a) + R * side).tolist(), "index": (a + 0.5 * (c - a) - R * side).tolist()}}
        sg = {"index": 1, "motion": motion, "side": "right", "object": name, "pose": pose, "target_pos": [0.33, float(a[1]) + 0.04],
              "expect_order": "a+ b+ b- a-"}
        if lift_m is not None:
            sg["lift_m"] = lift_m
        out = plan_subgoal_keyframes(self.scene, sg, out_json=str(self.tmp / f"{motion}.json"))
        return out, m, bodies, k

    def test_lay_phases_and_expectations(self):
        out, m, bodies, k = self._plan("lay")
        labels = [kf["label"] for kf in out["keyframes"]]
        for word in ("lift", "carry", "set down", "let go", "retreat"):
            self.assertTrue(any(word in l for l in labels), labels)
        exp = out["expect"]
        self.assertTrue(exp["released"] and exp["at_rest"])
        self.assertEqual(exp["target_axes"], "xy")
        self.assertEqual(exp["crossings"]["code"], "a+ b+ b- a-")
        # carried clear of the right leg: the link's underside 12 mm above the top of the strand under its path
        self.assertGreaterEqual(exp["lay"]["carry_z"], 2 * R + R + RM.LAY_CLEAR_M - 1e-6)
        self.assertGreaterEqual(exp["lay"]["lift_m"], RM.LAY_MIN_LIFT_M - 1e-6)
        names = {m.body(b).name for b in bodies}
        self.assertTrue(names <= set(exp["involved"]))       # the whole cable moves: no link is a disturbed bystander

    def test_drag_stays_low(self):
        out, *_ = self._plan("drag")
        self.assertEqual(out["expect"]["lay"]["kind"], "drag")
        self.assertAlmostEqual(out["expect"]["lay"]["lift_m"], RM.DRAG_LIFT_M, places=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)

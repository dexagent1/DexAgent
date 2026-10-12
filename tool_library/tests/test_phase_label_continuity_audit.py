"""phase_label_continuity_audit: well-formed single- and two-hand labels pass; malformed labels, tags that disagree
with the subgoal array, a hand going back a subgoal, an unknown side and a mislabelled acting hand fail."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures" / "episode_audits"))
import synth as S
import mujoco
import numpy as np
from tool_library.verifiers.phase_label_continuity_audit import phase_label_continuity_audit

tmp = S.tmpdir()
scene = str(S.SETUP)
m = mujoco.MjModel.from_xml_path(scene)
q0 = S.home(m)
wr, wl = S.wrist_adr(m, "right"), S.wrist_adr(m, "left")
T = 120
Q = np.repeat(q0[None], T, 0)
Q[:60, wr[1]] = q0[wr[1]] + np.linspace(0, 0.1, 60); Q[60:, wr[1]] = q0[wr[1]] + 0.1      # the right hand reaches
Q[60:, wl[1]] = q0[wl[1]] + np.linspace(0, 0.1, 60)                                      # then the left hand
L = ["right: reach for the box #1"] * 40 + ["right: grip box #1"] * 20 + ["left: reach for the lid #2"] * 30 + \
    ["left: hold the lid #3 | right: let go of box #3"] * 30
SG = [1] * 60 + [2] * 30 + [3] * 30
good = S.save(tmp / "good.npz", Q, L, subgoal=SG)
C = S.Cases()


def good_passes():
    r = phase_label_continuity_audit(scene, good)
    assert r["ok"], r["issues"]
    assert r["has_subgoal_array"] and r["n_phases"] == 4, r


def malformed_and_unknown_side_fail():
    Lb = list(L); Lb[10:20] = ["reach for the box"] * 10; Lb[70:80] = ["up: lift #2"] * 10
    r = phase_label_continuity_audit(scene, S.save(tmp / "mal.npz", Q, Lb, subgoal=SG))
    assert not r["ok"] and r["malformed"] == ["reach for the box"] and r["unknown_sides"] == ["up"], r


def tag_mismatch_and_order_fail():
    SGb = list(SG); SGb[50:55] = [2] * 5
    r = phase_label_continuity_audit(scene, S.save(tmp / "tag.npz", Q, L, subgoal=SGb))
    assert not r["ok"] and r["n_tag_mismatches"] == 5, r
    Lo = list(L); Lo[90:] = ["left: hold the lid #1 | right: let go of box #3"] * 30         # the left hand goes 2 -> 1
    r = phase_label_continuity_audit(scene, S.save(tmp / "ord.npz", Q, Lo))
    assert not r["ok"] and r["order_violations"][0]["side"] == "left" and r["order_violations"][0]["subgoal"] == 1, r["order_violations"]


def acting_hand_mislabelled_fails():
    Lm = ["left: reach for the box #1"] * 60 + L[60:]                                        # the right hand moves
    r = phase_label_continuity_audit(scene, S.save(tmp / "act.npz", Q, Lm, subgoal=SG))
    assert not r["ok"] and r["mislabelled"][0]["other"] == "right", r["mislabelled"]
    assert phase_label_continuity_audit(scene, S.save(tmp / "act2.npz", Q, Lm, subgoal=SG), check_acting_hand=False)["ok"]


def misuse():
    S.expect_value_error(lambda: phase_label_continuity_audit(scene, good, still_m=-1), "still_m")


C.run("single-hand and 'left: X | right: Y' labels with a matching subgoal array pass", good_passes)
C.run("a label without a side and an unknown side fail", malformed_and_unknown_side_fail)
C.run("tags disagreeing with the subgoal array, and a hand going back a subgoal, fail", tag_mismatch_and_order_fail)
C.run("a phase labelled for the left hand while only the right hand moves fails", acting_hand_mislabelled_fails)
C.run("a negative stillness threshold is refused", misuse)
C.exit()

"""deformable_integrity_audit: a capsule rope lying on the table passes (cable-style chain and constraint chain); a
rope folded onto itself, pressed into the table, stretched apart, and a segment passing through another between two
frames fail; a scene without a rope passes with a note. Box-link ribbons (construct_multi_strand_cable's flat strands,
each strand a chain hanging off one junction) are judged too: they passed vacuously before ("no cable or capsule chain
in the scene"); lying apart they pass, a strand lying through another or passing through it between frames fails."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures" / "episode_audits"))
import synth as S
import numpy as np
from tool_library.verifiers.deformable_integrity_audit import deformable_integrity_audit

tmp = S.tmpdir()
R, H, N = 0.005, 0.01, 10
QX = "0.7071068 0 0.7071068 0"                                      # capsule axis (local z) along world x


def cable_xml():
    body = ""
    for i in range(N - 1, -1, -1):
        j = '<freejoint name="rope_free"/>' if i == 0 else f'<joint name="rope_J{i}" type="ball"/>'
        pos = f"0 0.4 {R}" if i == 0 else f"{2 * H} 0 0"
        body = f'<body name="rope_B{i}" pos="{pos}">{j}<geom name="rope_G{i}" type="capsule" size="{R} {H}" pos="{H} 0 0" quat="{QX}" mass="0.005"/>{body}</body>'
    return f'<mujoco><worldbody><geom name="floor" type="plane" size="1 1 0.1"/>{body}</worldbody></mujoco>'


def chain_xml():
    bodies = "".join(f'<body name="seg{i}" pos="{i * 2 * H + H} 0.4 {R}" quat="{QX}"><freejoint name="seg{i}_free"/>'
                     f'<geom name="segG{i}" type="capsule" size="{R} {H}" mass="0.005"/></body>' for i in range(6))
    eqs = "".join(f'<connect body1="seg{i}" body2="seg{i + 1}" anchor="0 0 {H}"/>' for i in range(5))
    return f'<mujoco><worldbody><geom name="floor" type="plane" size="1 1 0.1"/>{bodies}</worldbody><equality>{eqs}</equality></mujoco>'


cable = tmp / "cable.xml"; cable.write_text(cable_xml())
chain = tmp / "chain.xml"; chain.write_text(chain_xml())
import mujoco
mc, mk = mujoco.MjModel.from_xml_path(str(cable)), mujoco.MjModel.from_xml_path(str(chain))
T = 20
C = S.Cases()


def lying_rope_passes():
    for sc, mm, name in ((cable, mc, "rope_B0"), (chain, mk, "seg0")):
        Q = np.repeat(mm.qpos0[None], T, 0)
        r = deformable_integrity_audit(str(sc), S.save(tmp / f"ok_{name}.npz", Q, "right: hold the rope #1"))
        assert r["ok"] and r["chains"][name]["segments"] >= 6, r


def fold_and_table_fail():
    Q = np.repeat(mc.qpos0[None], T, 0)
    a = int(mc.jnt_qposadr[mujoco.mj_name2id(mc, mujoco.mjtObj.mjOBJ_JOINT, "rope_J5")])
    Q[:, a:a + 4] = [0, 0, 0, 1]                                          # folded 180 deg back onto itself
    r = deformable_integrity_audit(str(cable), S.save(tmp / "fold.npz", Q, "right: hold the rope #1"))
    assert not r["ok"] and r["worst"]["rule"] == "self" and r["worst"]["value"] > 5, r["worst"]
    Q = np.repeat(mc.qpos0[None], T, 0); Q[:, 2] = 0.0                    # centre line on the tabletop: 5 mm inside it
    r = deformable_integrity_audit(str(cable), S.save(tmp / "table.npz", Q, "right: press the rope #1"))
    assert not r["ok"] and r["worst"]["rule"] == "table" and r["worst"]["value"] > 4, r["worst"]


def stretch_fails():
    Q = np.repeat(mk.qpos0[None], T, 0)
    for i in range(3, 6):
        a = int(mk.jnt_qposadr[mujoco.mj_name2id(mk, mujoco.mjtObj.mjOBJ_JOINT, f"seg{i}_free")])
        Q[10:, a] += 0.008                                               # an 8 mm gap on a 20 mm segment
    r = deformable_integrity_audit(str(chain), S.save(tmp / "stretch.npz", Q, "right: pull the rope #1"))
    assert not r["ok"] and r["worst"]["rule"] == "stretch" and r["worst"]["frame"] == 10, r["worst"]


def crossing_between_frames_fails():
    Q = np.repeat(mk.qpos0[None], T, 0)
    a5 = int(mk.jnt_qposadr[mujoco.mj_name2id(mk, mujoco.mjtObj.mjOBJ_JOINT, "seg5_free")])
    c1x = 3 * H
    Q[:, a5:a5 + 3] = [c1x, 0.4, R + 4 * R]                              # segment 5 lies across segment 1, above it ...
    Q[:, a5 + 3:a5 + 7] = [0.7071068, 0.7071068, 0, 0]                   # ... along y
    Q[10:, a5 + 2] = R - 4 * R                                           # ... and below it one frame later
    r = deformable_integrity_audit(str(chain), S.save(tmp / "cross.npz", Q, "right: tie the rope #1"))
    assert not r["ok"] and r["n_events"]["crossing"] >= 1, r["n_events"]
    assert any("between frames" in s for s in r["issues"]), r["issues"]


def no_rope_passes():
    r = deformable_integrity_audit(str(S.SETUP), S.save(tmp / "none.npz", np.repeat(mujoco.MjModel.from_xml_path(str(S.SETUP)).key_qpos[0][None], 5, 0), "right: x #1"))
    assert r["ok"] and "no cable" in r["note"], r


# box-link ribbons: two flat strands (8 mm wide, 1.2 mm thick, 8 links of 12 mm) on ball joints off one free junction
TH, W, LK, NL = 0.0006, 0.004, 0.006, 8


def ribbon_xml():
    def strand(tag, y):
        body = ""
        for i in range(NL - 1, -1, -1):
            pos = f"0.006 {y} {TH - 0.003}" if i == 0 else f"{2 * LK} 0 0"
            body = (f'<body name="rib_{tag}{i}" pos="{pos}"><joint name="rib_{tag}J{i}" type="ball"/>'
                    f'<geom name="rib_{tag}G{i}" type="box" size="{LK} {W} {TH}" pos="{LK} 0 0" mass="0.0005"/>{body}</body>')
        return body
    return ('<mujoco><worldbody><geom name="floor" type="plane" size="1 1 0.1"/><body name="rib_junction" pos="0 0.4 0.003">'
            '<freejoint name="rib_free"/><geom name="rib_junction_g" type="box" size="0.006 0.02 0.003" mass="0.002"/>'
            f'{strand("A", 0.01)}{strand("B", -0.01)}</body></worldbody></mujoco>')


ribbon = tmp / "ribbon.xml"; ribbon.write_text(ribbon_xml())
mr = mujoco.MjModel.from_xml_path(str(ribbon))


def swing_b(yaw_deg, tilt_deg=0.0, lift=0.0):
    """qpos with strand B swung about the vertical by yaw_deg at its first joint (across strand A), tilted up by tilt_deg
    along its length, and the whole ribbon raised by lift."""
    from scipy.spatial.transform import Rotation as Rot
    q = mr.qpos0.copy()
    q[2] += lift
    a = int(mr.jnt_qposadr[mujoco.mj_name2id(mr, mujoco.mjtObj.mjOBJ_JOINT, "rib_BJ0")])
    th = np.radians(yaw_deg)
    u = np.array([-np.sin(th), np.cos(th), 0.0])                      # horizontal, across the swung strand
    x = (Rot.from_rotvec(-np.radians(tilt_deg) * u) * Rot.from_euler("z", th)).as_quat()
    q[a:a + 4] = [x[3], x[0], x[1], x[2]]
    return q


def ribbon_lying_passes():
    Q = np.repeat(mr.qpos0[None], T, 0)
    r = deformable_integrity_audit(str(ribbon), S.save(tmp / "rib_ok.npz", Q, "right: press the ribbon #1"))
    assert r["ok"], r["issues"]
    ch = r["chains"]
    assert set(ch) == {"rib_A0", "rib_B0"} and all(c["profile"] == "box" and c["segments"] == NL for c in ch.values()), ch
    assert abs(ch["rib_A0"]["radius_m"] - TH) < 1e-9 and abs(ch["rib_A0"]["width_m"] - 2 * W) < 1e-9 and abs(ch["rib_A0"]["length_m"] - 2 * LK * NL) < 1e-6, ch


def strand_through_strand_fails():
    Q = np.repeat(swing_b(20.0)[None], T, 0)                         # B lies across A at the same height: 1.2 mm overlap
    r = deformable_integrity_audit(str(ribbon), S.save(tmp / "rib_through.npz", Q, "right: press the ribbon #1"))
    assert not r["ok"] and r["worst"]["rule"] == "self" and r["worst"]["chain"] == "rib_A0 x rib_B0" and r["worst"]["value"] > 1.0, r["worst"]


def strand_crossing_between_frames_fails():
    Q = np.repeat(swing_b(20.0, 3.0, lift=0.006)[None], T, 0)      # B passes over A ...
    Q[10:] = swing_b(20.0, -3.0, lift=0.006)                        # ... and under it one frame later
    r = deformable_integrity_audit(str(ribbon), S.save(tmp / "rib_cross.npz", Q, "right: drag the ribbon #1"))
    assert not r["ok"] and r["n_events"]["crossing"] >= 1 and r["n_events"]["self"] == 0, r["n_events"]
    assert any("rib_A0 x rib_B0 passes through itself between frames" in s_ for s_ in r["issues"]), r["issues"]
    Q = np.repeat(swing_b(20.0, 3.0, lift=0.006)[None], T, 0)      # over it the whole time: a crossing that stays physical
    r = deformable_integrity_audit(str(ribbon), S.save(tmp / "rib_over.npz", Q, "right: drag the ribbon #1"))
    assert r["ok"], r["issues"]


def built_ribbon_judged():
    from tool_library.skills.construct_multi_strand_cable import construct_multi_strand_cable
    b = construct_multi_strand_cable(str(tmp / "msc"), name="tassel", include_floor=True,
                                     strands=[{"label": f"t{k}", "width": 0.008, "centreline": [[0.0, 0.0], [0.15 * np.cos(a), 0.15 * np.sin(a)]]}
                                              for k, a in enumerate(np.radians([-25, 0, 25]))])
    mb = mujoco.MjModel.from_xml_path(b["xml"])
    Q = np.repeat(mb.qpos0[None], T, 0)
    r = deformable_integrity_audit(b["xml"], S.save(tmp / "msc.npz", Q, "right: press the tassel #1"))
    assert r["ok"] and len(r["chains"]) == 3 and all(c["profile"] == "box" for c in r["chains"].values()), r
    Q[:, 2] -= 0.004                                                  # the whole tassel 4 mm into the table
    r = deformable_integrity_audit(b["xml"], S.save(tmp / "msc_low.npz", Q, "right: press the tassel #1"))
    assert not r["ok"] and r["worst"]["rule"] == "table", r["worst"]


def physics_substep_penetration_cannot_hide_between_video_frames():
    path = tmp / "substep_penetration.npz"
    clear = np.repeat(mc.qpos0[None], 2, 0)
    raw = np.repeat(mc.qpos0[None], 4, 0)
    raw[1, 2] -= 0.005  # Video endpoints are clear; one physics state penetrates the table.
    np.savez(path, qpos=clear, labels=np.array(['right: carry #1'] * 2), fps=30.,
             initial_qpos=mc.qpos0, initial_time_s=2.,
             substep_qpos=raw, substep_time_s=2. + np.arange(1, 5) * .002)
    report = deformable_integrity_audit(str(cable), str(path))
    assert not report['ok'] and report['n_events']['table'], report
    assert report['recording_grid'] == 'physics_substeps' and report['n_states_checked'] == 5


def released_constraint_is_not_stretch():
    Q = np.repeat(mk.qpos0[None], T, 0)
    for i in range(3, 6):
        a = int(mk.jnt_qposadr[mujoco.mj_name2id(mk, mujoco.mjtObj.mjOBJ_JOINT, f'seg{i}_free')])
        Q[10:, a] += .008
    active = np.ones((T, mk.neq), dtype=bool)
    active[10:, 2] = False
    report = deformable_integrity_audit(str(chain), {'qpos': Q, 'fps': 30., 'eq_active': active})
    assert report['ok'] and report['n_events']['stretch'] == 0, report


C.run("a rope lying straight on the table passes (ball-joint cable and connect-linked chain)", lying_rope_passes)
C.run("a rope folded onto itself and a rope pressed 5 mm into the table fail", fold_and_table_fail)
C.run("a constraint chain pulled 8 mm apart fails on stretch", stretch_fails)
C.run("a segment passing through another between two frames fails", crossing_between_frames_fails)
C.run("a scene without a rope passes with a note", no_rope_passes)
C.run("box-link ribbon strands off one junction lying apart are found and pass", ribbon_lying_passes)
C.run("a ribbon strand lying through another strand of the same cable fails", strand_through_strand_fails)
C.run("a ribbon strand passing through another between frames fails; one crossing over it passes", strand_crossing_between_frames_fails)
C.run("construct_multi_strand_cable's flat strands are judged (a tassel pressed into the table fails)", built_ribbon_judged)
C.run("rope penetration between saved video frames fails on physics substeps", physics_substep_penetration_cannot_hide_between_video_frames)
C.run("a recorded released join is not falsely classified as stretch", released_constraint_is_not_stretch)
C.exit()

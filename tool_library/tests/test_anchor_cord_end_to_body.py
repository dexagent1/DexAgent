import sys, tempfile, os
import numpy as np
import mujoco
from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))  # runnable from anywhere
from tool_library.skills.anchor_cord_end_to_body import anchor_cord_end_to_body

fails = 0


def check(case, cond, msg=""):
    global fails
    if cond:
        print(f"[PASS] {case}")
    else:
        fails += 1
        print(f"[FAIL] {case} {msg}")


def chain_scene(path, n=20, seg=0.012, r=0.0012, block_pos=(-0.03, 0, 0.02), yaw=0.0, curve=0.0, key=True):
    """A cord of n capsule links (nested bodies on ball joints, free root) lying on a floor beside a fixed block."""
    links, x, y, th = [], 0.0, 0.0, yaw
    body = ""
    for i in range(n):
        if i == 0:
            body += f'<body name="c0" pos="0 0 {r}"><freejoint name="c_free"/>'
        else:
            body += f'<body name="c{i}" pos="{seg} 0 0" euler="0 0 {curve}"><joint name="c_j{i}" type="ball" damping="1e-3" armature="1e-5"/>'
        body += f'<geom type="capsule" fromto="0 0 0 {seg} 0 0" size="{r}" density="1100"/>'
    body += "</body>" * n
    nq = 7 + 4 * (n - 1)
    keyxml = ""
    if key:
        m = mujoco.MjModel.from_xml_string(f'<mujoco><compiler angle="radian"/><worldbody>{body}</worldbody></mujoco>')
        q = m.qpos0.copy(); q[0:3] = [0.0, 0.0, r]; q[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        keyxml = f'<keyframe><key name="home" qpos="{" ".join(f"{v:.9g}" for v in q)}"/></keyframe>'
    xml = (f'<mujoco><compiler angle="radian"/><option timestep="0.002"/><worldbody><geom name="floor" type="plane" size="1 1 0.01"/>'
           f'<body name="robot_base" pos="0.5 0.5 0.05"><geom type="box" size="0.02 0.02 0.02"/></body>'
           f'<body name="block" pos="{block_pos[0]} {block_pos[1]} {block_pos[2]}"><geom type="box" size="0.02 0.03 0.02"/></body>'
           f'<body name="slider" pos="0.3 0.3 0.02"><freejoint/><geom type="box" size="0.02 0.02 0.02" mass="0.1"/></body>'
           f'{body}</worldbody>{keyxml}</mujoco>')
    open(path, "w").write(xml)
    return n


def pull(xml, free_body, force, seconds=1.5, anchored="c0"):
    m = mujoco.MjModel.from_xml_path(xml); d = mujoco.MjData(m)
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, anchored); f = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, free_body)
    p0 = d.xpos[a].copy()
    for _ in range(int(seconds / m.opt.timestep)):
        d.xfrc_applied[f, :3] = force
        mujoco.mj_step(m, d)
    return float(np.linalg.norm(d.xpos[a] - p0)), np.isfinite(d.qpos).all()


with tempfile.TemporaryDirectory() as tmp:
    # 1) weld the cord's first link to the block: residual ~0 at the keyframe, and the end holds under a 0.5 N pull
    s1 = f"{tmp}/s1.xml"; n = chain_scene(s1, yaw=0.4, curve=0.05)
    r1 = anchor_cord_end_to_body(s1, f"{tmp}/s1a.xml", "c0", "block")
    moved, fin = pull(r1["out_xml"], f"c{n - 1}", [0.5, 0.0, 0.0])
    moved0, _ = pull(s1, f"c{n - 1}", [0.5, 0.0, 0.0])
    check("weld holds a light cord end under a 0.5 N pull", r1["residual_m"] < 1e-6 and moved < 0.002 and fin and moved0 > 0.02,
          f"residual {r1['residual_m']} moved {moved} unanchored {moved0}")

    # 2) anchoring the LAST (deepest) link of a longer curved cord, keyframe != qpos0, block far from the origin
    s2 = f"{tmp}/s2.xml"; n2 = chain_scene(s2, n=30, seg=0.01, block_pos=(0.35, 0.1, 0.02), yaw=-0.3, curve=0.03)
    r2 = anchor_cord_end_to_body(s2, f"{tmp}/s2a.xml", "c0", "block")
    moved2, fin2 = pull(r2["out_xml"], f"c{n2 - 1}", [0.0, -0.4, 0.0])
    check("weld holds the root of a 30-link cord whose keyframe differs from qpos0", r2["residual_m"] < 1e-6 and moved2 < 0.002 and fin2
          and not r2["warnings"], f"residual {r2['residual_m']} moved {moved2}")
    r2b = anchor_cord_end_to_body(s2, f"{tmp}/s2b.xml", f"c{n2 - 1}", "block")
    check("anchoring a deep link is warned about", r2b["residual_m"] < 1e-6 and len(r2b["warnings"]) == 1 and "29 links" in r2b["warnings"][0],
          str(r2b["warnings"]))

    # 3) a scene without keyframes, written in place (out_xml == scene_xml); the state size is unchanged
    s3 = f"{tmp}/s3.xml"; chain_scene(s3, n=12, key=False)
    nq0 = mujoco.MjModel.from_xml_path(s3).nq
    r3 = anchor_cord_end_to_body(s3, s3, "c3", "block", name="mid_clip")
    m3 = mujoco.MjModel.from_xml_path(s3)
    check("in-place anchor without keyframes keeps nq", m3.nq == nq0 and m3.neq == 1 and r3["equality"] == "mid_clip"
          and r3["residual_rad"] < 1e-4)

    # 4) refusals: the world/table/robot, a moving body, unknown params, a duplicate name
    s4 = f"{tmp}/s4.xml"; chain_scene(s4)
    for case, args, kw, word in (("robot anchor refused", ("c0", "robot_base"), {}, "robot"),
                                 ("moving anchor refused", ("c0", "slider"), {}, "can move"),
                                 ("world anchor refused", ("c0", "world"), {}, "world"),
                                 ("unknown parameter refused", ("c0", "block"), {"kind": "connect"}, "unknown"),
                                 ("duplicate name refused", ("c0", "block"), {"name": "mid_clip"}, "already")):
        src = s3 if case.startswith("duplicate") else s4
        try:
            anchor_cord_end_to_body(src, f"{tmp}/x.xml", *args, **kw)
            check(case, False, "no error")
        except ValueError as e:
            check(case, word in str(e), str(e))

sys.exit(1 if fails else 0)

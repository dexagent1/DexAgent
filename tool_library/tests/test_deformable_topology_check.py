"""Smoke test for deformable_topology_check: a jointed single-strand rope passes; fused links, unjoined strands, masked self-collision, pinned ends and self-intersection fail."""
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SETUP = ROOT / "setup"
TMP = Path(tempfile.mkdtemp())
fails = 0


def check(name, cond, msg=""):
    global fails
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + ("" if cond else f"  {str(msg)[:600]}"))
    fails += 0 if cond else 1


def bench(name, objects="", key_extra="", edit=None):
    """The setup's bench (setup/scene.xml) with objects added to the worldbody and their free-joint qpos appended to
    keyframe 'home', written to a temp file (meshes resolved from setup/)."""
    xml = (SETUP / "scene.xml").read_text()
    xml = xml.replace('<compiler angle="radian" />', f'<compiler angle="radian" meshdir="{SETUP}" texturedir="{SETUP}" />', 1)
    xml = xml.replace("</worldbody>", objects + "\n  </worldbody>", 1)
    if key_extra:
        xml = re.sub(r'(<key name="home" qpos=")([^"]*)', lambda m: m.group(1) + m.group(2) + " " + key_extra, xml, count=1)
    if edit:
        xml = edit(xml)
    p = TMP / f"{name}.xml"
    p.write_text(xml)
    return str(p)


CUBE = ('<body name="cube" pos="0 0.5 0.025"><freejoint name="cube_free"/>'
        '<geom name="cube_geom" type="box" size="0.025 0.025 0.025" mass="0.1" rgba="0.8 0.2 0.2 1"/></body>')
CUBE_Q = "0 0.5 0.025 1 0 0 0"


import json

from tool_library.verifiers.deformable_topology_check import deformable_topology_check as check_topo


def chain(prefix, n, x0=0.0, y=0.5, z=0.012, r=0.006, h=0.01, missing_joint=None):
    """A rope as MuJoCo's cable composite builds it: nested capsule links on ball joints, the first one free."""
    xml, close = "", ""
    for i in range(n):
        pos = f"{x0} {y} {z}" if i == 0 else f"{2 * h} 0 0"
        j = '<freejoint/>' if i == 0 else ("" if i == missing_joint else f'<joint name="{prefix}J_{i}" type="ball"/>')
        xml += (f'<body name="{prefix}B_{i}" pos="{pos}">{j}<geom name="{prefix}G_{i}" type="capsule" size="{r} {h}" quat="0.7071068 0 0.7071068 0" '
                f'mass="0.002" rgba="0.6 0.1 0.1 1"/>')
        close += "</body>"
    excl = "".join(f'<exclude body1="{prefix}B_{i}" body2="{prefix}B_{i + 1}"/>' for i in range(n - 1))
    return xml + close, excl


def scene(name, bodies, excl="", eq="", key=""):
    p = TMP / f"{name}.xml"
    p.write_text(f'<mujoco><worldbody><body name="table" pos="0 0.5 -0.02"><geom name="g_table" type="box" size="0.6 0.6 0.02"/></body>{bodies}'
                 f'</worldbody><contact>{excl}</contact><equality>{eq}</equality></mujoco>')
    return str(p)


def recon(name, roots):
    p = TMP / f"{name}.json"
    p.write_text(json.dumps({"objects": [{"id": "rope", "type": "deformable", "skill": "construct_rope", "root_body": roots[0],
                                          "bodies": [f"{r[:-2]}_{i}" for r in roots for i in range(12)]}]}))
    return str(p)


b, e = chain("rope", 12)
r = check_topo(scene("ok", b, e))
check("a 12-link rope on ball joints with free self-collision passes", r["ok"] and r["ropes"][0]["strands"] == 1 and r["ropes"][0]["ends"] == 2, r["ropes"])
b2, e2 = chain("rope", 12, missing_joint=6)
r = check_topo(scene("fused", b2, e2))
check("a link without its ball joint (a rigid rod in the rope) is caught", (not r["ok"]) and r["ropes"][0]["fused_links"] == ["ropeB_6"], r["issues"])
ba, ea = chain("ropea", 12, y=0.45)
bb, eb = chain("ropeb", 12, y=0.55)
rc = recon("two", ["ropeaB_0", "ropebB_0"])
r = check_topo(scene("two", ba + bb, ea + eb), recon=rc)
check("a rope left as two strands (cut cables never joined) is caught", (not r["ok"]) and r["ropes"][0]["strands"] == 2, r["ropes"])
r = check_topo(scene("joined", ba + bb, ea + eb, eq='<connect body1="ropeaB_11" body2="ropebB_0" anchor="0.01 0 0"/>'), recon=rc)
check("the same two cables joined by an active connect are one strand with two ends", r["ok"] and r["ropes"][0]["strands"] == 1 and r["ropes"][0]["ends"] == 2, r["ropes"])
nonadj = "".join(f'<exclude body1="ropeB_{i}" body2="ropeB_{j}"/>' for i in range(12) for j in range(i + 1, 12))
r = check_topo(scene("noself", b, nonadj))
check("a rope with self-collision excluded everywhere (it passes through itself) is caught", (not r["ok"]) and r["ropes"][0]["self_collision_frac"] == 0.0, r["issues"])
r = check_topo(scene("pinned", b, e, eq='<weld body1="ropeB_0" body2="table"/>'))
check("a rope end welded to the table at the start is caught", (not r["ok"]) and r["ropes"][0]["pinned"], r["issues"])
r = check_topo(scene("pinned_off", b, e, eq='<weld body1="ropeB_0" body2="table" active="false"/>'))
check("an inactive grip weld (switched on by a later phase) is allowed", r["ok"], r["issues"])
# a centreline that crosses itself at rest: a coil whose last links lie on the first ones
loop, close = "", ""
pts = [(0.0, 0.5, 0.012), (0.02, 0, 0), (0.02, 0, 0), (0.0, 0.02, 0), (-0.02, 0, 0), (-0.02, 0.0, 0), (0.0, -0.02, 0.004)]
for i, p in enumerate(pts):
    j = '<freejoint/>' if i == 0 else '<joint type="ball"/>'
    loop += f'<body name="coilB_{i}" pos="{p[0]} {p[1]} {p[2]}">{j}<geom name="coilG_{i}" type="sphere" size="0.008" mass="0.002"/>'
    close += "</body>"
ex = "".join(f'<exclude body1="coilB_{i}" body2="coilB_{i + 1}"/>' for i in range(len(pts) - 1))
r = check_topo(scene("crossing", loop + close, ex), min_links=5)
check("a rope whose links overlap a non-adjacent link at rest is caught", (not r["ok"]) and r["ropes"][0]["interpenetrations"], r["ropes"])
# a ribbon of three strands joined at one end (a braid's tail): a tree with three free ends
branch, bclose = "", ""
for i in range(6):
    branch += (f'<body name="ribB_{12 + i}" pos="{0.0 if i == 0 else 0.02} {0.02 if i == 0 else 0} 0"><joint type="ball"/>'
               f'<geom name="ribG_{12 + i}" type="capsule" size="0.006 0.01" quat="0.7071068 0 0.7071068 0" mass="0.002"/>')
    bclose += "</body>"
tree_b, tree_e = chain("rib", 12)
mid = '<geom name="ribG_6" type="capsule" size="0.006 0.01" quat="0.7071068 0 0.7071068 0" mass="0.002" rgba="0.6 0.1 0.1 1"/>'
assert mid in tree_b
tree_b = tree_b.replace(mid, mid + branch + bclose, 1)                  # a third strand leaves the middle of the chain
tree_e += '<exclude body1="ribB_6" body2="ribB_12"/>' + "".join(f'<exclude body1="ribB_{12 + i}" body2="ribB_{13 + i}"/>' for i in range(5))
rt = {"objects": [{"id": "ribbon", "type": "deformable", "root_body": "ribB_0", "bodies": [f"ribB_{i}" for i in range(18)]}]}  # not named like a rope
r = check_topo(scene("tree", tree_b, tree_e), recon=rt)
check("a three-ended ribbon is not a simple chain by default", (not r["ok"]) and r["ropes"][0]["ends"] == 3, r["issues"])
rt["objects"][0]["topology"] = {"strands": 1, "ends": 3}
r = check_topo(scene("tree", tree_b, tree_e), recon=rt)
check("the same ribbon passes when scene_recon declares one strand with three ends, and the report lists it",
      r["ok"] and r["ropes"][0]["declared"] == {"strands": 1, "ends": 3}, r["issues"])
# flat box links (a ribbon) whose last link lies inside the first one
flat, fclose = "", ""
fpts = [(0.0, 0.5, 0.01), (0.03, 0, 0), (0.03, 0, 0), (0.0, 0.03, 0), (-0.03, 0, 0), (-0.03, 0.0, 0), (0.004, -0.03, 0.0)]
for i, q in enumerate(fpts):
    j = '<freejoint/>' if i == 0 else '<joint type="ball"/>'
    flat += f'<body name="flatB_{i}" pos="{q[0]} {q[1]} {q[2]}">{j}<geom name="flatG_{i}" type="box" size="0.014 0.01 0.002" mass="0.002"/>'
    fclose += "</body>"
fx = "".join(f'<exclude body1="flatB_{i}" body2="flatB_{i + 1}"/>' for i in range(len(fpts) - 1))
r = check_topo(scene("flat", flat + fclose, fx), min_links=5)
check("box links that overlap a non-adjacent link at rest are caught", (not r["ok"]) and r["ropes"][0]["interpenetrations"], r["ropes"])
# a cord end held under a fixed tape dispenser: allowed only when scene_recon declares it, and never to the table
DISP = '<body name="dispenser" pos="0.0 0.5 0.03"><geom name="disp_g" type="box" size="0.03 0.02 0.03" contype="0" conaffinity="0"/></body>'
ra = {"objects": [{"id": "rope", "type": "deformable", "root_body": "ropeB_0", "bodies": [f"ropeB_{i}" for i in range(12)]}]}
weld = '<weld body1="ropeB_0" body2="dispenser"/>'
r = check_topo(scene("anch0", b + DISP, e, eq=weld), recon=ra)
check("a cord welded to a fixed object without a declaration is pinned", (not r["ok"]) and r["ropes"][0]["pinned"], r["issues"])
ra["objects"][0]["anchored_to"] = {"body": "dispenser", "reason": "Stage 1: the far end lies under the tape dispenser"}
r = check_topo(scene("anch1", b + DISP, e, eq=weld), recon=ra)
check("the same weld passes when scene_recon declares the anchor", r["ok"] and r["ropes"][0]["anchored_to"] == ["dispenser"], r["issues"])
ra["objects"][0]["anchored_to"] = {"body": "table"}
r = check_topo(scene("anch2", b + DISP, e, eq='<weld body1="ropeB_0" body2="table"/>'), recon=ra)
check("an anchor declared on the table is still pinned", (not r["ok"]) and r["ropes"][0]["pinned"], r["issues"])
r = check_topo(scene("named_missing", '<body name="cube2" pos="0 0.5 0.03"><freejoint/><geom type="box" size="0.02 0.02 0.02" mass="0.1"/></body>'), objects=["ribbon_junction"])
check("objects named but none recognised as a rope fails instead of passing silently", not r["ok"] and "give recon" in r["issues"][0], r["issues"])
CUBE_ONLY = '<body name="cube" pos="0 0.5 0.03"><freejoint/><geom name="c" type="box" size="0.02 0.02 0.02" mass="0.1"/></body>'
r = check_topo(scene("norope", CUBE_ONLY))
check("a scene without ropes passes with a note", r["ok"] and r["ropes"] == [] and r["note"], r)

print(f"{'OK' if fails == 0 else 'FAILED'}: deformable_topology_check ({fails} failure(s))")
sys.exit(1 if fails else 0)

"""Smoke test for hand_contact_pair_mask_check: solid geoms (named or not) pass under the executor's pairing rule; a
shell in a contact bit of its own stays out of the hand pairs; a solid geom opted out of them, and one-handed explicit
pairs, fail."""
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


from tool_library.verifiers.hand_contact_pair_mask_check import hand_contact_pair_mask_check as check_pairs

r = check_pairs(bench("cube", CUBE, CUBE_Q))
check("a named solid object is paired with both hands by the executor's rule", r["ok"] and r["hands"] == {"right": 19, "left": 19}, r["issues"])
r = check_pairs(bench("unnamed", CUBE.replace('name="cube_geom" ', ''), CUBE_Q))
check("an unnamed solid object geom is paired too (the executor names it)", r["ok"], r["issues"])
RING = ('<body name="ring" pos="0.1 0.5 0.02"><freejoint/><geom name="ring_body" type="cylinder" size="0.015 0.01" mass="0.01"/>'
        '<body name="ring_magnet"><geom name="ring_magnet_shell" type="sphere" size="0.03" contype="4" conaffinity="4" group="3" mass="1e-5"/></body></body>')
r = check_pairs(bench("magnet", CUBE + RING, CUBE_Q + " 0.1 0.5 0.02 1 0 0 0"))
check("a magnet shell in a contact bit of its own (4) is left out of the hand pairs", r["ok"] and not r["special_paired"], r["issues"])
r = check_pairs(bench("optout", CUBE.replace('name="cube_geom"', 'name="cube_geom_nohand"'), CUBE_Q))
check("a solid geom opted out of the hand pairs ('_nohand') is reported: the hand passes through it",
      (not r["ok"]) and r["unpaired"] and r["unpaired"][0]["geom"].startswith("cube_geom_nohand"), r["issues"])
FEET = ('<body name="cell" pos="0 0.6 0.0"><freejoint/><geom name="cell_can" type="cylinder" size="0.007 0.02" pos="0 0 0.02" mass="0.01"/>'
        '<geom type="sphere" size="0.0012" pos="0.003 0 0.00105" group="3"/></body>')
r = check_pairs(bench("feet", FEET, "0 0.6 0 1 0 0 0"))
check("unnamed support feet inside a paired can need no pair of their own", r["ok"], r["issues"])
hand = [f"robot_{s}_collision_hand_{s}_{p}" for s in ("left", "right") for p in
        ["palm_0", "thumb_0", "thumb_1", "thumb_2"] + [f"{f}_{i}" for f in ("index", "middle", "ring", "pinky") for i in range(3)]]
hull = [f"robot_{s}_collision_hand_{s}_{l}_hull" for s in ("left", "right") for l in ("hand_C_MC", "thumb_MC", "thumb_DP")]
pairs = "".join(f'<pair geom1="{h}" geom2="cube_geom"/>' for h in hand + hull)
r = check_pairs(bench("explicit", CUBE, CUBE_Q, lambda x: x.replace("</contact>", pairs + "</contact>", 1)), include_executor_pairs=False)
check("explicit pairs with every hand collision geom pass without the executor", r["ok"] and r["pairs"]["explicit"] >= 38, r["issues"])
pairs_r = "".join(f'<pair geom1="{h}" geom2="cube_geom"/>' for h in hand + hull if "_right_" in h)
r = check_pairs(bench("half", CUBE, CUBE_Q, lambda x: x.replace("</contact>", pairs_r + "</contact>", 1)), include_executor_pairs=False)
check("explicit pairs with one hand only leave the other hand passing through: caught", (not r["ok"]) and {u["side"] for u in r["unpaired"]} == {"left"}, r["unpaired"])

print(f"{'OK' if fails == 0 else 'FAILED'}: hand_contact_pair_mask_check ({fails} failure(s))")
sys.exit(1 if fails else 0)

import sys, tempfile
from pathlib import Path
from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))  # runnable from anywhere
from tool_library.verifiers.slot_insertion_test import slot_insertion_test

HDR = '<mujoco><option timestep="0.002"/><worldbody>'

def receptacle(d, w=0.12, depth=0.16, h=0.04, walls=True, block=False, yaw=0, free=False, name="rk"):
    g = [f'<geom type="box" size="{w/2+0.01} {depth/2} 0.003" pos="0 0 0.003" friction="0.7 0.02 0.002"/>',
         f'<geom type="box" size="{w/2+0.01} 0.003 {h/2}" pos="0 {-depth/2-0.003} {0.006+h/2}"/>']
    if walls:
        for s in (-1, 1):
            g.append(f'<geom type="box" size="0.003 {depth/2} {h/2}" pos="{s*(w/2+0.003)} 0 {0.006+h/2}"/>')
    if block:
        g.append(f'<geom type="box" size="{w/2} 0.003 {h/2}" pos="0 {depth*0.15} {0.006+h/2}"/>')
    sites = (f'<site name="{name}_floor" pos="0 0 0.006"/>'
             f'<site name="{name}_mouth" pos="0 {depth/2+0.02} {0.006+h/2}"/>')
    fj = '<freejoint/>' if free else ''
    xml = HDR + f'<body name="{name}" euler="0 0 {yaw}">{fj}{"".join(g)}{sites}</body></worldbody></mujoco>'
    p = Path(d) / f"{name}.xml"; p.write_text(xml); return str(p)

def insert(d, l=0.145, w=0.10, h=0.025, name="part"):
    xml = HDR + (f'<body name="{name}" pos="0.5 0 0.1"><freejoint/>'
                 f'<geom type="box" size="{w/2} {l/2} {h/2}" mass="0.5" friction="0.7 0.02 0.002"/></body>'
                 '</worldbody></mujoco>')
    p = Path(d) / f"{name}.xml"; p.write_text(xml); return str(p)

fails = 0
def check(case, cond, rep):
    global fails
    if cond: print(f"[PASS] {case}")
    else:
        fails += 1; print(f"[FAIL] {case}: {rep}")

with tempfile.TemporaryDirectory() as d:
    slots = [["rk_floor", "rk_mouth"]]
    r = slot_insertion_test(receptacle(d), insert(d), slots, insert_nose_local=[0, -1, 0])
    check("fitting slot seats, holds and pens the part", r["ok"] and r["slots"][0]["seated"], r)
    r = slot_insertion_test(receptacle(d, block=True, name="rk"), insert(d), slots, insert_nose_local=[0, -1, 0])
    check("a bar across the bay stops the part short", (not r["ok"]) and not r["slots"][0]["seated"] and r["issues"], r)
    r = slot_insertion_test(receptacle(d, walls=False, name="rk"), insert(d), slots, insert_nose_local=[0, -1, 0])
    check("a slot without side walls fails the side shove", (not r["ok"]) and not r["slots"][0]["penned"], r)
    rp = receptacle(d, w=0.08, depth=0.12, h=0.03, yaw=90, free=True, name="card")
    r = slot_insertion_test(rp, insert(d, l=0.11, w=0.07, h=0.02, name="cart"), [["card_floor", "card_mouth"]])
    check("rotated free receptacle, smaller part, auto nose axis", r["ok"], r)
    try:
        slot_insertion_test(receptacle(d), insert(d), [["nope", "rk_mouth"]])
        check("unknown site raises", False, "no error")
    except ValueError:
        check("unknown site raises ValueError", True, None)
sys.exit(1 if fails else 0)

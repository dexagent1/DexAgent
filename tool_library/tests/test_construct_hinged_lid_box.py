import sys, math, tempfile, numpy as np, mujoco
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root: runnable from anywhere
from tool_library.skills.construct_hinged_lid_box import construct_hinged_lid_box
fails = 0
def lowest_lid_z(b):
    m = mujoco.MjModel.from_xml_path(b["xml"]); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    g = m.geom(b["lid_geom"]).id
    R = d.geom_xmat[g].reshape(3, 3); c = d.geom_xpos[g]; s = m.geom_size[g]
    corners = [c + R @ (np.array([sx, sy, sz]) * s) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    return min(p[2] for p in corners), m, d
cases = [
    ("shut_default", dict(), 0.0),
    ("gift_box_rest_with_skirt", dict(length=0.25, width=0.23, height=0.045, lid_init_deg="rest"), None),
    ("chest_open_120_no_skirt", dict(length=0.40, width=0.30, height=0.20, lid_flap=False, lid_init_deg=120.0), 120.0),
    ("small_open_flat_180", dict(length=0.12, width=0.10, height=0.05, lid_init_deg=180.0, skirt_init_deg=60.0), 180.0),
]
for name, kw, expect in cases:
    with tempfile.TemporaryDirectory() as td:
        try:
            b = construct_hinged_lid_box(td, **kw)
            low, m, d = lowest_lid_z(b)
            q = math.degrees(d.qpos[m.joint(b["lid_joint"]).qposadr[0]])
            ok = abs(q - b["lid_init_deg"]) < 1e-3 and m.jnt_type[m.joint(b["lid_joint"]).id] == 3
            if expect is not None: ok &= abs(q - expect) < 1e-3
            if kw.get("lid_init_deg") == "rest": ok &= 180 < q < 200 and 0.0 <= low < 0.003
            if expect == 0.0:   # shut lid sits on the walls: bottom face at the wall top
                ok &= abs(low - DEFAULTS_H) < 1e-4 if (DEFAULTS_H := 0.06) else False
            if b["skirt_joint"]:
                ok &= abs(math.degrees(d.qpos[m.joint(b["skirt_joint"]).qposadr[0]]) - b["skirt_init_deg"]) < 1e-3
            # set to shut: qpos=0 puts lid on top of the walls
            d.qpos[m.joint(b["lid_joint"]).qposadr[0]] = 0.0; mujoco.mj_forward(m, d)
            g = m.geom(b["lid_geom"]).id
            ok &= abs(d.geom_xpos[g][2] - (kw.get("height", 0.06) + kw.get("thickness", 0.003) / 2)) < 1e-4
            print(("[PASS] " if ok else "[FAIL] ") + f"{name}: lid0={q:.2f} low={low:.4f} mass={b['mass_kg']}")
            fails += (not ok)
        except Exception as e:
            print(f"[FAIL] {name}: {e!r}"); fails += 1
for name, kw in [("unknown_param", dict(lenght=0.2)), ("init_outside_range", dict(lid_init_deg=250.0))]:
    try:
        construct_hinged_lid_box(tempfile.mkdtemp(), **kw); print(f"[FAIL] {name}: no error"); fails += 1
    except ValueError:
        print(f"[PASS] {name} raises ValueError")
sys.exit(1 if fails else 0)

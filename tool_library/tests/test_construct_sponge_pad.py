import sys, tempfile
import mujoco
from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))  # runnable from anywhere
from tool_library.skills.construct_sponge_pad import construct_sponge_pad
fails = 0
t = tempfile.mkdtemp()
for name, kw in {"default": {}, "small_thick": dict(length=0.10, width=0.06, thickness=0.03, nx=3, ny=2, stiffness_N_per_m=1200),
                 "large_fine": dict(length=0.15, width=0.10, nx=5, ny=4, stiffness_N_per_m=4000, mass=0.03)}.items():
    try:
        b = construct_sponge_pad(f"{t}/{name}", name=name, **kw)
        m = mujoco.MjModel.from_xml_path(b["xml"])
        nx, ny = kw.get("nx", 4), kw.get("ny", 3)
        assert len(b["cell_joints"]) == nx * ny
        ks = sum(float(m.joint(j).stiffness[0]) for j in b["cell_joints"])
        assert abs(ks - b["stiffness_N_per_m"]) < 1e-3 * b["stiffness_N_per_m"]
        if "mass" in kw:
            assert abs(b["mass_kg"] - kw["mass"]) < 0.02 * kw["mass"], b["mass_kg"]
        d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        top = max(d.geom_xpos[g][2] + m.geom_size[g][2] for g in range(m.ngeom))
        assert abs(top - b["size_m"][2]) < 1e-6, top
        print(f"[PASS] {name}: {nx*ny} cells, {b['mass_kg']*1000:.1f} g, travel {b['max_compression_m']*1000:.1f} mm")
    except Exception as e:
        fails += 1; print(f"[FAIL] {name}: {e}")
for bad in (dict(nx=0), dict(base_frac=0.99), dict(colour=1)):
    try:
        construct_sponge_pad(f"{t}/bad", **bad); fails += 1; print(f"[FAIL] accepted {bad}")
    except ValueError:
        print(f"[PASS] refuses {bad}")
sys.exit(1 if fails else 0)

import os, sys, tempfile
import mujoco
from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))  # runnable from anywhere
from tool_library.skills.bench_tool_large_arena import bench_tool_large_arena


def dense_scene(n, memory=None):
    """Two free bodies of n x n small overlapping boxes resting into each other: n^2 * n^2 contact candidates."""
    g1 = "".join(f'<geom type="box" size="0.004 0.004 0.004" pos="{0.006*i} {0.006*j} 0"/>' for i in range(n) for j in range(n))
    g2 = "".join(f'<geom type="box" size="0.004 0.004 0.004" pos="{0.006*i+0.002} {0.006*j+0.002} 0.006"/>' for i in range(n) for j in range(n))
    size = f'<size memory="{memory}"/>' if memory else ""
    return (f'<mujoco>{size}<option><flag gravity="disable"/></option><default><geom condim="6"/></default><worldbody>'
            f'<body name="a"><freejoint/>{g1}</body><body name="b" pos="0 0 0.0"><freejoint/>{g2}</body></worldbody></mujoco>')


def write(xml):
    fd, path = tempfile.mkstemp(suffix=".xml", dir=os.environ.get("TMPDIR"))
    os.write(fd, xml.encode()); os.close(fd)
    return path


fails = 0
# 1: a contact-dense scene that overflows the default arena inside the solver runs under the wrapper
p = write(dense_scene(12))
raised = False
from tool_library.verifiers.contact_check import contact_check
try:
    contact_check(p, "a", "b")
except Exception as e:
    raised = "stackAlloc" in str(e) or "memory" in str(e).lower()
r = bench_tool_large_arena("contact_check", {"scene": p, "a": "a", "b": "b"}, memory_mb=256)
res = r["result"] if isinstance(r["result"], dict) else {}
if raised and r["models_compiled"] >= 1 and "ok" in res and isinstance(res.get("contacts"), list):
    print(f"[PASS] dense scene: default arena overflows (mj_stackAlloc) while settling, the wrapped call settles and returns its report")
else:
    print(f"[FAIL] dense scene: raised={raised} compiled={r['models_compiled']} result={str(res)[:200]}"); fails += 1
# 2: the loaders are restored afterwards (default arena again), also after the wrapped tool raises
m = mujoco.MjModel.from_xml_path(p)
try:
    bench_tool_large_arena("probe_scene", {"scene": "/nonexistent/scene.xml"}, memory_mb=128)
    ok_raise = False
except Exception:
    ok_raise = True
m2 = mujoco.MjModel.from_xml_path(p)
if m.narena < 256 * 2**20 and m2.narena == m.narena and ok_raise:
    print(f"[PASS] loaders restored: arena back to {m.narena} bytes after a normal and a failing call")
else:
    print(f"[FAIL] loaders not restored: {m.narena} {m2.narena} raised={ok_raise}"); fails += 1
# 3: a model that already asks for more memory keeps its own; a small scene compiles with the floor
big = write(dense_scene(2, memory="300M"))
captured = {}
r3 = bench_tool_large_arena("compile_scene", {"scene_xml": big}, memory_mb=64)
spec = mujoco.MjSpec.from_file(big)
if int(spec.memory) == 300 * 2**20 and r3["models_compiled"] >= 1 and r3["result"].get("compiled", True):
    print("[PASS] larger own arena kept (300 MB > 64 MB floor) and the scene compiles")
else:
    print(f"[FAIL] own arena: {spec.memory} compiled={r3['models_compiled']} res={str(r3['result'])[:200]}"); fails += 1
# 4: misuse raises ValueError naming the parameter
bad = 0
for kw in ({"tool": "no_such_tool", "args": {}}, {"tool": "probe_scene", "args": [1]}, {"tool": "probe_scene", "args": {}, "memory_mb": 0},
           {"tool": "bench_tool_large_arena", "args": {}}):
    try:
        bench_tool_large_arena(**kw)
    except ValueError:
        bad += 1
if bad == 4:
    print("[PASS] misuse: unknown tool, non-dict args, zero memory, self-wrap all raise ValueError")
else:
    print(f"[FAIL] misuse raised {bad}/4"); fails += 1
for f in (p, big):
    os.remove(f)
sys.exit(1 if fails else 0)

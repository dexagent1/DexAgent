"""Render an exported view bundle with Blender Cycles (transparent film, PBR, tracked camera)."""
from __future__ import annotations

TOOL = {
    "name": "render_cycles",
    "title": "Render with Cycles",
    "type": "skill",
    "category": "Scene Runtime & Rendering",
    "summary": "Renders one exported view (scene.json + transforms.npz) to RGBA frames with Blender Cycles, in parallel chunks, skipping frames already on disk.",
    "description": (
        "Builds the Blender scene once from the bundle (principled materials, textures, "
        "transmission for see-through geoms, a near-fade so a wrist lens does not dissolve its "
        "own hand, sun + fill + world light from the plate), keyframes every moving geom, and "
        "lets Blender's animation loop render the frame range on a transparent film. The frames "
        "are PNG RGBA f%05d.png, ready for composite_renders. Runs the bundled stage4 renderer "
        "as a subprocess in an interpreter that has bpy, so it can be split into chunks across "
        "GPUs."
    ),
    "inputs": [
        {"name": "view_dir", "type": "path", "description": "A view folder written by export_blender_bundle."},
        {"name": "chunks", "type": "int", "description": "Parallel processes over the frame range (1)."},
        {"name": "python", "type": "str | None", "description": "Interpreter with bpy installed (default: current)."},
        {"name": "samples", "type": "int | None", "description": "Override Cycles samples in scene.json before rendering."},
        {"name": "env", "type": "dict | None", "description": "Extra environment, e.g. CUDA_VISIBLE_DEVICES per chunk as a list under 'gpus'."},
    ],
    "outputs": [
        {"name": "result", "type": "dict", "description": "frames dir, n_rendered, runtime_s, log paths."},
    ],
    "status": "tested",
    "entrypoint": "render_cycles",
    "requirements": ["bpy (Blender as a Python module, 4.x)", "the stage4_robot_data_generation package"],
    "tags": ["Blender", "Cycles", "photoreal", "render"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Knot the Rope",
                "Battery Insertion", "Drawing and Wiping", "Computer Installation", "Open the Bottle", "Scissor Cutting",
                "Biology Experiment"],
    "cli": "python -m tool_library.skills.render_cycles bundle/ego --chunks 2",
}

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from tool_library.common.stage4 import STAGE4_ROOT


def render_cycles(view_dir, chunks=1, python=None, samples=None, env=None):
    view = Path(view_dir)
    meta_f = view / "scene.json"
    assert meta_f.exists(), f"{view} is not an exported view (no scene.json)"
    assert (STAGE4_ROOT / "stage4" / "render_cycles.py").exists(), f"stage4 package missing at {STAGE4_ROOT}"
    if samples is not None:
        meta = json.loads(meta_f.read_text())
        meta["samples"] = int(samples)
        meta_f.write_text(json.dumps(meta, indent=1))
    n = int(json.loads(meta_f.read_text())["n_frames"])
    gpus = (env or {}).pop("gpus", None) if env else None
    t0 = time.time()
    procs, logs = [], []
    for i in range(int(chunks)):
        e = {**os.environ, **(env or {})}
        if gpus:
            e["CUDA_VISIBLE_DEVICES"] = str(gpus[i % len(gpus)])
        log = view / f"render_{i}.log"
        logs.append(str(log))
        cmd = [python or sys.executable, "-m", "stage4.render_cycles", str(view)] + ([f"{i}/{chunks}"] if chunks > 1 else [])
        procs.append(subprocess.Popen(cmd, cwd=STAGE4_ROOT, env=e, stdout=open(log, "w"), stderr=subprocess.STDOUT))
    rcs = [p.wait() for p in procs]
    frames = view / "frames"
    done = sum((frames / f"f{t:05d}.png").exists() for t in range(n))
    if any(rcs) or done < n:
        tail = "".join(Path(l).read_text()[-1500:] for l in logs)
        raise RuntimeError(f"render failed (exit codes {rcs}, {done}/{n} frames); logs: {logs}\n{tail}")
    return {"frames_dir": str(frames), "n_rendered": done, "runtime_s": round(time.time() - t0, 1), "logs": logs}


if __name__ == "__main__":
    a = sys.argv[1:]
    print(json.dumps(render_cycles(a[0], int(a[2]) if len(a) > 2 and a[1] == "--chunks" else 1), indent=1))

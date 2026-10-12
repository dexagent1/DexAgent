"""A MuJoCo planning copy of the cords Isaac Sim simulates, so the executor's rope moves have links to act on."""
from __future__ import annotations

TOOL = {
    "name": "build_cord_planning_scene",
    "title": "Build cord planning scene",
    "type": "skill",
    "category": "Trajectory & Execution",
    "summary": "Writes a copy of the Stage 2 scene with every cord that Isaac Sim simulates added as a MuJoCo planning copy "
               "(construct_rope-named links at the cord's settled, or a recorded, state), so execute_subgoal_plan's rope "
               "moves (lay, drag, pin) can pinch and carry its links; the cord's physics of record stays Isaac Sim's.",
    "description": (
        "Deformables are simulated in Isaac Sim (Stage 2's cord_json; execute_isaac_deformables in Stage 3). The "
        "executor's rope moves and plan_knot_moves address a cable's links by body name, so this tool adds each Isaac "
        "cord to a copy of scene.xml as a chain of the same links (construct_rope's names <name>B_first .. B_last, ball "
        "joints with light damping), laid at the cord's settled Stage 2 state, or at `state` (a recording's frame: "
        "after an earlier move, plan the next one from where Isaac Sim left the cord: state={'recording': "
        "isaac/isaac_recording.npz, 'frame': -1}). Execute the robot in this scene, then run execute_isaac_deformables "
        "on the executed scene and trajectory: the planning copy's columns are overwritten by the Isaac recording in "
        "the mirror the gates read, so what MuJoCo did to the copy is never evidence. Returns the scene path and the "
        "cords' body names."
    ),
    "inputs": [
        {"name": "scene_recon_json", "type": "path", "description": "Stage 2 scene_recon.json (its Isaac deformables and scene.xml)."},
        {"name": "out_xml", "type": "path", "description": "The scene to write (keyframes extended with the cords' state)."},
        {"name": "scene_xml", "type": "path | None", "description": "The scene to extend; default the Stage 2 scene.xml."},
        {"name": "state", "type": "dict | None", "description": "{recording, frame}: lay the cords as an Isaac recording left them. Default the settled Stage 2 state."},
    ],
    "outputs": [
        {"name": "result", "type": "dict", "description": "scene (path), cords {name: [link body names]}, slices {name: [first, last + 1] qpos}."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo", "Isaac Sim"],
    "tags": ["stage3", "deformable", "rope", "planning", "isaac sim"],
    "used_in": ["Knot the Rope"],
    "status": "tested",
    "entrypoint": "build_cord_planning_scene",
}

import json
from pathlib import Path

import numpy as np

from tool_library.common import isaac_cord as IC


def build_cord_planning_scene(scene_recon_json, out_xml, scene_xml=None, state=None):
    from tool_library.skills.execute_isaac_deformables import isaac_deformables
    rp = Path(scene_recon_json)
    recon = json.loads(rp.read_text())
    pairs = isaac_deformables(rp)
    if not pairs:
        raise ValueError("the Stage 2 scene has no deformable simulated in Isaac Sim")
    specs = [s for _, s in pairs]
    initial = None
    if state:
        z = np.load(state["recording"], allow_pickle=True)
        k = int(state.get("frame", -1))
        initial = {s["name"]: {"centres": z[f"cord_{s['name']}_centres"][k], "quats": z[f"cord_{s['name']}_quats"][k]} for s in specs}
    base = Path(scene_xml) if scene_xml else rp.parent / recon.get("scene_xml", "scene.xml")
    info = IC.add_mirrors(base, specs, out_xml, initial=initial, proxy=True)
    return {"scene": info["path"], "cords": {s["name"]: IC.body_names(s["name"], s["n_links"]) for s in specs},
            "slices": info["slices"]}

"""Run a hand-scene skill or verifier with an enlarged MuJoCo arena for contact-dense objects."""
from __future__ import annotations

TOOL = {
    "name": "hand_tool_large_arena",
    "title": "Hand tool with a large arena",
    "type": "skill",
    "category": "Grasp & Poses",
    "summary": "Run any library skill or verifier that builds floating-hand scenes (pose solvers, grasp generators, stress "
               "tests) with the hand scene compiled under a larger MuJoCo arena, so contact-dense objects (perforated "
               "racks, tip trays, thin-walled tubes made of many wedges) do not abort the solver with mj_stackAlloc overflow.",
    "description": (
        "The floating-hand scenes the grasp tools build pair every hand collision geom with every object geom at condim 6. "
        "With an object made of hundreds of small geoms (a perforated rack, a 96-well tip tray, a cup built from wedges) a "
        "seed that overlaps the object creates hundreds of contacts and the default arena that MuJoCo sizes from the model "
        "overflows inside mj_forward (FatalError mj_stackAlloc), which kills the whole tool call. This wrapper sets "
        "<size memory=...> on the base MJCF every hand scene is compiled from (tool_library.common.hand), for the duration "
        "of one call of the named tool, then restores it. The tool's own arguments pass through unchanged and its result "
        "is returned verbatim with the arena size and the number of hand scenes compiled during the call. Nothing about "
        "the physics changes except the memory available for contacts and constraints."
    ),
    "inputs": [
        {"name": "tool", "type": "str", "description": "Name of a library skill or verifier that takes a hand scene dict (e.g. pose_from_points, generate_rigid_grasps, grasp_lift_test)."},
        {"name": "args", "type": "dict", "description": "Keyword arguments for that tool, exactly as it takes them."},
        {"name": "memory_mb", "type": "int", "description": "Arena size in megabytes for every hand scene compiled during the call. Default 128 (range 8..4096)."},
    ],
    "outputs": [
        {"name": "out", "type": "dict", "description": "{tool, memory_mb, hand_scenes_built, narena_bytes (smallest arena of the compiled hand scenes), result (the tool's own return value)}."},
    ],
    "requirements": ["mujoco>=3.2"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "grasp", "hand scene", "memory", "contact-dense", "wrapper"],
    "status": "tested",
    "entrypoint": "hand_tool_large_arena",
}

import contextlib
import importlib

from tool_library.common import hand as H


@contextlib.contextmanager
def large_arena(memory_mb=128):
    """Context in which every hand scene built by tool_library.common.hand gets a memory_mb arena."""
    memory_mb = int(memory_mb)
    if not 8 <= memory_mb <= 4096:
        raise ValueError(f"memory_mb must be within 8..4096, got {memory_mb}")
    old_xml, old_build = H._SCENE_XML, H.build_hand_scene
    built = []
    tag = f'<size memory="{memory_mb}M"/>'

    def build(*a, **k):
        out = old_build(*a, **k)
        built.append(int(out[0].narena))
        return out

    H._SCENE_XML = old_xml.replace('<compiler angle="radian"/>', '<compiler angle="radian"/>\n  ' + tag, 1)
    if tag not in H._SCENE_XML:
        H._SCENE_XML = old_xml
        raise RuntimeError("hand scene base MJCF has no <compiler angle=\"radian\"/> line to anchor the size tag")
    H.build_hand_scene = build
    try:
        yield built
    finally:
        H._SCENE_XML, H.build_hand_scene = old_xml, old_build


def _resolve(tool):
    if not isinstance(tool, str) or not tool.isidentifier():
        raise ValueError(f"tool must be a library tool name, got {tool!r}")
    for pkg in ("tool_library.skills", "tool_library.verifiers"):
        try:
            mod = importlib.import_module(f"{pkg}.{tool}")
        except ModuleNotFoundError:
            continue
        fn = getattr(mod, getattr(mod, "TOOL", {}).get("entrypoint", tool), None)
        if callable(fn):
            return fn
    raise ValueError(f"tool {tool!r} is not a skill or verifier of the library")


def hand_tool_large_arena(tool, args=None, memory_mb=128):
    if tool == "hand_tool_large_arena":
        raise ValueError("tool must name another library tool, not this wrapper")
    if args is not None and not isinstance(args, dict):
        raise ValueError("args must be a dict of the tool's keyword arguments")
    fn = _resolve(tool)
    with large_arena(memory_mb) as built:
        res = fn(**(args or {}))
    return {"tool": tool, "memory_mb": int(memory_mb), "hand_scenes_built": len(built),
            "narena_bytes": min(built) if built else None, "result": res}

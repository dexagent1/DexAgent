TOOL = {
    "name": "bench_tool_large_arena",
    "title": "Bench tool with a large arena",
    "type": "skill",
    "category": "Trajectory & Execution",
    "summary": "Run any library skill or verifier that loads a bench scene from its MJCF (the subgoal executor, the planner's preview, the episode audits) with every model compiled during the call given a larger MuJoCo arena, so contact-dense states (a lid seated on a rim, snap latches pressed over their nubs, many small geoms touching at once) do not abort the call with mj_stackAlloc overflow.",
    "description": (
        "MuJoCo sizes the arena (the memory for contacts and constraints) from the model unless the MJCF sets <size memory>. "
        "A bench scene whose state packs hundreds of contacts (a lid with a gasket seated on a rim, a strap pressed over a nub, "
        "a hand on top of both) can overflow it inside mj_forward, and the FatalError kills the whole tool call even though "
        "nothing is wrong with the plan or the physics (the same frames run cleanly with more memory). For the duration of one "
        "call of the named tool this wrapper makes every model compiled through mujoco.MjModel.from_xml_path, "
        "mujoco.MjModel.from_xml_string and mujoco.MjSpec.compile get at least `memory_mb` megabytes of arena (a model that "
        "already asks for more keeps its own), then restores the original loaders, also when the tool raises. The tool's own "
        "arguments pass through unchanged and its result is returned verbatim, with the arena size used and how many models "
        "were compiled during the call. Nothing about the physics changes: only the memory available for contacts and "
        "constraints. It does not edit any scene file."
    ),
    "inputs": [
        {"name": "tool", "type": "str", "description": "Name of a library skill or verifier (e.g. execute_subgoal_plan, preview_keyframes, episode_physics_check)."},
        {"name": "args", "type": "dict", "description": "The tool's own arguments, passed through unchanged."},
        {"name": "memory_mb", "type": "int", "description": "Minimum arena per compiled model in megabytes. Default 256."},
    ],
    "outputs": [
        {"name": "result", "type": "dict", "description": "{result: the tool's return value verbatim, tool, memory_mb, models_compiled}."},
    ],
    "status": "tested",
    "entrypoint": "bench_tool_large_arena",
    "requirements": ["mujoco>=3.2"],
    "tags": ["stage3", "arena", "memory", "mj_stackAlloc", "contacts", "wrapper"],
}

import importlib

import mujoco


def _load_tool(name):
    if not isinstance(name, str) or not name.strip():
        raise ValueError("tool must be the name of a library skill or verifier")
    if name == "bench_tool_large_arena":
        raise ValueError("tool: bench_tool_large_arena cannot wrap itself")
    for kind in ("skills", "verifiers"):
        try:
            mod = importlib.import_module(f"tool_library.{kind}.{name}")
        except ModuleNotFoundError as e:
            if e.name and e.name.endswith(name):
                continue
            raise
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn
    raise ValueError(f"tool: no library skill or verifier named {name!r}")


def bench_tool_large_arena(tool, args, memory_mb=256):
    fn = _load_tool(tool)
    if not isinstance(args, dict):
        raise ValueError("args must be a dict of the wrapped tool's arguments")
    try:
        memory_mb = int(memory_mb)
    except (TypeError, ValueError):
        raise ValueError("memory_mb must be a positive integer (megabytes)")
    if memory_mb <= 0:
        raise ValueError("memory_mb must be a positive integer (megabytes)")
    want = memory_mb * 1024 * 1024
    count = {"n": 0}
    orig_path = mujoco.MjModel.from_xml_path
    orig_string = mujoco.MjModel.from_xml_string
    orig_compile = mujoco.MjSpec.compile

    def _compile(spec, *a, **k):
        if int(spec.memory) < want:
            spec.memory = want
        count["n"] += 1
        return orig_compile(spec, *a, **k)

    def _from_path(filename, assets=None):
        spec = mujoco.MjSpec.from_file(str(filename), assets=assets) if assets else mujoco.MjSpec.from_file(str(filename))
        return _compile(spec)

    def _from_string(xml, assets=None):
        spec = mujoco.MjSpec.from_string(xml, assets=assets) if assets else mujoco.MjSpec.from_string(xml)
        return _compile(spec)

    mujoco.MjModel.from_xml_path = staticmethod(_from_path)
    mujoco.MjModel.from_xml_string = staticmethod(_from_string)
    mujoco.MjSpec.compile = _compile
    try:
        out = fn(**args)
    finally:
        mujoco.MjModel.from_xml_path = orig_path
        mujoco.MjModel.from_xml_string = orig_string
        mujoco.MjSpec.compile = orig_compile
    return {"result": out, "tool": tool, "memory_mb": memory_mb, "models_compiled": count["n"]}

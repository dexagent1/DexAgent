"""Canonical robot reference for proposal scenes and rigid execution scenes."""
from functools import lru_cache
from pathlib import Path

from tool_library.common import scene_assert as SA


@lru_cache(maxsize=8)
def _runtime_reference(scene_path, mtime_ns):
    from tool_library.common.rigid_robot import compile_rigid
    return compile_rigid(scene_path, setup_scene=scene_path).model


def load_reference(model, setup=None):
    """Derive runtime structure from trusted setup, never from the submitted robot."""
    from tool_library.common.rigid_robot import is_rigid
    ref = dict(SA.load_setup(setup))
    ref["runtime"] = is_rigid(model)
    if ref["runtime"]:
        path = Path(ref["scene_path"]).resolve()
        ref["model"] = _runtime_reference(str(path), path.stat().st_mtime_ns)
    return ref

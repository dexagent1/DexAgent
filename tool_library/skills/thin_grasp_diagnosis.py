"""Can a lying part be pinched or wrapped from where it lies? If not, which primitive makes it graspable."""
from __future__ import annotations

TOOL = {
    "name": "thin_grasp_diagnosis",
    "title": "Thin grasp diagnosis",
    "type": "skill",
    "category": "Grasp & Poses",
    "summary": "Decide from geometry whether a part lying on the bench (a phone, a card, scissors, a strap, a flat ribbon) can "
               "be pinched or wrapped where it lies, and if not, name the primitive that makes it graspable (slide it to "
               "overhang the edge and pinch it from below, tilt it up, hook it through an opening, gather a strip into an "
               "arch) with the numbers it decided on.",
    "description": (
        "Every grasp skill assumes the pads can close on the part's sides or get under it. The hand's own sizes decide "
        "that (read from its collision capsules: pad radius, fingertip thickness, thumb-index span), against the part as "
        "it lies (its footprint and height above its support, the gap under it from rays cast up from the support, its "
        "through-openings from above, how far it overhangs or stays inside each table edge, its mass and the "
        "friction it has on the support). Verdicts: side_pinch (height >= pad radius + 1 mm off the table + 2 mm under "
        "the top edge: 10 mm for the Sharpa hand), pinch_under (a gap a fingertip thick + 2 mm), wrap, edge_pinch (the "
        "overhang the moment balance needs: fingers under the part deeper than the thumb on top, the thumb pressing at "
        "most 5 N, the finger a radius plus 3 mm past the table's edge; and no more overhang than keeps the centre of "
        "mass 12 mm on the table), slide (to the nearest table edge: a drag with pads pressed on its top needs the "
        "hand's friction 1.2 above the support's and a press N = mu_s m g / (mu_h - mu_s) within the 40 N wrist cap; a "
        "push needs a side face taller than a pad radius + 2 mm), tilt_up (pressing the overhanging end about the edge "
        "raises the far end by the lever ratio), hook (an opening wider than a fingertip + 3 mm; the thumb outside when "
        "the wall is 10 mm tall), strip_pinch (a flexible chain thinner than a side pinch needs). `recommend` orders the "
        "plans that work, with the tools that build their poses, or says nothing in the library makes it graspable (then "
        "report the task blocked with these measurements). Call it before trying grasps on a thin or flat part."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "Bench scene MJCF (Stage 2 scene.xml)."},
        {"name": "body", "type": "str", "description": "The part: an object's root body or a part of it (its subtree is measured)."},
        {"name": "side", "type": "str", "description": "Which hand ('left' or 'right'); its sizes are read from the scene. Default 'right'."},
        {"name": "state", "type": "str | list | None", "description": "A trajectory.npz (last frame) or a qpos: the part where an earlier subgoal left it. Default the scene's keyframe."},
        {"name": "out_json", "type": "str | None", "description": "Write the report here."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "geometry {height_m, support, footprint, edges {+x|-x|+y|-y: {overhang_m, edge_distance_m}}, under {max_gap_m, at}, openings [...], mass_kg, mu_support, flexible}, hand {pad_radius_m, tip_thickness_m, finger_thickness_m, span_m}, verdicts {side_pinch, pinch_under, wrap, edge_pinch, slide, tilt_up, hook, strip_pinch: {ok, why, ...numbers}}, graspable_as_lies, recommend [{plan, why}]."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["grasp", "thin-object", "flat", "diagnosis", "phone", "scissors", "ribbon", "stage3"],
    "status": "tested",
    "entrypoint": "thin_grasp_diagnosis",
}

import json
from pathlib import Path

import mujoco
import numpy as np

from tool_library.common import episode as E
from tool_library.common import nonprehensile as NP
from tool_library.common import plan as P


def thin_grasp_diagnosis(scene, body, side="right", state=None, out_json=None):
    model = mujoco.MjModel.from_xml_path(str(scene)) if isinstance(scene, (str, Path)) else scene
    data = mujoco.MjData(model)
    if state is None:
        if model.nkey:
            mujoco.mj_resetDataKeyframe(model, data, 0)
    elif isinstance(state, (str, Path)):
        data.qpos[:] = E.state_qpos(model, state)
    else:
        data.qpos[:] = np.asarray(state, float)
    mujoco.mj_forward(model, data)
    rigs = P.scene_rigs(model)
    if side not in rigs:
        raise ValueError(f"no {side} hand in the scene (hands: {sorted(rigs)})")
    rep = NP.diagnose(model, data, body, rigs[side])
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(rep, indent=1, default=float))
        rep["out_json"] = str(out_json)
    return rep

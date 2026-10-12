"""Construct an articulated drawer cabinet (ArtiCraft route): measured
structure in, URDF with prismatic drawers out, geometry QC included.

The agent supplies MEASURED structure: outer width/depth/height in metres, a
TOP-to-BOTTOM grid of drawer rows (each row: how many drawers across and its
share of the height), which drawer the demonstration moves, the travel, and
colours read off the video (carcass, drawer fronts, pulls). The tool renders
these into a parametric ArtiCraft `model.py` and compiles it through
ArtiCraft's own geometry QC (parts must not overlap in sampled poses, joint
origins must sit on geometry, each drawer must move along +y by at least half
its travel), giving a URDF plus closed/open preview renders. Without an
ArtiCraft checkout the same geometry is written as a URDF directly, and the
library's own checks (urdf_articulation_check, articulated_joint_travel_test
after port_urdf_asset) stand in for ArtiCraft's QC. Handles stand 32 mm proud
of the front so a robot finger fits behind the bar.

    from tool_library.skills.construct_articulated import construct_articulated
    out = construct_articulated(out_dir, width=0.47, depth=0.45, height=0.49,
                                rows=[{"cols": 2, "height_frac": 0.3}, {"cols": 1}, {"cols": 1}],
                                moving={"row": 1, "col": 0}, body_rgba=[0.58, 0.44, 0.25, 1])
    out["urdf"], out["joints"], out["moving_joint"], out["backend"]
"""
from __future__ import annotations

TOOL = {
    "name": "construct_articulated",
    "title": "Construct articulated cabinet",
    "type": "skill",
    "category": "Object Reconstruction",
    "summary": "Measured dims + a top-to-bottom drawer grid + measured colours -> ArtiCraft "
               "model.py compiled through ArtiCraft's geometry QC to a URDF with prismatic "
               "drawers and finger-clearing pulls; a built-in URDF writer when ArtiCraft is absent.",
    "description": (
        "Writes a parametric model.py against the ArtiCraft SDK: a carcass of five panels with "
        "shelf webs between rows and dividers between columns, one drawer box per cell (front, "
        "two sides, back, bottom) with two posts and a pull bar standing max(32 mm, 6 x bar "
        "radius) proud of the front, and a PRISMATIC articulation per drawer along +y with "
        "limits 0..travel_frac x depth. Panel thickness, row gap, side clearance and reveal "
        "scale with the dims. With backend='articraft' (default when ARTICRAFT_DIR or "
        "`articraft_dir` points at a checkout) the model is copied into a fresh record, "
        "compiled with `cli.compile_record`, and accepted only on 'failures=0': ArtiCraft's "
        "checks that no parts overlap in sampled poses, that joint origins lie on geometry and "
        "that each drawer moves >= 0.5 x travel along +y. With backend='builtin' the identical "
        "geometry is emitted as a URDF by this file (same names, materials and joints), so the "
        "tool works without ArtiCraft. Either way the result lists the joint names "
        "('carcass_to_drawer_r<i>c<j>'), the moving joint, the measured layout actually built, "
        "and closed/open preview PNGs rendered with MuJoCo. Colours are not defaulted silently: "
        "body_rgba and front_rgba should come from a colour measurement of the video, and the "
        "handle must differ from the body (a pull is too small to appear in a palette, so it "
        "is read by eye: black ~[0.05,0.05,0.05,1], steel ~[0.75,0.75,0.78,1])."
    ),
    "inputs": [
        {"name": "out_dir", "type": "str", "description": "Directory for model.py, model.urdf, previews and construct_articulated_params.json."},
        {"name": "width", "type": "float, metres", "description": "Outer width across the drawer fronts (measured)."},
        {"name": "depth", "type": "float, metres", "description": "Outer depth front to back; drawers pull out along +y (measured)."},
        {"name": "height", "type": "float, metres", "description": "Outer height (measured)."},
        {"name": "rows", "type": "list[dict]",
         "description": "TOP row first: [{'cols': int, 'height_frac': float}, ...]; height_frac defaults to equal shares."},
        {"name": "moving", "type": "dict",
         "description": "{'row': i, 'col': j} of the drawer the demonstration moves (0-based). Default row 0, col 0."},
        {"name": "travel_frac", "type": "float", "description": "Drawer travel as a fraction of depth. Default 0.7."},
        {"name": "body_rgba", "type": "list[float] | None", "description": "Carcass colour (measured). Default a pale board."},
        {"name": "front_rgba", "type": "list[float] | None", "description": "Drawer front colour (measured). Default near-white."},
        {"name": "handle_rgba", "type": "list[float] | None", "description": "Pull colour. Default dark hardware [0.12, 0.12, 0.13, 1]."},
        {"name": "name", "type": "str", "description": "Robot/object name in the URDF. Default 'egoscene_cabinet'."},
        {"name": "description", "type": "str", "description": "Appearance notes recorded with the asset."},
        {"name": "backend", "type": "str", "description": "'auto' (default), 'articraft' or 'builtin'."},
        {"name": "articraft_dir", "type": "str | None", "description": "ArtiCraft checkout; default env ARTICRAFT_DIR."},
        {"name": "articraft_python", "type": "str | None", "description": "Interpreter with ArtiCraft's deps; default <articraft_dir>/.venv/bin/python or env ARTICRAFT_PY."},
        {"name": "base_record", "type": "str", "description": "Record under data/records to clone. Default 'rec_drawer_cabinet_with_sliding_drawers_0001' or env ARTICRAFT_BASE_RECORD."},
        {"name": "preview", "type": "bool", "description": "Render closed/open PNGs with MuJoCo. Default True."},
        {"name": "timeout", "type": "float, seconds", "description": "ArtiCraft compile timeout (retried once on a cold filesystem). Default 600."},
    ],
    "outputs": [
        {"name": "urdf", "type": "str", "description": "The URDF (model.urdf in out_dir)."},
        {"name": "model_py", "type": "str", "description": "The ArtiCraft model.py that describes the same object."},
        {"name": "backend", "type": "str", "description": "'articraft' or 'builtin': which produced the URDF."},
        {"name": "qc", "type": "dict", "description": "ArtiCraft's compile summary line and report path, or the builtin writer's note."},
        {"name": "joints", "type": "list[str]", "description": "Prismatic joint names, top row first."},
        {"name": "moving_joint", "type": "str", "description": "The joint of the drawer the demo moves."},
        {"name": "layout", "type": "list[dict]", "description": "Per row: centre_z, row_h, cols as built."},
        {"name": "previews", "type": "list[str]", "description": "closed/open PNGs when preview=True."},
        {"name": "params_json", "type": "str", "description": "The parameters, for rebuilding."},
    ],
    "status": "tested",
    "entrypoint": "construct_articulated",
    "requirements": ["numpy", "mujoco", "ArtiCraft checkout (optional)"],
    "simulators": ["MuJoCo"],
    "tags": ['articulated', 'drawer', 'cabinet', 'urdf', 'articraft', 'prismatic', 'reconstruct', 'parametric', 'object reconstruction'],
    "used_in": ["Place Toy in Drawer"],
    "cli": "python -m tool_library.skills.construct_articulated <out_dir> <width> <depth> <height> <rows_json>",
}

import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import numpy as np

# The ArtiCraft template: parameterised ONLY via
# the PARAMS literal injected at the top.
_TEMPLATE = '''from __future__ import annotations

import json

from sdk import (
    ArticulatedObject,
    ArticulationType,
    AssetContext,
    Box,
    Cylinder,
    Inertial,
    Material,
    MotionLimits,
    Origin,
    TestContext,
    TestReport,
)

ASSETS = AssetContext.from_script(__file__)
HERE = ASSETS.asset_root

PARAMS = json.loads(r"""__PARAMS_JSON__""")

W = float(PARAMS["width"])
D = float(PARAMS["depth"])
H = float(PARAMS["height"])
ROWS = PARAMS["rows"]                    # top to bottom: {"cols": n, "height_frac": f}
TRAVEL = float(PARAMS.get("travel_frac", 0.7)) * D
BODY_RGBA = tuple(PARAMS.get("body_rgba", [0.92, 0.91, 0.88, 1.0]))
FRONT_RGBA = tuple(PARAMS.get("front_rgba", [0.95, 0.94, 0.92, 1.0]))
# Hardware is dark far more often than it is near-white, and a handle
# is too small to form its own colour cluster in the measured palette,
# so a pale default would go unnoticed on a cabinet whose real pulls
# are black. This is a PRIOR, not a measurement — pass handle_rgba to
# override it whenever the video shows otherwise.
HANDLE_RGBA = tuple(PARAMS.get("handle_rgba", [0.12, 0.12, 0.13, 1.0]))

T = max(0.006, min(0.02, 0.03 * min(W, H)))      # panel thickness
GAP = max(0.003, 0.01 * H)                        # row gap
CLEAR = max(0.002, 0.006 * W)                     # drawer side clearance
REVEAL = max(0.002, 0.008 * W)                    # front reveal
INNER_W = W - 2.0 * T


def _rows_layout():
    """[(center_z, row_h, cols)] top to bottom."""
    n = len(ROWS)
    fracs = [max(0.05, float(r.get("height_frac", 1.0 / n))) for r in ROWS]
    s = sum(fracs)
    avail = H - 2.0 * T - (n + 1) * GAP
    out = []
    z_top = H - T - GAP
    for r, f in zip(ROWS, fracs):
        row_h = avail * f / s
        out.append((z_top - row_h / 2.0, row_h, int(r.get("cols", 1))))
        z_top -= row_h + GAP
    return out


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject(name="egoscene_cabinet", assets=ASSETS)
    body = Material(name="body", rgba=BODY_RGBA)
    front_m = Material(name="front", rgba=FRONT_RGBA)
    handle = Material(name="handle", rgba=HANDLE_RGBA)
    model.materials.extend([body, front_m, handle])

    carcass = model.part("carcass")
    zc = H / 2.0
    carcass.visual(Box((T, D, H)),
                   origin=Origin(xyz=(-W / 2.0 + T / 2.0, 0.0, zc)),
                   material=body, name="left_side")
    carcass.visual(Box((T, D, H)),
                   origin=Origin(xyz=(W / 2.0 - T / 2.0, 0.0, zc)),
                   material=body, name="right_side")
    carcass.visual(Box((INNER_W, D, T)),
                   origin=Origin(xyz=(0.0, 0.0, H - T / 2.0)),
                   material=body, name="top_panel")
    carcass.visual(Box((INNER_W, D, T)),
                   origin=Origin(xyz=(0.0, 0.0, T / 2.0)),
                   material=body, name="bottom_panel")
    carcass.visual(Box((INNER_W, T, H - 2.0 * T)),
                   origin=Origin(xyz=(0.0, -D / 2.0 + T / 2.0, zc)),
                   material=body, name="rear_panel")
    carcass.inertial = Inertial.from_geometry(
        Box((W, D, H)), mass=max(0.6, 30.0 * W * D * H),
        origin=Origin(xyz=(0.0, 0.0, zc)))

    layout = _rows_layout()
    # shelf webs between rows + column dividers
    for i, (center_z, row_h, cols) in enumerate(layout):
        if i > 0:
            above_z = layout[i - 1][0] - layout[i - 1][1] / 2.0
            web_z = (above_z + center_z + row_h / 2.0) / 2.0
            carcass.visual(Box((INNER_W, D - T, max(0.004, GAP * 0.8))),
                           origin=Origin(xyz=(0.0, T / 2.0, web_z)),
                           material=body, name=f"web_{i}")
        cell_span = INNER_W / cols
        for c in range(1, cols):
            x = -INNER_W / 2.0 + c * cell_span
            carcass.visual(Box((T, D - T, row_h)),
                           origin=Origin(xyz=(x, T / 2.0, center_z)),
                           material=body, name=f"divider_r{i}c{c}")

    box_depth = 0.8 * D - T
    for i, (center_z, row_h, cols) in enumerate(layout):
        cell_span = INNER_W / cols
        for c in range(cols):
            name = f"drawer_r{i}c{c}"
            cell_cx = -INNER_W / 2.0 + (c + 0.5) * cell_span
            front_w = cell_span - 2.0 * REVEAL - (T if cols > 1 else 0.0)
            box_w = front_w - 2.0 * CLEAR
            box_h = max(0.01, row_h - 0.006)
            side_h = box_h - 0.004
            dt = min(T, 0.008)
            drawer = model.part(name)
            drawer.visual(Box((front_w, dt, row_h - 0.004)),
                          origin=Origin(xyz=(0.0, box_depth / 2.0 + dt / 2.0, 0.0)),
                          material=front_m, name="front")
            drawer.visual(Box((dt, box_depth, side_h)),
                          origin=Origin(xyz=(-box_w / 2.0 + dt / 2.0, 0.0, 0.0)),
                          material=body, name="side_left")
            drawer.visual(Box((dt, box_depth, side_h)),
                          origin=Origin(xyz=(box_w / 2.0 - dt / 2.0, 0.0, 0.0)),
                          material=body, name="side_right")
            drawer.visual(Box((box_w - 2.0 * dt, dt, side_h)),
                          origin=Origin(xyz=(0.0, -box_depth / 2.0 + dt / 2.0, 0.0)),
                          material=body, name="back")
            drawer.visual(Box((box_w - 2.0 * dt, box_depth - 2.0 * dt, dt)),
                          origin=Origin(xyz=(0.0, 0.0, -side_h / 2.0 + dt / 2.0)),
                          material=body, name="bottom")
            pr = max(0.003, min(0.006, 0.06 * row_h))
            span = 0.35 * front_w
            # HANDLE STANDOFF: the bar must sit far enough off the
            # drawer face for a ROBOT FINGER to pass behind it. A few
            # bar radii (under 1 cm) is thinner than any fingertip and
            # leaves the pull bar ungraspable: the fingertips end up
            # beside the bar with nowhere to go. Real drawer pulls stand
            # 3-4 cm proud; size the posts from the hand's own finger
            # thickness, not from the bar radius.
            post_len = max(0.032, 6.0 * pr)
            bar_y = box_depth / 2.0 + dt + post_len
            for off in (-span / 2.0, span / 2.0):
                drawer.visual(Cylinder(radius=pr * 0.7, length=post_len),
                              origin=Origin(xyz=(off,
                                                 bar_y - post_len / 2.0,
                                                 0.0),
                                            rpy=(-1.5707963, 0.0, 0.0)),
                              material=handle)
            drawer.visual(Cylinder(radius=pr, length=span + 2.0 * pr),
                          origin=Origin(xyz=(0.0, bar_y, 0.0),
                                        rpy=(0.0, 1.5707963, 0.0)),
                          material=handle, name="pull_bar")
            drawer.inertial = Inertial.from_geometry(
                Box((box_w, box_depth, box_h)),
                mass=max(0.05, 2.0 * box_w * box_depth * box_h * 30.0),
                origin=Origin(xyz=(0.0, 0.0, 0.0)))
            model.articulation(
                f"carcass_to_{name}",
                ArticulationType.PRISMATIC,
                parent="carcass",
                child=name,
                origin=Origin(xyz=(cell_cx, D / 2.0 - dt - box_depth / 2.0,
                                   center_z)),
                axis=(0.0, 1.0, 0.0),
                motion_limits=MotionLimits(effort=30.0, velocity=0.4,
                                           lower=0.0, upper=TRAVEL),
            )
    return model


def run_tests() -> TestReport:
    ctx = TestContext(object_model, asset_root=HERE)
    ctx.check_model_valid()
    ctx.check_mesh_files_exist()
    ctx.fail_if_articulation_origin_far_from_geometry(tol=0.02)
    ctx.fail_if_parts_overlap_in_sampled_poses(
        max_pose_samples=96, overlap_tol=0.004, overlap_volume_tol=0.0)
    for i, (center_z, row_h, cols) in enumerate(_rows_layout()):
        for c in range(cols):
            name = f"drawer_r{i}c{c}"
            ctx.expect_aabb_overlap(name, "carcass", axes="xy",
                                    min_overlap=0.2 * min(1.0, row_h))
            ctx.expect_joint_motion_axis(f"carcass_to_{name}", name,
                                         world_axis="y", direction="positive",
                                         min_delta=0.5 * TRAVEL)
    return ctx.report()


object_model = build_object_model()
'''

DEFAULT_BASE_RECORD = "rec_drawer_cabinet_with_sliding_drawers_0001"


# ---------------------------------------------------------------------------
# the same geometry as the template, in plain Python (builtin URDF writer)
# ---------------------------------------------------------------------------
def _dims(params: dict) -> dict:
    W, D, H = float(params["width"]), float(params["depth"]), float(params["height"])
    T = max(0.006, min(0.02, 0.03 * min(W, H)))
    return {"W": W, "D": D, "H": H, "T": T, "GAP": max(0.003, 0.01 * H),
            "CLEAR": max(0.002, 0.006 * W), "REVEAL": max(0.002, 0.008 * W),
            "INNER_W": W - 2.0 * T, "TRAVEL": float(params.get("travel_frac", 0.7)) * D}


def rows_layout(params: dict) -> list[dict]:
    """[(centre_z, row_h, cols)] top to bottom, exactly as the template lays it out."""
    d = _dims(params)
    ROWS = params["rows"]
    n = len(ROWS)
    fracs = [max(0.05, float(r.get("height_frac", 1.0 / n))) for r in ROWS]
    s = sum(fracs)
    avail = d["H"] - 2.0 * d["T"] - (n + 1) * d["GAP"]
    out = []
    z_top = d["H"] - d["T"] - d["GAP"]
    for r, f in zip(ROWS, fracs):
        row_h = avail * f / s
        out.append({"centre_z": z_top - row_h / 2.0, "row_h": row_h, "cols": int(r.get("cols", 1))})
        z_top -= row_h + d["GAP"]
    return out


def cabinet_parts(params: dict) -> dict:
    """links: name -> {visuals: [(name, kind, size, xyz, rpy, material)], mass, inertia_box};
    joints: [(name, parent, child, xyz, axis, lo, hi)]. Mirrors build_object_model()."""
    d = _dims(params)
    W, D, H, T, GAP, CLEAR, REVEAL, INNER_W, TRAVEL = (d[k] for k in
        ("W", "D", "H", "T", "GAP", "CLEAR", "REVEAL", "INNER_W", "TRAVEL"))
    zc = H / 2.0
    carcass = [("left_side", "box", (T, D, H), (-W / 2 + T / 2, 0.0, zc), (0, 0, 0), "body"),
               ("right_side", "box", (T, D, H), (W / 2 - T / 2, 0.0, zc), (0, 0, 0), "body"),
               ("top_panel", "box", (INNER_W, D, T), (0.0, 0.0, H - T / 2), (0, 0, 0), "body"),
               ("bottom_panel", "box", (INNER_W, D, T), (0.0, 0.0, T / 2), (0, 0, 0), "body"),
               ("rear_panel", "box", (INNER_W, T, H - 2 * T), (0.0, -D / 2 + T / 2, zc), (0, 0, 0), "body")]
    layout = rows_layout(params)
    for i, row in enumerate(layout):
        center_z, row_h, cols = row["centre_z"], row["row_h"], row["cols"]
        if i > 0:
            above_z = layout[i - 1]["centre_z"] - layout[i - 1]["row_h"] / 2.0
            web_z = (above_z + center_z + row_h / 2.0) / 2.0
            carcass.append((f"web_{i}", "box", (INNER_W, D - T, max(0.004, GAP * 0.8)),
                            (0.0, T / 2, web_z), (0, 0, 0), "body"))
        cell_span = INNER_W / cols
        for c in range(1, cols):
            x = -INNER_W / 2.0 + c * cell_span
            carcass.append((f"divider_r{i}c{c}", "box", (T, D - T, row_h), (x, T / 2, center_z),
                            (0, 0, 0), "body"))
    links = {"carcass": {"visuals": carcass, "mass": max(0.6, 30.0 * W * D * H),
                         "inertia_box": ((W, D, H), (0.0, 0.0, zc))}}
    joints = []
    box_depth = 0.8 * D - T
    for i, row in enumerate(layout):
        center_z, row_h, cols = row["centre_z"], row["row_h"], row["cols"]
        cell_span = INNER_W / cols
        for c in range(cols):
            name = f"drawer_r{i}c{c}"
            cell_cx = -INNER_W / 2.0 + (c + 0.5) * cell_span
            front_w = cell_span - 2.0 * REVEAL - (T if cols > 1 else 0.0)
            box_w = front_w - 2.0 * CLEAR
            box_h = max(0.01, row_h - 0.006)
            side_h = box_h - 0.004
            dt = min(T, 0.008)
            vis = [("front", "box", (front_w, dt, row_h - 0.004), (0.0, box_depth / 2 + dt / 2, 0.0), (0, 0, 0), "front"),
                   ("side_left", "box", (dt, box_depth, side_h), (-box_w / 2 + dt / 2, 0.0, 0.0), (0, 0, 0), "body"),
                   ("side_right", "box", (dt, box_depth, side_h), (box_w / 2 - dt / 2, 0.0, 0.0), (0, 0, 0), "body"),
                   ("back", "box", (box_w - 2 * dt, dt, side_h), (0.0, -box_depth / 2 + dt / 2, 0.0), (0, 0, 0), "body"),
                   ("bottom", "box", (box_w - 2 * dt, box_depth - 2 * dt, dt), (0.0, 0.0, -side_h / 2 + dt / 2), (0, 0, 0), "body")]
            pr = max(0.003, min(0.006, 0.06 * row_h))
            span = 0.35 * front_w
            post_len = max(0.032, 6.0 * pr)
            bar_y = box_depth / 2 + dt + post_len
            for k, off in enumerate((-span / 2, span / 2)):
                vis.append((f"post_{k}", "cylinder", (pr * 0.7, post_len), (off, bar_y - post_len / 2, 0.0),
                            (-1.5707963, 0.0, 0.0), "handle"))
            vis.append(("pull_bar", "cylinder", (pr, span + 2 * pr), (0.0, bar_y, 0.0), (0.0, 1.5707963, 0.0), "handle"))
            links[name] = {"visuals": vis, "mass": max(0.05, 2.0 * box_w * box_depth * box_h * 30.0),
                           "inertia_box": ((box_w, box_depth, box_h), (0.0, 0.0, 0.0))}
            joints.append((f"carcass_to_{name}", "carcass", name,
                           (cell_cx, D / 2 - dt - box_depth / 2, center_z), (0.0, 1.0, 0.0), 0.0, TRAVEL))
    return {"links": links, "joints": joints, "layout": layout}


def _box_inertia(size, mass):
    x, y, z = size
    return (mass / 12 * (y * y + z * z), mass / 12 * (x * x + z * z), mass / 12 * (x * x + y * y))


def write_urdf(params: dict, path, name: str = "egoscene_cabinet") -> str:
    """The template's geometry as a URDF, written here (the builtin backend)."""
    parts = cabinet_parts(params)
    mats = {"body": params.get("body_rgba", [0.92, 0.91, 0.88, 1.0]),
            "front": params.get("front_rgba", [0.95, 0.94, 0.92, 1.0]),
            "handle": params.get("handle_rgba", [0.12, 0.12, 0.13, 1.0])}
    f = lambda v: " ".join(f"{float(x):g}" for x in v)
    L = [f'<robot name="{name}">']
    for mn, rgba in mats.items():
        L.append(f'  <material name="{mn}">\n    <color rgba="{f(rgba)}" />\n  </material>')
    for ln, link in parts["links"].items():
        L.append(f'  <link name="{ln}">')
        size, org = link["inertia_box"]
        ixx, iyy, izz = _box_inertia(size, link["mass"])
        L.append(f'    <inertial>\n      <origin xyz="{f(org)}" rpy="0 0 0" />\n'
                 f'      <mass value="{link["mass"]:.6g}" />\n'
                 f'      <inertia ixx="{ixx:.6g}" ixy="0" ixz="0" iyy="{iyy:.6g}" iyz="0" izz="{izz:.6g}" />\n'
                 f'    </inertial>')
        for vn, kind, size, xyz, rpy, mat in link["visuals"]:
            geo = (f'<box size="{f(size)}" />' if kind == "box"
                   else f'<cylinder radius="{size[0]:.6g}" length="{size[1]:.6g}" />')
            for tag in ("visual", "collision"):
                L.append(f'    <{tag} name="{vn}">\n      <origin xyz="{f(xyz)}" rpy="{f(rpy)}" />\n'
                         f'      <geometry>\n        {geo}\n      </geometry>\n'
                         + (f'      <material name="{mat}">\n        <color rgba="{f(mats[mat])}" />\n      </material>\n' if tag == "visual" else "")
                         + f'    </{tag}>')
        L.append("  </link>")
    for jn, parent, child, xyz, axis, lo, hi in parts["joints"]:
        L.append(f'  <joint name="{jn}" type="prismatic">\n    <origin xyz="{f(xyz)}" rpy="0 0 0" />\n'
                 f'    <parent link="{parent}" />\n    <child link="{child}" />\n'
                 f'    <axis xyz="{f(axis)}" />\n'
                 f'    <limit effort="30" velocity="0.4" lower="{lo:g}" upper="{hi:.6g}" />\n  </joint>')
    L.append("</robot>")
    Path(path).write_text("\n".join(L) + "\n")
    return str(path)


# ---------------------------------------------------------------------------
def _articraft_compile(model_py: str, description: str, slug: str, articraft_dir: Path,
                       python: Path, base_record: str, timeout: float) -> tuple[Path, str, Path]:
    rec_id = f"rec_dexagent_{slug}_{uuid.uuid4().hex[:6]}"
    rec_dir = articraft_dir / "data" / "records" / rec_id
    base = articraft_dir / "data" / "records" / base_record
    if not base.exists():
        raise FileNotFoundError(f"ArtiCraft base record {base} not found")
    if rec_dir.exists():
        shutil.rmtree(rec_dir)
    shutil.copytree(base, rec_dir)
    (rec_dir / "revisions" / "rev_000001" / "model.py").write_text(model_py)
    (rec_dir / "description.txt").write_text(description)
    cmd = [str(python), "-m", "cli.compile_record", str(rec_dir)]
    try:
        proc = subprocess.run(cmd, cwd=str(articraft_dir), capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired:      # a cold file cache; the first try warms it
        proc = subprocess.run(cmd, cwd=str(articraft_dir), capture_output=True, text=True,
                              timeout=timeout)
    urdf = articraft_dir / "data" / "cache" / "record_materialization" / rec_id / "model.urdf"
    tail = ((proc.stdout or "") + (proc.stderr or ""))[-2500:]
    if proc.returncode != 0 or not urdf.exists() or "failures=0" not in tail:
        raise RuntimeError(f"ArtiCraft compile rejected the construction:\n{tail}")
    return urdf, tail, urdf.parent / "compile_report.json"


def _previews(urdf: str, out_dir: Path, joints: list[str]) -> list[str]:
    """Closed + all-open renders through port_urdf_asset + MuJoCo."""
    try:
        import mujoco
        from tool_library.skills.port_urdf_asset import port_urdf_asset
        info = port_urdf_asset(urdf, out_dir / "preview_mjcf", name="pv", include_floor=True)
        m = mujoco.MjModel.from_xml_path(info["xml"])
        d = mujoco.MjData(m)
        fp = info["footprint_m"]
        cx = 0.5 * (fp["x"][0] + fp["x"][1]) if fp else 0.0
        H = (fp["z"][1] - fp["z"][0]) if fp else 0.5
        cam = mujoco.MjvCamera()
        D = (fp["y"][1] - fp["y"][0]) if fp else 0.5
        cam.lookat[:] = [cx, 0.25 * D, 0.5 * H]
        cam.distance = 3.2 * max(H, (fp["x"][1] - fp["x"][0]) if fp else 0.5, D)
        cam.azimuth, cam.elevation = -55.0, -18.0      # from the +y side: the drawer fronts
        out = []
        from PIL import Image
        with mujoco.Renderer(m, height=480, width=640) as r:
            for tag, frac in (("closed", 0.0), ("open", 1.0)):
                mujoco.mj_resetData(m, d)
                for j in info["joints"]:
                    jid = m.joint(j["name"]).id
                    d.qpos[m.jnt_qposadr[jid]] = j["range"][0] + frac * (j["range"][1] - j["range"][0])
                mujoco.mj_forward(m, d)
                r.update_scene(d, camera=cam)
                p = out_dir / f"preview_{tag}.png"
                Image.fromarray(r.render()).save(p)
                out.append(str(p))
        return out
    except Exception as exc:          # a preview failure is not a build failure
        return [f"preview failed: {exc!r}"]


def construct_articulated(out_dir, width, depth, height, rows, moving=None,
                          travel_frac: float = 0.7, body_rgba=None, front_rgba=None,
                          handle_rgba=None, name: str = "egoscene_cabinet",
                          description: str = "", backend: str = "auto",
                          articraft_dir=None, articraft_python=None, base_record=None,
                          preview: bool = True, timeout: float = 600.0) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not isinstance(rows, list) or not rows:
        raise ValueError("rows must be a non-empty list of {'cols': int, 'height_frac': float}, TOP row first")
    moving = dict(moving or {"row": 0, "col": 0})
    mr, mc = int(moving.get("row", 0)), int(moving.get("col", 0))
    if not (0 <= mr < len(rows)) or not (0 <= mc < int(rows[mr].get("cols", 1))):
        raise ValueError(f"moving={moving} does not address a drawer in rows={rows}")
    if handle_rgba is not None and body_rgba is not None:
        hd = float(np.linalg.norm(np.asarray(handle_rgba[:3], float) - np.asarray(body_rgba[:3], float)))
        if hd < 0.06:
            raise ValueError("handle_rgba is the carcass colour: pulls are read by eye off the frame "
                             "(black ~[0.05,0.05,0.05,1], steel ~[0.75,0.75,0.78,1]), never from the palette")
    params = {"width": float(width), "depth": float(depth), "height": float(height),
              "rows": [{"cols": int(r.get("cols", 1)),
                        **({"height_frac": float(r["height_frac"])} if "height_frac" in r else {})}
                       for r in rows],
              "travel_frac": float(travel_frac)}
    for k, v in (("body_rgba", body_rgba), ("front_rgba", front_rgba), ("handle_rgba", handle_rgba)):
        if v is not None:
            params[k] = [float(x) for x in v]
    model_py = _TEMPLATE.replace("__PARAMS_JSON__", json.dumps(params, indent=1))
    if name != "egoscene_cabinet":
        model_py = model_py.replace('ArticulatedObject(name="egoscene_cabinet"', f'ArticulatedObject(name="{name}"')
    (out_dir / "model.py").write_text(model_py)
    (out_dir / "description.txt").write_text(str(description))

    adir = Path(articraft_dir or os.environ.get("ARTICRAFT_DIR", "")) if (articraft_dir or os.environ.get("ARTICRAFT_DIR")) else None
    have = adir is not None and (adir / "cli" / "compile_record.py").exists()
    if backend == "auto":
        backend = "articraft" if have else "builtin"
    qc = {}
    urdf_out = out_dir / "model.urdf"
    if backend == "articraft":
        if not have:
            raise FileNotFoundError("backend='articraft' needs an ArtiCraft checkout: pass articraft_dir "
                                    "or set ARTICRAFT_DIR")
        py = Path(articraft_python or os.environ.get("ARTICRAFT_PY") or (adir / ".venv" / "bin" / "python"))
        slug = "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")
        urdf, summary, rep = _articraft_compile(model_py, str(description), slug, adir, py,
                                                base_record or os.environ.get("ARTICRAFT_BASE_RECORD", DEFAULT_BASE_RECORD),
                                                float(timeout))
        shutil.copy(urdf, urdf_out)
        if rep.exists():
            shutil.copy(rep, out_dir / "compile_report.json")
        line = next((ln.strip() for ln in summary.splitlines() if "failures=" in ln), "")
        if rep.exists():
            try:
                line = json.loads(rep.read_text()).get("signal_bundle", {}).get("summary", line) or line
            except Exception:
                pass
        qc = {"backend": "articraft", "summary": line.splitlines()[0] if line else "",
              "report": str(out_dir / "compile_report.json") if rep.exists() else None,
              "record": str(urdf.parent)}
    elif backend == "builtin":
        write_urdf(params, urdf_out, name=name)
        qc = {"backend": "builtin",
              "note": "geometry written directly; run urdf_articulation_check and, after "
                      "port_urdf_asset, articulated_joint_travel_test in place of ArtiCraft's QC"}
    else:
        raise ValueError("backend must be 'auto', 'articraft' or 'builtin'")
    joints = [f"carcass_to_drawer_r{i}c{c}" for i, r in enumerate(params["rows"]) for c in range(r["cols"])]
    layout = rows_layout(params)
    previews = _previews(str(urdf_out), out_dir, joints) if preview else []
    out = {"urdf": str(urdf_out), "model_py": str(out_dir / "model.py"), "backend": backend,
           "qc": qc, "joints": joints, "moving_joint": f"carcass_to_drawer_r{mr}c{mc}",
           "layout": [{k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()} for row in layout],
           "previews": previews, "params_json": str(out_dir / "construct_articulated_params.json"),
           "note": ("+y is the cabinet front: drawers pull out along +y. Port to MJCF with "
                    "port_urdf_asset; in a scene the joints get the asset prefix.")}
    (out_dir / "construct_articulated_params.json").write_text(json.dumps(
        {**params, "moving": {"row": mr, "col": mc}, "name": name, "description": str(description)}, indent=1))
    return out


def rebuild(params_json, out_dir=None, **kw) -> dict:
    p = json.loads(Path(params_json).read_text())
    return construct_articulated(out_dir or Path(params_json).parent, p["width"], p["depth"], p["height"],
                                 p["rows"], p.get("moving"), p.get("travel_frac", 0.7),
                                 p.get("body_rgba"), p.get("front_rgba"), p.get("handle_rgba"),
                                 name=p.get("name", "egoscene_cabinet"),
                                 description=p.get("description", ""), **kw)


if __name__ == "__main__":
    import sys
    a = sys.argv[1:]
    print(json.dumps(construct_articulated(a[0], float(a[1]), float(a[2]), float(a[3]),
                                           json.loads(a[4]) if len(a) > 4 else [{"cols": 1}]), indent=1))

"""THREAD / ROPE / CABLE — a deformable linear object you can actually knot.

THE POINT
---------
A thread has no 6-DoF pose, so every tool in stage 2 aimed at rigid objects is
meaningless for it: there is nothing for `fit_object_poses` to fit and nothing
for a settle-drift check to call stable. What a thread has is a CENTRELINE and
a bending stiffness, and those are what this builder emits — a MuJoCo elastic
cable (`mujoco.elasticity.cable`) of N capsule segments that is inextensible
along its length, bends and twists with measured stiffness, and COLLIDES WITH
ITSELF, which is the one property without which a knot is not a knot.

The asset also carries two mocap "grips" (one per end, weldable on and off at
runtime) for scripted grabs — the same mechanism the replay machinery uses for
rigid objects. The DEMOS do not use them: a weld to a teleported mocap
prescribes where the end goes, which proves nothing about the cable. They drag
it with a capped spring-damper FORCE instead (drive.ThreadDriver), so the end
lags, the rest follows only through tension and bending, and letting go is
really letting go.

WHAT IS AND IS NOT MEASURABLE FROM VIDEO
    measurable  : length, radius, colour, where the ends are, the centreline
    NOT          : bending and twisting stiffness — those come from the
                   material, so they are numbers you set (and can fit later by
                   matching how the real thread hangs).
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

from .mjcf import (FLOOR_XML, LIGHTS_XML, free_camera, load, merge_params,
                     mjcf_header, rgba, rollout, save_params, save_stills,
                     vec, write_video)

# ---------------------------------------------------------------------------
# DEFAULTS — metres, kilograms, seconds.
# ---------------------------------------------------------------------------
DEFAULTS = {
    "name": "thread",

    # ---- size ----
    "length":    0.60,      # total centreline length
    "segments":  40,        # capsule count. Knots need ~40+; drag needs ~20
    "radius":    0.0035,    # thread radius (also its collision radius)

    # ---- appearance ----
    "rgba":      [0.88, 0.24, 0.20, 1.0],
    # END CAPS / FERRULES. A stanchion rope, a shoelace and a charger cable
    # all have them, and they are not decoration: they are where a hand
    # grabs, and the two ends are the only parts of a rope you can tell apart
    # in a video. `cap_length` is metres of cap at EACH end, snapped to whole
    # capsules (a cap shorter than one segment cannot be drawn on a composite
    # cable — raise `segments` if the snap loses too much). Per-end overrides
    # exist because real ropes are not symmetric: this bench's rope carries a
    # 62 mm pointed ferrule at one end and a 44 mm clasp at the other.
    "cap_length":     0.0,
    "cap_length_a":   None,     # None -> cap_length
    "cap_length_b":   None,
    "cap_rgba":       [0.83, 0.68, 0.24, 1.0],
    "cap_radius_scale": 1.0,    # a ferrule is usually a touch fatter
    # MATERIAL, not colour. A measured albedo is only half of what makes a
    # brass ferrule look like metal and velvet look like velvet — the other
    # half is the highlight, and MuJoCo's stock specular 0.5 / shininess 0.5
    # splits the difference and gets both wrong (the rope reads as glossy
    # plastic, the ferrule as cream-coloured resin).
    "specular":       0.12,     # velvet: almost none
    "shininess":      0.1,
    # 0.9 is a mirror: it renders a (0.54, 0.51, 0.29) brass at
    # (0.96, 0.91, 0.51), i.e. cream. 0.35 with a high shininess is a
    # tight glint on a still-brass surface.
    "cap_specular":   0.35,     # polished metal
    "cap_shininess":  0.85,
    # INVISIBLE by default: the mocap grips are an optional scripted-grab rig
    # that the force drivers do not use, and a green ball floating beside the
    # thread in every render reads as part of the object.
    "grip_rgba": [0.15, 0.75, 0.35, 0.0],

    # ---- placement: the centreline starts here and runs along +x ----
    "pos":       [-0.30, 0.0, 0.32],
    "yaw_deg":   0.0,
    # THE MEASURED SHAPE, if there is one. A rope has no 6-DoF pose — its
    # pose IS its centreline — so a clip's fitted "pose" is this polyline
    # (world metres, any number of points >= 2, resampled by arclength at
    # load). Carried in the params sheet so the asset knows how it was lying
    # in the video; `reset_measured()` lays it back down. None -> straight.
    "centreline": None,
    # THE MEASURED SHAPE AS THE BUILT SHAPE. A list of >= 3 world points: the
    # cable is BUILT along this polyline (MJCF `vertex`), one capsule per pair
    # of consecutive points, so its compiled pose (qpos0) already lies on the
    # measured centreline -- a scene that stages it has the rope where the
    # video has it in every keyframe, with no keyframe surgery -- and its
    # stress-free shape is that curve (a rope that has taken a set). `segments`,
    # `length`, `pos` and `yaw_deg` then follow from the points. None (the
    # default) builds it straight along +x from `pos`, as before.
    "rest_vertices": None,

    # ---- material (NOT recoverable from video — set, then fit to the hang) --
    # Bending modulus, and the number that decides whether you have a thread
    # or a steel rod. Measured by dropping the cable in a tight S and looking
    # at what it does (span of the 58 cm cable afterwards):
    #     1e4 -> 0.7 cm   limp string, collapses into a heap
    #     1e5 -> 1.6 cm   shoelace  <- default
    #     1e6 -> 18 cm    stiff cord, keeps big curves
    #     4e6 -> springs back out, effectively a rod
    # All of them self-collide (46-48 simultaneous non-neighbour contacts in
    # a heap), which is the property a knot actually needs.
    "bend_stiffness":  1e5,
    "twist_stiffness": 5e4,
    "joint_damping":   0.012,
    # DRY FRICTION IN THE ROPE'S OWN JOINTS — the single most important knob
    # for "does it behave like a real fabric rope". A stanchion rope holds a
    # shape you fold into it and does not scoot when nudged, because the fibres
    # rub. That internal rubbing is `joint_frictionloss` (N·m per ball dof).
    # BUT MuJoCo's friction constraint is SOFT by default (0.02 s reference),
    # which on an 18 g segment is a weak damper that never reaches its bound —
    # so frictionloss alone reads as "no effect" (measured: identical hang at
    # 0.05 vs 5 N·m). It only becomes real dry friction with a STIFF reference
    # (`joint_fric_solref` below). With both, a laid loop holds and a lifted
    # arch stays open; without them the rope springs out of every fold. See
    # `joint_armature` for the numerical companion that keeps a light,
    # dry-friction ball dof from ringing.
    "joint_frictionloss": 0.02,
    "joint_armature":     1e-5,
    "joint_fric_solref":  [0.001, 1.0],
    "joint_fric_solimp":  [0.999, 0.9999, 0.0001, 0.5, 2],
    "density":         1100.0,  # kg/m3 -> mass follows from radius+length
    "friction":        [0.7, 0.02, 0.001],
    # condim 4 gives sliding + torsion only: a smooth cord squeezed between
    # two fingertips then ROLLS out, because nothing resists rolling. A cord
    # meant to be pinched wants condim 6 and a real roll coefficient.
    "condim":          4,
    # A 15 mm segment of 3.5 mm thread weighs 0.6 g, and MuJoCo's contact
    # stiffness scales with that mass, so at the default softness a tensioned
    # thread sinks ~10 mm into the table it is lying on.
    "contact_solref":  [0.004, 1.0],

    # ---- rig ----
    "grips":          True,    # mocap grip at each end (weld on/off = grasp)
    # A grip weld as stiff as a thread segment is light (0.6 g) yanks that
    # segment at hundreds of m/s the moment the mocap moves, and MuJoCo
    # resets the state — the rope silently teleports back to its spawn pose
    # mid-video. 0.01 is a compliant hand: no instability at any timestep.
    "grip_solref":    [0.01, 1.0],
    "timestep":       0.002,
    "include_floor":  True,
}


def build(params: dict | None = None, out_dir: str | Path = ".") -> dict:
    p = merge_params(DEFAULTS, params)
    name = str(p["name"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    verts = None
    if p.get("rest_vertices") is not None:
        verts = np.asarray(p["rest_vertices"], float)
        if verts.ndim != 2 or verts.shape[1] != 3 or len(verts) < 3:
            raise ValueError("rest_vertices: a list of >= 3 [x, y, z] points (metres)")
        seglen = np.linalg.norm(np.diff(verts, axis=0), axis=1)
        if float(seglen.min()) < 1e-5:
            raise ValueError("rest_vertices: two consecutive points coincide (a zero-length capsule)")
        p = dict(p, segments=int(len(verts)), length=round(float(seglen.sum()), 6),
                 pos=[float(v) for v in verts[0]],
                 rest_vertices=[[round(float(v), 6) for v in q] for q in verts])
    n = int(p["segments"])
    first, last = f"{name}B_first", f"{name}B_last"

    grips = welds = ""
    if p["grips"]:
        grips = "".join(f"""
    <body name="{name}_grip_{e}" mocap="true" pos="{vec(p['pos'])}">
      <geom type="sphere" size="{p['radius']*2.2:.5f}"
            rgba="{rgba(p['grip_rgba'])}" contype="0" conaffinity="0"/>
    </body>""" for e in ("a", "b"))
        welds = "".join(f"""
    <weld name="{name}_hold_{e}" body1="{name}_grip_{e}" body2="{b}"
          relpose="0 0 0 1 0 0 0" solref="{vec(p['grip_solref'])}"
          active="false"/>""" for e, b in (("a", first), ("b", last)))

    floor = FLOOR_XML if p["include_floor"] else ""
    if verts is None:
        shape = (f'curve="s" count="{n} 1 1" size="{p["length"]}" '
                 f'offset="{vec(p["pos"])}"')
    else:
        shape = 'vertex="' + " ".join(f"{v:.6f}" for v in verts.ravel()) + '"'
    # MATERIALS, not geom attributes: `specular`/`shininess` live on a
    # <material> in MuJoCo's schema, and a geom carrying them is a hard XML
    # error. The geom keeps its own rgba (which wins over the material's), so
    # the material here contributes only the highlight.
    xml = mjcf_header(f"{name}_cable", timestep=p["timestep"]) + f"""
  <extension><plugin plugin="mujoco.elasticity.cable"/></extension>

  <asset>
    <material name="{name}_mat" rgba="{rgba(p['rgba'])}"
              specular="{float(p['specular']):.3f}"
              shininess="{float(p['shininess']):.3f}"/>
    <material name="{name}_capmat" rgba="{rgba(p['cap_rgba'])}"
              specular="{float(p['cap_specular']):.3f}"
              shininess="{float(p['cap_shininess']):.3f}"
              reflectance="0.15"/>
  </asset>

  <worldbody>{LIGHTS_XML}{floor}{grips}
    <!-- MJCF lays the cable straight along +x from `offset`; <composite> has
         no rotation attribute, so yaw (and any measured centreline) is
         applied at runtime by set_centreline() below. -->
    <composite prefix="{name}" type="cable" {shape}
               initial="free">
      <plugin plugin="mujoco.elasticity.cable">
        <config key="twist" value="{p['twist_stiffness']}"/>
        <config key="bend"  value="{p['bend_stiffness']}"/>
        <config key="vmax"  value="0"/>
      </plugin>
      <joint kind="main" damping="{p['joint_damping']}"
             frictionloss="{p['joint_frictionloss']}"
             armature="{p['joint_armature']}"
             solreffriction="{vec(p['joint_fric_solref'])}"
             solimpfriction="{' '.join(str(v) for v in p['joint_fric_solimp'])}"/>
      <geom type="capsule" size="{p['radius']}" rgba="{rgba(p['rgba'])}"
            material="{name}_mat"
            condim="{int(p['condim'])}" density="{p['density']}"
            friction="{vec(p['friction'])}"
            solref="{vec(p['contact_solref'])}"/>
    </composite>
  </worldbody>
{f'  <equality>{welds}{chr(10)}  </equality>' if welds else ''}
</mujoco>
"""
    xml_path = out_dir / f"{name}.xml"
    xml_path.write_text(xml)
    caps = _paint_caps(xml_path, name, n, p)
    if p["centreline"] is not None:
        p = dict(p, centreline=[[float(v) for v in q]
                                for q in np.asarray(p["centreline"], float)])
    save_params(out_dir / f"{name}_params.json", p)

    return {
        "xml": str(xml_path),
        "params_json": str(out_dir / f"{name}_params.json"),
        "caps": caps,
        # count="N" vertices -> N-1 capsule bodies: B_first, B_1..B_{N-3},
        # B_last. Assuming B_1..B_{N-2} is an off-by-one that only shows up
        # as a KeyError on the very last body.
        "bodies": {"first": first, "last": last,
                   "segments": [f"{name}B_{i}" for i in range(1, n - 2)]},
        "grips": ({"a": f"{name}_grip_a", "b": f"{name}_grip_b"}
                  if p["grips"] else {}),
        "welds": ({"a": f"{name}_hold_a", "b": f"{name}_hold_b"}
                  if p["grips"] else {}),
        "segment_length_m": round(p["length"] / max(n - 1, 1), 5),
        "laid": verts is not None,
        "note": ("grab = teleport grip_<e> onto the end body's CURRENT pose "
                 "(mocap_pos/mocap_quat), then eq_active[hold_<e>] = 1; "
                 "release = 0. The cable is inextensible: pulling both grips "
                 "further apart than `length` just tensions it."),
    }


def _paint_caps(xml_path: Path, name: str, n: int, p: dict) -> dict:
    """Recolour (and optionally fatten) the capsules at each end IN PLACE.

    `<composite>` emits one `<geom>` template for the whole cable, so there is
    no MJCF spelling for "the last three capsules are brass". The spec API is:
    MjSpec expands the composite into real named bodies/geoms (`<name>G0` ..
    `<name>G{n-2}`), so the end geoms can be edited and the whole thing written
    back as a self-contained XML — plugin, welds and all. Doing it here rather
    than after `spec.compile()` in every consumer is deliberate: a scene that
    forgets a post-compile paint step renders an all-red rope and nothing
    complains.
    """
    seg_len = float(p["length"]) / max(n - 1, 1)
    la = p["cap_length_a"] if p["cap_length_a"] is not None else p["cap_length"]
    lb = p["cap_length_b"] if p["cap_length_b"] is not None else p["cap_length"]
    ka = int(round(float(la) / seg_len))
    kb = int(round(float(lb) / seg_len))
    if float(la) > 0:
        ka = max(ka, 1)
    if float(lb) > 0:
        kb = max(kb, 1)
    scale = float(p["cap_radius_scale"])
    ngeom = n - 1                                   # G0 .. G{n-2}
    ka, kb = min(ka, ngeom // 2), min(kb, ngeom // 2)
    if ka <= 0 and kb <= 0 and abs(scale - 1.0) < 1e-9:
        return {"a_segments": 0, "b_segments": 0, "geoms": []}
    import mujoco
    a_names = [f"{name}G{i}" for i in range(ka)]
    b_names = [f"{name}G{ngeom - 1 - i}" for i in range(kb)]
    spec = mujoco.MjSpec.from_file(str(xml_path))
    painted = []
    for g in spec.geoms:
        if (g.name or "") in a_names or (g.name or "") in b_names:
            g.rgba = [float(x) for x in p["cap_rgba"]]
            g.material = f"{name}_capmat"
            if abs(scale - 1.0) > 1e-9:
                g.size = [float(p["radius"]) * scale, 0.0, 0.0]
            painted.append(g.name)
    xml_path.write_text(spec.to_xml())
    return {"a_segments": ka, "b_segments": kb,
            "a_length_m": round(ka * seg_len, 4),
            "b_length_m": round(kb * seg_len, 4),
            "geoms": painted}


def rebuild(params_json: str | Path, out_dir: str | Path | None = None) -> dict:
    from .mjcf import load_params
    p = load_params(params_json)
    return build(p, out_dir or Path(params_json).parent)


# ---------------------------------------------------------------------------
def grab(model, data, built: dict, end: str) -> None:
    """Snap a grip onto its end AT ITS CURRENT POSE, then engage the weld.

    Order matters: activate first and the weld yanks the thread back to
    wherever the mocap was left, which looks like the rope teleporting.
    """
    g = model.body(built["grips"][end]).id
    b = model.body(built["bodies"]["first" if end == "a" else "last"]).id
    data.mocap_pos[model.body_mocapid[g]] = data.xpos[b]
    data.mocap_quat[model.body_mocapid[g]] = data.xquat[b]
    data.eq_active[model.equality(built["welds"][end]).id] = 1


def release(model, data, built: dict, end: str) -> None:
    data.eq_active[model.equality(built["welds"][end]).id] = 0


def move_grip(model, data, built: dict, end: str, pos) -> None:
    g = model.body(built["grips"][end]).id
    data.mocap_pos[model.body_mocapid[g]] = np.asarray(pos, float)


def _quat_x_to(dirv: np.ndarray) -> np.ndarray:
    """Quaternion (w,x,y,z) rotating +x onto `dirv`, with a stable roll."""
    v = np.asarray(dirv, float)
    v = v / (np.linalg.norm(v) + 1e-12)
    x = np.array([1.0, 0.0, 0.0])
    c = float(np.dot(x, v))
    if c > 1 - 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if c < -1 + 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    ax = np.cross(x, v)
    s = np.linalg.norm(ax)
    ang = np.arctan2(s, c)
    ax /= s
    return np.concatenate([[np.cos(ang / 2)], np.sin(ang / 2) * ax])


def set_centreline(model, data, built: dict, points) -> np.ndarray:
    """Lay the cable along an arbitrary polyline — THE hook for real clips.

    `points` is any (k>=2, 3) centreline (a mask skeleton lifted to 3D, a
    hand-drawn path, a straight line); it is resampled by arclength to the
    cable's own body count. The first body's free joint takes the position and
    tangent; every later ball joint takes the rotation from one tangent to the
    next, which is what makes the cable follow the curve instead of snapping
    back to the straight configuration MJCF built it in.
    """
    import mujoco
    pts = np.asarray(points, float)
    names = ([built["bodies"]["first"]] + built["bodies"]["segments"]
             + [built["bodies"]["last"]])
    nb = len(names)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    want = np.linspace(0.0, s[-1], nb + 1)
    path = np.stack([np.interp(want, s, pts[:, i]) for i in range(3)], axis=1)
    tang = np.diff(path, axis=0)
    quats = np.stack([_quat_x_to(t) for t in tang])

    jid = model.body(names[0]).jntadr[0]
    adr = model.jnt_qposadr[jid]
    data.qpos[adr:adr + 3] = path[0]
    data.qpos[adr + 3:adr + 7] = quats[0]
    for i in range(1, nb):
        jid = model.body(names[i]).jntadr[0]
        adr = model.jnt_qposadr[jid]
        prev_conj = quats[i - 1] * np.array([1.0, -1.0, -1.0, -1.0])
        rel = np.zeros(4)
        mujoco.mju_mulQuat(rel, prev_conj, quats[i])
        data.qpos[adr:adr + 4] = rel
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    return path


def reset_straight(model, data, built: dict, params: dict) -> np.ndarray:
    """Lay it straight from `pos` at `yaw_deg` — the params sheet's own pose."""
    a = np.radians(float(params["yaw_deg"]))
    p0 = np.asarray(params["pos"], float)
    p1 = p0 + float(params["length"]) * np.array([np.cos(a), np.sin(a), 0.0])
    return set_centreline(model, data, built, np.stack([p0, p1]))


def reset_measured(model, data, built: dict, params: dict) -> np.ndarray:
    """Lay it on the MEASURED centreline from the params sheet.

    This is a rope's equivalent of `fit_object_poses` writing a 6-DoF pose:
    the polyline IS the pose, so the reconstruction is only in the scene once
    this has been called. Falls back to the straight pose when the sheet
    carries no measurement, so a hand-built asset still lands somewhere sane.
    """
    pts = params.get("centreline")
    if pts is None:
        return reset_straight(model, data, built, params)
    return set_centreline(model, data, built, np.asarray(pts, float))


def centreline(model, data, built: dict) -> np.ndarray:
    ids = [model.body(built["bodies"]["first"]).id] \
        + [model.body(b).id for b in built["bodies"]["segments"]] \
        + [model.body(built["bodies"]["last"]).id]
    return data.xpos[ids].copy()


def preview(built: dict, out_dir: str | Path) -> list[str]:
    """Two stills: as built (straight, or along its measured rest vertices), and settled in a loose S."""
    import mujoco
    from .mjcf import key_states, load_params
    m = load(built["xml"])
    dat = mujoco.MjData(m)
    p = load_params(built["params_json"])
    L = float(p["length"])
    laid = p.get("rest_vertices") is not None

    def straight(mm, dd):
        if laid:                                  # qpos0 already lies on the measured vertices
            mujoco.mj_resetData(mm, dd)
            mujoco.mj_forward(mm, dd)
        else:
            reset_straight(mm, dd, built, p)

    def s_curve(mm, dd):
        u = np.linspace(0, 1, 10)
        reset_straight(mm, dd, built, p)          # anchors the pose/height
        base = np.asarray(p["pos"], float)
        set_centreline(mm, dd, built, np.stack(
            [base[0] + L * 0.32 * np.sin(2 * np.pi * u),
             base[1] + L * 0.55 * u,
             np.full_like(u, p["radius"] * 1.2)], axis=1))

    look = (np.asarray(p["rest_vertices"], float).mean(axis=0) if laid
            else [np.asarray(p["pos"], float)[0] + L * 0.35, 0, 0.05])
    return key_states(
        m, dat, [("as_built" if laid else "straight", straight), ("settled_S", s_curve)],
        out_dir, f"{Path(built['xml']).stem}_preview",
        camera=free_camera(look, L * 1.8 if not laid else max(0.3, float(np.ptp(np.asarray(p["rest_vertices"])[:, :2], axis=0).max()) * 2.0),
                           110, -30),
        settle_steps=400)


def demo(built: dict, out_mp4: str | Path, fps: int = 30,
         width: int = 640, height: int = 480) -> dict:
    """Lift, swing, coil, pull taut and cross the ends — every one of those
    from a CAPPED spring-damper force on an end segment, never a mocap weld
    dragging it along a prescribed path."""
    import mujoco
    from .mjcf import clear_forces, load_params
    from .drivers import ThreadDriver
    m = load(built["xml"])
    dat = mujoco.MjData(m)
    p = load_params(built["params_json"])
    reset_straight(m, dat, built, p)
    for _ in range(int(0.6 / m.opt.timestep)):    # let it land first
        mujoco.mj_step(m, dat)
    L = float(p["length"])
    drv = ThreadDriver(m, dat, built, height=float(p["radius"]), span=L)
    cam = free_camera([0.0, 0.0, 0.06], L * 2.2, 110, -22)

    def control(t, model, data):
        clear_forces(data)
        return drv.step(t)

    frames = rollout(m, dat, seconds=drv.T["loop"] + 1.0, fps=fps,
                     control=control, camera=cam, width=width, height=height,
                     spin_deg_per_s=3.0)
    out = write_video(frames, out_mp4, fps=fps)
    out["stills"] = save_stills(frames, Path(out_mp4).parent,
                                f"{Path(built['xml']).stem}_still", n=5)
    out["measured"] = {k: (round(float(v), 3) if isinstance(v, float) else v)
                       for k, v in drv.meas.items()}
    out["verdict"] = drv.verdict()
    return out


# ---------------------------------------------------------------------------

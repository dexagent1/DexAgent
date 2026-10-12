"""HELICAL COMPRESSION SPRING — the reusable elastic-contact skill.

THE POINT
---------
A battery holder, a retractable pen, a toaster lever and a pop-up lid all
contain the same mechanism: a coil of steel that pushes back proportionally
to how far you squeeze it, up to a hard stop where the coils touch. None of
that is in the pixels — a video shows a few silver arcs — so like the screw
thread and the crease, the spring is CONSTRUCTED from numbers:

    stiffness    N/m    how hard it pushes back per metre of squeeze
    free_length  m      how long it stands relaxed
    coil_radius  m      how fat the coil is
    wire_radius  m      how fat the wire is
    n_coils      -      how many active turns

Everything else (solid height, per-coil travel, helix pitch, the Euler-
stability armature) is DERIVED from those. Meeting the next spring in a
different object means calling `build()` with different numbers — never
editing code. That is the whole contract of this package.

HOW IT IS MODELLED
------------------
A serial chain of one body PER COIL, each on a slide joint along the spring
axis (+z from the mount face). n joints in series, each with joint stiffness
n*k, compose to exactly k overall — and because each joint's range is
[-(pitch - wire_d), 0], the spring goes COIL-BOUND at its real solid height:
a hard stop the solver enforces, not a script. Each body carries one visual
helix turn, so squeezing the spring visibly closes the gaps between turns,
exactly like the real part. The battery (or finger) presses on a metal TIP
PLATE carried by the last body — the only colliding geom, so contacts are a
clean disc-on-face problem instead of mesh-on-mesh.

The passive spring force is real physics, not a controller: push the plate
with any force and the solver decides the deflection; let go and the stored
energy comes back out (an inserted-then-released battery is genuinely
EJECTED). `probe()` measures the as-built stiffness with a quasi-static
force ramp and reports it against the request — the number is honest.

STABILITY (why armature is auto)
--------------------------------
The bench world integrates with plain Euler at 2 ms. A 0.1 g coil segment on
a 1250 N/m joint rings at 3.5 kHz and detonates instantly. The cure is joint
armature sized FROM the joint stiffness (omega*dt <= ~0.28), with damping as
a critical-ratio `zeta` — both derived per build, never constants.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .mjcf import (FLOOR_XML, LIGHTS_XML, clear_forces, free_camera, load,
                     merge_params, mjcf_header, rgba, rollout, save_params,
                     save_stills, write_mesh, write_video)

# ---------------------------------------------------------------------------
# DEFAULTS — metres, kilograms, seconds. The first five keys ARE the spring;
# the rest is derived behaviour and appearance.
# ---------------------------------------------------------------------------
DEFAULTS = {
    "name": "spring",

    # ---- THE SPRING (what you measure or choose) ----
    "stiffness":   250.0,     # N/m, of the WHOLE spring (a AA-holder coil
                              #   takes ~2-3 N at full squeeze: 250-450)
    "free_length": 0.0115,    # relaxed length of the coil stack
    "coil_radius": 0.0042,    # helix centreline radius
    "wire_radius": 0.00045,   # the steel itself
    "n_coils":     5,         # active turns (one chain body each)

    # ---- tip & seat ----
    "plate_radius":    None,  # None -> coil_radius + wire_radius: the disc
                              #   the pressing object actually touches
    "plate_thickness": 0.0008,
    "seat_turns":      0.9,   # a flat "ground" turn drawn at the base so the
                              #   spring does not appear to float off its wall

    # ---- appearance: bright zinc-plated steel ----
    "rgba":       [0.80, 0.82, 0.84, 1.0],
    "specular":   0.7,
    "shininess":  0.9,

    # ---- dynamics ----
    "moving_mass": 0.0008,    # total translating mass (a real spring ~0.5 g)
    "zeta":        0.25,      # damping ratio per joint: 0.1 boings, 1.0 dead
    "armature":    None,      # None -> derived for Euler @ 2 ms (see module
                              #   doc); only override for a softer/stiffer
                              #   integrator budget
    "friction_tip": 0.85,     # plate contact friction (slide). High on
                              #   purpose: whatever presses a spring usually
                              #   does it at a tilt, and the tip must hold
                              #   the press inside the friction cone

    # ---- standalone-build extras (ignored by emit()) ----
    "include_floor": True,
    "timestep": 0.002,        # match the bench world: plain Euler @ 2 ms is
                              #   what the asset must survive
}


# ------------------------------ derived numbers ----------------------------
def derived(p: dict) -> dict:
    """Every number the five spring keys imply — one place, used by emit,
    probe and any parent builder sizing a pocket around the spring."""
    wire_d = 2.0 * float(p["wire_radius"])
    n = int(p["n_coils"])
    if n < 2:
        raise ValueError("n_coils must be >= 2 (one body per coil)")
    pitch = (float(p["free_length"]) - wire_d) / n
    if pitch <= wire_d:
        raise ValueError(
            f"free_length {p['free_length']} m cannot hold {n} coils of "
            f"{wire_d*1e3:.2f} mm wire — the spring would be born solid "
            f"(pitch {pitch*1e3:.2f} <= wire {wire_d*1e3:.2f} mm)")
    travel_per = pitch - wire_d
    k_joint = float(p["stiffness"]) * n
    # Euler @ timestep: omega = sqrt(k/arm) must stay ~<0.28/dt
    dt = float(p.get("timestep", 0.002))
    arm = (float(p["armature"]) if p.get("armature") is not None
           else max(0.01, k_joint * (dt / 0.28) ** 2))
    damping = 2.0 * float(p["zeta"]) * float(np.sqrt(k_joint * arm))
    plate_r = (float(p["plate_radius"]) if p.get("plate_radius") is not None
               else float(p["coil_radius"]) + float(p["wire_radius"]))
    return {
        "wire_d": wire_d, "pitch": pitch, "n": n,
        "travel_per_coil_m": travel_per,
        "travel_m": n * travel_per,
        "solid_length_m": float(p["free_length"]) - n * travel_per,
        "k_joint": k_joint, "armature": arm, "damping": damping,
        "plate_radius": plate_r,
        "tip_free_z": float(p["free_length"]) + float(p["plate_thickness"]),
    }


# ------------------------------ helix mesh ---------------------------------
def _helix(coil_radius, pitch, turns, wire_radius, steps_per_turn=48,
           ring=10):
    """One visual coil segment (common.helix_tube, kept here so a future
    conical spring only changes this function)."""
    from .mjcf import helix_tube
    return helix_tube(coil_radius, pitch, turns, wire_radius,
                      steps_per_turn=steps_per_turn, ring=ring)


# ------------------------------ the skill ----------------------------------
def emit(params: dict, mesh_dir: str | Path, name: str | None = None) -> dict:
    """THE EMBEDDING API — what another builder calls to put a spring in its
    own MJCF (a battery case calls it once per slot).

    Writes the helix meshes into `mesh_dir` and returns MJCF fragments:
      assets_xml   <mesh>/<material> entries for the parent's <asset>
      chain_xml    the serial coil bodies + tip plate, to be nested inside a
                   parent body; the spring stands along +z from that body's
                   local origin (z=0 = mount face). Rotate the PARENT frame
                   to point the spring anywhere.
      joints       the n slide joints, base->tip (for reading compression)
      tip_body / plate_geom, and every `derived()` number.

    All element names are prefixed with `name`, so two springs in one model
    never collide.
    """
    p = merge_params(DEFAULTS, params)
    name = name or str(p["name"])
    d = derived(p)
    mesh_dir = Path(mesh_dir)
    mesh_dir.mkdir(parents=True, exist_ok=True)

    turn = _helix(p["coil_radius"], d["pitch"], 1.0, p["wire_radius"])
    turn_stl = write_mesh(turn, mesh_dir / f"{name}_turn.stl")
    seat = _helix(p["coil_radius"], d["wire_d"] * 0.55, float(p["seat_turns"]),
                  p["wire_radius"])
    seat_stl = write_mesh(seat, mesh_dir / f"{name}_seat.stl")

    assets_xml = f"""
    <mesh name="{name}_turn" file="{turn_stl}"/>
    <mesh name="{name}_seat" file="{seat_stl}"/>
    <material name="{name}_steel" rgba="{rgba(p['rgba'])}"
              specular="{float(p['specular']):.3f}"
              shininess="{float(p['shininess']):.3f}" reflectance="0.3"/>"""

    m_per = float(p["moving_mass"]) / d["n"]
    joints = [f"{name}_c{i}" for i in range(1, d["n"] + 1)]
    # innermost first: tip plate rides the last coil. The plate COLLIDES as
    # a BOX under a visual cylinder: a disc pressed by another disc (a
    # battery's flat end) is a degenerate libccd pair whose contact normal
    # wanders, and a cell pressed on a cylinder plate at a tilt FLIPPED
    # straight through it (measured); a box face has one normal.
    inner = f"""
      <body name="{name}_tip" pos="0 0 {d['pitch']:.6f}">
        <inertial pos="0 0 0" mass="{m_per:.6f}"
                  diaginertia="1e-08 1e-08 1e-08"/>
        <geom name="{name}_plate" type="box"
              size="{d['plate_radius']*0.72:.6f} {d['plate_radius']*0.72:.6f} {float(p['plate_thickness'])/2:.6f}"
              pos="0 0 {float(p['wire_radius']) + float(p['plate_thickness'])/2:.6f}"
              rgba="{rgba(p['rgba'])}" group="3"
              friction="{float(p['friction_tip']):.3f} 0.005 0.0001"/>
        <geom type="cylinder"
              size="{d['plate_radius']:.6f} {float(p['plate_thickness'])/2:.6f}"
              pos="0 0 {float(p['wire_radius']) + float(p['plate_thickness'])/2:.6f}"
              material="{name}_steel" rgba="{rgba(p['rgba'])}"
              contype="0" conaffinity="0" mass="0"/>
      </body>"""
    for i in range(d["n"], 0, -1):
        z = float(p["wire_radius"]) if i == 1 else d["pitch"]
        inner = f"""
      <body name="{name}_coil{i}" pos="0 0 {z:.6f}">
        <inertial pos="0 0 {d['pitch']/2:.6f}" mass="{m_per:.6f}"
                  diaginertia="1e-08 1e-08 1e-08"/>
        <joint name="{name}_c{i}" type="slide" axis="0 0 1"
               range="{-d['travel_per_coil_m']:.6f} 0" stiffness="{d['k_joint']:.3f}"
               springref="0" damping="{d['damping']:.4f}"
               armature="{d['armature']:.5f}" limited="true"
               solreflimit="{2 * float(p.get('timestep', 0.002)):.4f} 1" solimplimit="0.99 0.999 0.0001"/>
        <geom type="mesh" mesh="{name}_turn" material="{name}_steel"
              rgba="{rgba(p['rgba'])}" contype="0" conaffinity="0"/>{inner}
      </body>"""
    chain_xml = f"""
      <geom type="mesh" mesh="{name}_seat" material="{name}_steel"
            rgba="{rgba(p['rgba'])}" contype="0" conaffinity="0"/>{inner}"""

    return {
        "assets_xml": assets_xml, "chain_xml": chain_xml,
        "joints": joints, "tip_body": f"{name}_tip",
        "plate_geom": f"{name}_plate",
        "meshes": [turn_stl, seat_stl], **d,
    }


def compression(model, data, joints: list[str]) -> float:
    """Metres of squeeze right now (0 = relaxed) — the live readout every
    demo label uses."""
    return -sum(float(data.qpos[model.joint(j).qposadr[0]]) for j in joints)


# ------------------------------ standalone build ---------------------------
def build(params: dict | None = None, out_dir: str | Path = ".") -> dict:
    """The spring alone, mounted upright on a base block — the form the
    skill is TESTED in (probe + demo run on this)."""
    p = merge_params(DEFAULTS, params)
    name = str(p["name"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frag = emit(p, out_dir, name)
    floor = FLOOR_XML if p["include_floor"] else ""
    xml = mjcf_header(f"{name}_asset", timestep=p["timestep"]) + f"""
  <asset>{frag['assets_xml']}
  </asset>
  <worldbody>{LIGHTS_XML}{floor}
    <body name="{name}_mount" pos="0 0 0">{frag['chain_xml']}
    </body>
  </worldbody>
</mujoco>
"""
    xml_path = out_dir / f"{name}.xml"
    xml_path.write_text(xml)
    save_params(out_dir / f"{name}_params.json", p)
    return {"xml": str(xml_path),
            "params_json": str(out_dir / f"{name}_params.json"),
            "mount_body": f"{name}_mount", **{k: v for k, v in frag.items()
                                              if k != "chain_xml"},
            "note": (f"passive spring: push {frag['plate_geom']} with any force and read "
                     "compression(); joints hard-stop at solid height "
                     f"({frag['solid_length_m']*1e3:.1f} mm)")}


def rebuild(params_json: str | Path, out_dir: str | Path | None = None):
    from .mjcf import load_params
    return build(load_params(params_json), out_dir or Path(params_json).parent)


# ------------------------------ QC: measured stiffness ---------------------
def probe(built: dict, forces=(0.5, 1.0, 1.5, 2.0)) -> dict:
    """Quasi-static force ramp on the tip plate -> measured N/m.

    The one number the whole skill promises is `stiffness`; this measures it
    in the compiled model (joint limits, armature and damping all in the
    loop) instead of trusting the algebra. Also presses to solid to verify
    the coil-bound stop."""
    import mujoco
    m = load(built["xml"])
    d = mujoco.MjData(m)
    tip = m.body(built["tip_body"]).id
    js = built["joints"]
    pts = []
    for F in forces:
        mujoco.mj_resetData(m, d)
        for _ in range(3000):                       # 6 s: fully settled
            clear_forces(d)
            d.xfrc_applied[tip, 2] = -float(F)
            mujoco.mj_step(m, d)
        pts.append((float(F), compression(m, d, js)))
    # solid-stop check: 10x the last force must NOT compress past travel
    mujoco.mj_resetData(m, d)
    for _ in range(3000):
        clear_forces(d)
        d.xfrc_applied[tip, 2] = -10.0 * float(forces[-1])
        mujoco.mj_step(m, d)
    x_solid = compression(m, d, js)
    # fit ONLY the points still on the spring (past ~85% travel the hard
    # stop carries the load and the slope means the STOP, not the spring)
    lin = [(f, x) for f, x in pts if x < 0.85 * built["travel_m"]]
    F = np.array([q[0] for q in lin])
    X = np.array([q[1] for q in lin])
    k_meas = float(np.polyfit(X, F, 1)[0]) if len(lin) > 1 else float("nan")
    p = json.loads(Path(built["params_json"]).read_text())
    out = {
        "requested_N_per_m": float(p["stiffness"]),
        "measured_N_per_m": round(k_meas, 1),
        "error_pct": round(100 * (k_meas / float(p["stiffness"]) - 1), 1),
        "force_deflection": [(f, round(x * 1e3, 3)) for f, x in pts],
        "travel_m": built["travel_m"],
        "solid_stop_m": round(x_solid, 6),
        # 0.3 mm of constraint softness at 10x load is a hard stop; 3 mm is
        # a limit that is not doing its job
        "solid_stop_ok": bool(x_solid <= built["travel_m"] + 3e-4),
    }
    return out


# ------------------------------ demo ---------------------------------------
def demo(built: dict, out_mp4: str | Path, fps: int = 30) -> dict:
    """Press-hold-release with a capped force (never a position servo — the
    spring must be free to refuse), then a light tap: squeeze, coil-bind,
    spring-back, ring-down. The video IS the elasticity check."""
    import mujoco
    m = load(built["xml"])
    d = mujoco.MjData(m)
    tip = m.body(built["tip_body"]).id
    js, travel = built["joints"], built["travel_m"]
    p = json.loads(Path(built["params_json"]).read_text())
    F_solid = 1.6 * float(p["stiffness"]) * travel   # enough to coil-bind

    def control(t, model, data):
        clear_forces(data)
        if t < 1.0:
            phase, F = "relaxed", 0.0
        elif t < 2.5:
            phase, F = "pressed (capped force)", F_solid * min(1, (t - 1.0))
        elif t < 3.5:
            phase, F = "held at solid height", F_solid
        elif t < 5.0:
            phase, F = "RELEASED - stored energy returns", 0.0
        elif t < 5.2:
            phase, F = "light tap", 0.3 * F_solid
        else:
            phase, F = "ring-down", 0.0
        if F:
            data.xfrc_applied[tip, 2] = -F
        c = compression(model, data, js)
        return (f"{phase}   squeeze {c*1e3:4.1f}/{travel*1e3:.1f} mm   "
                f"F {F:3.1f} N")

    cam = free_camera([0, 0, built["tip_free_z"] * 0.5],
                      built["tip_free_z"] * 9.0, 90, -12)
    frames = rollout(m, d, seconds=6.5, fps=fps, control=control, camera=cam,
                     width=640, height=480)
    out = write_video(frames, out_mp4, fps=fps)
    out["stills"] = save_stills(frames, Path(out_mp4).parent,
                                Path(out_mp4).stem, n=4)
    return out


# ---------------------------------------------------------------------------

"""MICROPIPETTE FAMILY — an Eppendorf-style single-channel pipette whose
mechanisms really work, the 1000 ul tip it picks up, and the hinged-lid
tip box the tips stand in.

WHAT IS MECHANISM HERE (none of it is in a scan):

  PLUNGER, TWO STAGES   a slide against a soft main spring to the FIRST
                        STOP (~2.5 N at ~12 mm — the aspirate stroke), then
                        a separate preloaded stiff spring (the blow-out
                        pin) the stem only meets at that depth: the force
                        step a thumb feels is a second body's preload, not
                        a lookup table.
  TIP SNAP-FIT          real tips stretch over the nose cone. The give is
                        explicit, latchbox-style: two spring-loaded JAWS on
                        the nose (radial slides) that the tip collar's
                        inner lip forces in and then captures — attach and
                        eject are both measured forces, and a mounted tip
                        carries the pipette's weight and a shake.
  EJECTOR               a sleeve on its own sprung slide whose bottom edge
                        presses the mounted collar down past the jaws:
                        pressing the ejector button IS how the tip leaves.
  TIP BOX LID           a friction hinge at the back of the base; the lid
                        stays where a push leaves it, exactly like the
                        photo's open TipOne box.

No actuators anywhere: every demo motion is an applied wrench.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .mjcf import (FLOOR_XML, LIGHTS_XML, ForceGrip, apply_wrench,
                     clear_forces, free_camera, load, merge_params,
                     mjcf_header, revolve_profile, rgba, save_params,
                     save_stills, shell_ring, vec, write_mesh, write_video)

_SNAP_ATTR = ('friction="0.5 0.01 0.0002" condim="3" '
              'solref="0.002 1" solimp="0.95 0.99 0.0005"')


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _qz(deg):
    h = np.radians(deg) / 2
    return np.array([np.cos(h), 0.0, 0.0, np.sin(h)])


def _qy(deg):
    h = np.radians(deg) / 2
    return np.array([np.cos(h), 0.0, np.sin(h), 0.0])


# ---------------------------------------------------------------------------
# THE TIP — 1000 ul, shared by the pipette (what it grabs) and the box
# (what it holds). Local frame: +z up, z=0 at the FINE END, collar at top.
# ---------------------------------------------------------------------------
TIP_DEFAULTS = {
    "name": "ptip",
    "length":      0.0760,
    "collar_or":   0.0045,    # collar outer radius (rests on the tray hole)
    "collar_ir":   0.0041,    # entrance bore
    "collar_len":  0.0090,
    "lip_in":      0.0008,    # the inner lip the nose jaws snap past —
                              #   generous, because contact force scales
                              #   with penetration and a 0.24 mm nominal
                              #   interference transmitted only ~0.3 N
                              #   through the solver (measured); at ~0.5 mm
                              #   the snap carries newtons
    "lip_depth":   0.0020,    # below the collar's top edge
    "tip_rgba":    [0.62, 0.72, 0.93, 0.55],
    "density":     1010.0,
    "friction":    "0.6 0.01 0.0002",
    "timestep":    0.001,
    "include_floor": True,
    "pos": [0.0, 0.0, 0.0],
}


def _tip_body_xml(p: dict, prefix: str, pos, mesh_name: str,
                  free: bool = True) -> str:
    L = float(p["length"])
    cor, cir, cl = (float(p["collar_or"]), float(p["collar_ir"]),
                    float(p["collar_len"]))
    lip, lipd = float(p["lip_in"]), float(p["lip_depth"])
    if cir >= cor:
        raise ValueError(f"collar_ir {cir*1e3:.2f} mm >= collar_or {cor*1e3:.2f} mm — the collar needs a wall; "
                         f"shrink collar_ir or enlarge collar_or")
    if lip >= cir:
        raise ValueError(f"lip_in {lip*1e3:.2f} mm closes the {cir*1e3:.2f} mm bore — shrink lip_in")
    if cl >= L:
        raise ValueError(f"collar_len {cl*1e3:.1f} mm is not shorter than the tip length {L*1e3:.1f} mm")
    tc = rgba(p["tip_rgba"])
    # solref 4 ms EVERYWHERE on the tip: MuJoCo's default 20 ms timeconst
    # rests ANY body ~g*tau^2 = 4 mm deep regardless of mass — deeper than
    # every feature on a gram-scale part, so feet sank through their own
    # spheres and contact normals catapulted the tip out of whatever seat
    # it was given (measured, in four different socket designs)
    gA = (f'density="{p["density"]}" friction="{p["friction"]}" condim="3" '
          f'solref="0.004 1"')
    inv = 'rgba="0 0 0 0"'
    g = [f'      <geom name="{prefix}_vis" type="mesh" mesh="{mesh_name}" '
         f'rgba="{tc}" contype="0" conaffinity="0" group="2" density="1"/>']
    # collar: a shell the nose really enters
    g += shell_ring(f"{prefix}_collar", cir, cor - cir, L - cl, L,
                    "0 0 0 0", gA, n=8)
    # ROUNDED entrance rim: the descending jaws tolerate only ~0.3 mm of
    # misalignment and the guide chain allows ~0.6, so on a square-edged
    # entrance they landed flat, stored the whole press force and blew
    # (measured). On this ring they cam inward and compress their own
    # springs instead.
    for i in range(8):
        th = np.radians(22.5 + 45.0 * i)
        g.append(f'      <geom name="{prefix}_rim{i}" type="capsule" '
                 f'size="0.0005 {0.42 * cir:.6f}" '
                 f'pos="{vec(((cir + 0.0002) * np.cos(th), (cir + 0.0002) * np.sin(th), L - 0.0003))}" '
                 f'zaxis="{vec((-np.sin(th), np.cos(th), 0.0))}" '
                 f'rgba="0 0 0 0" friction="0.3 0.008 0.0002" condim="3" '
                 f'solref="0.006 1" density="{p["density"]}"/>')
    # the inner lip: SIXTEEN tangential capsules — near-torus. The snap
    # interference must survive EVERY yaw of the tip (nothing resists a
    # snap-fit spinning): an 8-chord ring's junctions bulge to r=3.62 mm,
    # zero interference against 3.6 mm jaws, and the mounted tip slid off
    # under its own weight whenever a junction drifted over a jaw
    # (measured). At 16 chords the worst-case interference is 0.22 mm.
    # zaxis, NOT euler: MJCF euler is an INTRINSIC sequence and
    # euler="90 0 th" spins the capsule about its own axis before tipping
    # it — every lip came out parallel to y and gripped the descending
    # nose like a chuck. Contact is STIFF and ABSOLUTE (negative solref):
    # mass-scaled stiffness on this near-massless pair is softer than the
    # jaw springs it must transmit.
    for i in range(16):
        th = np.radians(11.25 + 22.5 * i)
        g.append(f'      <geom name="{prefix}_lip{i}" type="capsule" '
                 f'size="{lip:.6f} 0.0009" '
                 f'pos="{vec((cir * np.cos(th), cir * np.sin(th), L - lipd))}" '
                 f'zaxis="{vec((-np.sin(th), np.cos(th), 0.0))}" '
                 f'rgba="0 0 0 0" friction="0.3 0.008 0.0002" condim="3" '
                 f'solref="-20000 -40" solimp="0.95 0.99 0.0005" '
                 f'density="{p["density"]}"/>')
    # tapering shaft
    for i, (r, z0, z1) in enumerate((
            (0.0036, L - cl - 0.014, L - cl + 0.001),
            (0.0028, L - cl - 0.030, L - cl - 0.013),
            (0.0018, 0.006, L - cl - 0.029),
            (0.0009, 0.0015, 0.0065))):
        g.append(f'      <geom name="{prefix}_s{i}" type="cylinder" '
                 f'size="{r:.6f} {(z1 - z0) / 2:.6f}" '
                 f'pos="0 0 {(z0 + z1) / 2:.6f}" {inv} {gA}/>')
    # the foot is a SPHERE: a hair-thin cylinder's end-face on a plate is
    # a degenerate contact (measured: 0.6 mm sink, then NaN), sphere-box
    # is the best-conditioned pair there is. Lowest point = local z 0.
    g.append(f'      <geom name="{prefix}_foot" type="sphere" '
             f'size="0.0011" pos="0 0 0.0011" {inv} {gA}/>')
    # a PLAIN freejoint: free-joint damping looks harmless and is not —
    # on this body's 1e-8 kg*m^2 spin axis even 0.0002 N*m*s/rad NaN'd the
    # solver the moment two external forces overlapped (measured; with
    # zero damping the same run is rock stable)
    fj = f'      <freejoint name="{prefix}_free"/>\n' if free else ""
    return (f'    <body name="{prefix}" pos="{vec(pos)}">\n{fj}'
            + "\n".join(g) + "\n    </body>")


def _tip_mesh(p: dict, out_dir: Path, name: str) -> str:
    L = float(p["length"])
    cor, cir, cl = (float(p["collar_or"]), float(p["collar_ir"]),
                    float(p["collar_len"]))
    prof = [(3e-4, 0.0), (0.0011, 0.001), (0.0020, L - cl - 0.030),
            (0.0030, L - cl - 0.013), (0.0038, L - cl + 0.001),
            (cor, L - cl + 0.004), (cor, L), (cir, L),
            (cir, L - cl * 0.6), (3e-4, L - cl * 0.6)]
    return write_mesh(revolve_profile(prof, sections=32),
                      out_dir / "meshes" / f"{name}.obj")


def build_pipette_tip(params: dict | None = None,
                      out_dir: str | Path = ".") -> dict:
    p = merge_params(TIP_DEFAULTS, params)
    name = str(p["name"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _tip_mesh(p, out_dir, f"{name}_lathe")
    body = _tip_body_xml(p, name, np.asarray(p["pos"], float)
                         + [0, 0, 0.0], f"{name}_lathe_m")
    floor = FLOOR_XML if p["include_floor"] else ""
    xml = (mjcf_header(f"{name}_tip", timestep=p["timestep"],
                       meshdir="meshes")
           + f"""  <asset>
    <mesh name="{name}_lathe_m" file="{name}_lathe.obj"/>
  </asset>
  <worldbody>{LIGHTS_XML}{floor}
{body}
  </worldbody>
</mujoco>
""")
    xml_path = out_dir / f"{name}.xml"
    xml_path.write_text(xml)
    m = load(xml_path)
    save_params(out_dir / f"{name}_params.json", p)
    built = {"xml": str(xml_path),
             "params_json": str(out_dir / f"{name}_params.json"),
             "name": name, "kind": "pipette_tip",
             "tip_body": name, "length_m": float(p["length"]),
             "collar_or_m": float(p["collar_or"]),
             "lip_ir_m": float(p["collar_ir"]) - float(p["lip_in"]),
             "mass_kg": round(float(m.body(name).subtreemass[0]), 5)}
    (out_dir / f"{name}_built.json").write_text(json.dumps(built, indent=1))
    return built


# ---------------------------------------------------------------------------
# THE MICROPIPETTE. Local frame: +z up the barrel, z=0 at the NOSE TIP.
# ---------------------------------------------------------------------------
PIPETTE_DEFAULTS = {
    "name": "pipette",
    # ---- nose + tip interface (must agree with the tip's numbers) ----
    "nose_seat_z": 0.0075,    # where the mounted collar's TOP lands. The
                              #   tip's hollow is only ~8 mm deep before
                              #   its solid shaft begins — a 16 mm seat
                              #   asked the nose to pass through plastic
                              #   (measured: hard stop at 8 mm, then 18 N
                              #   of stored spring let go sideways). Real
                              #   noses also seat 6-10 mm in.
    "jaw_out":     0.0038,    # jaws' relaxed outer reach — sets the snap
                              #   against the tip's lip_ir. THE ORDERING
                              #   RULE: the plain nose passes the lip ring
                              #   (+-0.2 mm of centring) BEFORE the jaws
                              #   reach the collar entrance, and the jaws'
                              #   entry clearance (0.3 mm) must EXCEED that
                              #   centring slop — when it was the other way
                              #   round they landed flat on the rim and the
                              #   stored press force blew the model apart
    "jaw_r":       0.0006,
    "jaw_travel":  0.0009,
    "jaw_stiffness": 9000.0,
    "jaw_damping": 8.0,
    "jaw_armature": 0.010,
    # ---- ejector ----
    "eject_travel": 0.0100,
    "eject_stiffness": 800.0,
    "eject_damping": 3.0,
    # ---- plunger, two stages ----
    "stroke1":     0.0120,    # to the first stop
    "stroke2":     0.0055,    # first stop -> blow-out bottom
    "spring1":     220.0,     # N/m   (~2.6 N at the first stop)
    "blowout_preload_N": 5.0,
    "blowout_stiffness": 2600.0,
    # ---- looks / body ----
    "handle_r":    0.0130,
    "handle_len":  0.1050,
    "total_len":   0.2900,
    "body_rgba":   [0.93, 0.93, 0.92, 1.0],
    "grey_rgba":   [0.52, 0.53, 0.55, 1.0],
    "blue_rgba":   [0.16, 0.32, 0.75, 1.0],
    "accent_rgba": [0.80, 0.12, 0.12, 1.0],
    "density":     820.0,     # plastic shell, mostly hollow
    "friction":    "0.7 0.01 0.0002",
    "pos": [0.0, 0.0, 0.0],
    "yaw_deg": 0.0,
    "spin_deg": 0.0,          # lying: which side faces up
    "layout": "lying",        # lying | upright (upright needs a holder)
    "timestep": 0.001,
    "include_floor": True,
}


def build_micropipette(params: dict | None = None,
                       out_dir: str | Path = ".") -> dict:
    p = merge_params(PIPETTE_DEFAULTS, params)
    name = str(p["name"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seat = float(p["nose_seat_z"])
    jaw_out, jaw_r = float(p["jaw_out"]), float(p["jaw_r"])
    hr, hl, TL = (float(p["handle_r"]), float(p["handle_len"]),
                  float(p["total_len"]))
    wc, gc = rgba(p["body_rgba"]), rgba(p["grey_rgba"])
    bc, ac = rgba(p["blue_rgba"]), rgba(p["accent_rgba"])
    gA = f'density="{p["density"]}" friction="{p["friction"]}" condim="3"'
    off = 'contype="0" conaffinity="0"'
    s1, s2 = float(p["stroke1"]), float(p["stroke2"])
    pre = float(p["blowout_preload_N"])
    kb = float(p["blowout_stiffness"])

    # the tip's lip must actually contest the jaws
    tip_lip_ir = float(TIP_DEFAULTS["collar_ir"]) - float(TIP_DEFAULTS["lip_in"])
    interf = jaw_out - tip_lip_ir
    if interf <= 0:
        raise ValueError(f"jaw_out {1000*jaw_out:.2f} mm never meets the "
                         f"tip's lip ({1000*tip_lip_ir:.2f} mm) — no snap.")
    if interf >= float(p["jaw_travel"]):
        raise ValueError(f"snap interference {1000*interf:.2f} mm >= "
                         f"jaw_travel — the jaws cannot compress past the "
                         f"lip, the tip can never mount.")
    snap_N = 4 * float(p["jaw_stiffness"]) * interf   # all four jaws

    # ---------------- main body ------------------------------------------
    g = []
    # nose cone (enters the tip's collar; the seat shoulder stops it)
    # every nose section stays UNDER the tip lip's inner radius — the lip
    # is only ever contested by the sprung jaws, never by rigid plastic
    # (a 3.7 mm section here jammed 0.2 mm into the lip and the tip could
    # not seat at any force worth calling a press). The leading element is
    # a CAPSULE: its hemisphere is the lead-in that cams a millimetre of
    # approach error into the bore — a flat cylinder edge just perched on
    # the collar rim and levered the tip crooked (measured). Nose friction
    # is PP-on-PP low, or the insertion force reads as jam, not snap.
    nA = (f'density="{p["density"]}" friction="0.35 0.008 0.0002" '
          f'condim="3" solref="0.004 1"')
    # a SLIM, CONSTANT-radius nose column (3.1 mm against the lip ring's
    # 3.3): the lip centres it to +-0.2 mm several millimetres before the
    # jaws arrive at the entrance — guidance must always be tighter than
    # the snap's own entry clearance
    g.append(f'      <geom name="{name}_nose0" type="capsule" '
             f'size="0.0031 0.0011" pos="0 0 0.0043" rgba="{wc}" {nA}/>')
    g.append(f'      <geom name="{name}_nose1" type="cylinder" '
             f'size="0.0031 {(seat - 0.0050) / 2:.6f}" '
             f'pos="0 0 {(seat + 0.0050) / 2:.6f}" rgba="{wc}" {nA}/>')
    g.append(f'      <geom name="{name}_seat" type="cylinder" '
             f'size="0.0050 0.0012" pos="0 0 {seat + 0.0012:.6f}" '
             f'rgba="{wc}" {gA}/>')
    # slim zone the ejector sleeve rides over
    g.append(f'      <geom name="{name}_stem0" type="cylinder" '
             f'size="0.0036 {0.010:.6f}" pos="0 0 {seat + 0.0124:.6f}" '
             f'rgba="{wc}" {gA}/>')
    # lower shaft rising to the handle (section heights scale with total_len
    # relative to the 0.29 m default, so shorter pipettes stay buildable)
    k = TL / 0.29
    for i, (r, z0, z1) in enumerate(((0.0052, seat + 0.024, 0.075 * k),
                                     (0.0068, 0.075 * k, 0.115 * k),
                                     (0.0085, 0.115 * k, TL - hl - 0.020))):
        g.append(f'      <geom name="{name}_shaft{i}" type="cylinder" '
                 f'size="{r:.6f} {(z1 - z0) / 2:.6f}" '
                 f'pos="0 0 {(z0 + z1) / 2:.6f}" rgba="{wc}" {gA}/>')
    # the 1000-ul blue volume collar
    g.append(f'      <geom name="{name}_volband" type="cylinder" '
             f'size="0.0088 0.0100" pos="0 0 {TL - hl - 0.010:.6f}" '
             f'rgba="{bc}" {gA}/>')
    # handle
    g.append(f'      <geom name="{name}_handle" type="cylinder" '
             f'size="{hr:.6f} {hl / 2:.6f}" pos="0 0 {TL - hl / 2:.6f}" '
             f'rgba="{wc}" {gA}/>')
    # display window + red brand band (front = -y)
    g.append(f'      <geom name="{name}_display" type="box" '
             f'size="0.0080 0.0012 0.0110" '
             f'pos="0 {-hr + 0.0004:.6f} {TL - hl + 0.030:.6f}" '
             f'euler="8 0 0" rgba="0.15 0.16 0.18 1" {off} density="1"/>')
    g.append(f'      <geom name="{name}_brand" type="box" '
             f'size="0.0060 0.0008 0.0028" '
             f'pos="0 {-hr + 0.0002:.6f} {TL - hl + 0.052:.6f}" '
             f'rgba="{ac}" {off} density="1"/>')
    # the grey finger hook, curling backward (+y) from the handle top
    for i, (dy, dz, sy, sz) in enumerate(((0.0105, 0.004, 0.0060, 0.0040),
                                          (0.0190, 0.012, 0.0085, 0.0028),
                                          (0.0255, 0.024, 0.0028, 0.0085))):
        g.append(f'      <geom name="{name}_hook{i}" type="box" '
                 f'size="0.0060 {sy:.6f} {sz:.6f}" '
                 f'pos="0 {hr - 0.004 + dy:.6f} {TL - 0.020 + dz:.6f}" '
                 f'rgba="{gc}" {gA}/>')
    # grey top cap the plunger emerges from
    g.append(f'      <geom name="{name}_topcap" type="cylinder" '
             f'size="0.0075 0.0060" pos="0 0 {TL + 0.006:.6f}" '
             f'rgba="{gc}" {gA}/>')

    # internal second-stage pin: contype/conaffinity 2 — it may touch
    # NOTHING but the plunger stem (both live inside the handle shell)
    zpin = TL - 0.052
    pin = (f'      <body name="{name}_pin" pos="0 0 0">\n'
           f'        <joint name="{name}_blow" type="slide" axis="0 0 -1" '
           f'range="0 {s2:.6f}" stiffness="{kb}" '
           f'springref="{-pre / kb:.6f}" damping="4" armature="0.006" '
           # STIFF limits on every internal slide: the preload lives on the
           # range stop, and MuJoCo's default soft limit let it sink 1.7 mm
           # past the floor (measured) — the whole two-stage feel leaks away
           f'solreflimit="0.004 1" solimplimit="0.99 0.999 0.0002"/>\n'
           # a BOX top face: the stem meets it as sphere-on-box. Coaxial
           # cylinder-on-cylinder faces generate NO contact in libccd and
           # the blow-out stage silently never engaged (measured: the pin
           # sat untouched through a 22 N press)
           f'        <geom name="{name}_pintop" type="box" '
           f'size="0.0030 0.0030 0.0015" pos="0 0 {zpin:.6f}" '
           f'rgba="0.3 0.3 0.3 1" contype="2" conaffinity="2" '
           f'solref="-30000 -60" density="600"/>\n'
           f'      </body>')

    # plunger: blue button + stem, on its own sprung slide
    zstem_bot = zpin + 0.0015 + s1 + 0.0011   # meets the pin at q = s1
    plunger = (
        f'      <body name="{name}_plunger" pos="0 0 0">\n'
        f'        <joint name="{name}_plunge" type="slide" axis="0 0 -1" '
        f'range="-0.0005 {s1 + s2 + 0.0012:.6f}" '
        f'stiffness="{p["spring1"]}" damping="2.5" armature="0.008" '
        f'solreflimit="0.004 1" solimplimit="0.99 0.999 0.0002"/>\n'
        f'        <geom name="{name}_button" type="cylinder" '
        f'size="0.0060 0.0045" pos="0 0 {TL + 0.0165:.6f}" '
        f'rgba="{bc}" density="400" friction="{p["friction"]}"/>\n'
        f'        <geom name="{name}_pshaft" type="cylinder" '
        f'size="0.0028 {0.012:.6f}" pos="0 0 {TL + 0.002:.6f}" '
        f'rgba="{gc}" {off} density="400"/>\n'
        # sphere bottom lands exactly where the old cylinder face was, so
        # the stem still meets the pin at q = stroke1
        f'        <geom name="{name}_stembot" type="sphere" '
        f'size="0.0022" pos="0 0 {zstem_bot + 0.0011:.6f}" '
        f'rgba="0.3 0.3 0.3 1" contype="2" conaffinity="2" '
        f'solref="-30000 -60" density="600"/>\n'
        f'      </body>')

    # ejector: grey button high on the back, rod down the barrel's back,
    # sleeve around the slim zone above the nose
    sleeve_bot = seat + 0.0015
    ej = [f'        <geom name="{name}_ejbtn" type="box" '
          f'size="0.0050 0.0035 0.0090" '
          f'pos="0 {hr + 0.0015:.6f} {TL - 0.006:.6f}" '
          f'rgba="{gc}" density="400" friction="{p["friction"]}"/>',
          f'        <geom name="{name}_ejrod" type="box" '
          f'size="0.0022 0.0014 {0.055:.6f}" '
          f'pos="0 {0.0086:.6f} {TL - hl - 0.085 * k:.6f}" '
          f'euler="3.5 0 0" rgba="{gc}" {off} density="300"/>']
    ej += shell_ring(f"{name}_sleeve", 0.0038, 0.0010,
                     sleeve_bot, sleeve_bot + 0.0140, gc,
                     f'density="500" friction="{p["friction"]}" condim="3"',
                     n=8)
    ejector = (
        f'      <body name="{name}_ejector" pos="0 0 0">\n'
        f'        <joint name="{name}_eject" type="slide" axis="0 0 -1" '
        f'range="-0.0002 {p["eject_travel"]}" '
        f'stiffness="{p["eject_stiffness"]}" '
        f'damping="{p["eject_damping"]}" armature="0.006" '
        f'solreflimit="0.004 1" solimplimit="0.99 0.999 0.0002"/>\n'
        + "\n".join(ej) + "\n      </body>")

    # FOUR snap jaws on the nose, each on its own radial slide, low enough
    # that the captured lip rests fully above them. Two jaws were escaped
    # for free: a ring can WORM past two point catches by tilting (how you
    # take a ring off a peg), and the mounted tip slid off under its own
    # weight (measured). Four posts mean any tilt still contests at least
    # two springs at once.
    jaws = []
    for i, ang in enumerate((45.0, 135.0, 225.0, 315.0)):
        cth, sth = np.cos(np.radians(ang)), np.sin(np.radians(ang))
        jaws.append(
            f'      <body name="{name}_jaw{i}" pos="0 0 0">\n'
            f'        <joint name="{name}_jawslide{i}" type="slide" '
            f'axis="{-cth:.5f} {-sth:.5f} 0" range="0 {p["jaw_travel"]}" '
            f'stiffness="{p["jaw_stiffness"]}" '
            f'damping="{p["jaw_damping"]}" '
            f'armature="{p["jaw_armature"]}" '
            f'solreflimit="0.004 1" solimplimit="0.99 0.999 0.0002"/>\n'
            # the SNAP force lives in the jaw's own spring; the contact
            # itself is soft — a 2 ms contact on a 15 kN/m sprung jaw at a
            # 1 ms step detonated the chain at first touch (measured)
            # a SPHERE, not a capsule: the mounted lip must REST ON TOP of
            # the jaw so the contact normal carries an axial component —
            # geometric retention. Beside a capsule's cylindrical waist the
            # normal is purely radial and the tip creep-slid off through
            # MuJoCo's regularized friction at ~1 mm/s (measured). NEGATIVE
            # solref = ABSOLUTE contact stiffness: the mass-scaled form on
            # this near-massless pair was softer than the jaw spring itself
            # and capped the snap at 0.7 N (measured).
            f'        <geom name="{name}_jawg{i}" type="sphere" '
            f'size="0.0009" '
            f'pos="{(jaw_out - 0.0009) * cth:.6f} '
            f'{(jaw_out - 0.0009) * sth:.6f} {seat - 0.0033:.6f}" '
            f'rgba="{wc}" friction="0.3 0.008 0.0002" condim="3" '
            f'solref="-20000 -40" solimp="0.95 0.99 0.0005" '
            f'density="900"/>\n'
            f'      </body>')

    # ---------------- root pose ------------------------------------------
    pos = np.asarray(p["pos"], float)
    if str(p["layout"]) == "lying":
        q = _quat_mul(_quat_mul(_qz(float(p["yaw_deg"])), _qy(90.0)),
                      _qz(float(p["spin_deg"])))
        pos = pos + [0.0, 0.0, hr]
    else:
        q = _quat_mul(_qz(float(p["yaw_deg"])), np.array([1.0, 0, 0, 0]))
    floor = FLOOR_XML if p["include_floor"] else ""
    xml = (mjcf_header(f"{name}_micropipette", timestep=p["timestep"]) + f"""
  <worldbody>{LIGHTS_XML}{floor}
    <body name="{name}" pos="{vec(pos)}"
          quat="{' '.join(f'{v:.8f}' for v in q)}">
      <freejoint name="{name}_free"/>
{chr(10).join(g)}
{pin}
{plunger}
{ejector}
{chr(10).join(jaws)}
    </body>
  </worldbody>
</mujoco>
""")
    xml_path = out_dir / f"{name}.xml"
    xml_path.write_text(xml)
    m = load(xml_path)
    save_params(out_dir / f"{name}_params.json", p)
    built = {
        "xml": str(xml_path),
        "params_json": str(out_dir / f"{name}_params.json"),
        "name": name, "kind": "micropipette",
        "pipette_body": name, "plunger_body": f"{name}_plunger",
        "ejector_body": f"{name}_ejector",
        "plunger_joint": f"{name}_plunge", "eject_joint": f"{name}_eject",
        "blow_joint": f"{name}_blow",
        "jaw_joints": [f"{name}_jawslide{i}" for i in range(4)],
        "nose_seat_z_m": seat, "total_len_m": TL,
        "stroke1_mm": round(1000 * s1, 2), "stroke2_mm": round(1000 * s2, 2),
        "first_stop_N": round(float(p["spring1"]) * s1, 2),
        "second_stage_step_N": round(pre, 2),
        "tip_snap_N": round(snap_N, 2),
        "snap_interference_mm": round(1000 * interf, 3),
        "mass_kg": round(float(m.body(name).subtreemass[0]), 4),
        "note": (f"plunger: {1000*s1:.0f} mm to the first stop "
                 f"({float(p['spring1'])*s1:.1f} N), then the blow-out "
                 f"pin's {pre:.1f} N preload must yield for the last "
                 f"{1000*s2:.1f} mm. Tip mounts by pressing the nose in: "
                 f"the collar lip forces both jaws in ({snap_N:.1f} N) and "
                 f"is captured above them; the ejector sleeve pushes it "
                 f"back past the jaws. All springs are joints — nothing is "
                 f"actuated."),
    }
    (out_dir / f"{name}_built.json").write_text(json.dumps(built, indent=1))
    return built


# ---------------------------------------------------------------------------
# THE TIP BOX — navy base, pale-blue tray of 96 collars, friction-hinged
# translucent lid. Wells in `free_wells` hold REAL tips (free bodies) over
# real tray sockets; the rest are moulded visuals.
# ---------------------------------------------------------------------------
TIPBOX_DEFAULTS = {
    "name": "tipbox",
    "length": 0.127, "width": 0.096,
    "base_h": 0.072,
    "wall_t": 0.0030,
    "tray_t": 0.0040,
    "rows": 8, "cols": 12, "pitch": 0.0090,
    "hole_r": 0.00375,        # bears 0.75 mm of collar per side AND guides
                              #   the 3.6 mm shaft within 0.15 mm — at 4.0
                              #   the collar's 0.4 mm perch let a settling
                              #   tip skitter off and wedge 12 deg crooked
                              #   against the neighbouring cells (measured)
    "lid_h": 0.059,           # deep enough that the CLOSED lid clears the
                              #   standing tips' collars — a 45 mm lid came
                              #   down ON the free tips and wedged (measured)
    "lid_t": 0.0022,
    "lid_deg": 100.0,          # where the lid starts (0 = closed)
    "lid_open_max": 118.0,
    "lid_friction": 0.055,     # what makes it STAY where it is pushed
    "lid_damping": 0.02,
    "lid_mass": 0.001,         # kg, spread over the lid's plates by volume:
                               #   MuJoCo derives the inertia from them (a
                               #   hand-written 1e-7 kg m^2 was 38x too small)
    "free_wells": [],          # [[row, col]] wells holding REAL tips
    "tip_params": {},
    "base_rgba": [0.093, 0.11, 0.30, 1.0],
    "tray_rgba": [0.60, 0.72, 0.92, 1.0],
    "lid_rgba":  [0.86, 0.89, 0.93, 0.40],
    "stub_rgba": [0.62, 0.72, 0.93, 0.75],
    "label_rgba": [0.75, 0.78, 0.85, 1.0],
    "density": 480.0,
    "friction": "0.85 0.015 0.0004",
    "pos": [0.0, 0.0, 0.0],
    "yaw_deg": 0.0,
    "timestep": 0.001,
    "include_floor": True,
}


def tipbox_well_xy(p: dict, row: int, col: int) -> tuple[float, float]:
    pitch = float(p["pitch"])
    return (-(int(p["cols"]) - 1) / 2 * pitch + int(col) * pitch,
            -(int(p["rows"]) - 1) / 2 * pitch + int(row) * pitch)


def build_tip_box(params: dict | None = None,
                  out_dir: str | Path = ".") -> dict:
    p = merge_params(TIPBOX_DEFAULTS, params)
    name = str(p["name"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    L, W, H = float(p["length"]), float(p["width"]), float(p["base_h"])
    wt, tt = float(p["wall_t"]), float(p["tray_t"])
    rows, cols, pitch = int(p["rows"]), int(p["cols"]), float(p["pitch"])
    hr = float(p["hole_r"])
    bc, yc = rgba(p["base_rgba"]), rgba(p["tray_rgba"])
    lc, sc = rgba(p["lid_rgba"]), rgba(p["stub_rgba"])
    # 4 ms solref to match the gram-scale tips this box exists to hold
    # (see _tip_body_xml) — the default 20 ms swallows their features
    gA = (f'density="{p["density"]}" friction="{p["friction"]}" condim="3" '
          f'solref="0.004 1"')

    tp = merge_params(TIP_DEFAULTS, dict(p["tip_params"],
                                         include_floor=False))
    _tip_mesh(tp, out_dir, f"{name}_tiplathe")
    tray_top = H
    # a seated tip rests TAPER-IN-FUNNEL, like a ball on a tee: eight
    # slanted capsules under the well form a V the shaft's widest step
    # settles into — line contact, self-centring, nothing to wedge on.
    # Every rim/pad/cage bearing tried before this one ejected the tip:
    # the collar's polygonal corners out-reach the 4.5 mm half-cell, and a
    # foot-on-pad stand let the foot walk off the pad edge and spring the
    # shaft out of its cage (all measured).
    # THE SEAT RULE, learned the hard way: a free tip may only ever be
    # supported by faces whose normals are VERTICAL (the platform it
    # stands on) or HORIZONTAL (the cage that stops it leaning). Slanted
    # funnels, sub-millimetre rims and small pads all pumped the tip out —
    # any support with an up-slope component turns a lateral squeeze into
    # lift, and a gram-scale body ratchets straight out of the well
    # (measured, in three different geometries).
    plat_top = tray_top - (float(tp["length"]) - float(tp["collar_len"])) \
        + 0.005                         # collar rides ~5 mm proud
    tip_z = plat_top + 0.0002           # sphere foot's lowest point = z 0
    free_wells = {(int(r), int(c)) for r, c in p["free_wells"]}
    if float(tp["length"]) - float(tp["collar_len"]) > H - 0.004:
        raise ValueError(
            f"a {1000*float(tp['length']):.0f} mm tip hangs "
            f"{1000*(float(tp['length'])-float(tp['collar_len'])):.0f} mm "
            f"below the tray — deeper than the {1000*H:.0f} mm base. "
            f"Raise base_h.")

    stub_col_r = min(0.0044, pitch - float(tp["collar_or"]) / np.cos(np.pi / 8) - 0.0003)
    if stub_col_r < 0.001:
        raise ValueError(f"pitch {1000 * pitch:.1f} mm leaves no room for the standing tips' collars next to a free "
                         f"tip's {2000 * float(tp['collar_or']):.1f} mm collar; raise pitch")
    g = []
    # base walls + floor
    for i, (sx, sy, px, py) in enumerate((
            (L / 2, wt / 2, 0.0, W / 2 - wt / 2),
            (L / 2, wt / 2, 0.0, -(W / 2 - wt / 2)),
            (wt / 2, W / 2 - wt, L / 2 - wt / 2, 0.0),
            (wt / 2, W / 2 - wt, -(L / 2 - wt / 2), 0.0))):
        g.append(f'      <geom name="{name}_wall{i}" type="box" '
                 f'size="{vec((sx, sy, H / 2))}" '
                 f'pos="{vec((px, py, H / 2))}" rgba="{bc}" {gA}/>')
    g.append(f'      <geom name="{name}_floor" type="box" '
             f'size="{vec((L / 2 - wt, W / 2 - wt, 0.0015))}" '
             f'pos="0 0 0.0015" rgba="{bc}" {gA}/>')
    # brand plate on the front face
    g.append(f'      <geom name="{name}_label" type="box" '
             f'size="0.020 0.0006 0.006" '
             f'pos="0 {-W / 2 - 0.0004:.6f} {0.30 * H:.6f}" '
             f'rgba="{rgba(p["label_rgba"])}" contype="0" conaffinity="0" '
             f'density="1"/>')
    # tray: cell-by-cell like the rack — solid cells + stub visuals, real
    # sockets under the free wells
    fx, fy = (cols * pitch) / 2, (rows * pitch) / 2
    zc = tray_top - tt / 2
    for r in range(rows):
        for c in range(cols):
            x, y = tipbox_well_xy(p, r, c)
            if (r, c) in free_wells:
                # A FREE tip is held SHAFT-AND-FOOT, not collar-on-rim: its
                # fine end stands on a pad and the guide boxes cage the
                # 3.6 mm shaft section with 0.3 mm of play over 20 mm of
                # depth (tilt capped at ~2 deg). Bearing the 9 mm collar on
                # a sub-millimetre hole rim was measured to fail twice
                # over: the polygonal collar's corners out-reach the cell
                # and wedge against the neighbours, and any settle bounce
                # ratchets the tip up and over crooked. The rim here is
                # VISUAL only.
                s = (pitch / 2 - hr) / 2
                for k, (ex, ey, hx, hy) in enumerate((
                        (x, y + hr + s, pitch / 2, s),
                        (x, y - hr - s, pitch / 2, s),
                        (x + hr + s, y, s, hr),
                        (x - hr - s, y, s, hr))):
                    g.append(f'      <geom name="{name}_t{r}_{c}_{k}" '
                             f'type="box" size="{vec((hx, hy, tt / 2))}" '
                             f'pos="{vec((ex, ey, zc))}" rgba="{yc}" '
                             f'contype="0" conaffinity="0" density="30"/>')
                # cell-wide platform (the foot cannot reach its edge)...
                g.append(f'      <geom name="{name}_p{r}_{c}" type="box" '
                         f'size="{vec((0.0044, 0.0044, (plat_top - 0.004) / 2))}" '
                         f'pos="{vec((x, y, (plat_top + 0.004) / 2))}" '
                         f'rgba="{bc}" {gA}/>')
                # ...and a plumb-walled cage around the widest shaft step
                for k in range(4):
                    th = np.radians(45.0 + 90.0 * k)
                    g.append(f'      <geom name="{name}_q{r}_{c}_{k}" '
                             f'type="box" size="0.0010 0.0042 0.0110" '
                             f'pos="{vec((x + 0.0052 * np.cos(th), y + 0.0052 * np.sin(th), tray_top - 0.0112))}" '
                             f'euler="0 0 {np.degrees(th):.2f}" '
                             f'rgba="{yc}" {gA}/>')
            else:
                g.append(f'      <geom name="{name}_t{r}_{c}" type="box" '
                         f'size="{vec((pitch / 2, pitch / 2, tt / 2))}" '
                         f'pos="{vec((x, y, zc))}" rgba="{yc}" {gA}/>')
                # a moulded tip collar standing in the well (visual),
                # matched to the height the REAL free tips sit at
                g.append(f'      <geom name="{name}_stub{r}_{c}" '
                         f'type="cylinder" size="0.0044 0.0068" '
                         f'pos="{vec((x, y, tray_top + 0.0088))}" '
                         f'rgba="{sc}" contype="0" conaffinity="0" '
                         f'density="30"/>')
                # ...and its collision, so what is drawn is also there: the
                # stubs stood 15.6 mm above the tray with nothing a finger
                # could touch. Same height, a radius that leaves a free
                # tip's octagonal collar (corners at collar_or / cos 22.5)
                # 0.3 mm in a neighbouring well; massless (the visual
                # stub carries the mass), group 3
                g.append(f'      <geom name="{name}_stubc{r}_{c}" '
                         f'type="cylinder" size="{stub_col_r:.6f} 0.0068" '
                         f'pos="{vec((x, y, tray_top + 0.0088))}" '
                         f'group="3" rgba="0 0 0 0" density="0" '
                         f'friction="{p["friction"]}" condim="3" solref="0.004 1"/>')
    # tray margins
    for i, (ex, ey, hx, hy) in enumerate((
            (0.0, (fy + W / 2 - wt) / 2, L / 2 - wt, (W / 2 - wt - fy) / 2),
            (0.0, -(fy + W / 2 - wt) / 2, L / 2 - wt, (W / 2 - wt - fy) / 2),
            ((fx + L / 2 - wt) / 2, 0.0, (L / 2 - wt - fx) / 2, fy),
            (-(fx + L / 2 - wt) / 2, 0.0, (L / 2 - wt - fx) / 2, fy))):
        if hx > 5e-4 and hy > 5e-4:
            g.append(f'      <geom name="{name}_tm{i}" type="box" '
                     f'size="{vec((hx, hy, tt / 2))}" '
                     f'pos="{vec((ex, ey, zc))}" rgba="{yc}" {gA}/>')

    # ---- the lid: hinge at the back top edge of the base ----------------
    lh, lt = float(p["lid_h"]), float(p["lid_t"])
    start = float(p["lid_deg"])
    ca, sa = np.cos(np.radians(-start / 2)), np.sin(np.radians(-start / 2))
    # 2 mm of skirt clearance all round: at 0.8 mm a slammed-shut lid
    # WEDGED onto the base and no honest reopening force would move it
    lg = 0.0020
    # the lid's plates, each with its share of lid_mass by volume: MuJoCo then
    # derives the lid's centre of mass and inertia from its own geometry
    plates = [("lidtop", (L / 2 + lt + lg, W / 2 + lt + lg, lt / 2), (0.0, -W / 2 - lg, lh - lt / 2))]
    for i, (sx, sy, px, py) in enumerate((
            (L / 2 + lt + lg, lt / 2, 0.0, -W - 2 * lg - lt / 2),
            (L / 2 + lt + lg, lt / 2, 0.0, -lg + lt / 2),
            (lt / 2, W / 2 + lg, L / 2 + lg + lt / 2, -W / 2 - lg),
            (lt / 2, W / 2 + lg, -(L / 2 + lg + lt / 2), -W / 2 - lg))):
        plates.append((f"lidw{i}", (sx, sy, lh / 2 - 0.001), (px, py, lh / 2)))
    if float(p["lid_mass"]) <= 0:
        raise ValueError("lid_mass must be > 0")
    vols = [8 * sz[0] * sz[1] * sz[2] for _, sz, _ in plates]
    gL = (f'friction="{p["friction"]}" condim="3" solref="0.004 1"')
    lidg = [f'        <geom name="{name}_{nm}" type="box" size="{vec(sz)}" pos="{vec(ps)}" rgba="{lc}" '
            f'mass="{float(p["lid_mass"]) * v / sum(vols):.8g}" {gL}/>'
            for (nm, sz, ps), v in zip(plates, vols)]
    hinge_z = H - 0.032         # closed: the lid's inner ceiling must sit
    tips_top = tip_z + float(tp["length"])
    if hinge_z + lh - lt < tips_top + 0.003:
        raise ValueError(
            f"lid_h {1000*lh:.0f} mm leaves the closed lid's ceiling at "
            f"{1000*(hinge_z+lh-lt):.1f} mm — under the standing tips "
            f"({1000*tips_top:.1f} mm). Raise lid_h.")
    lid = f"""
    <body name="{name}_lid" pos="{vec((0.0, W / 2 + lg, hinge_z))}"
          quat="{ca:.8f} {sa:.8f} 0 0">
      <joint name="{name}_hinge" type="hinge" axis="-1 0 0"
             range="-2 {p['lid_open_max']}" ref="{start:.3f}"
             damping="{p['lid_damping']}"
             frictionloss="{p['lid_friction']}" armature="0.0004"/>
{chr(10).join(lidg)}
    </body>"""

    tips_xml, tips = [], []
    for i, (r, c) in enumerate(sorted(free_wells)):
        x, y = tipbox_well_xy(p, r, c)
        pref = f"{name}_tip{i}"
        tips_xml.append(_tip_body_xml(
            tp, pref,
            [x + float(p["pos"][0]), y + float(p["pos"][1]),
             tip_z + float(p["pos"][2])], f"{name}_tiplathe_m"))
        tips.append({"body": pref, "row": r, "col": c})

    free = f'      <freejoint name="{name}_free"/>\n'
    floor = FLOOR_XML if p["include_floor"] else ""
    xml = (mjcf_header(f"{name}_tipbox", timestep=p["timestep"],
                       meshdir="meshes")
           + f"""  <asset>
    <mesh name="{name}_tiplathe_m" file="{name}_tiplathe.obj"/>
  </asset>
  <worldbody>{LIGHTS_XML}{floor}
    <body name="{name}" pos="{vec(p['pos'])}" euler="0 0 {p['yaw_deg']}">
{free}{chr(10).join(g)}{lid}
    </body>
{chr(10).join(tips_xml)}
  </worldbody>
</mujoco>
""")
    xml_path = out_dir / f"{name}.xml"
    xml_path.write_text(xml)
    m = load(xml_path)
    save_params(out_dir / f"{name}_params.json", p)
    built = {
        "xml": str(xml_path),
        "params_json": str(out_dir / f"{name}_params.json"),
        "name": name, "kind": "tipbox",
        "box_body": name, "lid_body": f"{name}_lid",
        "lid_joint": f"{name}_hinge",
        "tray_top_m": tray_top, "tip_seat_z_m": round(tip_z, 5),
        "tip_collar_top_m": round(tip_z + float(tp["length"]), 5),
        "free_tips": tips, "rows": rows, "cols": cols, "pitch_m": pitch,
        "lid_start_deg": start, "lid_open_max_deg": float(p["lid_open_max"]),
        "mass_kg": round(float(m.body(name).subtreemass[0]), 4),
        "note": (f"lid hinge: 0 = closed, +ve = open, friction "
                 f"{p['lid_friction']} N*m holds it wherever a push leaves "
                 f"it. free_tips stand in REAL sockets (collar on the tray "
                 f"hole, shaft guided below) and can be pressed onto a "
                 f"pipette nose and lifted out; the other 95 wells are "
                 f"moulded visuals. tipbox_well_xy(params,row,col) locates "
                 f"any well."),
    }
    (out_dir / f"{name}_built.json").write_text(json.dumps(built, indent=1))
    return built


def rebuild(params_json: str | Path, out_dir: str | Path | None = None) -> dict:
    params_json = Path(params_json)
    p = json.loads(params_json.read_text())
    out = out_dir or params_json.parent
    if "stroke1" in p:
        return build_micropipette(p, out)
    if "lid_h" in p:
        return build_tip_box(p, out)
    return build_pipette_tip(p, out)


# ---------------------------------------------------------------------------
# demos
# ---------------------------------------------------------------------------
def _jadr(model, suffix):
    for i in range(model.njnt):
        n = model.joint(i).name or ""
        if n == suffix or n.endswith(suffix):
            return int(model.joint(i).qposadr[0])
    return None


def demo_micropipette(built: dict, out_mp4: str | Path,
                      seconds: float = 22.0, fps: int = 30) -> dict:
    """The pipette's whole life in one force-driven take: press onto a tip
    standing in a box socket (measured snap), lift and shake it (the snap
    carries it), run the plunger through both stages (measured stops),
    then press the ejector and watch the tip leave."""
    import mujoco
    from .mjcf import rollout
    pp = json.loads(Path(built["params_json"]).read_text())
    tmp = Path(built["xml"]).parent / "demo_world"
    bx = build_tip_box({"name": "demobox", "free_wells": [[4, 6]],
                        "lid_deg": 102.0, "timestep": 0.0005}, tmp)
    # splice the pipette in, held upright above the free tip
    px, py = tipbox_well_xy(json.loads(Path(bx["params_json"]).read_text()),
                            4, 6)
    tip_top = float(bx["tip_collar_top_m"])
    pip = build_micropipette(dict(pp, layout="upright",
                                  pos=[0, 0, 0], include_floor=False),
                             tmp / "pip")
    inner = Path(pip["xml"]).read_text()
    inner = inner.split("<worldbody>")[1].split("</worldbody>")[0]
    inner = "\n".join(ln for ln in inner.splitlines() if "<light" not in ln)
    hover0 = tip_top + 0.020
    inner = inner.replace('pos="0.000000 0.000000 0.000000"',
                          f'pos="{vec((px, py, hover0))}"', 1)
    xml_p = Path(bx["xml"])
    xml_p.write_text(xml_p.read_text().replace(
        "  </worldbody>", f"{inner}\n  </worldbody>"))
    model = load(xml_p)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    name = built["name"]
    pb = model.body(name).id
    plb = model.body(f"{name}_plunger").id
    ejb = model.body(f"{name}_ejector").id
    tipb = model.body(bx["free_tips"][0]["body"]).id
    grip = ForceGrip.sized(model, pb, stiff=2.0, max_force=28.0,
                           max_torque=1.2)
    seat = float(built["nose_seat_z_m"])
    qup = np.array([1.0, 0, 0, 0])
    log = {"snap_seen_N": None, "carry_gap_mm": None, "first_stop": None,
           "second_stop": None, "released_mm": None, "eject_sep_mm": None,
           "phases": []}
    st = {"phase": "press", "t0": 0.0, "z": hover0, "f": 0.0}

    def goto(ph, t):
        log["phases"].append((ph, round(t, 2)))
        st["phase"], st["t0"], st["f"], st["ok"] = ph, t, 0.0, 0

    def nose_state(d):
        """(axial gap mm, lateral offset mm) of the tip collar's top vs its
        seat on the nose. BOTH matter: the axial gap alone read 'mounted'
        when the nose had slid down BESIDE the collar and the two merely
        shared a height (measured)."""
        R = d.xmat[pb].reshape(3, 3)
        tip_top_w = (d.xpos[tipb]
                     + d.xmat[tipb].reshape(3, 3)
                     @ np.array([0, 0, float(TIP_DEFAULTS["length"])]))
        v = R.T @ (tip_top_w - d.xpos[pb])
        return (float(1000 * (v[2] - seat)),
                float(1000 * np.hypot(v[0], v[1])))

    def nose_gap(d):
        return nose_state(d)[0]

    def control(t, m, d):
        clear_forces(d)
        ph, el = st["phase"], t - st["t0"]
        q = float(d.qpos[_jadr(m, f"{name}_plunge")])
        # thumb forces act ALONG THE BARREL: a world-vertical push at the
        # top of a 29 cm lever is an inverted pendulum — any lean grows the
        # tipping moment faster than the grip's restoring torque, and the
        # pipette went over at 3 N (measured)
        axis_dn = -d.xmat[pb].reshape(3, 3)[:, 2]
        if ph == "press":
            # the seat is 36 mm below the hover (20 mm standoff + 16 mm of
            # nose inside the collar) — the target must be allowed past it.
            # Descent is rate * dt (a /0.001 here once made it 10 mm PER
            # STEP and the grip hammered the tip out of its socket), and
            # it SLOWS to 3 mm/s for the last stretch so the snap happens
            # at a controlled force instead of a slam.
            gap, lat = nose_state(d)
            rate = 0.012 if gap < -6.0 else 0.003
            st["z"] = max(hover0 - 0.048,
                          st["z"] - rate * float(m.opt.timestep))
            grip.hold(d, target_pos=[px, py, st["z"]], target_quat=qup)
            f = float(np.linalg.norm(grip.last[:3]))
            # MOUNTED means seated-and-staying: one bounced sample of a
            # small gap once ended the press mid-flight and the lift left
            # without the tip (measured)
            st["ok"] = st.get("ok", 0) + 1 if (gap > -0.8 and lat < 2.0) \
                else 0
            if st["ok"] >= 120:
                log["snap_seen_N"] = round(max(log.get("snap_seen_N") or 0,
                                               f), 1)
                goto("lift", t)
            if f > 6.0 and log["snap_seen_N"] is None:
                log["snap_seen_N"] = round(f, 1)
            if el > 7.0:
                goto("abort", t)
            return (f"press the nose into the tip: {f:.1f} N "
                    f"(seat gap {gap:.1f} mm)")
        if ph == "lift":
            u = min(1.0, (el / 2.5) ** 2)   # gentle start off the seat
            # a slow 4 mm sway, not a shake: the retention proof is the
            # seat-gap number, and violent wiggling read as instability
            dx = 0.004 * np.sin(2 * np.pi * 0.8 * max(0.0, el - 2.5)) \
                if el > 2.5 else 0.0
            grip.hold(d, target_pos=[px + dx, py, st["z"] + 0.070 * u],
                      target_quat=qup)
            log["carry_gap_mm"] = round(nose_gap(d), 2)
            if el > 4.0:
                log["gap_before_eject"] = round(nose_gap(d), 2)
                goto("eject", t)     # eject while the mount is FRESH — the
                #                      ejector pushing the tip off IS the
                #                      point; the plunger runs after, empty
            return (f"lift + sway — the snap carries the tip "
                    f"(seat gap {nose_gap(d):.1f} mm)")
        if ph == "plunge1":
            # rise smoothly off the eject spot first — a hold target that
            # jumps a dozen centimetres at a phase boundary is a
            # whole-body yank through the grip spring
            if st.get("p1z") is None:
                st["p1z"] = float(d.xpos[pb][2])
            u = min(1.0, el / 1.2)
            s_ = u * u * (3 - 2 * u)
            zt = st["p1z"] + s_ * ((st["z"] + 0.070) - st["p1z"])
            grip.hold(d, target_pos=[px + 0.075, py - 0.055, zt],
                      target_quat=qup)
            if el > 1.2:
                st["f"] = min(4.0, st["f"] + 2.0 * float(m.opt.timestep))
                apply_wrench(d, plb, force=st["f"] * axis_dn)
            if el > 3.7:
                log["first_stop"] = {"N": round(st["f"], 2),
                                     "mm": round(1000 * q, 2)}
                goto("plunge2", t)
            return (f"thumb: {st['f']:.1f} N — plunger {1000*q:.1f} mm "
                    f"(first stop at {built['stroke1_mm']} mm)")
        if ph == "plunge2":
            grip.hold(d, target_pos=[px + 0.075, py - 0.055,
                                     st["z"] + 0.070], target_quat=qup)
            st["f"] = min(22.0, max(st["f"], 4.0)
                          + 9.0 * float(m.opt.timestep))
            apply_wrench(d, plb, force=st["f"] * axis_dn)
            if el > 2.5:
                log["second_stop"] = {"N": round(st["f"], 2),
                                      "mm": round(1000 * q, 2)}
                goto("release", t)
            return (f"past the stop: {st['f']:.1f} N — plunger "
                    f"{1000*q:.1f} mm (blow-out)")
        if ph == "release":
            grip.hold(d, target_pos=[px + 0.075, py - 0.055,
                                     st["z"] + 0.070], target_quat=qup)
            # the thumb EASES off: cutting 22 N in one tick fires the
            # whole sprung stack at ~30 g (el-based, because goto zeroes
            # st["f"] on phase entry)
            f_rel = max(0.0, 22.0 - 30.0 * el)
            if f_rel > 0:
                apply_wrench(d, plb, force=f_rel * axis_dn)
            if el > 2.2:
                log["released_mm"] = round(1000 * q, 2)
                goto("done", t)
            return f"released — the springs return it ({1000*q:.1f} mm)"
        if ph == "eject":
            # TWO legs — translate CLEAR of the box at carry height, then
            # straight down to tip-end ~12 mm off the floor — and squeeze
            # only once ARRIVED, on a RAMP, held until the lip is clearly
            # past the jaws. A diagonal path wipes the hanging tip off on
            # the box rim, a mid-air squeeze reads as flailing, and a
            # squeeze cut on the jaws' own slope limit-cycles (all
            # measured, here or on the bench). Released quasi-statically
            # the tip just drops and stays.
            dt_ = float(m.opt.timestep)
            z_eject = 0.0125 + float(TIP_DEFAULTS["length"]) - seat
            gx, gy = px + 0.075, py - 0.055    # open floor, clear of the
            #                                    box AND its open lid
            if st.get("e0") is None:
                st["e0"] = d.xpos[pb].copy()
                st["ez"] = float(st["e0"][2])
            if el < 1.5:
                u = min(1.0, el / 1.4)
                s_ = u * u * (3 - 2 * u)
                e0 = st["e0"]
                tgt = [e0[0] + s_ * (gx - e0[0]),
                       e0[1] + s_ * (gy - e0[1]),
                       e0[2] + s_ * ((st["z"] + 0.075) - e0[2])]
                st["ez"] = float(tgt[2])
            else:
                st["ez"] = max(z_eject, st["ez"] - 0.060 * dt_)
                tgt = [gx, gy, st["ez"]]
            grip.hold(d, target_pos=tgt, target_quat=qup)
            sep = -nose_gap(d)             # mounted = 0, ejected = large
            arrived = float(d.xpos[pb][2]) < z_eject + 0.004
            if arrived and st["f"] == 0.0 and sep > 3.0:
                goto("abort", t)
                return "LOST THE TIP before the ejector — abort"
            if arrived and sep < 7.0:
                st["f"] = min(18.0, st["f"] + 8.0 * dt_)
                apply_wrench(d, ejb, force=st["f"] * axis_dn)
            st["ok"] = st["ok"] + 1 if (st["f"] > 0 and sep > 6.0) else 0
            if st["ok"] * dt_ > 0.25:
                log["eject_sep_mm"] = round(sep, 1)
                goto("plunge1", t)
            if el > 9.0:
                goto("abort", t)
            return (f"squeeze the ejector: the tip drops off "
                    f"({sep:.1f} mm)" if st["f"] > 0
                    else "carry it clear, then straight down")
        grip.hold(d, target_pos=[px + 0.075, py - 0.055, st["z"] + 0.070],
                  target_quat=qup)
        return "done — tip dropped beside the box, pipette empty"

    cam = free_camera([px, py, bx["tray_top_m"] + 0.06], 0.42, 125, -14)
    frames = rollout(model, data, seconds, fps, control, camera=cam,
                     width=640, height=480)
    out = write_video(frames, out_mp4, fps=fps)
    ok = (log["carry_gap_mm"] is not None and abs(log["carry_gap_mm"]) < 2.0
          and log["first_stop"] and log["second_stop"]
          and log["first_stop"]["mm"] > 0.7 * built["stroke1_mm"]
          and log["second_stop"]["mm"] > built["stroke1_mm"] + 2.0
          and (log["released_mm"] or 99) < 2.0
          and abs(log.get("gap_before_eject") or 99) < 2.5
          and log["eject_sep_mm"] is not None)
    verdict = ("PASS: snap-mounted under measured force, carried through a "
               "shake, both plunger stages at their designed depths, spring "
               "return, ejected by the ejector alone" if ok
               else f"FAIL: {log}")
    stills = save_stills(frames, Path(out_mp4).parent,
                         Path(out_mp4).stem, n=4)
    return {**out, "measured": log, "verdict": verdict, "stills": stills}


def demo_tip_box(built: dict, out_mp4: str | Path, seconds: float = 12.0,
                 fps: int = 30) -> dict:
    """Close the lid with a fingertip push, then reopen it; the friction
    hinge holds every pose, and the free tips ride it out in their wells."""
    import mujoco
    from .mjcf import rollout
    model = load(built["xml"])
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    name = built["name"]
    lb = model.body(f"{name}_lid").id
    bb = model.body(built["box_body"]).id
    ja = _jadr(model, f"{name}_hinge")
    log = {"closed_deg": None, "reopened_deg": None, "rest_deg": None}

    def control(t, m, d):
        clear_forces(d)
        deg = float(np.degrees(d.qpos[ja]))
        R = d.xmat[lb].reshape(3, 3)
        if 1.0 < t < 9.0:
            # the OTHER hand steadies the base: the lid push exceeds the
            # 1.2 N the whole box weighs, and without this the "open the
            # lid" force simply threw the box (measured: floor contact on
            # the lid top, then NaN)
            apply_wrench(d, bb, force=[0, 0, -4.0])
        # push at the lid's far (front) edge, along the lid's own normal
        edge = d.xpos[lb] + R @ np.array([0.0, -float(
            json.loads(Path(built["params_json"]).read_text())["width"]),
            float(built.get("lid_h", 0.045))])
        if t < 1.0:
            return f"the box as found: lid open at {deg:.0f} deg"
        if t < 4.0:
            if deg > 3.0:
                # eased and speed-guarded in BOTH directions: a constant
                # torque closes this lid at ~60 rad/s and the slam through
                # the free tips NaN'd the solver (measured)
                s = float(np.clip((deg - 8.0) / 40.0, 0.15, 1.0))
                w = float(np.degrees(d.qvel[m.jnt_dofadr[
                    m.joint(f"{built['name']}_hinge").id]]))
                if w > -140.0:
                    apply_wrench(d, lb, force=-2.2 * s * R[:, 2],
                                 point=edge, com=d.xpos[lb])
            else:
                log["closed_deg"] = round(deg, 1)
            return f"push the lid shut ({deg:.0f} deg)"
        if t < 5.0:
            log["closed_deg"] = round(deg, 1)
            return f"closed at {deg:.0f} deg"
        if t < 9.0:
            tgt = float(built["lid_start_deg"]) - 3.0
            if deg < tgt:
                # the push EASES OFF approaching the target — a constant
                # 3.6 N slams the lid through its 118-deg stop and the
                # impact NaNs the solver (measured)
                s = float(np.clip((tgt + 12.0 - deg) / 45.0, 0.12, 1.0))
                w = float(np.degrees(d.qvel[m.jnt_dofadr[
                    m.joint(f"{built['name']}_hinge").id]]))
                if w < 140.0:
                    apply_wrench(d, lb, force=3.0 * s * R[:, 2], point=edge,
                                 com=d.xpos[lb])
            else:
                log["reopened_deg"] = round(deg, 1)
            return f"lift it open again ({deg:.0f} deg)"
        log["rest_deg"] = round(deg, 1)
        return f"hands off — friction holds it at {deg:.0f} deg"

    p = json.loads(Path(built["params_json"]).read_text())
    cam = free_camera([0, 0, 0.06], 0.42, 115, -18)
    frames = rollout(model, data, seconds, fps, control, camera=cam,
                     width=640, height=480)
    out = write_video(frames, out_mp4, fps=fps)
    seated = []
    for tinfo in built["free_tips"]:
        tb = model.body(tinfo["body"]).id
        x, y = tipbox_well_xy(p, tinfo["row"], tinfo["col"])
        bb = model.body(built["box_body"]).id
        q = (data.xmat[bb].reshape(3, 3).T
             @ (data.xpos[tb] - data.xpos[bb]))
        seated.append(bool(np.hypot(q[0] - x, q[1] - y) < 0.004))
    ok = (log["closed_deg"] is not None and log["closed_deg"] < 6.0
          and log["reopened_deg"] is not None
          and log["rest_deg"] is not None
          and abs(log["rest_deg"] - log["reopened_deg"]) < 8.0
          and all(seated))
    verdict = ("PASS: pushed closed, pushed open, holds its pose on hinge "
               "friction alone, tips undisturbed" if ok
               else f"FAIL: {log} seated={seated}")
    stills = save_stills(frames, Path(out_mp4).parent,
                         Path(out_mp4).stem, n=4)
    return {**out, "measured": {**log, "tips_seated": seated},
            "verdict": verdict, "stills": stills}


def preview(built: dict, out_dir: str | Path) -> list[str]:
    import mujoco
    from PIL import Image
    model = load(built["xml"])
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    span = max(0.14, 1.5 * float(built.get("total_len_m", 0)
                                 or built.get("base_h", 0.1) * 2.2))
    cam = free_camera([0, 0, span / 5], max(0.2, 1.8 * span), 120, -22)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(model, height=480, width=640) as r:
        r.update_scene(data, camera=cam)
        pth = out_dir / f"{built['name']}_preview.png"
        Image.fromarray(r.render().copy()).save(pth)
    return [str(pth)]


# ---------------------------------------------------------------------------

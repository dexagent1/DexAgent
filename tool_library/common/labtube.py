"""MICROCENTRIFUGE TUBE FAMILY — the 1.5 ml flip-cap tube, the rack that
holds a grid of them, and the cylindrical decapper that pops them open.

THE POINT
---------
A lab tube is not "a small bottle": its lid is a FLIP CAP on a moulded
hinge, held shut by a bead on the plug that snaps past a lip inside the
bore. Opening one is a force event (a few newtons at the tab), and the
whole reason the DECAPPER exists is that thumbs are clumsy: you drop its
hollow skirt over the cap, twist so an internal ledge slides under the
cap's tab, and pry — the skirt's far rim presses DOWN on the tube's rim
flange while the ledge lifts the tab, so the tube itself feels almost no
net force and stays seated in the rack. All three of those facts are
mechanisms, none of them survives a scan, so they are CONSTRUCTED from
measured numbers exactly like the bottle thread and the latchbox snap.

HOW THE CAP SNAP IS MODELLED (the latchbox doctrine, miniaturised)
------------------------------------------------------------------
The physical give when a cap pops is the plug flexing. That give is an
explicit DOF, never contact squish:

    <hinge fold>     0 deg = CLOSED, +ve = flipping open
      <slide flex>   the plug's own radial springiness, 0 = relaxed

and the snap is geometry against that spring: a bead on the plug's far
side stands `bead - nub` clear of the bore's lip, so passing costs
`flex_stiffness x interference` newtons — printed at build time, the same
force in both directions, which is why a closed tube stays closed upside
down and why re-closing is also a snap.

No actuators exist in any of these assets: every motion in every demo is
an applied wrench (common.ForceGrip / apply_wrench).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .mjcf import (FLOOR_XML, LIGHTS_XML, ForceGrip, apply_wrench,
                     clear_forces, free_camera, load, merge_params,
                     mjcf_header, revolve_profile, rgba, save_params,
                     save_stills, shell_ring, vec, write_mesh, write_video)

# ---------------------------------------------------------------------------
# MICROTUBE — 1.5 ml Eppendorf-style flip-cap tube. Standard catalogue
# dimensions; the photo's tubes are these (white translucent, cap tab
# opposite the hinge).
# ---------------------------------------------------------------------------
TUBE_DEFAULTS = {
    "name": "microtube",
    "total_len":   0.0390,    # tip to rim top
    "body_r":      0.0054,    # cylindrical section outer radius
    "cone_len":    0.0175,    # conical lower section
    "tip_r":       0.0018,
    "rim_r":       0.0066,    # the rim FLANGE the rack (and decapper) bear on
    "rim_t":       0.0018,
    "bore_r":      0.0045,    # inner mouth the plug seals into
    "bore_depth":  0.0080,    # how deep the mouth is modelled hollow
    "cap_r":       0.0057,    # cap disc — SMALLER than rim_r so a decapper
                              #   skirt can swallow the cap yet rest on the rim
    "cap_t":       0.0030,
    "plug_r":      0.0038,
    "plug_depth":  0.0030,
    "tab_len":     0.0042,    # the pry tab, mounted at the disc's TOP edge
    "tab_w":       0.0050,
    "tab_t":       0.0013,
    "hinge_gap":   0.0022,    # rim edge -> hinge pivot (the moulded strap)
    # ---- the snap: bead on the plug vs lip in the bore. BOTH are capsules
    # (rounded), because the pop is a CAM action: two box corners meeting
    # square-on either wedge solid or slip for free (measured: 0.54 N
    # instead of the designed 4.5), while round-on-round converts the pry
    # smoothly into flex compression at the designed force. ----
    "bead_out":    0.0005,    # bead proud of the plug, radially (capsule r)
    "nub_in":      0.0007,    # bore lip proud of the bore wall (capsule r).
                              #   Kept at plug_r-flush: a lip proud of the
                              #   PLUG (not just the bead) squeezes it all
                              #   the time and the "closed" cap rested
                              #   visibly ajar at 3.5 deg (measured)
    "flex_travel": 0.0012,
    "flex_stiffness": 9000.0, # N/m: pop force = stiffness x interference
    "flex_damping": 6.0,
    "flex_armature": 0.010,   # what makes a sub-gram plug on a kN/m spring
                              #   integrable at 1 ms (latchbox's remedy)
    "fold_range_deg": 185.0,
    "fold_damping": 0.002,    # moulded hinges are LOSSY. At 0.0004 an
                              #   open cap swung as a near-undamped
                              #   pendulum for tens of seconds, and a rack
                              #   of them slowly pumped their own sockets
                              #   until a tube went non-finite at t~54 s
                              #   (measured, on the bench)
    "fold_frictionloss": 0.002,    # an opened cap must stay where it fell
    "fold_armature": 0.0005,
    "cap_start_deg": 0.0,     # 0 = closed; ~165 = lying fully open
    # ---- appearance / physics ----
    "tube_rgba":  [0.93, 0.94, 0.96, 0.55],
    "cap_rgba":   [0.93, 0.94, 0.96, 0.80],
    "density":    1050.0,
    "friction":   "0.7 0.01 0.0002",
    "pos":        [0.0, 0.0, 0.0],
    "yaw_deg":    0.0,
    "layout":     "lying",    # a lone tube cannot stand on its cone:
                              #   lying | upright (upright only makes sense
                              #   seated in a rack or a holder)
    "timestep":   0.001,
    "include_floor": True,
}

# geoms that ARE the snap: stiff, so the pop force is the flex spring's and
# not the solver's squish
_SNAP_ATTR = ('friction="0.4 0.01 0.0002" condim="3" '
              'solref="0.002 1" solimp="0.95 0.99 0.0005"')


def _tube_snap_numbers(p: dict) -> dict:
    """The three numbers that decide whether the cap can work at all, with
    loud errors — a cap that compiles but can never pop reads as a physics
    bug two hours later."""
    bead_outer = float(p["plug_r"]) + float(p["bead_out"])
    nub_inner = float(p["bore_r"]) - float(p["nub_in"])
    interf = bead_outer - nub_inner
    if bead_outer >= float(p["bore_r"]):
        raise ValueError(
            f"plug_r+bead_out = {1000*bead_outer:.2f} mm >= bore_r "
            f"{1000*float(p['bore_r']):.2f} mm — the beaded plug cannot enter "
            f"the bore at all. Shrink plug_r/bead_out or open bore_r.")
    if interf <= 0:
        raise ValueError(
            f"bead ({1000*bead_outer:.2f} mm) never reaches the bore lip "
            f"({1000*nub_inner:.2f} mm): interference {1000*interf:.2f} mm — "
            f"the cap closes but nothing holds it. Raise bead_out or nub_in.")
    if interf >= float(p["flex_travel"]):
        raise ValueError(
            f"snap interference {1000*interf:.2f} mm >= flex_travel "
            f"{1000*float(p['flex_travel']):.2f} mm — the plug cannot flex "
            f"far enough to ever pass the lip. Raise flex_travel.")
    if float(p["cap_r"]) >= float(p["rim_r"]):
        raise ValueError(
            f"cap_r {1000*float(p['cap_r']):.2f} mm >= rim_r "
            f"{1000*float(p['rim_r']):.2f} mm — a decapper skirt that "
            f"swallows the cap would swallow the rim too and land on the "
            f"rack instead of the flange it must pry against.")
    return {"interference_m": interf,
            "pop_force_N": float(p["flex_stiffness"]) * interf}


def _tube_body_xml(p: dict, prefix: str, pos, yaw_deg: float,
                   cap_deg: float, mesh_name: str,
                   euler: tuple | None = None) -> str:
    """One complete tube (free body) + its cap chain, as attachable XML.
    Local frame: z=0 at the TIP, +z to the rim; the cap hinge sits on the
    +x side, the tab and the snap on the -x side. `euler` overrides the
    plain yaw (a freejoint must sit on a top-level body, so a lying tube
    is pitched HERE, never by a wrapper body)."""
    tl, br, cl = float(p["total_len"]), float(p["body_r"]), float(p["cone_len"])
    rr, rt = float(p["rim_r"]), float(p["rim_t"])
    bore_r, bd = float(p["bore_r"]), float(p["bore_depth"])
    cr, ct = float(p["cap_r"]), float(p["cap_t"])
    pr, pd = float(p["plug_r"]), float(p["plug_depth"])
    tube_c, cap_c = rgba(p["tube_rgba"]), rgba(p["cap_rgba"])
    # 4 ms solref on every tube geom (the gram-scale rule, see TOOLS.md):
    # the default 20 ms let a hard ledge-drag bury the sub-gram tab and
    # catapult-NaN the cap on release (measured, on the bench)
    gA = (f'density="{p["density"]}" friction="{p["friction"]}" condim="3" '
          f'solref="0.004 1"')
    inv = 'rgba="0 0 0 0"'                    # collision under a mesh visual

    g = [f'      <geom name="{prefix}_vis" type="mesh" mesh="{mesh_name}" '
         f'rgba="{tube_c}" contype="0" conaffinity="0" group="2" '
         f'density="1"/>',
         f'      <geom name="{prefix}_tip" type="sphere" '
         f'size="{p["tip_r"]:.6f}" pos="0 0 {p["tip_r"]:.6f}" {inv} {gA}/>',
         f'      <geom name="{prefix}_conea" type="cylinder" '
         f'size="{0.55*br:.6f} {0.0035:.6f}" pos="0 0 {0.0065:.6f}" '
         f'{inv} {gA}/>',
         f'      <geom name="{prefix}_coneb" type="cylinder" '
         f'size="{0.85*br:.6f} {(cl-0.010)/2:.6f}" '
         f'pos="0 0 {(cl+0.010)/2:.6f}" {inv} {gA}/>',
         f'      <geom name="{prefix}_bodyc" type="cylinder" '
         f'size="{br:.6f} {(tl-bd-cl)/2:.6f}" '
         f'pos="0 0 {(tl-bd+cl)/2:.6f}" {inv} {gA}/>']
    # the hollow mouth: a shell the plug really enters
    g += shell_ring(f"{prefix}_bore", bore_r, br - bore_r + 0.0002,
                    tl - bd, tl - rt, "0 0 0 0", gA, n=10)
    g += shell_ring(f"{prefix}_rimf", bore_r, rr - bore_r,
                    tl - rt, tl, "0 0 0 0", gA, n=10)
    # the bore LIP the bead snaps past — far side (-x), 1 mm down the bore;
    # a capsule lying tangentially, so the bead CAMS over it
    g.append(f'      <geom name="{prefix}_lip" type="capsule" '
             f'size="{float(p["nub_in"]):.6f} 0.0018" euler="90 0 0" '
             f'pos="{vec((-bore_r, 0.0, tl - 0.0010))}" '
             f'rgba="0 0 0 0" {_SNAP_ATTR} density="{p["density"]}"/>')
    # the moulded hinge strap, visual
    hx = rr + float(p["hinge_gap"])
    g.append(f'      <geom name="{prefix}_strap" type="box" '
             f'size="{vec((p["hinge_gap"] / 2 + 0.0006, 0.0016, 0.0007))}" '
             f'pos="{vec((rr + float(p["hinge_gap"]) / 2, 0.0, tl - 0.0007))}" '
             f'rgba="{cap_c}" density="{p["density"]}"/>')

    # ---- cap chain: fold hinge at the rim top, flex slide inside --------
    start = float(cap_deg)
    ca, sa = np.cos(np.radians(start / 2)), np.sin(np.radians(start / 2))
    # the disc centre must land ON THE TUBE AXIS when closed, so measured
    # from the hinge it sits at -(rim_r + hinge_gap) — using cap_r here
    # once left the plug 0.9 mm off-axis and the bead never met the lip
    disc_cx = -hx
    # z in the cap frame: 0 IS the rim-top plane (the hinge height). The
    # disc lives ABOVE it (its underside rests on the rim), the plug hangs
    # BELOW it into the bore. Drawing the disc below the hinge plane buried
    # it inside the rim ring, and the "closed" cap rested visibly ajar on
    # that phantom contact (measured: 3.5 deg at rest).
    cap = [
        f'        <geom name="{prefix}_capdisc" type="cylinder" '
        f'size="{cr:.6f} {ct / 2:.6f}" '
        f'pos="{vec((disc_cx, 0.0, ct / 2))}" rgba="{cap_c}" {gA}/>',
        # the plug tapers (a square-edged cylinder wedges against the bore
        # ring on the way out — measured: the cap stuck at 17 deg)
        f'        <geom name="{prefix}_plug" type="cylinder" '
        f'size="{pr:.6f} {pd / 4:.6f}" '
        f'pos="{vec((disc_cx, 0.0, -pd / 4))}" rgba="{cap_c}" {gA}/>',
        f'        <geom name="{prefix}_plugtip" type="cylinder" '
        f'size="{0.84 * pr:.6f} {pd / 4:.6f}" '
        f'pos="{vec((disc_cx, 0.0, -3 * pd / 4))}" rgba="{cap_c}" {gA}/>',
        # the snap bead, 0.3 mm of free rise below the lip: any more arc
        # travel and the hinge circle's inward drift spends the radial
        # interference before it is contested (measured: pop at 0.22 N)
        f'        <geom name="{prefix}_bead" type="capsule" '
        f'size="{float(p["bead_out"]):.6f} 0.0016" euler="90 0 0" '
        f'pos="{vec((disc_cx - pr, 0.0, -0.0025))}" '
        f'rgba="{cap_c}" {_SNAP_ATTR} density="{p["density"]}"/>',
        # the pry tab at the disc's TOP edge — what the decapper ledge lifts
        f'        <geom name="{prefix}_tab" type="box" '
        f'size="{vec((p["tab_len"] / 2, p["tab_w"] / 2, p["tab_t"] / 2))}" '
        f'pos="{vec((disc_cx - cr - float(p["tab_len"]) / 2 + 0.0006, 0.0, ct - float(p["tab_t"]) / 2))}" '
        f'rgba="{cap_c}" {gA}/>']
    ex, ey, ez = euler if euler is not None else (0.0, 0.0, yaw_deg)
    return f"""
    <body name="{prefix}" pos="{vec(pos)}"
          euler="{ex:.3f} {ey:.3f} {ez:.3f}">
      <freejoint name="{prefix}_free"/>
{chr(10).join(g)}
      <body name="{prefix}_cap" pos="{vec((hx, 0.0, tl))}"
            quat="{ca:.8f} 0 {sa:.8f} 0">
        <joint name="{prefix}_fold" type="hinge" axis="0 1 0"
               range="-2 {p['fold_range_deg']}" ref="{start:.3f}"
               damping="{p['fold_damping']}"
               frictionloss="{p['fold_frictionloss']}"
               armature="{p['fold_armature']}"/>
        <inertial pos="{vec((disc_cx, 0.0, -0.002))}" mass="0.0003"
                  diaginertia="2e-9 2e-9 2e-9"/>
        <body name="{prefix}_capface">
          <joint name="{prefix}_flex" type="slide" axis="1 0 0"
                 range="0 {p['flex_travel']}"
                 stiffness="{p['flex_stiffness']}"
                 damping="{p['flex_damping']}"
                 armature="{p['flex_armature']}"
                 solreflimit="0.004 1" solimplimit="0.99 0.999 0.0002"/>
{chr(10).join(cap)}
        </body>
      </body>
    </body>"""


def _tube_mesh(p: dict, out_dir: Path, name: str) -> str:
    """The lathe visual: closed cone-to-rim outline with a real open mouth."""
    tl, br, cl = float(p["total_len"]), float(p["body_r"]), float(p["cone_len"])
    rr, rt = float(p["rim_r"]), float(p["rim_t"])
    bore_r, bd = float(p["bore_r"]), float(p["bore_depth"])
    prof = [(1e-4, 0.0), (float(p["tip_r"]), 0.0008), (0.85 * br, cl),
            (br, cl + 0.002), (br, tl - rt), (rr, tl - rt), (rr, tl),
            (bore_r, tl), (bore_r, tl - bd), (1e-4, tl - bd)]
    return write_mesh(revolve_profile(prof, sections=40),
                      out_dir / "meshes" / f"{name}.obj")


def build_microtube(params: dict | None = None,
                    out_dir: str | Path = ".") -> dict:
    p = merge_params(TUBE_DEFAULTS, params)
    name = str(p["name"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snap = _tube_snap_numbers(p)
    _tube_mesh(p, out_dir, f"{name}_lathe")

    pos = np.asarray(p["pos"], float)
    if str(p["layout"]) == "lying":
        # resting on its side: the tube body itself is pitched 90 deg (a
        # freejoint may only sit on a top-level body, so no wrapper), and
        # the cylinder's own radius is the footprint height
        wrap = _tube_body_xml(
            p, name, [pos[0], pos[1], pos[2] + float(p["body_r"])], 0.0,
            float(p["cap_start_deg"]), f"{name}_lathe_m",
            euler=(0.0, 90.0, float(p["yaw_deg"])))
    else:
        wrap = _tube_body_xml(p, name, pos, float(p["yaw_deg"]),
                              float(p["cap_start_deg"]), f"{name}_lathe_m")
    floor = FLOOR_XML if p["include_floor"] else ""
    xml = (mjcf_header(f"{name}_microtube", timestep=p["timestep"],
                       meshdir="meshes")
           + f"""  <asset>
    <mesh name="{name}_lathe_m" file="{name}_lathe.obj"/>
  </asset>
  <worldbody>{LIGHTS_XML}{floor}
{wrap}
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
        "name": name, "kind": "microtube",
        "tube_body": name, "cap_body": f"{name}_cap",
        "fold_joint": f"{name}_fold", "flex_joint": f"{name}_flex",
        "closed_deg": 0.0, "open_deg": float(p["fold_range_deg"]),
        "snap_interference_mm": round(1000 * snap["interference_m"], 3),
        "snap_force_N": round(snap["pop_force_N"], 2),
        "tube_mass_kg": round(float(m.body(name).subtreemass[0]), 5),
        "rim_top_m": round(float(p["total_len"]), 5),
        "rim_r_m": float(p["rim_r"]), "cap_r_m": float(p["cap_r"]),
        "tab_reach_m": round(float(p["hinge_gap"]) + 2 * float(p["cap_r"])
                             + float(p["tab_len"]), 5),
        "note": (f"fold joint: 0 deg = closed, +ve = open; the flex slide is "
                 f"the plug's own give. Popping the cap costs "
                 f"{snap['pop_force_N']:.1f} N at the bead "
                 f"({1000*snap['interference_m']:.2f} mm past the bore lip), "
                 f"both directions — a closed tube stays closed upside down. "
                 f"The tab (for the decapper's ledge) is at the disc TOP on "
                 f"the far side from the hinge."),
    }
    (out_dir / f"{name}_built.json").write_text(json.dumps(built, indent=1))
    return built


# ---------------------------------------------------------------------------
# TUBE RACK — the blue translucent 80/96-well grid. Wells that must really
# accept a tube get a square through-opening + a guide shaft (sockets);
# every other well is a solid cell with a moulded-hole visual, because 96
# real shafts is ~500 needless geoms in a scene that inserts into three.
# ---------------------------------------------------------------------------
RACK_DEFAULTS = {
    "name": "tuberack",
    "rows": 5, "cols": 16,          # the classic 80-well 2:1 blue rack
    "pitch": 0.0138,
    "length": 0.245, "width": 0.105,
    "height": 0.052,
    "plate_t": 0.0070,
    "wall_t": 0.0035,
    "hole_r": 0.0056,               # moulded hole look (visual)
    "socket_half": 0.0056,          # square opening half-width at sockets:
                                    #   passes body_r, catches rim_r
    "shaft_depth": 0.018,
    "rack_rgba": [0.13, 0.30, 0.86, 0.62],
    "hole_rgba": [0.03, 0.07, 0.30, 0.95],
    "density": 380.0,               # ribbed polypropylene, mostly air:
                                    #   chosen so the compiled rack lands
                                    #   near the real ~150 g
    "friction": "0.85 0.015 0.0004",
    "rack_free": True,
    "tubes": [],                    # [{row, col, cap_deg, yaw_deg}] seated
    "open_sockets": [],             # [[row, col]] wells given a REAL opening
                                    #   + guide shaft but no tube — where a
                                    #   demo (or a robot) will insert one
    "tube_params": {},              # overrides for the seated tubes' build
    "pos": [0.0, 0.0, 0.0],
    "yaw_deg": 0.0,
    "timestep": 0.001,
    "include_floor": True,
}


def rack_well_xy(p: dict, row: int, col: int) -> tuple[float, float]:
    """Well centre in the rack's own frame (x along length, origin centre)."""
    pitch = float(p["pitch"])
    x0 = -(float(p["cols"]) - 1) / 2 * pitch
    y0 = -(float(p["rows"]) - 1) / 2 * pitch
    return x0 + int(col) * pitch, y0 + int(row) * pitch


def build_tube_rack(params: dict | None = None,
                    out_dir: str | Path = ".") -> dict:
    p = merge_params(RACK_DEFAULTS, params)
    name = str(p["name"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, cols, pitch = int(p["rows"]), int(p["cols"]), float(p["pitch"])
    L, W, H = float(p["length"]), float(p["width"]), float(p["height"])
    pt, wt = float(p["plate_t"]), float(p["wall_t"])
    sh, sd = float(p["socket_half"]), float(p["shaft_depth"])
    rc, hc = rgba(p["rack_rgba"]), rgba(p["hole_rgba"])
    gA = (f'density="{p["density"]}" friction="{p["friction"]}" condim="3" '
          f'solref="0.004 1"')
    if cols * pitch > L - 2 * wt or rows * pitch > W - 2 * wt:
        raise ValueError(f"{rows}x{cols} wells at {1000*pitch:.1f} mm pitch "
                         f"do not fit a {1000*L:.0f}x{1000*W:.0f} mm rack")

    tp = merge_params(TUBE_DEFAULTS, dict(p["tube_params"],
                                          include_floor=False))
    _tube_snap_numbers(tp)
    mesh = _tube_mesh(tp, out_dir, f"{name}_tubelathe")
    rim_seat = H  # tube rim flange rests ON the plate top
    tube_z = rim_seat - (float(tp["total_len"]) - float(tp["rim_t"]))

    sockets = {(int(t["row"]), int(t["col"])) for t in p["tubes"]}
    sockets |= {(int(r), int(c)) for r, c in p["open_sockets"]}
    for t in p["tubes"]:
        if not (0 <= int(t["row"]) < rows and 0 <= int(t["col"]) < cols):
            raise ValueError(f"tube at ({t['row']},{t['col']}) is outside the "
                             f"{rows}x{cols} grid")

    g = []
    # walls + a low internal floor (tips hang well above it)
    for i, (sx, sy, px, py) in enumerate((
            (L / 2, wt / 2, 0.0, W / 2 - wt / 2),
            (L / 2, wt / 2, 0.0, -(W / 2 - wt / 2)),
            (wt / 2, W / 2 - wt, L / 2 - wt / 2, 0.0),
            (wt / 2, W / 2 - wt, -(L / 2 - wt / 2), 0.0))):
        g.append(f'      <geom name="{name}_wall{i}" type="box" '
                 f'size="{vec((sx, sy, H / 2))}" '
                 f'pos="{vec((px, py, H / 2))}" rgba="{rc}" {gA}/>')
    g.append(f'      <geom name="{name}_base" type="box" '
             f'size="{vec((L / 2, W / 2, 0.0015))}" pos="0 0 0.0015" '
             f'rgba="{rc}" {gA}/>')
    # the top plate, cell by cell
    zc = H - pt / 2
    for r in range(rows):
        for c in range(cols):
            x, y = rack_well_xy(p, r, c)
            if (r, c) in sockets:
                # four edge strips leave a through-opening, and four 45-deg
                # chamfer boxes make it an OCTAGON: a square hole's diagonal
                # is 1.41x its width, and a shaken tube's rim found exactly
                # that corner and left through it (measured: every tube on
                # the floor). The octagon caps the escape radius everywhere.
                s = (pitch / 2 - sh) / 2
                for k, (ex, ey, hx, hy) in enumerate((
                        (x, y + sh + s, pitch / 2, s),
                        (x, y - sh - s, pitch / 2, s),
                        (x + sh + s, y, s, sh),
                        (x - sh - s, y, s, sh))):
                    g.append(f'      <geom name="{name}_c{r}_{c}_{k}" '
                             f'type="box" size="{vec((hx, hy, pt / 2))}" '
                             f'pos="{vec((ex, ey, zc))}" rgba="{rc}" {gA}/>')
                # guide shaft below the opening (with the same chamfers,
                # slightly wider so the plate rim is what actually bears)
                for k, (ex, ey, hx, hy) in enumerate((
                        (x, y + sh + 0.001, sh + 0.002, 0.001),
                        (x, y - sh - 0.001, sh + 0.002, 0.001),
                        (x + sh + 0.001, y, 0.001, sh),
                        (x - sh - 0.001, y, 0.001, sh))):
                    g.append(f'      <geom name="{name}_s{r}_{c}_{k}" '
                             f'type="box" size="{vec((hx, hy, sd / 2))}" '
                             f'pos="{vec((ex, ey, H - pt - sd / 2))}" '
                             f'rgba="{rc}" {gA}/>')
                for k in range(4):
                    th = np.radians(45.0 + 90.0 * k)
                    # euler = th exactly: the THIN side must point radially
                    # (th+90 lays the long side across the well — 2 mm
                    # inside the hanging tube, held only by symmetry)
                    for tag, rad, hz, zz in (
                            ("k", sh + 0.0009, pt / 2, zc),
                            ("q", sh + 0.0014, sd / 2, H - pt - sd / 2)):
                        g.append(
                            f'      <geom name="{name}_{tag}{r}_{c}_{k}" '
                            f'type="box" '
                            f'size="{vec((0.0008, 0.55 * sh, hz))}" '
                            f'pos="{vec((x + rad * np.cos(th), y + rad * np.sin(th), zz))}" '
                            f'euler="0 0 {np.degrees(th):.2f}" '
                            f'rgba="{rc}" {gA}/>')
            else:
                g.append(f'      <geom name="{name}_c{r}_{c}" type="box" '
                         f'size="{vec((pitch / 2, pitch / 2, pt / 2))}" '
                         f'pos="{vec((x, y, zc))}" rgba="{rc}" {gA}/>')
                g.append(f'      <geom name="{name}_h{r}_{c}" '
                         f'type="cylinder" '
                         f'size="{p["hole_r"]:.6f} 0.0004" '
                         f'pos="{vec((x, y, H + 0.0004))}" rgba="{hc}" '
                         f'contype="0" conaffinity="0" density="1"/>')
    # margins between the well field and the walls
    fx = (cols * pitch) / 2
    fy = (rows * pitch) / 2
    for i, (ex, ey, hx, hy) in enumerate((
            (0.0, (fy + (W / 2)) / 2, L / 2 - wt, (W / 2 - fy) / 2),
            (0.0, -(fy + (W / 2)) / 2, L / 2 - wt, (W / 2 - fy) / 2),
            ((fx + L / 2) / 2, 0.0, (L / 2 - fx) / 2, fy),
            (-(fx + L / 2) / 2, 0.0, (L / 2 - fx) / 2, fy))):
        if hx > 1e-4 and hy > 1e-4:
            g.append(f'      <geom name="{name}_m{i}" type="box" '
                     f'size="{vec((hx, hy, pt / 2))}" '
                     f'pos="{vec((ex, ey, zc))}" rgba="{rc}" {gA}/>')

    tubes_xml, tube_list = [], []
    for i, t in enumerate(p["tubes"]):
        x, y = rack_well_xy(p, int(t["row"]), int(t["col"]))
        pref = f"{name}_tube{i}"
        tubes_xml.append(_tube_body_xml(
            tp, pref, [x + float(p["pos"][0]), y + float(p["pos"][1]),
                       tube_z + float(p["pos"][2])],
            float(t.get("yaw_deg", 0.0)), float(t.get("cap_deg", 0.0)),
            f"{name}_tubelathe_m"))
        tube_list.append({"body": pref, "cap_body": f"{pref}_cap",
                          "fold_joint": f"{pref}_fold",
                          "row": int(t["row"]), "col": int(t["col"]),
                          "cap_deg": float(t.get("cap_deg", 0.0)),
                          "yaw_deg": float(t.get("yaw_deg", 0.0))})

    free = f'      <freejoint name="{name}_free"/>\n' if p["rack_free"] else ""
    floor = FLOOR_XML if p["include_floor"] else ""
    xml = (mjcf_header(f"{name}_rack", timestep=p["timestep"],
                       meshdir="meshes")
           + f"""  <asset>
    <mesh name="{name}_tubelathe_m" file="{name}_tubelathe.obj"/>
  </asset>
  <worldbody>{LIGHTS_XML}{floor}
    <body name="{name}" pos="{vec(p['pos'])}" euler="0 0 {p['yaw_deg']}">
{free}{chr(10).join(g)}
    </body>
{chr(10).join(tubes_xml)}
  </worldbody>
</mujoco>
""")
    xml_path = out_dir / f"{name}.xml"
    xml_path.write_text(xml)
    m = load(xml_path)
    save_params(out_dir / f"{name}_params.json", p)
    snap = _tube_snap_numbers(tp)
    built = {
        "xml": str(xml_path),
        "params_json": str(out_dir / f"{name}_params.json"),
        "name": name, "kind": "tuberack",
        "rack_body": name, "tubes": tube_list,
        "rows": rows, "cols": cols, "pitch_m": pitch,
        "plate_top_m": H, "tube_seat_z_m": round(tube_z, 5),
        "socket_wells": sorted(list(sockets)),
        "rack_mass_kg": round(float(m.body(name).subtreemass[0]), 4),
        "tube_mass_kg": (round(float(m.body(tube_list[0]["body"])
                                     .subtreemass[0]), 5)
                         if tube_list else None),
        "snap_force_N": round(snap["pop_force_N"], 2),
        "note": (f"tubes HANG by the rim flange on the plate at z="
                 f"{H:.3f} m; wells listed in socket_wells have real "
                 f"through-openings + guide shafts, all other cells are "
                 f"solid with moulded-hole visuals. rack_well_xy(params, "
                 f"row, col) gives any well centre for placing/inserting."),
    }
    (out_dir / f"{name}_built.json").write_text(json.dumps(built, indent=1))
    return built


# ---------------------------------------------------------------------------
# TUBE OPENER (decapper) — a hollow two-tier cylinder. The skirt swallows
# the CAP but not the RIM FLANGE, so pressing down pins the tube; a window
# in the skirt admits the cap's tab; the low ledge beside the window slides
# UNDER the tab on a twist, and a tilt then pries the cap open against its
# own snap while the far rim bears down on the flange — net force on the
# tube ~ zero, which is why the tube stays in the rack.
# ---------------------------------------------------------------------------
OPENER_DEFAULTS = {
    "name": "tubeopener",
    "skirt_ir": 0.0061,       # MUST be between the tube's cap_r and rim_r
    "skirt_t": 0.0024,
    "skirt_h": 0.0150,
    "window_deg": [322.0, 38.0],   # the tab slot, centred on +x
    "ledge_deg": [38.0, 76.0],     # the low step beside it (twist CCW to
                                   #   slide it under the tab)
    "ledge_h": 0.0012,
    "ledge_slot_h": 0.0075,        # clearance above the ledge the tab rises
                                   #   through while prying: the tab needs
                                   #   ~0.3 mm of rise per degree of cap
                                   #   fold, and a 4 mm slot stopped the pry
                                   #   at 7 deg — under the snap's own
                                   #   release angle (measured)
    "flare_out": 0.0024,           # the funnel legs under the rim: how far
                                   #   they flare outward...
    "flare_drop": 0.0008,          # ...and how far below the rim their
                                   #   CENTRELINE ends (the capsule radius
                                   #   0.8 mm hangs further). The skirt-to-
                                   #   cap clearance is 0.5 mm; no force-
                                   #   guided descent centres that well, so
                                   #   round legs cam the cap in from +-2 mm.
                                   #   drop+radius MUST stay under the
                                   #   tube's rim_t, or the legs land on the
                                   #   rack plate and the rim never reaches
                                   #   the flange it pries against (measured)
    "collar_h": 0.0090,            # full ring above the windows
    "grip_r": 0.0075,
    "grip_h": 0.0170,
    "opener_rgba": [0.13, 0.38, 0.85, 1.0],
    "density": 1150.0,
    "friction": "0.8 0.015 0.0004",
    "pos": [0.0, 0.0, 0.0],
    "yaw_deg": 0.0,
    "timestep": 0.001,
    "include_floor": True,
}


def build_tube_opener(params: dict | None = None,
                      out_dir: str | Path = ".") -> dict:
    p = merge_params(OPENER_DEFAULTS, params)
    name = str(p["name"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ir, t = float(p["skirt_ir"]), float(p["skirt_t"])
    sh, ch = float(p["skirt_h"]), float(p["collar_h"])
    lh, lsh = float(p["ledge_h"]), float(p["ledge_slot_h"])
    oc = rgba(p["opener_rgba"])
    gA = (f'density="{p["density"]}" friction="{p["friction"]}" condim="3" '
          f'solref="0.004 1"')
    w0, w1 = [float(v) for v in p["window_deg"]]
    l0, l1 = [float(v) for v in p["ledge_deg"]]

    # skirt wall with BOTH cutouts absent, then the ledge and the upper
    # bridge added back over the ledge span
    g = shell_ring(f"{name}_skirt", ir, t, 0.0, sh, oc, gA, n=14,
                   gap_deg=(w0, l1))
    lmid = np.radians((l0 + l1) / 2)
    ln = 2 * (ir + t / 2) * np.tan(np.radians((l1 - l0) / 2)) * 0.9
    for tag, z0, z1 in ((f"{name}_ledge", 0.0, lh),
                        (f"{name}_bridge", lh + lsh, sh)):
        g.append(f'      <geom name="{tag}" type="box" '
                 f'size="{vec((t / 2, ln / 2, (z1 - z0) / 2))}" '
                 f'pos="{vec(((ir + t / 2) * np.cos(lmid), (ir + t / 2) * np.sin(lmid), (z0 + z1) / 2))}" '
                 f'euler="0 0 {np.degrees(lmid):.3f}" '
                 f'rgba="{oc}" {_SNAP_ATTR} density="{p["density"]}"/>')
    # funnel legs below the rim (capsules: round, so they CAM the cap
    # toward the bore instead of catching it on a square step)
    fo, fd = float(p["flare_out"]), float(p["flare_drop"])
    for i in range(14):
        ang = 360.0 * (i + 0.5) / 14
        a = (ang - w0) % 360.0
        if a <= (l1 - w0) % 360.0:
            continue
        th = np.radians(ang)
        c, s = np.cos(th), np.sin(th)
        g.append(f'      <geom name="{name}_leg{i}" type="capsule" '
                 f'size="0.0008" fromto="'
                 f'{vec(((ir + 0.0002) * c, (ir + 0.0002) * s, 0.0012))} '
                 f'{vec(((ir + fo) * c, (ir + fo) * s, -fd))}" '
                 f'rgba="{oc}" {gA}/>')
    # full collar above the windows, then the grip
    g += shell_ring(f"{name}_collar", ir, t, sh, sh + ch, oc, gA, n=14)
    g.append(f'      <geom name="{name}_shoulder" type="cylinder" '
             f'size="{ir + t:.6f} 0.0012" '
             f'pos="0 0 {sh + ch + 0.0012:.6f}" rgba="{oc}" {gA}/>')
    g.append(f'      <geom name="{name}_grip" type="cylinder" '
             f'size="{p["grip_r"]:.6f} {float(p["grip_h"]) / 2:.6f}" '
             f'pos="0 0 {sh + ch + 0.0024 + float(p["grip_h"]) / 2:.6f}" '
             f'rgba="{oc}" {gA}/>')
    g.append(f'      <geom name="{name}_dimple" type="cylinder" '
             f'size="{0.55 * float(p["grip_r"]):.6f} 0.0004" '
             f'pos="0 0 {sh + ch + 0.0028 + float(p["grip_h"]):.6f}" '
             f'rgba="{rgba([0.06, 0.16, 0.5, 1.0])}" contype="0" '
             f'conaffinity="0" density="1"/>')

    floor = FLOOR_XML if p["include_floor"] else ""
    xml = (mjcf_header(f"{name}_opener", timestep=p["timestep"]) + f"""
  <worldbody>{LIGHTS_XML}{floor}
    <body name="{name}" pos="{vec(p['pos'])}" euler="0 0 {p['yaw_deg']}">
      <freejoint name="{name}_free"/>
{chr(10).join(g)}
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
        "name": name, "kind": "tubeopener",
        "opener_body": name,
        "skirt_ir_m": ir, "skirt_h_m": sh,
        "window_center_deg": (w0 + (w1 - w0) % 360 / 2) % 360,
        "engage_twist_deg": -((l0 + l1) / 2 - (w0 + (w1 - w0) % 360 / 2)
                              % 360),
        "ledge_top_m": lh,
        "total_h_m": round(sh + ch + 0.0024 + float(p["grip_h"]) + 0.0008, 4),
        "mass_kg": round(float(m.body(name).subtreemass[0]), 4),
        "note": (f"USE: window (centred {((w0 + (w1 - w0) % 360 / 2) % 360):.0f} deg "
                 f"in the opener frame) over the cap's tab, lower until the "
                 f"skirt rests on the tube's rim flange, twist "
                 f"{(l0 + l1) / 2 - ((w0 + (w1 - w0) % 360 / 2) % 360):.0f} deg "
                 f"CCW so the ledge slides under the tab, then tilt about "
                 f"the far rim: the ledge lifts the tab, the far rim presses "
                 f"the flange, the cap pops, the tube stays put."),
    }
    (out_dir / f"{name}_built.json").write_text(json.dumps(built, indent=1))
    return built


def rebuild(params_json: str | Path, out_dir: str | Path | None = None) -> dict:
    """Any of the three, from its own params file."""
    params_json = Path(params_json)
    p = json.loads(params_json.read_text())
    out = out_dir or params_json.parent
    if "skirt_ir" in p:
        return build_tube_opener(p, out)
    if "rows" in p:
        return build_tube_rack(p, out)
    return build_microtube(p, out)


# ---------------------------------------------------------------------------
# state readout + demos
# ---------------------------------------------------------------------------
def _jadr(model, suffix):
    for i in range(model.njnt):
        n = model.joint(i).name or ""
        if n == suffix or n.endswith(suffix):
            return int(model.joint(i).qposadr[0])
    return None


def cap_state(model, data, tube_prefix: str) -> dict:
    """The gate a tube demo is scored on: the fold angle and the flex bow."""
    import mujoco
    mujoco.mj_forward(model, data)
    fold = _jadr(model, f"{tube_prefix}_fold")
    flex = _jadr(model, f"{tube_prefix}_flex")
    return {"fold_deg": (round(float(np.degrees(data.qpos[fold])), 2)
                         if fold is not None else None),
            "flex_mm": (round(float(1000 * data.qpos[flex]), 3)
                        if flex is not None else None)}


def tube_seated(model, data, built_rack: dict, tube: dict) -> dict:
    """Is this tube still hanging in its own well? Measured in the RACK's
    frame so a rack that slid during the demo does not fail its tubes."""
    import mujoco
    mujoco.mj_forward(model, data)

    def _b(suffix):
        for i in range(model.nbody):
            n = model.body(i).name or ""
            if n == suffix or n.endswith(suffix):
                return i
        raise KeyError(suffix)
    rb = _b(built_rack["rack_body"])
    tb = _b(tube["body"])
    pr = data.xpos[rb]
    Rr = data.xmat[rb].reshape(3, 3)
    q = Rr.T @ (data.xpos[tb] - pr)
    x, y = rack_well_xy(json.loads(Path(built_rack["params_json"])
                                   .read_text()),
                        tube["row"], tube["col"])
    lat = float(np.hypot(q[0] - x, q[1] - y))
    up = float((Rr.T @ data.xmat[tb].reshape(3, 3))[2, 2])
    return {"lateral_off_mm": round(1000 * lat, 2),
            "upright_cos": round(up, 4),
            "seated": bool(lat < 0.004 and up > 0.9)}


def demo_microtube(built: dict, out_mp4: str | Path, seconds: float = 9.0,
                   fps: int = 30) -> dict:
    """Pop the cap open with a pry force under the tab (a thumbnail), let it
    fall open, then snap it shut again with a press — while a second wrench
    pins the tube the way a real thumb-and-fingers hold does. Forces only."""
    import mujoco
    from .mjcf import rollout
    # a one-well rack IS the holder — the natural single-tube situation
    rp = {"name": "demo1w", "rows": 1, "cols": 1, "pitch": 0.016,
          "length": 0.05, "width": 0.05, "height": 0.045,
          "tubes": [{"row": 0, "col": 0, "cap_deg": 0.0}],
          "tube_params": {k: v for k, v in
                          json.loads(Path(built["params_json"]).read_text())
                          .items() if k in
                          {"total_len", "body_r", "cone_len", "tip_r",
                           "rim_r", "rim_t", "bore_r", "cap_r", "cap_t",
                           "plug_r", "plug_depth", "bead_out", "nub_in",
                           "flex_travel", "flex_stiffness", "tube_rgba",
                           "cap_rgba"}},
          "rack_free": False}
    tmp = Path(built["xml"]).parent / "demo_world"
    rk = build_tube_rack(rp, tmp)
    model = load(rk["xml"])
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    tube = rk["tubes"][0]
    tb = model.body(tube["body"]).id
    cb = model.body(f"{tube['body']}_capface").id
    tab_g = model.geom(f"{tube['body']}_tab").id
    disc_g = model.geom(f"{tube['body']}_capdisc").id
    # The holding hand is a CONSTANT press, not a spring: a spring stiff
    # enough to out-pull the 7 N pry is unstable on a 3-gram body at this
    # timestep (explicit xfrc), and a stable one lets the tube ride 2 cm
    # up before it balances. A thumb pressing straight down + the rack's
    # own guide shaft is exactly how a person does it, and the shaft —
    # geometry, not gains — carries the side loads.
    log = {"popped_at_N": None, "max_open_deg": 0.0, "reclosed_deg": None}

    def control(t, m, d):
        clear_forces(d)
        fold = float(np.degrees(d.qpos[_jadr(m, f"{tube['body']}_fold")]))
        log["max_open_deg"] = max(log["max_open_deg"], fold)
        apply_wrench(d, tb, force=[0.0, 0.0, -10.0])
        cz = d.xmat[cb].reshape(3, 3)[:, 2]      # cap face normal
        if t < 4.0:
            f = min(7.0, 3.0 * t)             # ramp a pry force under the tab
            if fold < 100.0:
                apply_wrench(d, cb, force=f * cz,
                             point=d.geom_xpos[tab_g], com=d.xpos[cb])
                if log["popped_at_N"] is None and fold > 6.0:
                    log["popped_at_N"] = round(f, 2)
            return f"pry under the tab: {f:.1f} N (cap {fold:.0f} deg)"
        if t < 5.5:
            return f"released — cap rests at {fold:.0f} deg"
        if t < 8.2:                            # press it shut again — at the
            # TAB, the longest lever over the bead, exactly where a thumb
            # lands (pressing mid-disc left the bead parked ON the lip)
            apply_wrench(d, cb, force=-8.0 * cz,
                         point=d.geom_xpos[tab_g], com=d.xpos[cb])
            return f"press shut: cap {fold:.0f} deg"
        log["reclosed_deg"] = round(fold, 1)
        return f"closed again at {fold:.0f} deg"

    cam = free_camera([0.0, 0.0, rk["plate_top_m"] + 0.01], 0.16, 135, -22)
    frames = rollout(model, data, seconds, fps, control, camera=cam,
                     width=640, height=480)
    out = write_video(frames, out_mp4, fps=fps)
    end = cap_state(model, data, tube["body"])
    ok = (log["max_open_deg"] > 80.0 and log["reclosed_deg"] is not None
          and abs(log["reclosed_deg"]) < 8.0)
    measured = {**log, "end_state": end,
                "snap_force_designed_N": built["snap_force_N"]}
    verdict = ("PASS: popped open past 80 deg under a measured pry force and "
               "snapped shut again" if ok else
               f"FAIL: max_open {log['max_open_deg']:.0f} deg, reclosed "
               f"{log['reclosed_deg']}")
    stills = save_stills(frames, Path(out_mp4).parent,
                         Path(out_mp4).stem, n=4)
    return {**out, "measured": measured, "verdict": verdict, "stills": stills}


def demo_rack(built: dict, out_mp4: str | Path, seconds: float = 10.0,
              fps: int = 30) -> dict:
    """Shake the loaded rack, then drop a spare tube into a socket well from
    a height — retention and insertion are the two things a rack is for."""
    import mujoco
    from .mjcf import rollout
    p = json.loads(Path(built["params_json"]).read_text())
    if not built["tubes"]:
        raise ValueError("demo_rack needs at least one seated tube")
    # the drop-in tube gets its OWN, EMPTY socket well (dropping onto an
    # occupied well is a collision, not an insertion)
    occupied = {(t["row"], t["col"]) for t in built["tubes"]}
    empty = [(r, c) for r, c in built["socket_wells"]
             if (r, c) not in occupied]
    if empty:
        drop_rc = empty[0]
        p2 = dict(p, include_floor=True)
    else:
        drop_rc = next((r, c) for r in range(int(p["rows"]))
                       for c in range(int(p["cols"]))
                       if (r, c) not in occupied)
        p2 = dict(p, include_floor=True,
                  open_sockets=list(p["open_sockets"]) + [list(drop_rc)])
    x, y = rack_well_xy(p, *drop_rc)
    tmp = Path(built["xml"]).parent / "demo_world"
    rk = build_tube_rack(p2, tmp)
    # splice the drop tube into the world xml (same lathe mesh)
    tp = merge_params(TUBE_DEFAULTS, dict(p["tube_params"],
                                          include_floor=False))
    drop = _tube_body_xml(tp, f"{p['name']}_drop", [x, y, rk["plate_top_m"]
                                                    + 0.030],
                          0.0, 0.0, f"{p['name']}_tubelathe_m")
    xml_p = Path(rk["xml"])
    xml_p.write_text(xml_p.read_text().replace(
        "  </worldbody>", f"{drop}\n  </worldbody>"))
    model = load(xml_p)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    rb = model.body(rk["rack_body"]).id
    grip = ForceGrip(model, rb, kp=260.0, kd=22.0, max_force=60.0,
                     kr=4.0, kdr=0.35, max_torque=4.0)
    p0 = data.xpos[rb].copy()
    q0 = data.xquat[rb].copy()

    def control(t, m, d):
        clear_forces(d)
        if t < 1.0:
            return "settle"
        if t < 5.0:                     # lateral shake, growing amplitude
            a = 0.009 * min(1.0, (t - 1.0) / 1.0)
            dx = a * np.sin(2 * np.pi * 1.5 * (t - 1.0))
            grip.hold(d, target_pos=p0 + [dx, 0, 0], target_quat=q0)
            return f"shake the rack ({1000*a:.0f} mm at 1.5 Hz)"
        if t < 7.0:                     # tilt it 15 deg and back
            u = np.sin(np.pi * (t - 5.0) / 2.0)
            th = np.radians(15.0 * u)
            qt = np.array([np.cos(th / 2), 0, np.sin(th / 2), 0])
            q = np.zeros(4)
            mujoco.mju_mulQuat(q, qt, q0)
            grip.hold(d, target_pos=p0 + [0, 0, 0.004 * u], target_quat=q)
            return f"tilt {np.degrees(th):.0f} deg"
        grip.hold(d, target_pos=p0, target_quat=q0)
        return "set down — did every tube stay seated?"

    cam = free_camera([p0[0], p0[1], 0.05], 0.42, 115, -24)
    frames = rollout(model, data, seconds, fps, control, camera=cam,
                     width=640, height=480)
    out = write_video(frames, out_mp4, fps=fps)
    seats = [tube_seated(model, data, rk, t) for t in rk["tubes"]]
    dseat = tube_seated(model, data, rk,
                        {"body": f"{p['name']}_drop", "row": drop_rc[0],
                         "col": drop_rc[1]})
    ok = all(s["seated"] for s in seats) and dseat["seated"]
    measured = {"tubes_after_shake": seats, "dropped_tube": dseat}
    verdict = ("PASS: every tube rode out the shake and the dropped tube "
               "self-seated in the socket well" if ok else
               f"FAIL: {measured}")
    stills = save_stills(frames, Path(out_mp4).parent,
                         Path(out_mp4).stem, n=4)
    return {**out, "measured": measured, "verdict": verdict, "stills": stills}


def demo_opener(built_opener: dict, out_mp4: str | Path,
                tube_params: dict | None = None, seconds: float = 14.0,
                fps: int = 30) -> dict:
    """The decapper's whole use, forces only: lower over a closed tube in a
    rack, twist the ledge under the tab, pry — the cap pops while the tube
    stays seated — then lift away and lay the cap fully open."""
    import mujoco
    from .mjcf import rollout
    op = json.loads(Path(built_opener["params_json"]).read_text())
    rp = {"name": "openerdemo", "rows": 1, "cols": 2, "pitch": 0.016,
          "length": 0.065, "width": 0.05, "height": 0.045,
          "tubes": [{"row": 0, "col": 0, "cap_deg": 0.0, "yaw_deg": 180.0}],
          "tube_params": dict(tube_params or {}), "rack_free": False}
    tmp = Path(built_opener["xml"]).parent / "demo_world"
    rk = build_tube_rack(rp, tmp)
    # splice the opener into the rack world, hovering ALIGNED over the tube:
    # the demo proves the decapper's mechanism, not force-field freight
    # handling of a 7-gram cylinder across the bench
    x0_, y0_ = rack_well_xy(json.loads(Path(rk["params_json"]).read_text()),
                            0, 0)
    spawn = (x0_, y0_, rk["plate_top_m"] + 0.0018 + 0.036)
    opener_xml = Path(built_opener["xml"]).read_text()
    inner = opener_xml.split("<worldbody>")[1].split("</worldbody>")[0]
    inner = "\n".join(ln for ln in inner.splitlines()
                      if "<light" not in ln and 'name="floor"' not in ln)
    inner = inner.replace('pos="' + vec(op["pos"]) + '"',
                          f'pos="{vec(spawn)}"', 1)
    xml_p = Path(rk["xml"])
    xml_p.write_text(xml_p.read_text().replace(
        "  </worldbody>", f"{inner}\n  </worldbody>"))
    model = load(xml_p)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    tube = rk["tubes"][0]
    ob = model.body(built_opener["opener_body"]).id
    # transport uses a mass-sized grip (hand-picked gains detonate a 7-gram
    # body — see ForceGrip.sized); the twist and the pry are DIRECT wrenches,
    # because a stability-bounded grip on something this light cannot
    # out-muscle even millinewton-metre contact friction, and a real hand
    # twists a decapper with its fingers, not with a spring.
    # transport forces sized to the object: a 7-gram cylinder weighs 0.07 N,
    # so a 10 N ceiling let the descent STORE a newton-scale spring in the
    # first snag and ping the opener across the bench (measured)
    grip = ForceGrip.sized(model, ob, max_force=1.5, max_torque=0.02)
    H = rk["plate_top_m"]
    x0, y0 = rack_well_xy(json.loads(Path(rk["params_json"]).read_text()),
                          0, 0)
    seat_z = H + 0.0018                    # skirt rim on the tube's flange
    hover = np.array([x0, y0, seat_z + 0.036])
    qz = lambda deg: np.array([np.cos(np.radians(deg) / 2), 0, 0,  # noqa
                               np.sin(np.radians(deg) / 2)])
    twist = float(built_opener["engage_twist_deg"])
    log = {"pop_at_Nm": None, "max_fold": 0.0, "twist_deg": None,
           "phases": []}
    # CONDITION-gated state machine, not a clock: a clock that says "now
    # twist" while the skirt is still 6 mm up twists thin air (measured)
    st = {"phase": "hover", "t0": 0.0, "z": float(hover[2]), "yaw": 0.0,
          "tq": 0.0}

    def goto(ph, t):
        log["phases"].append((ph, round(t, 2)))
        st["phase"], st["t0"] = ph, t

    def yaw_of(d):
        R = d.xmat[ob].reshape(3, 3)
        return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))

    tbid = model.body(tube["body"]).id

    def tab_dir(d):
        """World direction the cap's TAB points (the tube's local -x). The
        tube hangs FREE and the twisting skirt drags its yaw around by
        flange friction, so the opener must CHASE the tab, not assume it —
        with a fixed target the tab rotated away and the ledge engaged air
        (measured, once the rack sockets stopped accidentally wedging the
        tube)."""
        R = d.xmat[tbid].reshape(3, 3)
        v = -R[:, 0]
        n = float(np.hypot(v[0], v[1])) + 1e-12
        return np.array([v[0] / n, v[1] / n, 0.0])

    def control(t, m, d):
        clear_forces(d)
        dt = float(m.opt.timestep)
        fold = float(np.degrees(d.qpos[_jadr(m, f"{tube['body']}_fold")]))
        log["max_fold"] = max(log["max_fold"], fold)
        ph, el = st["phase"], t - st["t0"]
        tabd = tab_dir(d)
        tab_deg = float(np.degrees(np.arctan2(tabd[1], tabd[0])))
        if ph == "hover":
            grip.hold(d, target_pos=hover, target_quat=qz(tab_deg))
            if el > 0.8:
                goto("lower", t)
            return "decapper over the closed tube (window to the tab)"
        if ph == "lower":
            st["z"] = max(seat_z + 0.0001, st["z"] - 0.018 * dt)
            grip.hold(d, target_pos=[x0, y0, st["z"]],
                      target_quat=qz(tab_deg))
            if d.xpos[ob][2] < seat_z + 0.0008 or el > 3.5:
                goto("twist" if d.xpos[ob][2] < seat_z + 0.002 else "abort",
                     t)
            return "lower: skirt swallows the cap, lands on the rim flange"
        if ph == "twist":
            st["yaw"] = max(twist, st["yaw"] - 40.0 * dt)
            tgt = tab_deg + st["yaw"]      # offset from the LIVE tab
            grip.hold(d, target_pos=[x0, y0, seat_z + 0.0002],
                      target_quat=qz(tgt))
            err = tgt - yaw_of(d)
            if abs(err) > 3.0:             # fingers assist the weak grip
                apply_wrench(d, ob, torque=[0, 0, np.sign(err) * 0.010])
            log["twist_deg"] = round(yaw_of(d) - tab_deg, 1)
            if yaw_of(d) - tab_deg < twist + 5.0:
                goto("pry", t)
            elif el > 3.0:
                goto("abort", t)
            return (f"twist {abs(twist):.0f} deg past the tab — the ledge "
                    f"slides under it")
        if ph == "pry":
            if fold < 8.0 and el < 3.5:
                grip.hold(d, target_pos=[x0, y0, seat_z], axes=(1, 1, 0),
                          target_quat=qz(tab_deg + twist))
                apply_wrench(d, ob, force=[0, 0, -1.2])
                # pry only against ENGAGEMENT: a free-spinning opener means
                # the ledge missed — cut the torque, fail honestly, no
                # 7-gram helicopter (measured, before this guard)
                if float(np.linalg.norm(d.cvel[ob, :3])) < 8.0:
                    st["tq"] = min(0.06, st["tq"] + 0.02 * dt)
                    # the tilt LIFTS the tab side while the far rim presses
                    # the flange: torque axis = -(z x tab_dir)
                    axis = -np.cross([0.0, 0.0, 1.0], tabd)
                    apply_wrench(d, ob, torque=st["tq"] * axis)
                    if log["pop_at_Nm"] is None and fold > 6.0:
                        log["pop_at_Nm"] = round(st["tq"], 3)
                    return f"pry: {st['tq']:.2f} N*m tilt (cap {fold:.0f} deg)"
                return "NOT ENGAGED — pry aborted"
            goto("lift" if fold >= 8.0 else "abort", t)
            return f"POPPED — cap at {fold:.0f} deg"
        if ph == "lift":
            u = min(1.0, el / 1.5)
            grip.hold(d, target_pos=[x0, y0, seat_z + 0.002 + 0.05 * u],
                      target_quat=qz(twist))
            if el > 2.0:
                goto("nudge", t)
            return f"lift away — the ledge drags the cap open ({fold:.0f} deg)"
        if ph == "nudge":
            grip.hold(d, target_pos=[x0 - 0.030, y0, seat_z + 0.052],
                      target_quat=qz(twist))
            if fold < 150.0 and el < 2.0:
                cb = m.body(f"{tube['body']}_capface").id
                cz = d.xmat[cb].reshape(3, 3)[:, 2]
                tab_g = m.geom(f"{tube['body']}_tab").id
                apply_wrench(d, cb, force=0.8 * cz,
                             point=d.geom_xpos[tab_g], com=d.xpos[cb])
                return f"a fingertip lays the cap open ({fold:.0f} deg)"
            return f"done — cap {fold:.0f} deg, tube still in its well"
        grip.hold(d, target_pos=[x0, y0, seat_z + 0.05], target_quat=qz(0))
        return f"ABORTED in {ph} — see measured phases"

    cam = free_camera([x0, y0, H + 0.015], 0.19, 140, -16)
    frames = rollout(model, data, seconds, fps, control, camera=cam,
                     width=640, height=480)
    out = write_video(frames, out_mp4, fps=fps)
    seat = tube_seated(model, data, rk, rk["tubes"][0])
    end = cap_state(model, data, tube["body"])
    ok = log["max_fold"] > 60.0 and seat["seated"]
    measured = {**log, "end_cap": end, "tube": seat}
    verdict = ("PASS: the decapper popped the cap (no direct force ever "
               "touched the cap until it was already open) and the tube "
               "never left its well" if ok else f"FAIL: {measured}")
    stills = save_stills(frames, Path(out_mp4).parent,
                         Path(out_mp4).stem, n=4)
    return {**out, "measured": measured, "verdict": verdict, "stills": stills}


def preview(built: dict, out_dir: str | Path) -> list[str]:
    """One still of the asset as built — the cheapest look."""
    import mujoco
    from PIL import Image
    model = load(built["xml"])
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    span = max(0.12, 1.6 * float(built.get("total_h_m", 0)
                                 or built.get("plate_top_m", 0)
                                 or built.get("rim_top_m", 0.1)))
    cam = free_camera([0, 0, span / 4], max(0.18, 2.2 * span), 120, -24)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with mujoco.Renderer(model, height=480, width=640) as r:
        r.update_scene(data, camera=cam)
        pth = out_dir / f"{built['name']}_preview.png"
        Image.fromarray(r.render().copy()).save(pth)
        paths.append(str(pth))
    return paths


# ---------------------------------------------------------------------------

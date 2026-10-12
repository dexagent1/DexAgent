"""Contact margin/gap check: do the scene's margins and gaps detect contacts when intended without pushing bodies apart?"""
from __future__ import annotations

TOOL = {
    "name": "contact_margin_gap_check",
    "title": "Contact margin/gap check",
    "type": "verifier",
    "category": "Scene Compile Checks",
    "summary": "Margins and gaps are judged under the semantics this MuJoCo actually has (measured on a probe): no collidable pair "
               "pushes from farther than `max_active_m` (bodies float, fingers stop short), no negative values, and every body an "
               "adhesion actuator pulls has a margin so it attracts before touching.",
    "description": (
        "The classic reading of MuJoCo's margin and gap (contacts detected inside margin, pushing only inside margin - gap, "
        "so margin == gap detects without force) is not what MuJoCo 3.x does: measured on two spheres, the two geoms' margins "
        "add up, a contact pushes as soon as the distance is below that margin, and gap only adds an inactive band beyond it "
        "(detected up to margin + gap, no force). So margin == gap, written to 'detect early without pushing', keeps the "
        "contact active and repels inside the margin: an object rests that far above its support and fingers stop that far "
        "short. And a margin of 0 detects nothing before touching, so an adhesion (magnet) actuator, which acts through its "
        "body's contacts, cannot attract from a distance. The check measures the semantics once (`margin_semantics` in the "
        "report), then for every collidable geom or explicit pair with a non-zero margin or gap computes, against the "
        "geoms it can collide with, the distance at which the contact starts pushing and the distance at which it is "
        "detected, and fails a geom whose contacts with margin-free geoms (supports, hands) push from farther than "
        "`max_active_m`, an explicit pair that does, negative margins or gaps, and adhesion bodies whose geoms have no "
        "margin; two margin-carrying geoms (studs of two bricks) push from the sum, which is a note since they rarely face "
        "each other. Geoms of adhesion bodies may push from their margin by design (the magnet range). At "
        "keyframe `key` it also lists the contacts that are already pushing across a gap (dist > 0) as evidence."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF file, XML string, or compiled model."},
        {"name": "max_active_m", "type": "float", "description": "Largest distance at which a contact may start pushing, metres. Default 0.001."},
        {"name": "key", "type": "int | str", "description": "Keyframe for the evidence of contacts pushing across a gap. Default 0."},
    ],
    "outputs": [
        {"name": "report", "type": "dict",
         "description": "ok; issues; margin_semantics {combine, gap_band, version}; geoms: [{geom, body, margin, gap, adhesion, worst_active_m, "
                        "worst_detect_m, partner, problem}]; pairs: [{pair, margin, gap, problem}]; hovering: [{geom1, geom2, dist_m, "
                        "active_m}]; worst."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["compile", "contact", "margin", "gap", "adhesion", "magnet"],
    "status": "tested",
    "entrypoint": "contact_margin_gap_check",
    "cli": "python -m tool_library.verifiers.contact_margin_gap_check scene.xml",
}

import json
import sys

import mujoco
import numpy as np

from tool_library.common import compile_checks as C


def contact_margin_gap_check(scene, max_active_m=0.001, key=0):
    C.require_scene(scene)
    model, err = C.compile_scene(scene)
    if model is None:
        return C.compile_failure("fix the error MuJoCo names", err, geoms=[])
    sem = C.margin_semantics()
    adh = C.adhesion_bodies(model)
    issues, notes, rows, prow = [], [], [], []
    marg = np.asarray(model.geom_margin, float)
    gap = np.asarray(model.geom_gap, float)
    paired = {}
    for p in range(model.npair):
        paired.setdefault(int(model.pair_geom1[p]), []).append(p)
        paired.setdefault(int(model.pair_geom2[p]), []).append(p)
    special = [g for g in range(model.ngeom) if (marg[g] != 0 or gap[g] != 0) and (C.collidable(model, g) or g in paired)]
    for g in special:
        b = int(model.geom_bodyid[g])
        is_adh = b in adh
        # judged against what the geom meets in practice (supports, hands and other margin-free geoms: the contact pushes from
        # this geom's own margin); two margin-carrying geoms push from the sum, which is noted, since such geoms (studs of
        # two bricks) usually never face each other
        worst_a, worst_d, partner = 0.0, 0.0, None
        both = []
        for h in range(model.ngeom):
            if h == g or not C.can_collide(model, g, h):
                continue
            a, dd = C.pair_distances(float(marg[g]), float(gap[g]), float(marg[h]), float(gap[h]), sem)
            if marg[h] != 0 and int(model.geom_bodyid[h]) not in adh:
                both.append((a, h))
                continue
            if a > worst_a or partner is None:
                worst_a, worst_d, partner = a, dd, h
        if partner is None and both:
            a, h = min(both)
            worst_a, worst_d, partner = a, C.pair_distances(float(marg[g]), float(gap[g]), float(marg[h]), float(gap[h]), sem)[1], h
        if both and not is_adh:
            a, h = max(both)
            if a > max_active_m + 1e-12:
                notes.append(f"geom '{C.geom_name(model, g)}' and '{C.geom_name(model, h)}' both carry a margin: if they ever face each other "
                             f"their contact pushes from {1000 * a:.2f} mm (the margins add)")
        probs = []
        if marg[g] < 0 or gap[g] < 0:
            probs.append(f"negative margin/gap ({marg[g]:g}, {gap[g]:g}): contacts start only after penetrating that deep")
        if not is_adh and partner is not None and worst_a > max_active_m + 1e-12:
            semtxt = ("margins add and gap only extends detection beyond the margin" if sem["combine"] == "sum" and sem["gap_band"]
                      else "this MuJoCo's margin semantics")
            intent = (" (margin == gap was meant to detect without pushing, but under " + semtxt + " the contact stays active and "
                      "repels inside the margin)") if marg[g] > 0 and abs(marg[g] - gap[g]) < 1e-12 else ""
            fix = (f"; to detect early without force use margin=0 gap={worst_a:g}" if sem["gap_band"]
                   else "; lower the margin or raise the gap")
            probs.append(f"contacts with '{C.geom_name(model, partner)}' push from {1000 * worst_a:.2f} mm{intent}: the body rests or is "
                         f"held that far from what it touches (an object floats, fingers stop short){fix}")
        rows.append({"geom": C.geom_name(model, g), "body": C.body_name(model, b), "margin": float(marg[g]), "gap": float(gap[g]),
                     "adhesion": is_adh, "worst_active_m": worst_a, "worst_detect_m": worst_d,
                     "partner": C.geom_name(model, partner) if partner is not None else None, "problem": "; ".join(probs) or None})
    for p in range(model.npair):
        pm, pg = float(model.pair_margin[p]), float(model.pair_gap[p])
        if pm == 0 and pg == 0:
            continue
        g1, g2 = int(model.pair_geom1[p]), int(model.pair_geom2[p])
        a = pm if sem["gap_band"] else pm - pg
        probs = []
        if pm < 0 or pg < 0:
            probs.append(f"negative pair margin/gap ({pm:g}, {pg:g})")
        if a > max_active_m + 1e-12 and int(model.geom_bodyid[g1]) not in adh and int(model.geom_bodyid[g2]) not in adh:
            probs.append(f"the explicit pair pushes from {1000 * a:.2f} mm: its geoms are held that far apart")
        prow.append({"pair": C.name_of(model, mujoco.mjtObj.mjOBJ_PAIR, p, "pair"), "geom1": C.geom_name(model, g1),
                     "geom2": C.geom_name(model, g2), "margin": pm, "gap": pg, "problem": "; ".join(probs) or None})
    # adhesion bodies need a margin to attract before touching
    for b in sorted(adh):
        gs = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == b and C.collidable(model, g)]
        if gs and all(marg[g] <= 0 for g in gs):
            issues.append(f"body '{C.body_name(model, b)}' is pulled by an adhesion actuator but none of its geoms has a margin: a contact "
                          "is detected only once the surfaces touch, so the magnet attracts nothing from a distance; give its geom "
                          "margin = the attraction range (gap adds only inactive contacts)")
    group = {}
    for r in rows:
        if r["problem"]:
            group.setdefault(r["problem"], []).append(r["geom"])
    for prob, gs in group.items():
        issues.append(f"{len(gs)} geom(s) ({gs[:4]}{'...' if len(gs) > 4 else ''}): {prob}")
    for r in prow:
        if r["problem"]:
            issues.append(f"pair '{r['pair']}' ({r['geom1']} - {r['geom2']}): {r['problem']}")
    # evidence: contacts already pushing across a gap at the initial state
    hovering = []
    if special or prow:
        data = C.key_data(model, key)
        for i in range(data.ncon):
            c = data.contact[i]
            if float(c.dist) > 1e-5 and float(c.dist) < float(c.includemargin):
                hovering.append({"geom1": C.geom_name(model, int(c.geom1)), "geom2": C.geom_name(model, int(c.geom2)),
                                 "dist_m": float(c.dist), "active_m": float(c.includemargin)})
    rows.sort(key=lambda r: r["problem"] is None)
    seen = set()
    notes = [n for n in notes if not (n.split("'")[1] + n.split("'")[3] in seen or seen.add(n.split("'")[1] + n.split("'")[3]))][:10]
    return C.jsonable({"ok": not issues, "compiled": True, "issues": issues, "notes": notes, "margin_semantics": sem, "geoms": rows[:60],
                       "n_geoms_with_margin_or_gap": len(rows), "pairs": prow[:40], "hovering": hovering[:30],
                       "worst": rows[0] if rows and rows[0]["problem"] else None})


def main(argv=None):
    rep = contact_margin_gap_check((sys.argv[1:] if argv is None else argv)[0])
    print(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

"""Physics options check: the scene simulates with the setup's physics, not an easier one."""
from __future__ import annotations

TOOL = {
    "name": "physics_options_check",
    "title": "Physics options check",
    "type": "verifier",
    "category": "Scene Assertions",
    "summary": "Compares a scene's simulation options with the setup's: gravity, timestep (finer is allowed), integrator, friction cone, impratio, no-slip iterations, solver and iterations, the disable/enable flags (contact, gravity, constraint, override...) and medium density/viscosity/wind, and with require_execution_contact also the executor's contact model (elliptic cones, >= 5 no-slip iterations).",
    "description": (
        "A scene can make a task easy without touching any object: turning gravity down, contacts off, the friction "
        "cone pyramidal (pyramidal cones once made grasps seat with zero finger force), the solver to one iteration, or "
        "the air into syrup that holds a released object. The reference is the setup (default: the repository's "
        "setup/): setup.yaml's physics block (timestep, integrator, impratio) and the setup scene's compiled options for "
        "everything else. Rules: gravity equal; timestep not larger than the setup's (a finer step is allowed, e.g. "
        "0.25 ms for screw threads) and not below min_timestep; integrator equal; impratio equal; cone not weaker "
        "(elliptic is stronger than pyramidal); noslip_iterations, iterations and ls_iterations not fewer; solver "
        "equal; no disable flag the setup does not set; no 'override' enable flag (it replaces every contact's "
        "solref/solimp/friction/margin); density, viscosity and wind zero when the setup's are. With "
        "require_execution_contact=True (a Stage 3 execution model written by build_execution_model) the cone must be "
        "elliptic and noslip_iterations >= 5, as the executor sets them."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "Scene MJCF."},
        {"name": "setup", "type": "str | None", "description": "Setup folder (or its setup.yaml / scene.xml). Default: the repository's setup/."},
        {"name": "require_execution_contact", "type": "bool", "description": "Also require the executor's contact model (elliptic cone, noslip >= 5). Default False."},
        {"name": "min_timestep", "type": "float", "description": "Smallest timestep accepted, seconds (a step this fine stalls every rollout). Default 1e-5."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; scene {option: value}; reference {option: value}; violations [{option, scene, reference, rule}]."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "pyyaml"],
    "simulators": ["MuJoCo"],
    "tags": ["assertion", "anti-tampering", "physics", "options", "gravity", "timestep", "cone", "stage2", "stage3"],
    "status": "tested",
    "entrypoint": "physics_options_check",
}

import mujoco

from tool_library.common import scene_assert as SA

CONES = {int(mujoco.mjtCone.mjCONE_PYRAMIDAL): "pyramidal", int(mujoco.mjtCone.mjCONE_ELLIPTIC): "elliptic"}
INTEGRATORS = {int(v): k for k, v in (("euler", mujoco.mjtIntegrator.mjINT_EULER), ("rk4", mujoco.mjtIntegrator.mjINT_RK4),
                                       ("implicit", mujoco.mjtIntegrator.mjINT_IMPLICIT), ("implicitfast", mujoco.mjtIntegrator.mjINT_IMPLICITFAST))}
SOLVERS = {int(mujoco.mjtSolver.mjSOL_PGS): "PGS", int(mujoco.mjtSolver.mjSOL_CG): "CG", int(mujoco.mjtSolver.mjSOL_NEWTON): "Newton"}


def _flags(enum, value):
    return [n for n in dir(enum) if n.startswith("mj") and not n.endswith("NUM") and int(getattr(enum, n)) & int(value)
            and bin(int(getattr(enum, n))).count("1") == 1]


def _opts(model):
    o = model.opt
    return {"gravity": [float(x) for x in o.gravity], "timestep": float(o.timestep), "integrator": INTEGRATORS.get(int(o.integrator), int(o.integrator)),
            "cone": CONES.get(int(o.cone), int(o.cone)), "impratio": float(o.impratio), "noslip_iterations": int(o.noslip_iterations),
            "iterations": int(o.iterations), "ls_iterations": int(o.ls_iterations), "solver": SOLVERS.get(int(o.solver), int(o.solver)),
            "disableflags": int(o.disableflags), "enableflags": int(o.enableflags), "density": float(o.density),
            "viscosity": float(o.viscosity), "wind": [float(x) for x in o.wind]}


def physics_options_check(scene, setup=None, require_execution_contact=False, min_timestep=1e-5):
    model, err = SA.try_load_model(scene)
    if model is None:
        return SA.report([f"the scene does not compile: {err}"], compiled=False)
    from tool_library.verifiers._reference import load_reference
    st = load_reference(model, setup)
    ref = _opts(st["model"])
    phys = (st["cfg"] or {}).get("physics") or {}
    if "timestep" in phys:
        ref["timestep"] = float(phys["timestep"])
    if "integrator" in phys:
        ref["integrator"] = str(phys["integrator"]).lower()
    if "impratio" in phys:
        ref["impratio"] = float(phys["impratio"])
    if "cone" in phys:
        ref["cone"] = str(phys["cone"]).lower()
    if "noslip_iterations" in phys:
        ref["noslip_iterations"] = int(phys["noslip_iterations"])
    sc = _opts(model)
    v = []

    def bad(opt, rule):
        v.append({"option": opt, "scene": sc[opt], "reference": ref[opt], "rule": rule})

    if SA.differs(sc["gravity"], ref["gravity"], 1e-9, 1e-9):
        bad("gravity", "must equal the setup's")
    if sc["timestep"] > ref["timestep"] * (1 + 1e-9):
        bad("timestep", "must not be larger than the setup's")
    if sc["timestep"] < min_timestep:
        v.append({"option": "timestep", "scene": sc["timestep"], "reference": min_timestep, "rule": f"must be at least {min_timestep}"})
    if sc["integrator"] != ref["integrator"]:
        bad("integrator", "must equal the setup's")
    if abs(sc["impratio"] - ref["impratio"]) > 1e-9 * max(1.0, ref["impratio"]):
        bad("impratio", "must equal the setup's")
    if ref["cone"] == "elliptic" and sc["cone"] != "elliptic":
        bad("cone", "must be elliptic like the setup's")
    for opt in ("noslip_iterations", "iterations", "ls_iterations"):
        if sc[opt] < ref[opt]:
            bad(opt, "must not be fewer than the setup's")
    if sc["solver"] != ref["solver"]:
        bad("solver", "must equal the setup's")
    new_dis = int(sc["disableflags"]) & ~int(ref["disableflags"])
    if new_dis:
        v.append({"option": "disableflags", "scene": _flags(mujoco.mjtDisableBit, new_dis), "reference": _flags(mujoco.mjtDisableBit, ref["disableflags"]),
                  "rule": "the scene disables what the setup simulates"})
    ovr = int(mujoco.mjtEnableBit.mjENBL_OVERRIDE)
    if int(sc["enableflags"]) & ovr and not int(ref["enableflags"]) & ovr:
        v.append({"option": "enableflags", "scene": _flags(mujoco.mjtEnableBit, sc["enableflags"]), "reference": _flags(mujoco.mjtEnableBit, ref["enableflags"]),
                  "rule": "the contact override replaces every contact's parameters"})
    for opt in ("density", "viscosity"):
        if abs(sc[opt] - ref[opt]) > 1e-12:
            bad(opt, "a medium the setup does not have drags held and released objects")
    if SA.differs(sc["wind"], ref["wind"], 1e-12, 0.0):
        bad("wind", "must equal the setup's")
    if require_execution_contact:
        if sc["cone"] != "elliptic":
            v.append({"option": "cone", "scene": sc["cone"], "reference": "elliptic", "rule": "the executor's contact model uses elliptic cones (pyramidal ones seat grasps with no finger force)"})
        if sc["noslip_iterations"] < 5:
            v.append({"option": "noslip_iterations", "scene": sc["noslip_iterations"], "reference": 5, "rule": "the executor's contact model uses at least 5 no-slip iterations"})
    issues = [f"option {x['option']} is {x['scene']} in the scene but {x['reference']} in the setup: {x['rule']}; "
              f"set it back in <option> (and <flag>)" for x in v]
    return SA.report(issues, scene=sc, reference=ref, violations=v, setup=str(st["dir"]))

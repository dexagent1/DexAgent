"""A generic dexterous-hand rig for MuJoCo scenes.

Any hand MJCF that follows the convention below can be used by the grasp
skills and verifiers:

* a 6-DoF "virtual wrist" chain of joints named ``<side>_pos_x``, ``_pos_y``,
  ``_pos_z``, ``_rot_x``, ``_rot_y``, ``_rot_z`` (slide, slide, slide, hinge,
  hinge, hinge, applied in that order), each with a position actuator named
  ``<joint>_position``;
* finger joints with position actuators ``<joint>_position``;
* one site per fingertip named ``<side>_<finger>_tip`` (knuckle sites
  ``<side>_<finger>_pip`` / ``_mcp`` are optional);
* collision geoms named ``collision_hand_<side>_<finger>_<k>`` that ship with
  contype/conaffinity 0 and are switched on through explicit contact pairs.

The Sharpa and Shadow hands used in the paper both follow it; a different hand
is supported by passing a ``HandSpec`` with its own name patterns. Everything
that drives the hand goes through the actuators (``ctrl``), never through
external forces: an object moves only because a finger pushed it.

Which object geoms the hand touches (``build_hand_scene``): the hand behaves
like a geom with contype = conaffinity = ``HAND_CONTACT_BITS`` (1, MuJoCo's
default), so an object geom gets a hand pair only when its own bits can meet
those: a visual mesh (0/0) never does, and a geom an asset put in a collision
bit of its own (a ring's magnet sphere in bit 4, meant only for ring-ring
contact) stays out of the hand's way. An object with no collision bits at all
is paired whole (a hand-scene stand-in made only of meshes). A geom can also
opt out explicitly: its default class name ends in ``nohand``
(``<default class="nohand">`` or ``"magnet_nohand"``), its name ends in
``_nohand``, or it is listed (by name or name prefix) in ``no_hand_pairs``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
WRIST = ("x", "y", "z", "rx", "ry", "rz")
_WRIST_JOINT = {"x": "pos_x", "y": "pos_y", "z": "pos_z",
                "rx": "rot_x", "ry": "rot_y", "rz": "rot_z"}

# A plain-literal description of a hand's naming. `{side}` and `{finger}` are
# substituted. The defaults match the Sharpa and Shadow MJCFs.
DEFAULT_HAND_SPEC = {
    "actuator_suffix": "_position",
    "tip_site": "{side}_{finger}_tip",
    "knuckle_sites": ["{side}_{finger}_pip", "{side}_{finger}_mcp"],
    "collision_geom": "collision_hand_{side}_",
    "palm_body_suffixes": ["_hand_C_MC", "_palm"],
    "fingers": list(FINGERS),
}

# How a finger joint behaves when the hand "closes" to a grip fraction.
# Sharpa names flexion joints *_FE / *_PIP / *_DIP / *_IP, abduction *_AA;
# Shadow uses FFJ1..4, MFJ, RFJ, LFJ1..5, THJ1..5 (J4 = abduction, THJ5 =
# thumb opposition, LFJ5 = the pinky's palm arch).
_FLEX_SUFFIXES = ("_FE", "_PIP", "_DIP", "_IP", "pinky_CMC")
_THUMB_OPPOSE = ("thumb_CMC_AA", "thumb_MCP_AA")
_SHADOW_ABDUCT = re.compile(r"^(FF|MF|RF|LF)J4$")
_SHADOW_JOINT = re.compile(r"^(FF|MF|RF|LF|TH)J[1-5]$")
# Shadow finger-name prefixes -> the finger they belong to
_SHADOW_FINGER = {"FF": "index", "MF": "middle", "RF": "ring", "LF": "pinky", "TH": "thumb"}


def joint_role(short: str) -> str:
    """'flex' | 'abduct' | 'thumb_oppose' | 'pinky_arch' | 'other' for a
    finger joint's short name (the part after '<side>_')."""
    if any(s in short for s in _THUMB_OPPOSE) or short == "THJ5":
        return "thumb_oppose"
    if short.endswith("_AA") or _SHADOW_ABDUCT.match(short) or short == "THJ3":
        return "abduct"
    if short == "LFJ5":
        return "pinky_arch"
    if any(s in short for s in _FLEX_SUFFIXES) or _SHADOW_JOINT.match(short):
        return "flex"
    return "other"


def finger_of_joint(short: str) -> str | None:
    """Which finger a short joint name belongs to, or None."""
    for f in FINGERS:
        if short.startswith(f):
            return f
    m = re.match(r"^(FF|MF|RF|LF|TH)J", short)
    return _SHADOW_FINGER[m.group(1)] if m else None


def euler_to_mat(e) -> np.ndarray:
    """Rotation of the wrist chain: Rx(rx) Ry(ry) Rz(rz) (intrinsic XYZ)."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler("XYZ", np.asarray(e, float)).as_matrix()


def mat_to_euler(R) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(np.asarray(R, float)).as_euler("XYZ")


def nearest_euler_branch(e, prev) -> np.ndarray:
    """The intrinsic-XYZ triple equal to `e` as a rotation that is nearest to
    `prev` joint by joint. (rx, ry, rz) and (rx±pi, pi-ry, rz±pi) are the same
    orientation but half a turn apart at every joint; feeding servos the far
    branch spins the wrist round for no change in pose."""
    e, prev = np.asarray(e, float), np.asarray(prev, float)
    alt = np.array([e[0] - np.sign(e[0] or 1.0) * np.pi,
                    np.pi - e[1] if e[1] >= 0 else -np.pi - e[1],
                    e[2] - np.sign(e[2] or 1.0) * np.pi])
    best, bd = e, None
    for cand in (e, alt):
        for k0 in (-1, 0, 1):
            for k2 in (-1, 0, 1):
                c = cand + np.array([2 * np.pi * k0, 0.0, 2 * np.pi * k2])
                d = float(np.abs(c - prev).max())
                if bd is None or d < bd:
                    bd, best = d, c
    return best


@dataclass
class HandRig:
    """Indices of one hand attached with prefix `prefix` (for example
    'hand_' -> joints 'hand_right_pos_x', ...)."""
    model: object
    prefix: str
    side: str
    spec: dict
    wrist_act: dict = field(default_factory=dict)
    wrist_qpos: dict = field(default_factory=dict)
    finger_act: dict = field(default_factory=dict)      # short joint name -> actuator id
    finger_range: dict = field(default_factory=dict)
    finger_qpos: dict = field(default_factory=dict)
    fingertip_sites: dict = field(default_factory=dict)  # finger -> site id
    knuckle_sites: dict = field(default_factory=dict)
    root_body: int = -1
    palm_body: int = -1
    body_ids: set = field(default_factory=set)
    geom_ids: set = field(default_factory=set)
    collision_gids: set = field(default_factory=set)
    rigid_wrist: bool = False

    @classmethod
    def build(cls, model, prefix: str, side: str, spec: dict | None = None) -> "HandRig":
        spec = dict(DEFAULT_HAND_SPEC, **(spec or {}))
        rig = cls(model=model, prefix=prefix, side=side, spec=spec)
        suf = spec["actuator_suffix"]
        head = f"{prefix}{side}_"
        yaw = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, head + "base_yaw")
        rig.rigid_wrist = yaw >= 0 and int(model.body_jntnum[yaw]) == 0 and mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, head + "pos_x") < 0
        for dof, jn in _WRIST_JOINT.items():
            if rig.rigid_wrist:
                break
            jname = f"{prefix}{side}_{jn}"
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname + suf)
            if jid < 0 or aid < 0:
                raise ValueError(f"hand '{prefix}': wrist joint/actuator '{jname}' not found; "
                                 "the hand MJCF needs the pos_x..rot_z virtual wrist")
            rig.wrist_qpos[dof] = int(model.jnt_qposadr[jid])
            rig.wrist_act[dof] = int(aid)
        for j in range(model.njnt):
            jn = model.joint(j).name or ""
            if not jn.startswith(head):
                continue
            short = jn[len(head):]
            if short in _WRIST_JOINT.values():
                continue
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, jn + suf)
            if aid < 0:
                continue
            rig.finger_act[short] = int(aid)
            lo, hi = model.jnt_range[j]
            rig.finger_range[short] = (float(lo), float(hi))
            rig.finger_qpos[short] = int(model.jnt_qposadr[j])
        for fin in spec["fingers"]:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                                    prefix + spec["tip_site"].format(side=side, finger=fin))
            if sid >= 0:
                rig.fingertip_sites[fin] = int(sid)
            for pat in spec["knuckle_sites"]:
                k = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                                      prefix + pat.format(side=side, finger=fin))
                if k >= 0:
                    rig.knuckle_sites.setdefault(fin, int(k))
        if not rig.fingertip_sites:
            raise ValueError(f"hand '{prefix}': no fingertip sites named like "
                             f"{spec['tip_site']}")
        rig.root_body = yaw if rig.rigid_wrist else int(model.body(f"{head}base_tx").id) if mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"{head}base_tx") >= 0 else int(
            model.jnt_bodyid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{head}pos_x")])
        # the hand's own bodies: everything under the prefix except an arm sharing it (a bench station is
        # 'robot_right_' for both the arm 'robot_right_arm_*' and the hand 'robot_right_right_*')
        rig.body_ids = {b for b in range(model.nbody) if (model.body(b).name or "").startswith(prefix)
                        and "_arm_" not in (model.body(b).name or "")[len(prefix) - 1:]}
        rig.geom_ids = {g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in rig.body_ids}
        cpat = prefix + spec["collision_geom"].format(side=side)
        rig.collision_gids = {g for g in rig.geom_ids
                              if (model.geom(g).name or "").startswith(cpat)}
        for b in sorted(rig.body_ids):
            nm = model.body(b).name or ""
            if any(nm.endswith(s) for s in spec["palm_body_suffixes"]):
                rig.palm_body = int(b)
                break
        return rig

    # -- reading and writing the hand's state -------------------------------
    def wrist_pos(self, data) -> np.ndarray:
        if self.rigid_wrist:
            return data.xpos[self.root_body].copy()
        return np.array([data.qpos[self.wrist_qpos[k]] for k in ("x", "y", "z")], float)

    def wrist_euler(self, data) -> np.ndarray:
        if self.rigid_wrist:
            return mat_to_euler(data.xmat[self.root_body].reshape(3, 3))
        return np.array([data.qpos[self.wrist_qpos[k]] for k in ("rx", "ry", "rz")], float)

    def teleport(self, data, pos, euler):
        """Write the wrist pose straight into qpos (for initialisation only)."""
        if self.rigid_wrist:
            raise ValueError("a runtime wrist is fixed to its arm: solve the arm, or use a proposal scene for grasp search")
        for k, v in zip(("x", "y", "z"), np.asarray(pos, float)):
            data.qpos[self.wrist_qpos[k]] = float(v)
        for k, v in zip(("rx", "ry", "rz"), np.asarray(euler, float)):
            data.qpos[self.wrist_qpos[k]] = float(v)

    def wrist_ctrl(self, ctrl, pos, euler):
        if self.rigid_wrist:
            raise ValueError("a runtime wrist has no world-pose actuators; command the arm motors")
        for k, v in zip(("x", "y", "z"), np.asarray(pos, float)):
            ctrl[self.wrist_act[k]] = float(v)
        for k, v in zip(("rx", "ry", "rz"), np.asarray(euler, float)):
            ctrl[self.wrist_act[k]] = float(v)

    def finger_targets(self, grip: float) -> dict:
        """Joint targets for a uniform closure: flexion joints at `grip` of
        their range, abduction neutral, thumb at a fixed opposition."""
        g = float(np.clip(grip, 0.0, 1.0))
        out = {}
        for short, (lo, hi) in self.finger_range.items():
            role = joint_role(short)
            if role == "thumb_oppose":
                out[short] = lo + 0.4 * (hi - lo)
            elif role == "abduct":
                out[short] = 0.0 if lo <= 0.0 <= hi else 0.5 * (lo + hi)
            elif role == "pinky_arch":
                out[short] = lo + 0.3 * g * (hi - lo)
            elif role == "flex":
                out[short] = lo + g * (hi - lo)
        return out

    def finger_ctrl(self, ctrl, targets: dict):
        for k, v in targets.items():
            if k in self.finger_act:
                lo, hi = self.finger_range[k]
                ctrl[self.finger_act[k]] = float(np.clip(v, lo, hi))

    def set_fingers(self, data, targets: dict):
        for k, v in targets.items():
            if k in self.finger_qpos:
                data.qpos[self.finger_qpos[k]] = float(v)

    def flexion_joints(self, finger: str) -> list:
        return [k for k in self.finger_qpos
                if finger_of_joint(k) == finger and joint_role(k) == "flex"]

    def knuckle_point(self, data, finger: str) -> np.ndarray:
        """A point up the finger from the tip: the finger's knuckle site when
        the asset has one, else the origin of the fingertip link's parent
        body (the joint before the tip), which every hand has."""
        if finger in self.knuckle_sites:
            return np.asarray(data.site_xpos[self.knuckle_sites[finger]], float).copy()
        if finger not in self.fingertip_sites:
            raise ValueError(f"finger '{finger}' has no tip site")
        b = int(self.model.site_bodyid[self.fingertip_sites[finger]])
        parent = int(self.model.body_parentid[b])
        return np.asarray(data.xpos[parent], float).copy()

    def tip_geoms(self, finger: str | None = None) -> set:
        """Collision geoms on the fingertip link(s): the parts that are meant to touch. Visual meshes
        are left out when the link has collision primitives: mj_geomDistance returns a false zero for
        mesh-mesh pairs, which once made a cinch believe a pad touched a bottle 4 cm away."""
        out = set()
        for fin, sid in self.fingertip_sites.items():
            if finger is not None and fin != finger:
                continue
            b = int(self.model.site_bodyid[sid])
            geoms = {g for g in self.geom_ids if int(self.model.geom_bodyid[g]) == b}
            # every collision geom of the tip link: the thumb tip's visual hull (handgeom.add_hand_hulls) sits half a
            # millimetre proud of its pad capsule, so it is the geom that touches (pad_radius reads the capsule only)
            col = geoms & self.collision_gids
            out |= col or geoms
        return out


def hand_mass(model, rig: HandRig) -> float:
    return float(sum(model.body_mass[b] for b in rig.body_ids))


FINGER_FORCE_NM = 0.8      # torque cap of a finger joint servo: a position servo with kp 300 and no cap
WRIST_FORCE_N = 40.0       # presses with hundreds of newtons, and a cap that merely slips in reality tears
WRIST_TORQUE_NM = 3.0      # out of its thread (measured: 170 N on one pad, the bottle dragged 5 cm)
WRIST_ROT_ARMATURE = 0.05  # reflected inertia of the wrist's rotation joints (shipped 1.0: a 17 N m servo
                           # torque went into accelerating the armature, not the cap)


FINGER_ARMATURE = 0.05     # a finger joint's reflected inertia; the shipped 1.0 makes a torque-capped finger
                           # take seconds to close (0.8 Nm / 1.0 kg m^2 = 0.8 rad/s^2)


FINGER_KP = None           # finger servo stiffness (N m / rad); None keeps the asset's (300). A saturated stiff servo
                           # cannot balance a pinch: two fingers at the cap push harder than one thumb.


def limit_hand_forces(model, rig: HandRig, finger_force_nm: float = FINGER_FORCE_NM, wrist_force_n: float = WRIST_FORCE_N,
                      wrist_torque_nm: float = WRIST_TORQUE_NM, finger_armature: float = FINGER_ARMATURE,
                      wrist_rot_armature: float = WRIST_ROT_ARMATURE, finger_kp: float | None = FINGER_KP) -> dict:
    """Clamp the hand's actuator forces to what a real hand and arm deliver, and give the finger joints a
    plausible reflected inertia with the servo damping recomputed for it (critical). 0 or None leaves an
    actuator unlimited / the armature as shipped."""
    for k, a in rig.finger_act.items():
        if finger_force_nm:
            model.actuator_forcelimited[a] = 1
            model.actuator_forcerange[a] = [-float(finger_force_nm), float(finger_force_nm)]
        if finger_kp:
            model.actuator_gainprm[a][0] = float(finger_kp); model.actuator_biasprm[a][1] = -float(finger_kp)
        if finger_armature:
            j = int(model.actuator_trnid[a][0]); dof = int(model.jnt_dofadr[j])
            model.dof_armature[dof] = float(finger_armature)
            kp = float(model.actuator_gainprm[a][0])
            model.actuator_biasprm[a][2] = -2.0 * np.sqrt(kp * float(finger_armature))
    fcap = wrist_force_n if wrist_force_n is None or np.isscalar(wrist_force_n) else list(wrist_force_n)   # scalar or [fx, fy, fz]
    for k, a in rig.wrist_act.items():
        if k in ("x", "y", "z"):
            lim = fcap if fcap is None or np.isscalar(fcap) else fcap[("x", "y", "z").index(k)]
        else:
            lim = wrist_torque_nm
        if lim:
            model.actuator_forcelimited[a] = 1
            model.actuator_forcerange[a] = [-float(lim), float(lim)]
        if k in ("rx", "ry", "rz") and wrist_rot_armature:
            j = int(model.actuator_trnid[a][0]); dof = int(model.jnt_dofadr[j])
            model.dof_armature[dof] = float(wrist_rot_armature)
            kp = float(model.actuator_gainprm[a][0])
            inertia = float(wrist_rot_armature) + 0.012 * max(hand_mass(model, rig), 0.2)   # hand about its wrist
            model.actuator_biasprm[a][2] = -2.0 * np.sqrt(kp * inertia) * 1.15
    return {"finger_force_nm": finger_force_nm, "wrist_force_n": wrist_force_n, "wrist_torque_nm": wrist_torque_nm,
            "finger_armature": finger_armature, "wrist_rot_armature": wrist_rot_armature, "finger_kp": finger_kp}


def set_wrist_gain(model, rig: HandRig, kp: float = 10000.0) -> dict:
    """Stiffen the wrist servos so the hand does not droop under its own
    weight: a position servo's steady-state error is load/kp, and a 1.9 kg
    hand on kp=1000 hangs 18 mm below every pose it is told to hold. kv is
    set 15% above critical for the hand's actual mass."""
    m = max(hand_mass(model, rig), 0.2)
    kv = 2.0 * np.sqrt(float(kp) * m) * 1.15
    for a in rig.wrist_act.values():
        model.actuator_gainprm[a][0] = float(kp)
        model.actuator_biasprm[a][1] = -float(kp)
        model.actuator_biasprm[a][2] = -float(kv)
    return {"kp": float(kp), "kv": round(float(kv), 1), "hand_mass_kg": round(m, 3)}


# ---------------------------------------------------------------------------
# scenes: floor + objects + one hand, with explicit hand-object contact pairs
# ---------------------------------------------------------------------------
_SCENE_XML = """<mujoco model="{name}">
  <compiler angle="radian"/>
  <option timestep="{dt}" integrator="implicitfast" cone="elliptic" impratio="5" noslip_iterations="5"/>
  <size memory="256M"/>
  <visual><global offwidth="1280" offheight="960"/></visual>
  <worldbody>
    <light name="key" pos="0.4 -0.5 1.2" dir="-0.3 0.4 -1" diffuse=".7 .7 .7"/>
    <light name="fill" pos="-0.6 0.4 0.9" dir="0.5 -0.3 -1" diffuse=".3 .3 .3"/>
    {floor}
  </worldbody>
</mujoco>
"""
TABLE_PAIR = {"solref": (0.01, 1.0), "friction": (1.0, 1.0, 0.1, 0.0, 0.0), "condim": 3}   # the bench's hand-table pairs (setup/scene.xml)
# hand-object contact friction (sliding, torsional, rolling) with condim 6. MuJoCo's torsional and rolling coefficients
# are lengths (the contact patch's radius): a soft fingertip pad presses a patch of about a centimetre, so torsion
# 0.02 m (a generous 2 cm) and rolling 0.001 m. The earlier (1.2, 1.0, 0.1) let a two-finger pinch hold any torque about
# its axis: measured, the battery-cell and hard-drive pinches of verified episodes lifted only with those values.
HAND_PAIR_FRICTION = (1.2, 0.02, 0.001)
HAND_CONTACT_BITS = 1      # the hand meets an object geom as a default MuJoCo geom would (contype = conaffinity = 1)
NO_HAND_TAG = "nohand"     # class-name / geom-name suffix that keeps a geom out of the hand's pairs


def hand_may_touch(contype, conaffinity, hand_bits=HAND_CONTACT_BITS) -> bool:
    """MuJoCo's own contact rule between an object geom and a hand made of geoms with contype = conaffinity =
    `hand_bits`: they collide when either one's contype meets the other's conaffinity."""
    hb = int(hand_bits)
    return bool((int(contype) & hb) or (int(conaffinity) & hb))


def _opted_out(name, classname, no_hand_pairs=()) -> bool:
    nm, cls = str(name or ""), str(classname or "")
    if cls.endswith(NO_HAND_TAG) or nm.endswith("_" + NO_HAND_TAG):
        return True
    return any(nm == p or nm.startswith(p) for p in (no_hand_pairs or ()) if p)


def hand_pair_geoms(spec_geoms, obj_prefixes, hand_bits=HAND_CONTACT_BITS, no_hand_pairs=()):
    """Names of the object geoms (MjSpec geoms under one of `obj_prefixes`) the hand gets explicit pairs with, and a
    dict {name: reason} of the ones left out. Rules in the module docstring: the geom's masks must meet the hand's
    bits, an opt-out (class or name ending in 'nohand', or `no_hand_pairs`) wins, and an object none of whose geoms
    has any collision bit is paired whole."""
    keep, skipped = [], {}
    by_obj = {}
    for g in spec_geoms:
        nm = g.name or ""
        pre = max((p for p in obj_prefixes if nm.startswith(p)), key=len, default=None)     # 'ring_b_' before 'ring_'
        if pre is None:
            continue
        by_obj.setdefault(pre, []).append(g)
    for pre, geoms in by_obj.items():
        any_bits = any(int(g.contype) or int(g.conaffinity) for g in geoms)
        for g in geoms:
            cls = g.classname.name if getattr(g, "classname", None) is not None else ""
            if _opted_out(g.name, cls, no_hand_pairs):
                skipped[g.name] = "opted out"
            elif any_bits and not hand_may_touch(g.contype, g.conaffinity, hand_bits):
                skipped[g.name] = ("visual (contype/conaffinity 0)" if not (int(g.contype) or int(g.conaffinity))
                                   else f"contype {int(g.contype)} / conaffinity {int(g.conaffinity)} never meet the hand's bits {int(hand_bits)}")
            else:
                keep.append(g.name)
    return keep, skipped


def hand_span(model, rig) -> float:
    """How far any of the hand's collision geoms (bounding sphere included) can be from one of its fingertip sites, with
    the fingers open: a hand whose fingertips are on a set of points has no geometry farther than this from them."""
    d = mujoco.MjData(model)
    rig.teleport(d, np.zeros(3), np.zeros(3))
    rig.set_fingers(d, {k: lo for k, (lo, hi) in rig.finger_range.items()})
    mujoco.mj_kinematics(model, d)
    tips = np.array([d.site_xpos[s] for s in rig.fingertip_sites.values()], float)
    gids = sorted(rig.collision_gids) or sorted(rig.geom_ids)
    if not len(tips) or not gids:
        return 0.25
    P = np.asarray(d.geom_xpos[gids], float)
    R = np.asarray(model.geom_rbound[gids], float)
    return float((np.linalg.norm(P[:, None, :] - tips[None], axis=2) + R[:, None]).max())


def _region_geoms(sp, names, region, hand_name, side, spec_d):
    """The subset of `names` (object geoms) whose bounding sphere comes within the pair region: {"center": xyz} or
    {"points": [xyz, ...]} (centre = their mean) with "radius_m" (default: the points' spread plus the hand's span
    plus "margin_m", 0.02). Positions are the objects' compiled initial poses."""
    m = sp.compile()
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    pts = np.atleast_2d(np.asarray(region.get("points", [region.get("center")]), float))
    if pts.size == 0 or not np.all(np.isfinite(pts)):
        raise ValueError("pair_region needs 'center' [x, y, z] or 'points' [[x, y, z], ...]")
    c = np.asarray(region.get("center", pts.mean(axis=0)), float)
    r = region.get("radius_m")
    if r is None:
        rig = HandRig.build(m, f"{hand_name}_", side, spec_d)
        r = float(np.linalg.norm(pts - c, axis=1).max()) + hand_span(m, rig) + float(region.get("margin_m", 0.02))
    keep = []
    for nm in names:
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, nm)
        if g < 0:
            continue
        rb = float(m.geom_rbound[g]) if float(m.geom_rbound[g]) > 0 else 1e9
        if float(np.linalg.norm(d.geom_xpos[g] - c)) - rb <= float(r):
            keep.append(nm)
    return keep, float(r), c
# the floor has the bench table's contact (setup/scene.xml's default geom: solref 0.005, solimp 0.99 0.999 0.0005, MuJoCo's
# default friction): with MuJoCo's softer 0.02 / 0.9 0.95 an upright AA cell's terminal bumps sank and the same clamp
# that held it on the bench tipped it in the hand scene (measured by the grasp robustness package)
_FLOOR = ('<geom name="floor" type="plane" size="2 2 .05" pos="0 0 0" '
          'rgba="0.62 0.60 0.57 1" friction="1 0.005 0.0001" solref="0.005 1" solimp="0.99 0.999 0.0005"/>')


def _floor_xml(floor) -> str:
    """The hand scene's support: True = the infinite plane at z = 0; a dict {center, half_extents[, quat]} = the bench
    table as the finite box it is (same contact), so a hand may reach under a part that overhangs its edge; False =
    none."""
    if isinstance(floor, dict):
        c = [float(v) for v in floor["center"]]; h = [float(v) for v in floor["half_extents"]]
        q = [float(v) for v in floor.get("quat", (1.0, 0.0, 0.0, 0.0))]
        return (f'<geom name="floor" type="box" size="{h[0]} {h[1]} {h[2]}" pos="{c[0]} {c[1]} {c[2]}" '
                f'quat="{q[0]} {q[1]} {q[2]} {q[3]}" rgba="0.62 0.60 0.57 1" friction="1 0.005 0.0001" '
                'solref="0.005 1" solimp="0.99 0.999 0.0005"/>')
    return _FLOOR if floor else ""


def floor_surface(model, d, floor_gid):
    """The floor's top surface: (point on it, unit up normal, footprint) where footprint is None for a plane and
    (centre of the top face, x axis, y axis, half x, half y) for a box floor (a finite table: points beyond its edge
    are not under it)."""
    p0 = np.asarray(d.geom_xpos[floor_gid], float)
    R = np.asarray(d.geom_xmat[floor_gid], float).reshape(3, 3)
    if int(model.geom_type[floor_gid]) != int(mujoco.mjtGeom.mjGEOM_BOX):
        return p0, R[:, 2], None
    hx, hy, hz = (float(v) for v in model.geom_size[floor_gid])
    top = p0 + R[:, 2] * hz
    return top, R[:, 2], (top, R[:, 0], R[:, 1], hx, hy)


def _over_floor(W, fp):
    """Mask of the points W [n, 3] that lie over the floor's footprint (all of them for a plane)."""
    if fp is None:
        return np.ones(len(W), bool)
    c, ex, ey, hx, hy = fp
    return (np.abs((W - c) @ ex) <= hx) & (np.abs((W - c) @ ey) <= hy)


def _attach_asset(spec, xml_path, name, pos=(0, 0, 0), yaw_deg=0.0, quat=None):
    """Attach an MJCF asset under prefix `name_`, without its own lights or
    ground plane (a second floor at the asset's own height corrupts every
    clearance number afterwards)."""
    src = Path(xml_path)
    txt = src.read_text()
    txt = re.sub(r"[ \t]*<light\b[^/]*/>\s*\n?", "", txt)
    txt = re.sub(r"[ \t]*<geom\b[^>]*type=\"plane\"[^>]*/>\s*\n?", "", txt)
    import os as _os
    import uuid as _uuid
    # a name no other process uses: parallel hand-scene builds once raced on one file and read it empty or missing
    stripped = src.with_name(src.stem + f"__attach_{name}_{_os.getpid()}_{_uuid.uuid4().hex[:8]}.xml")
    stripped.write_text(txt)
    try:
        child = mujoco.MjSpec.from_file(str(stripped))
    finally:
        try:
            stripped.unlink()
        except OSError:
            pass
    # objects are found by their name prefix (contact pairs, grasp targets, clearances): an unnamed geom would stay
    # invisible to the hand, which then closes straight through it
    used = {g.name for g in child.geoms if g.name}
    for i, g in enumerate(child.geoms):
        if not g.name:
            nm = f"geom{i}"
            while nm in used:
                nm += "_"
            g.name = nm; used.add(nm)
    f = spec.worldbody.add_frame()
    f.pos = [float(v) for v in pos]
    if quat is not None:
        f.quat = [float(v) for v in quat]
    else:
        yaw = np.radians(float(yaw_deg))
        f.quat = [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]
    from tool_library.common.contact_defaults import asset_default_names
    contact_names = asset_default_names(child, src, prefix=f"{name}_")
    spec.attach(child, prefix=f"{name}_", frame=f)
    return contact_names


def _hand_self_geom_pairs(model, gids):
    """Visible link pairs, independent of whether physics contacts are installed.

    Planning scenes deliberately omit many explicit contact pairs. Their
    geometric open/clamp checks must still see exactly the same hand links as
    the execution and standalone grasp models.
    """
    gids = sorted(g for g in gids
                  if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                  and model.geom_group[g] <= 2 and model.geom_rgba[g, 3] > 0
                  and "collision_hand_" not in (model.geom(g).name or ""))
    for i, g in enumerate(gids):
        bg = int(model.geom_bodyid[g]); pg = int(model.body_parentid[bg])
        for h in gids[i + 1:]:
            bh = int(model.geom_bodyid[h]); ph = int(model.body_parentid[bh])
            if bg == bh or bg in (ph, int(model.body_parentid[ph])) or bh in (pg, int(model.body_parentid[pg])):
                continue
            yield g, h


def add_hand_self_contacts(spec):
    """Make the visible hand links physically collide with one another.

    Coarse pad capsules are useful for object contact but overlap the adjacent
    finger/palm even in a valid assembled hand. Self contact instead uses the
    existing visible meshes, so it matches the independent visual audit. Only
    the same body and its parent/grandparent are mechanically exempt.
    """
    for body in spec.bodies:
        if "left_" not in (body.name or "") and "right_" not in (body.name or ""):
            continue
        for i, geom in enumerate(body.geoms):
            if not geom.name and geom.type == mujoco.mjtGeom.mjGEOM_MESH and geom.group <= 2:
                geom.name = f"{body.name}__self_visual_{i}"
    m = spec.compile()
    existing = {frozenset((p.geomname1, p.geomname2)) for p in spec.pairs}
    added = []
    for b in range(m.nbody):
        name = m.body(b).name or ""
        if not name.endswith(("left_hand_C_MC", "right_hand_C_MC")):
            continue
        prefix = name[:-len("hand_C_MC")]
        gids = [g for g in range(m.ngeom)
                if (m.body(int(m.geom_bodyid[g])).name or "").startswith(prefix)
                and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                and m.geom_group[g] <= 2 and m.geom_rgba[g, 3] > 0
                and "collision_hand_" not in (m.geom(g).name or "")]
        for g, h in _hand_self_geom_pairs(m, gids):
            gn, hn = m.geom(g).name, m.geom(h).name
            if not gn or not hn:
                raise ValueError("visible hand self-contact requires named geoms")
            key = frozenset((gn, hn))
            if key in existing:
                continue
            p = spec.add_pair(); p.name = f"handself__{gn}__{hn}"
            p.geomname1, p.geomname2 = gn, hn
            p.condim = 3; p.solref = [0.006, 1.]
            p.solimp = [0.99, 0.999, 0.001, 0.5, 2.]
            p.friction = [0.5, 0.5, 0.005, 0.0001, 0.0001]
            existing.add(key); added.append(p.name)
    return added


def build_hand_scene(objects, hand_xml, side="right", hand_name="hand", *,
                     spec=None, floor=True, timestep=0.002, contact_pairs=True,
                     pair_solref=(0.006, 1.0), friction=HAND_PAIR_FRICTION,
                     wrist_kp=10000.0, settle_s=0.8, park=(0.0, 0.0, 1.5), joint_init=None,
                     grasp_body=None, finger_force_nm=FINGER_FORCE_NM, wrist_force_n=WRIST_FORCE_N,
                     wrist_torque_nm=WRIST_TORQUE_NM, gravcomp=True, hand_bits=HAND_CONTACT_BITS,
                     no_hand_pairs=(), pair_region=None):
    """Compile floor + objects + one hand.

    objects: list of {"name", "xml", "pos"=(0,0,0), "yaw_deg"=0 | "quat", "hold"=False} (hold: the object is held in
    the air at this moment and keeps its pose through the settle).
    Returns (model, data0, rig, ids) where data0 has the objects settled and
    the hand parked at `park` with a mildly open posture, and ids holds
    "hand_bids", "hand_gids", "obj_gids" (all objects), "grasp_gids" (the LAST
    object, the one being grasped), "floor_gid", "n_pairs", "unpaired" (object
    geom -> why it got no hand pair).

    With contact_pairs, every hand collision geom gets an explicit pair with
    the floor and with every object geom the hand may touch: the hand assets
    ship with all contacts off, so without pairs the fingers close straight
    through an object. An object geom is paired when its contype/conaffinity
    meet `hand_bits` and it does not opt out (class or name ending in
    'nohand', or listed in `no_hand_pairs`); see the module docstring.
    `pair_region` ({"center": xyz} or {"points": [xyz, ...]}, optional
    "radius_m" / "margin_m") pairs only the object geoms near the contact
    region: a scene of 75 ring walls made every hand-scene step pay for 1400
    pairs. The default radius is the points' spread plus the hand's own span,
    so nothing the posed hand can reach is left out."""
    base = _SCENE_XML.format(name="hand_scene", dt=float(timestep), floor=_floor_xml(floor))
    sp = mujoco.MjSpec.from_string(base)
    rigid_contact_names = []
    for o in objects:
        rigid_contact_names.extend(_attach_asset(sp, o["xml"], o["name"], o.get("pos", (0, 0, 0)),
                                                o.get("yaw_deg", 0.0), o.get("quat")))
    from tool_library.common.contact_defaults import add_rigid_pairs
    add_rigid_pairs(sp, rigid_contact_names)
    hand = mujoco.MjSpec.from_file(str(hand_xml))
    from tool_library.common.handgeom import add_hand_hulls
    add_hand_hulls(hand)                    # palm / thumb collision = the hull of what the camera sees
    sp.attach(hand, prefix=f"{hand_name}_", frame=sp.worldbody.add_frame())
    if gravcomp:
        # the bench's arm carries the hand's weight (build_execution_model): the floating test hand must too, or a
        # pitched pose saturates the wrist servo's torque and the fingertips sag off the planned path
        for b in sp.bodies:
            if (b.name or "").startswith(f"{hand_name}_"):
                b.gravcomp = 1.0
    spec_d = dict(DEFAULT_HAND_SPEC, **(spec or {}))
    cpat = f"{hand_name}_" + spec_d["collision_geom"].format(side=side)
    hand_geoms = [g.name for g in sp.geoms if (g.name or "").startswith(cpat)]
    obj_prefixes = tuple(f"{o['name']}_" for o in objects)
    obj_geoms, unpaired = hand_pair_geoms(sp.geoms, obj_prefixes, hand_bits, no_hand_pairs)
    touchable = set(obj_geoms)               # what clearances and overlaps are measured against (the region aside)
    region_info = None
    if contact_pairs and pair_region:
        near, r_m, c = _region_geoms(sp, obj_geoms, dict(pair_region), hand_name, side, spec_d)
        for nm in obj_geoms:
            if nm not in near:
                unpaired[nm] = "outside the pair region"
        obj_geoms = near
        region_info = {"center": [round(float(v), 4) for v in c], "radius_m": round(r_m, 4), "n_geoms": len(near)}
    n_pairs = 0
    if contact_pairs:
        for hg in hand_geoms:
            for og in obj_geoms + (["floor"] if floor else []):
                pr = sp.add_pair()
                pr.name = f"grip__{hg}__{og}"
                pr.geomname1, pr.geomname2 = hg, og
                if og == "floor":
                    # the hand on the table as the bench has it (its scenes carry hand-table pairs solref 0.01,
                    # friction 1, condim 3): stiffer, grippier floor pairs let fingers that rest on the table close
                    # onto a pen in stress tests and miss it on the bench
                    pr.solref = list(TABLE_PAIR["solref"]); pr.friction = list(TABLE_PAIR["friction"]); pr.condim = int(TABLE_PAIR["condim"])
                else:
                    pr.solref = list(pair_solref)
                    pr.solimp = [0.99, 0.999, 0.001, 0.5, 2]
                    pr.friction = [friction[0], friction[0], friction[1], friction[2], friction[2]]
                    pr.condim = 6
                n_pairs += 1
    add_hand_self_contacts(sp)
    model = sp.compile()
    data0 = mujoco.MjData(model)
    rig = HandRig.build(model, f"{hand_name}_", side, spec)
    rig.teleport(data0, np.asarray(park, float), np.zeros(3))
    open_t = {k: lo + 0.25 * (hi - lo) for k, (lo, hi) in rig.finger_range.items()}
    rig.set_fingers(data0, open_t)
    # the servos must be TOLD to hold the park pose, or ctrl=0 drags the hand
    # to the origin (through the object) while the scene settles
    hold_ctrl(data0, rig, np.asarray(park, float), np.zeros(3), open_t)
    if wrist_kp:
        set_wrist_gain(model, rig, kp=float(wrist_kp))
    limit_hand_forces(model, rig, finger_force_nm, wrist_force_n, wrist_torque_nm)
    # the wrist mount stack is solver plumbing: hide it in renders
    for g in range(model.ngeom):
        if "_base_" in (model.body(model.geom_bodyid[g]).name or ""):
            model.geom_rgba[g, 3] = 0.0
    held = []
    for jn, v in (joint_init or {}).items():     # the objects' internal joints as the scene has them now
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            raise ValueError(f"joint_init: no joint {jn!r} in the hand scene")
        adr = int(model.jnt_qposadr[jid])
        v = np.atleast_1d(np.asarray(v, float))
        data0.qpos[adr:adr + len(v)] = v
        if int(model.jnt_type[jid]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            held.append((adr, int(model.jnt_dofadr[jid]), float(v[0])))
    mujoco.mj_forward(model, data0)
    # objects held in the air at this moment (by the other hand, mid-handover): their free joints keep the pose the
    # scene gives them; gravity still acts in every test, so a grasp is never proven on a floating part
    in_air = []
    for o in objects:
        if not o.get("hold"):
            continue
        for j in range(model.njnt):
            if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE) and (model.body(int(model.jnt_bodyid[j])).name or "").startswith(f"{o['name']}_"):
                a, dof = int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])
                in_air.append((a, dof, data0.qpos[a:a + 7].copy()))
    if settle_s and settle_s > 0:          # measure against the SETTLED scene
        for _ in range(int(settle_s / model.opt.timestep)):
            mujoco.mj_step(model, data0)
            # a hinge or slide set by joint_init is the state the poses are solved in (a lid held at 140 deg, a
            # drawer pulled out): settling must not let it fall back to its rest
            for adr, dof, v in held:
                data0.qpos[adr] = v; data0.qvel[dof] = 0.0
            for adr, dof, q in in_air:
                data0.qpos[adr:adr + 7] = q; data0.qvel[dof:dof + 6] = 0.0
        data0.qvel[:] = 0.0
        mujoco.mj_forward(model, data0)
    floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor") if floor else -1
    obj_gids = {g for g in range(model.ngeom)
                if (model.geom(g).name or "").startswith(obj_prefixes)}
    # distances and contacts are measured against the geoms the hand can touch only: a visual mesh shell gives
    # mj_geomDistance false zeros against another mesh and is never what the physics touches, and a geom in a
    # collision bit of its own (a magnet sphere filling a ring's bore) would make every pose in the bore "overlap"
    touch = {g for g in obj_gids if (model.geom(g).name or "") in touchable}
    coll = {g for g in obj_gids if model.geom_contype[g] or model.geom_conaffinity[g]}
    obj_gids = touch or coll or obj_gids
    grasp_prefix = f"{objects[-1]['name']}_" if objects else ""
    grasp_gids = {g for g in obj_gids if (model.geom(g).name or "").startswith(grasp_prefix)}
    if grasp_body:                           # a part of an asset (a cap, a drawer front): its own subtree only
        gb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, grasp_body)
        if gb < 0:
            raise ValueError(f"grasp_body {grasp_body!r} is not in the hand scene")
        sub = {gb}
        changed = True
        while changed:
            changed = False
            for b in range(model.nbody):
                if int(model.body_parentid[b]) in sub and b not in sub:
                    sub.add(b); changed = True
        part = {g for g in obj_gids if int(model.geom_bodyid[g]) in sub}
        if part:
            grasp_gids = part
    ids = {"hand_bids": set(rig.body_ids), "hand_gids": set(rig.geom_ids),
           "obj_gids": obj_gids, "grasp_gids": grasp_gids or obj_gids,
           "grasp_prefix": grasp_prefix, "floor_gid": int(floor_gid), "n_pairs": n_pairs,
           "unpaired": unpaired, "pair_region": region_info,
           # free joints of the parts held in the air at this moment (qpos address, dof address): the stress tests
           # hold them as the other hand does until this hand has closed (a handover)
           "held_free": [(int(a_), int(d_)) for a_, d_, _ in in_air]}
    return model, data0, rig, ids


def body_geoms(model, body: str) -> set:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    if bid < 0:
        raise ValueError(f"body '{body}' is not in the scene")
    return {g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == bid}


def subtree_geoms(model, body: str) -> set:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    if bid < 0:
        raise ValueError(f"body '{body}' is not in the scene")
    bodies, stack = set(), [bid]
    while stack:
        b = stack.pop()
        bodies.add(b)
        stack += [c for c in range(model.nbody) if model.body_parentid[c] == b and c != b]
    return {g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies}


# ---------------------------------------------------------------------------
# posing, measuring
# ---------------------------------------------------------------------------
def hold_ctrl(data, rig: HandRig, pos, euler, fingers: dict | None):
    rig.wrist_ctrl(data.ctrl, pos, euler)
    if fingers:
        rig.finger_ctrl(data.ctrl, fingers)


def apply_pose(model, data, rig: HandRig, pose: dict, fingers: dict | None = None):
    """Put the hand at a pose in qpos AND ctrl, so the servos hold it there."""
    pos, eul = np.asarray(pose["pos"], float), np.asarray(pose["euler"], float)
    rig.teleport(data, pos, eul)
    tg = fingers if fingers is not None else (pose.get("finger_targets") or {})
    rig.set_fingers(data, tg)
    hold_ctrl(data, rig, pos, eul, tg)
    mujoco.mj_forward(model, data)


PRESS_DEPTH_RAD = 0.3      # how far past touch the gripping fingers are commanded: with the servo torque capped the
                           # clamp force is the cap, and the depth is how far an object may shift before a pad lets go


def press_targets(rig: HandRig, closed: dict, grip_fingers=(), depth_rad: float = PRESS_DEPTH_RAD, squeezed: dict | None = None,
                  max_travel_m: float | None = None) -> dict:
    """Clamp targets: the flexion joints of the gripping fingers (all fingers when none are named) are pushed
    `depth_rad` past the closed (just-touching) targets, clipped to their ranges; abduction and opposition stay.
    A tool's own squeezed targets are kept where they go deeper. A negative `depth_rad` drives the grip fingers
    toward extension instead (spreading what they hold, the rings of scissors): squeezed targets, which flex, do not
    apply then. `max_travel_m` caps the pads' displacement instead (a one-sided push: pads driven 8 mm into a 4 mm
    lid plate tunnel through it and push the wrong way)."""
    if max_travel_m is not None and depth_rad > 0:
        d = mujoco.MjData(rig.model)
        for k, v in closed.items():
            if k in rig.finger_qpos:
                d.qpos[rig.finger_qpos[k]] = float(v)
        mujoco.mj_kinematics(rig.model, d)
        grip = set(grip_fingers or FINGERS)
        tips0 = {f: d.site_xpos[rig.fingertip_sites[f]].copy() for f in grip if f in rig.fingertip_sites}
        lo_s, hi_s = 0.0, float(depth_rad)
        for _ in range(12):                              # bisection on the depth for the travel cap
            mid = 0.5 * (lo_s + hi_s)
            trial = press_targets(rig, closed, grip_fingers, mid, None)
            for k, v in trial.items():
                if k in rig.finger_qpos:
                    d.qpos[rig.finger_qpos[k]] = float(v)
            mujoco.mj_kinematics(rig.model, d)
            travel = max((np.linalg.norm(d.site_xpos[rig.fingertip_sites[f]] - tips0[f]) for f in tips0), default=0.0)
            if travel > max_travel_m:
                hi_s = mid
            else:
                lo_s = mid
        depth_rad = lo_s
    out = dict(closed)
    grip = set(grip_fingers or FINGERS)
    for k, v in closed.items():
        fin = finger_of_joint(k)
        if fin in grip and joint_role(k) in ("flex", "pinky_arch"):
            lo, hi = rig.finger_range[k]
            deeper = float(np.clip(v + float(depth_rad) * np.sign(hi - lo if hi != lo else 1.0), min(lo, hi), max(lo, hi)))
            out[k] = deeper
    if squeezed and float(depth_rad) >= 0.0:
        for k, v in squeezed.items():
            if k in out and finger_of_joint(k) in grip and joint_role(k) in ("flex", "pinky_arch"):
                lo, hi = rig.finger_range[k]
                out[k] = float(max(out[k], v)) if hi >= lo else float(min(out[k], v))
    return out


def pad_positions(rig: HandRig, targets: dict) -> dict:
    """Fingertip site positions (in the hand's own frame: the wrist at the origin) for a set of finger targets."""
    d = mujoco.MjData(rig.model)
    for k, v in targets.items():
        if k in rig.finger_qpos:
            d.qpos[rig.finger_qpos[k]] = float(v)
    mujoco.mj_kinematics(rig.model, d)
    return {f: d.site_xpos[s].copy() for f, s in rig.fingertip_sites.items()}


IDLE_SPREAD_RAD = (0.0, 0.15, -0.15, 0.3, -0.3)   # abductions tried for a curled idle finger, smallest first


def spread_idle_fingers(model, d, rig: HandRig, idle_fingers, tol_m=0.001) -> float:
    """Abduct curled idle fingers apart until their visible meshes stop intersecting: at equal flexion the ring and pinky
    synergy crosses by 5-7 mm, while either one abducted 0.15 rad away from the other is clear at every fold. Greedy from
    the outermost finger, each abduction taken from IDLE_SPREAD_RAD (within range); writes d.qpos and returns the idle
    fingers' remaining self-overlap (m)."""
    idle = [f for f in ("pinky", "ring", "middle", "index") if f in set(idle_fingers)]
    kinematic_forward(model, d)                       # the caller may just have written the fold into qpos
    depth = hand_self_overlap(model, d, rig, fingers=idle)
    for f in idle:
        if depth <= tol_m:
            break
        key = next((k for k in rig.finger_qpos if finger_of_joint(k) == f and joint_role(k) == "abduct"), None)
        if key is None:
            continue
        adr = rig.finger_qpos[key]; lo, hi = rig.finger_range[key]
        best = (depth, float(d.qpos[adr]))
        for a in IDLE_SPREAD_RAD[1:]:
            d.qpos[adr] = float(np.clip(a, lo, hi))
            kinematic_forward(model, d)
            v = hand_self_overlap(model, d, rig, fingers=idle)
            if v < best[0] - 1e-5:
                best = (v, float(d.qpos[adr]))
        d.qpos[adr] = best[1]
        kinematic_forward(model, d)
        depth = best[0]
    return depth


def open_targets(rig: HandRig, closed: dict, frac: float = 0.55, curl=()) -> dict:
    """Back every flexion joint of a closed pose off toward its extended end
    (the low end of its range, the convention finger_targets closes from)
    by `frac` of the way; abduction and thumb opposition stay. Fingers named
    in `curl` are folded into the palm instead: an idle finger that opens
    wide knocks the object it is not holding (measured on a cord pinch, the
    open middle/ring/pinky shoved the cord 160 mm aside)."""
    out, curl = {}, set(curl or ())
    for k, v in (closed or {}).items():
        if k not in rig.finger_range:
            continue
        lo, hi = rig.finger_range[k]
        role = joint_role(k)
        # an idle THUMB opens rather than curls: folded across the palm it
        # sits exactly where the object goes
        if finger_of_joint(k) in curl and finger_of_joint(k) != "thumb":
            out[k] = float(np.clip(0.0, lo, hi)) if role == "abduct" else (float(hi) if role in ("flex", "pinky_arch") else float(v))
            continue
        if role in ("flex", "pinky_arch"):
            out[k] = float(v + (lo - v) * float(frac))
        else:
            out[k] = float(v)
    if curl:
        # Idle fingers cannot be driven to their independent joint maxima:
        # that synergy crosses the ring/pinky meshes by over a centimetre.
        # Keep active fingers' requested opening and choose a compact, valid
        # idle posture. This only changes automatically generated idle targets.
        # Only the idle fingers' own overlap (with each other, the palm and the active fingers) is scored: an overlap
        # between two active fingers is the same at every fold and would otherwise pin the fold at its first value.
        d = mujoco.MjData(rig.model)
        idle = {k for k in out if finger_of_joint(k) in curl and finger_of_joint(k) != "thumb"}
        idle_fingers = {finger_of_joint(k) for k in idle}
        best = None
        for fold in (.95, .85, .65, .45, .25, .1, 0.):
            candidate = dict(out)
            for key in idle:
                lo, hi = rig.finger_range[key]
                if joint_role(key) in ("flex", "pinky_arch"):
                    candidate[key] = float(lo + fold * (hi - lo))
            for key, value in candidate.items():
                d.qpos[rig.finger_qpos[key]] = value
            kinematic_forward(rig.model, d)
            depth = hand_self_overlap(rig.model, d, rig)
            if best is None or depth < best[0]:
                best = (depth, candidate)
            if depth <= .001:
                break
        out = best[1]
    return out


ONE_SIDED_PRESS_M = 0.004  # pad travel of a one-sided press (no thumb opposing the grip fingers): a few millimetres keep
                           # the pads on their side of a thin part (8 mm into a 4 mm lid plate tunnelled through it)


def zero_wrist_frame(rig: HandRig) -> np.ndarray:
    """World rotation of the wrist frame with every wrist joint at zero: the frame pad_positions measures in (the
    root body's parent frame times its own offset, as plan.root_frame reads it)."""
    m = rig.model
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    b = int(rig.root_body)
    parent = int(m.body_parentid[b])
    Rp = np.asarray(d.xmat[parent], float).reshape(3, 3) if parent > 0 else np.eye(3)
    Rb = np.zeros(9)
    mujoco.mju_quat2Mat(Rb, np.asarray(m.body_quat[b], float))
    return Rp @ Rb.reshape(3, 3)


def pad_shift_world(rig: HandRig, closed: dict, press: dict, euler) -> dict:
    """{finger: world displacement (m) of its fingertip site} when the fingers go from `closed` to `press` with the
    wrist at the world orientation `euler` (the pose's own). pad_positions measures in the hand's frame at zero wrist
    joints, so the displacement is turned by the pose's rotation before it meets a world direction such as a contact
    normal: compared raw, a fingers-down index whose flexion lifts its pad 6.7 mm off a ribbon strip read as a 5 mm
    press into it, the planner kept the finger press and the pad left the strip (15.6 mm off at the hold)."""
    p0, p1 = pad_positions(rig, closed), pad_positions(rig, press)
    R = euler_to_mat(np.asarray(euler, float)) @ zero_wrist_frame(rig).T
    return {f: R @ (p1[f] - p0[f]) for f in p0}


def executor_clamp(rig: HandRig, pose: dict) -> dict:
    """The clamp the Stage 3 executor drives for a pose, shared by SubgoalPlanner._grip_from_pose and the grasp stress
    tests (grasp_robust.executor_grip, stress_rollout) so they press the same way:
    {closed, press, wrist_press (world offset in m, or None), one_sided, grip_fingers, grip_posed, pad_into_m}.

    The press is press_targets (flexion past touch, the pose's own squeezed targets kept where deeper; press_depth_rad
    and press_travel_m from the pose). A one-sided grip (no thumb opposing the grip fingers: a push, a cradle, a
    fingertip press) presses by ONE_SIDED_PRESS_M of pad travel. When that flexion moves the pads off their surfaces
    (mean travel along minus the contact normals, in the world, below zero: a fingers-down fingertip, a flat hand on a
    floor, fingers under a lid) it is no press: the fingers stay as posed (grip_posed), the wrist moves the press travel
    into the surface along the mean normal, and the approach keeps the grip fingers as posed too (approach_open_targets),
    since opening them is the reverse motion and drives the pads into the surface on the way down."""
    closed = {k: float(v) for k, v in (pose.get("finger_targets") or {}).items() if k in rig.finger_range}
    squeezed = {k: float(v) for k, v in (pose.get("finger_targets_squeezed") or {}).items() if k in rig.finger_range}
    grip = list(pose.get("grip_fingers") or [])
    one_sided = bool(grip) and "thumb" not in grip
    travel = pose.get("press_travel_m", ONE_SIDED_PRESS_M if one_sided else None)
    press = press_targets(rig, closed, grip, depth_rad=float(pose.get("press_depth_rad", PRESS_DEPTH_RAD)), squeezed=squeezed,
                          max_travel_m=None if travel is None else float(travel))
    wrist_press, posed, into = None, False, None
    normals = {f: np.asarray(n, float) for f, n in (pose.get("normals") or {}).items()}
    if one_sided and normals and pose.get("euler") is not None:
        shift = pad_shift_world(rig, closed, press, pose["euler"])
        along = [float(shift[f] @ -(n / max(np.linalg.norm(n), 1e-9))) for f, n in normals.items() if f in shift]
        if along:
            into = float(np.mean(along))
        if along and into < 0:
            press, posed = dict(closed), True
            nbar = np.mean([n / max(np.linalg.norm(n), 1e-9) for n in normals.values()], axis=0)
            if travel and float(travel) > 0 and np.linalg.norm(nbar) > 0.5:
                wrist_press = [float(v) for v in -nbar / np.linalg.norm(nbar) * float(travel)]
    # Additional squeezing must not command one finger through another. Find
    # the first collision-free part of the closure path, shared by execution
    # and every standalone stress test. Object contact still supplies the grip.
    probe = mujoco.MjData(rig.model)
    for key, value in closed.items():
        probe.qpos[rig.finger_qpos[key]] = value
    kinematic_forward(rig.model, probe)
    if hand_self_overlap(rig.model, probe, rig) <= 0.001:
        previous = 0.
        for alpha in np.linspace(0.05, 1., 20):
            for key, value in press.items():
                probe.qpos[rig.finger_qpos[key]] = closed.get(key, value) + alpha * (value - closed.get(key, value))
            kinematic_forward(rig.model, probe)
            if hand_self_overlap(rig.model, probe, rig) > 0.001:
                press = {key: closed.get(key, value) + previous * (value - closed.get(key, value)) for key, value in press.items()}
                break
            previous = float(alpha)
    return {"closed": closed, "press": press, "wrist_press": wrist_press, "one_sided": one_sided, "grip_fingers": grip,
            "grip_posed": posed, "pad_into_m": into}


def approach_open_targets(rig: HandRig, closed: dict, frac: float, grip_fingers=(), grip_posed: bool = False) -> dict:
    """The open shape the executor reaches and descends with: open_targets with the idle fingers curled (when grip
    fingers are named); with grip_posed (executor_clamp: a one-sided press whose flexion lifts the pads) the grip fingers
    keep their posed targets, because opening them pushes the pads into the surface before the hand arrives (a
    fingers-down index opened 55% reached 6.9 mm into a ribbon strip and shoved it 9.4 mm)."""
    grip = list(grip_fingers or [])
    opened = open_targets(rig, closed, float(frac), curl=[f for f in FINGERS if grip and f not in grip])
    if grip_posed:
        for k, v in closed.items():
            if k in opened and finger_of_joint(k) in grip:
                opened[k] = float(v)
    return opened


def palm_frame(model, data0, rig: HandRig, grip: float = 0.55):
    """(palm body id, palm normal in that body's frame): the normal of the
    pocket between the palm and the curling fingers, measured by closing the
    hand, so it is the direction the palm faces an object it holds."""
    if rig.palm_body < 0:
        return None, None
    d = mujoco.MjData(model)
    d.qpos[:] = data0.qpos
    rig.teleport(d, np.zeros(3), np.zeros(3))

    def tips_at(g):
        rig.set_fingers(d, rig.finger_targets(g))
        mujoco.mj_forward(model, d)
        return ({f: np.asarray(d.site_xpos[s], float).copy() for f, s in rig.fingertip_sites.items()},
                np.asarray(d.xpos[rig.palm_body], float).copy())

    t0, P = tips_at(0.0)
    t1, _ = tips_at(0.85)
    m0 = np.mean([v for f, v in t0.items() if f != "thumb"], axis=0)
    m1 = np.mean([v for f, v in t1.items() if f != "thumb"], axis=0)
    f_ax = (m0 - P) / (np.linalg.norm(m0 - P) + 1e-12)
    curl = m1 - m0
    n = curl - np.dot(curl, f_ax) * f_ax
    n /= np.linalg.norm(n) + 1e-12
    rig.set_fingers(d, rig.finger_targets(grip))
    mujoco.mj_forward(model, d)
    R0 = np.asarray(d.xmat[rig.palm_body], float).reshape(3, 3)
    return int(rig.palm_body), R0.T @ n


def geom_distance(model, data, g1, g2, distmax=0.05, fromto=None) -> float:
    """mj_geomDistance with its false zeros removed: MuJoCo 3.10 returns an exact 0.0 for some separated box-box and
    hull-hull/hull-box pairs (measured: a thumb-tip hull 2-25 mm off a bar read 0.0, so a cinch took the thumb for
    touching, the pose listed it as gripping and the other hand's mirror pose, read correctly, did not). A real touch
    reads a small non-zero number, so an exact zero is re-measured with the first geom nudged 1 micrometre along each axis
    both ways and the smallest non-zero reading is kept (grasp_robust.geom_distance does the same)."""
    g1, g2 = int(g1), int(g2)
    dist = float(mujoco.mj_geomDistance(model, data, g1, g2, float(distmax), fromto))
    if dist != 0.0:
        return dist
    keep = np.array(data.geom_xpos[g1], float)
    best, ft = None, (np.zeros(6) if fromto is not None else None)
    for k in range(6):
        data.geom_xpos[g1] = keep + np.eye(3)[k % 3] * (1e-6 if k < 3 else -1e-6)
        v = float(mujoco.mj_geomDistance(model, data, g1, g2, float(distmax), ft))
        if v != 0.0 and (best is None or v < best):
            best = v
            if fromto is not None:
                fromto[:] = ft
    data.geom_xpos[g1] = keep
    return 0.0 if best is None else best


def hand_geom_distance(model, data, g1, g2, distmax=0.05) -> float:
    return geom_distance(model, data, g1, g2, distmax)


def clearance(model, data, rig: HandRig, ids: dict, ignore_tips=(), floor=True) -> dict:
    """Deepest interpenetration (mm) of the hand's COLLISION geoms with the
    objects and the floor in the static pose, excluding the pads of the
    fingers in `ignore_tips` (a pad on the surface is the point). Visual
    shells are reported separately: they are larger than the collision
    primitives and do not decide viability."""
    keep = set()
    for f in ignore_tips:
        keep |= rig.tip_geoms(f)
    targets = [("object", sorted(ids["obj_gids"]))]
    if floor and ids.get("floor_gid", -1) >= 0:
        targets.append(("floor", [ids["floor_gid"]]))
    worst = {"object_mm": 0.0, "floor_mm": 0.0}
    vis = {"object_mm": 0.0, "floor_mm": 0.0}
    where = {}
    for g in rig.geom_ids:
        if g in keep:
            continue
        live = g in rig.collision_gids
        for tag, gl in targets:
            for og in gl:
                pen = -hand_geom_distance(model, data, g, og, 0.0) * 1000.0
                if not live:
                    vis[f"{tag}_mm"] = max(vis[f"{tag}_mm"], pen)
                elif pen > worst[f"{tag}_mm"]:
                    worst[f"{tag}_mm"] = pen
                    where[tag] = [model.geom(g).name, model.geom(og).name]
    return {"object_mm": round(worst["object_mm"], 2), "floor_mm": round(worst["floor_mm"], 2),
            "visual_object_mm": round(vis["object_mm"], 2),
            "visual_floor_mm": round(vis["floor_mm"], 2), "worst_pair": where}


def finger_contacts(model, data, rig: HandRig, target_gids) -> dict:
    """How many fingers touch the target geoms, and the total normal force."""
    tg = set(target_gids)
    seen, force = set(), 0.0
    buf = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        hand = g1 if g1 in rig.geom_ids else (g2 if g2 in rig.geom_ids else None)
        other = g2 if hand == g1 else g1
        if hand is None or other not in tg:
            continue
        nm = model.geom(hand).name or ""
        for f in FINGERS:
            if f"_{f}_" in nm:
                seen.add(f)
        mujoco.mj_contactForce(model, data, i, buf)
        force += abs(float(buf[0]))
    return {"n_finger": len(seen), "fingers": sorted(seen), "force_N": round(force, 1)}


def wrist_body(rig: HandRig) -> int:
    """The rotating wrist frame, for either virtual or rigid arm-mounted hands."""
    if rig.model is not None and "rz" in rig.wrist_act:
        return int(rig.model.jnt_bodyid[int(rig.model.actuator_trnid[rig.wrist_act["rz"]][0])])
    return int(rig.root_body)


def object_in_wrist(data, rig: HandRig, bid: int) -> np.ndarray:
    """Object position in the wrist frame: a held object keeps a constant
    offset however far the hand travels or turns, so drift is measured here.
    The frame is the body the wrist's last rotation moves (rig.root_body is the
    translation base, which never turns: a held part read as drifting by its
    lever times the angle whenever the wrist was turned)."""
    p = np.asarray(data.xpos[bid], float)
    m = rig.model
    if m is not None and "rz" in rig.wrist_act:
        wb = wrist_body(rig)
        return np.asarray(data.xmat[wb], float).reshape(3, 3).T @ (p - np.asarray(data.xpos[wb], float))
    R = np.asarray(data.xmat[rig.root_body], float).reshape(3, 3)
    return R.T @ (p - rig.wrist_pos(data))


# ---------------------------------------------------------------------------
# fingertip inverse kinematics: put named pads on named points
# ---------------------------------------------------------------------------
IK_ITERS = 320            # damped-least-squares steps per seed
IK_LAMBDA = 3e-3          # DLS damping
IK_MAX_STEP = 0.06        # |dq| cap per joint per step (rad or m)
# a pad is ~15 mm of rubber around the fingertip site: the SITE stops this
# far outside the surface so the PAD rests on it (measured: driving the site
# to the surface buried the pad and the servos answered with 10 kN)
TIP_STANDOFF_M = 0.015
# interpenetration up to this depth is "a pad on a surface", deeper is pushed
# out in the null space of the fingertip task
PEN_OK_M = 0.004
SEED_BACK_M = 0.13


def pin_finger_shape(rig: HandRig, d) -> set:
    """Zero every abduction joint (parallel fingers) and set the thumb to its
    opposition convention; returns the qpos addresses IK must leave alone."""
    pinned = set()
    for short, adr in rig.finger_qpos.items():
        role = joint_role(short)
        lo, hi = rig.finger_range.get(short, (-9.0, 9.0))
        if role == "abduct":
            d.qpos[adr] = float(np.clip(0.0, lo, hi))
            pinned.add(int(adr))
        elif role == "thumb_oppose":
            d.qpos[adr] = float(lo + 0.55 * (hi - lo))
            pinned.add(int(adr))
    return pinned


def _ik_dofs(model, rig: HandRig, pinned=frozenset(), fix_orientation=False):
    q2d, q2r = {}, {}
    for j in range(model.njnt):
        q2d[int(model.jnt_qposadr[j])] = int(model.jnt_dofadr[j])
        lo, hi = model.jnt_range[j]
        q2r[int(model.jnt_qposadr[j])] = ((float(lo), float(hi)) if model.jnt_limited[j]
                                          else (-1e9, 1e9))
    wrist = ("x", "y", "z") if fix_orientation else ("x", "y", "z", "rx", "ry", "rz")
    order = [rig.wrist_qpos[k] for k in wrist]
    order += [rig.finger_qpos[k] for k in sorted(rig.finger_qpos)
              if int(rig.finger_qpos[k]) not in pinned]
    lims = [q2r[a] for a in order]
    for i in range(len(wrist)):
        lims[i] = (-1e9, 1e9)
    return np.array([q2d[a] for a in order], int), np.array(order, int), lims


def _escape_rows(model, d, dofs, hand_bids, obj_gids, keep_gids, max_rows=10, floor_gid=None, vis=None):
    """Secondary-task rows n^T J dq = depth that push PENETRATING hand
    geometry back out of the object and the floor. The pads in keep_gids may
    press into the object (that is their job) but not into the floor; with
    `vis` (visual_floor_hulls) the hand's visible meshes are pushed up out of
    the floor plane too, as the Stage 3 penetration audits judge them."""
    rows, rhs, cons = [], [], []
    jacp = np.zeros((3, model.nv)); jac_other = np.zeros((3, model.nv))
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        self_contact = b1 in hand_bids and b2 in hand_bids
        if float(c.dist) >= -(0.0005 if self_contact else PEN_OK_M):
            continue
        h1 = b1 in hand_bids and g2 in obj_gids
        h2 = b2 in hand_bids and g1 in obj_gids
        if self_contact:
            # Visible self contact never receives a fingertip/object exception.
            cons.append((max(0., -float(c.dist) - 0.0005), g1, np.asarray(c.pos, float),
                         np.asarray(c.frame[:3], float), 1.0, b2))
            continue
        if not (h1 or h2):
            continue
        hg = g1 if h1 else g2
        other = g2 if h1 else g1
        if hg in keep_gids and other != floor_gid:
            continue
        cons.append((float(-c.dist) - PEN_OK_M, hg, np.asarray(c.pos, float),
                     np.asarray(c.frame[:3], float), 1.0 if h1 else -1.0, None))
    if vis and floor_gid is not None and floor_gid >= 0:
        p0, n_up, fp = floor_surface(model, d, floor_gid)
        for g, V in vis:
            W = V @ np.asarray(d.geom_xmat[g], float).reshape(3, 3).T + np.asarray(d.geom_xpos[g], float)
            h = np.where(_over_floor(W, fp), (W - p0) @ n_up, np.inf)    # past a box floor's edge nothing is under it
            k = int(np.argmin(h))
            if not np.isfinite(h[k]):
                continue
            if -float(h[k]) > PEN_OK_M:
                # a row pushing the lowest vertex up along the floor normal (frame: from the floor into the hand, sgn -1)
                cons.append((-float(h[k]) - PEN_OK_M, g, W[k], -n_up, -1.0, None))
    cons.sort(key=lambda t: -t[0])
    for depth, hg, pos, nrm, sgn, other_body in cons[:int(max_rows)]:
        mujoco.mj_jac(model, d, jacp, None, pos, int(model.geom_bodyid[hg]))
        if other_body is not None:
            mujoco.mj_jac(model, d, jac_other, None, pos, other_body)
            rows.append(nrm @ (jac_other[:, dofs] - jacp[:, dofs]))
        else:
            rows.append((-sgn * nrm) @ jacp[:, dofs])
        rhs.append(depth)
    if not rows:
        return None, None, 0.0
    return np.asarray(rows), np.asarray(rhs), max(c[0] for c in cons)


def visual_floor_hulls(model, rig: HandRig) -> list:
    """[(geom, hull vertices in the geom frame)] for the hand's visible mesh geoms (its non-collision meshes), cached
    on the rig: what the Stage 3 penetration audits (object_visual_penetration_audit, swept_collision_audit) test
    against the table."""
    cached = getattr(rig, "_visual_hulls", None)
    if cached is not None:
        return cached
    out = []
    for g in sorted(rig.geom_ids - set(rig.collision_gids)):
        if int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        mid = int(model.geom_dataid[g])
        if mid < 0:
            continue
        a, nv = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
        V = np.asarray(model.mesh_vert[a:a + nv], float)
        if len(V) > 64:
            try:
                from scipy.spatial import ConvexHull
                V = V[ConvexHull(V).vertices]
            except Exception:
                pass
        out.append((g, V))
    rig._visual_hulls = out
    return out


def visual_floor_depth(model, d, rig: HandRig, floor_gid: int) -> float:
    """Deepest point (m, 0 when clear) of the hand's visible meshes below the floor plane (a box floor: below its top
    and over its footprint)."""
    if floor_gid is None or floor_gid < 0:
        return 0.0
    p0, n_up, fp = floor_surface(model, d, floor_gid)
    worst = 0.0
    for g, V in visual_floor_hulls(model, rig):
        W = V @ np.asarray(d.geom_xmat[g], float).reshape(3, 3).T + np.asarray(d.geom_xpos[g], float)
        over = _over_floor(W, fp)
        if over.any():
            worst = max(worst, -float(np.min((W[over] - p0) @ n_up)))
    return worst


def _geom_finger(model, g) -> str | None:
    """The finger whose link carries geom g (by its body name), or None for the palm."""
    name = model.body(int(model.geom_bodyid[g])).name or ""
    return next((f for f in FINGERS if f"_{f}_" in name), None)


def hand_self_overlap(model, d, rig: HandRig, fingers=None) -> float:
    """Deepest visible non-adjacent hand-link intersection, including pads. With `fingers`, only the pairs that involve a
    link of one of those fingers (their overlap with each other, the palm and the other fingers): what folding those
    fingers can change, not what the posed grip fingers do to each other."""
    pairs = getattr(rig, "_self_pairs", None)
    if pairs is None:
        pairs = list(_hand_self_geom_pairs(model, rig.geom_ids))
        rig._self_pairs = pairs
    owner = None
    if fingers is not None:
        fingers = set(fingers)
        owner = getattr(rig, "_self_pair_fingers", None)
        if owner is None:
            owner = {pair: (_geom_finger(model, pair[0]), _geom_finger(model, pair[1])) for pair in pairs}
            rig._self_pair_fingers = owner
    worst = 0.
    for g, h in pairs:
        if owner is not None and not fingers.intersection(owner[(g, h)]):
            continue
        if np.linalg.norm(d.geom_xpos[g] - d.geom_xpos[h]) > model.geom_rbound[g] + model.geom_rbound[h]:
            continue
        worst = max(worst, -float(mujoco.mj_geomDistance(model, d, g, h, 0., None)))
    return worst


def deepest_overlap(model, d, rig: HandRig, ids: dict, keep_tips=FINGERS, visual_floor=False, self_fingers=None) -> float:
    """Deepest hand-vs-object/floor interpenetration (m) over the hand's
    COLLISION geoms; the pads of `keep_tips` are excused against the objects
    (pressing a part is their job) but never against the floor (a pad 6-10 mm
    in the table read 0 before). With visual_floor the hand's visible meshes
    are measured against the floor plane as well, as the Stage 3 penetration
    audits measure them. mj_geomDistance, because the hand asset ships with
    contype 0; bounding-sphere prefilter because this runs once per IK seed. `self_fingers` limits the hand's
    self-overlap term to the pairs involving those fingers (hand_self_overlap)."""
    tip = set()
    for f in keep_tips:
        tip |= rig.tip_geoms(f)
    P = np.asarray(d.geom_xpos, float)
    R = np.asarray(model.geom_rbound, float)
    fl = ids.get("floor_gid", -1)
    others = list(ids["obj_gids"]) + ([fl] if fl >= 0 else [])
    worst = hand_self_overlap(model, d, rig, fingers=self_fingers)
    for g in rig.collision_gids:
        for og in others:
            if og != fl and (g in tip or float(np.linalg.norm(P[g] - P[og])) > float(R[g] + R[og]) + 1e-4):
                continue
            dist = mujoco.mj_geomDistance(model, d, g, og, 0.0, None)
            if dist < -worst:
                worst = -dist
    if visual_floor and fl >= 0:
        worst = max(worst, visual_floor_depth(model, d, rig, fl))
    return float(worst)


def seed_sweep(centroid, approach_dirs, back_m=SEED_BACK_M, yaw_step_deg=30,
               pitches_deg=(-90, -60, -30, 0, 30, 60, 90)):
    """Wrist seeds: back along each approach direction, swept over yaw and
    pitch. Multi-start is what keeps a local minimum from passing as the
    answer."""
    out = []
    for ap in approach_dirs:
        ap = np.asarray(ap, float)
        ap = ap / (np.linalg.norm(ap) + 1e-12)
        base = np.asarray(centroid, float) + ap * float(back_m)
        for ydeg in range(0, 360, int(yaw_step_deg)):
            for pdeg in pitches_deg:
                out.append((base, [np.pi, np.radians(pdeg), np.radians(ydeg)]))
    return out


def kinematic_forward(model, d, contacts=False):
    """What the kinematic solvers read (body, site and geom poses, Jacobians and, with `contacts`, the contact list)
    without the constraint solver mj_forward runs after it. The results are identical; the cost is not: with 370
    contacts in a hand scene (three magnetic rings of 25 geoms each on the floor) the solver is 15 of mj_forward's
    15.2 ms, and the fingertip IK calls it once per iteration of every seed."""
    mujoco.mj_kinematics(model, d)
    mujoco.mj_comPos(model, d)
    if contacts:
        mujoco.mj_collision(model, d)


def solve_fingertip_ik(model, data0, rig: HandRig, ids: dict, targets: dict, seeds,
                       face_at=None, face_dir=None, parallel=False, iters=IK_ITERS,
                       lam=IK_LAMBDA, max_step=IK_MAX_STEP, fix_orientation=False,
                       time_budget_s=None, stats=None):
    """Drive the fingertip sites in `targets` {finger: xyz} onto their points
    from every seed, keep the rest of the hand out of the objects (null-space
    escape task), and return the best solution:
    (worst_residual_m, residual_per_finger, overlap_m, qpos, palm_facing).
    `fix_orientation` keeps each seed's wrist orientation (only the wrist position and the
    fingers move): fingers that must stay level under a thin lid, a hand whose orientation
    the arm's reach dictates. `time_budget_s` stops trying further seeds once that much time
    has passed (the first seed always runs) and returns the best so far; a dict passed as
    `stats` receives seeds_total, seeds_tried, elapsed_s and stopped ('time_budget' or None)."""
    import time as _time
    t_start = _time.monotonic()
    seeds = list(seeds)
    fingers = [f for f in targets if f in rig.fingertip_sites]
    if not fingers:
        raise ValueError(f"no target finger has a tip site; fingers are {sorted(rig.fingertip_sites)}")
    sites = [int(rig.fingertip_sites[f]) for f in fingers]
    goal = np.concatenate([np.asarray(targets[f], float) for f in fingers])
    probe = mujoco.MjData(model)
    probe.qpos[:] = data0.qpos
    pinned = pin_finger_shape(rig, probe) if parallel else frozenset()
    dofs, qadr, lims = _ik_dofs(model, rig, pinned, fix_orientation=fix_orientation)
    n = len(dofs)
    tip_gids = rig.tip_geoms()
    hand_bids = set(ids["hand_bids"])
    obj_gids = set(ids["obj_gids"]) | ({ids["floor_gid"]} if ids.get("floor_gid", -1) >= 0 else set())
    palm_bid, palm_n = palm_frame(model, data0, rig)
    vis = visual_floor_hulls(model, rig) if ids.get("floor_gid", -1) >= 0 else None
    d = mujoco.MjData(model)
    jac = np.zeros((3, model.nv))
    best = None
    n_tried, stopped = 0, None
    for seed_pos, seed_eul in seeds:
        if n_tried and time_budget_s is not None and _time.monotonic() - t_start > float(time_budget_s):
            stopped = "time_budget"
            break
        n_tried += 1
        d.qpos[:] = data0.qpos
        d.qvel[:] = 0.0
        rig.teleport(d, np.asarray(seed_pos, float), np.asarray(seed_eul, float))
        for k, a in rig.finger_qpos.items():
            lo, hi = rig.finger_range[k]
            d.qpos[a] = lo + 0.25 * (hi - lo)
        if parallel:
            pin_finger_shape(rig, d)
        kinematic_forward(model, d, contacts=False)
        for it in range(int(iters)):
            err = goal - np.concatenate([np.asarray(d.site_xpos[s], float) for s in sites])
            J = np.zeros((3 * len(sites), n))
            for i, s in enumerate(sites):
                mujoco.mj_jacSite(model, d, jac, None, s)
                J[3 * i:3 * i + 3, :] = jac[:, dofs]
            JT = J.T
            Jinv = JT @ np.linalg.inv(J @ JT + lam * np.eye(J.shape[0]))
            dq = Jinv @ err
            if it > 0.25 * iters:
                A, b, _ = _escape_rows(model, d, dofs, hand_bids, obj_gids, tip_gids, floor_gid=ids.get("floor_gid", -1), vis=vis)
                if A is not None:
                    dq_sec = A.T @ np.linalg.solve(A @ A.T + 1e-4 * np.eye(A.shape[0]), b)
                    dq = dq + (np.eye(n) - Jinv @ J) @ dq_sec
            dq = np.clip(dq, -max_step, max_step)
            for i, a in enumerate(qadr):
                lo, hi = lims[i]
                d.qpos[a] = float(np.clip(d.qpos[a] + dq[i], lo, hi))
            # contacts only once the escape task reads them (from the next iteration on)
            kinematic_forward(model, d, contacts=(it + 1) > 0.25 * iters)
        res = {f: float(np.linalg.norm(np.asarray(d.site_xpos[s], float) - np.asarray(targets[f], float)))
               for f, s in zip(fingers, sites)}
        worst = max(res.values())
        pen = deepest_overlap(model, d, rig, ids, keep_tips=tuple(fingers), visual_floor=True)
        face = 0.0
        if palm_n is not None:
            pn = np.asarray(d.xmat[palm_bid], float).reshape(3, 3) @ palm_n
            if face_dir is not None:
                face = float(pn @ np.asarray(face_dir, float))
            elif face_at is not None:
                v = np.asarray(face_at, float) - np.asarray(d.xpos[palm_bid], float)
                nv = float(np.linalg.norm(v))
                face = float(pn @ (v / nv)) if nv > 1e-6 else 0.0
        # a hand more than 4 mm into the scene ranks last whatever its residual
        # (a finger through a 4 mm lid plate reports only 4 mm), then residual
        # to a tolerance, then overlap to 2 mm, then the palm toward the object
        key = (bool(pen > 0.004), round(max(worst, 0.002), 3), round(max(pen, 0.002) * 500) / 500.0, -round(face, 2))
        if best is None or key < best[0]:
            best = (key, worst, res, pen, d.qpos.copy(), face)
        if worst <= 0.001 and pen <= 0.001 and face >= 0.85:
            break                                   # a seed that hits the points cleanly with the palm square on: the rest cannot rank higher
    if stats is not None:
        stats.update({"seeds_total": len(seeds), "seeds_tried": n_tried,
                      "elapsed_s": round(_time.monotonic() - t_start, 2), "stopped": stopped})
    if best is None:
        raise RuntimeError("no seeds given")
    _, worst, res, pen, qpos, face = best
    return worst, res, pen, qpos, face


def pad_radius(model, rig: HandRig, finger: str = "index") -> float:
    """Radius of the fingertip's collision capsule (the pad), or TIP_STANDOFF_M / 2 when the hand has none."""
    for g in rig.tip_geoms(finger):
        if g in rig.collision_gids and int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            return float(model.geom_size[g][0])
    return TIP_STANDOFF_M / 2


def finger_ik(model, d, rig: HandRig, targets: dict, iters: int = 80, lam: float = 1e-4, max_step: float = 0.05) -> dict:
    """Move ONLY the joints of the fingers in `targets` {finger: xyz} so their tip sites reach the points,
    the wrist and the other fingers staying put. Damped least squares with limit clipping. Returns the
    residual per finger (m); the joints are left in `d`."""
    fingers = [f for f in targets if f in rig.fingertip_sites]
    keys = [k for k in rig.finger_qpos if finger_of_joint(k) in fingers]
    qadr = [rig.finger_qpos[k] for k in keys]
    dofs = [int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{rig.prefix}{rig.side}_{k}")]) for k in keys]
    lims = [rig.finger_range[k] for k in keys]
    sites = [int(rig.fingertip_sites[f]) for f in fingers]
    goal = np.concatenate([np.asarray(targets[f], float) for f in fingers])
    jac = np.zeros((3, model.nv))
    kinematic_forward(model, d)
    for _ in range(int(iters)):
        err = goal - np.concatenate([np.asarray(d.site_xpos[s], float) for s in sites])
        if np.linalg.norm(err) < 2e-4:
            break
        J = np.zeros((3 * len(sites), len(dofs)))
        for i, s in enumerate(sites):
            mujoco.mj_jacSite(model, d, jac, None, s)
            J[3 * i:3 * i + 3, :] = jac[:, dofs]
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(J.shape[0]), err)
        dq = np.clip(dq, -max_step, max_step)
        for i, a in enumerate(qadr):
            lo, hi = lims[i]
            d.qpos[a] = float(np.clip(d.qpos[a] + dq[i], lo, hi))
        kinematic_forward(model, d)
    mujoco.mj_forward(model, d)                     # the caller gets a fully computed state, as before
    return {f:float(np.linalg.norm(np.asarray(d.site_xpos[s], float) - np.asarray(targets[f], float))) for f, s in zip(fingers, sites)}


def curl_free_fingers_clear(model, d, rig: HandRig, ids: dict, used: set, fracs=(0.65, 0.85, 0.95, 0.45, 0.25, 0.1)) -> float:
    """Fold the fingers that got no target by the first fraction in `fracs` whose fold does not drive them
    into the objects or the floor (least overlap wins when all do). Returns the fraction used."""
    best = None
    # the free fingers' fold is judged by what it can change: their own self-overlap, not the grip fingers' with each other
    free = [f for f in FINGERS if f not in set(used) and f in rig.fingertip_sites]
    for fr in fracs:
        curl_free_fingers(rig, d, used, fr)
        mujoco.mj_forward(model, d)
        ov = deepest_overlap(model, d, rig, ids, keep_tips=tuple(used))
        if best is None or ov < best[0] - 1e-6:
            best = (ov, fr)
        if ov <= 0.002:
            return fr
    curl_free_fingers(rig, d, used, best[1])
    spread_idle_fingers(model, d, rig, free)
    mujoco.mj_forward(model, d)
    return best[1]


def curl_free_fingers(rig: HandRig, d, used: set, frac: float = 0.65):
    """Fold the fingers that got no target into the palm (no splay)."""
    for k, adr in rig.finger_qpos.items():
        fin = finger_of_joint(k)
        if fin is None or fin in used:
            continue
        lo, hi = rig.finger_range[k]
        if joint_role(k) == "abduct":
            d.qpos[adr] = float(np.clip(0.0, lo, hi))
        elif joint_role(k) in ("flex", "pinky_arch"):
            d.qpos[adr] = lo + float(frac) * (hi - lo)


def pose_from_data(model, d, rig: HandRig, side: str) -> dict:
    return {"side": side,
            "pos": [float(d.qpos[rig.wrist_qpos[k]]) for k in ("x", "y", "z")],
            "euler": [float(d.qpos[rig.wrist_qpos[k]]) for k in ("rx", "ry", "rz")],
            "finger_targets": {k: float(d.qpos[a]) for k, a in rig.finger_qpos.items()}}


def cinch_fingers(model, d, rig: HandRig, obj_gids, fingers=None, step_rad=0.012,
                  max_steps=200, touch_m=-0.0008, max_pen_m=0.0018, max_link_pen_m=0.006, avoid_gids=()) -> dict:
    """Curl each finger's flexion joints toward the closed synergy until its
    pad just touches the object (true gap by mj_geomDistance). A step that
    goes deeper than max_pen_m or that makes the gap WORSE is reverted and
    the finger stops: past a point the flexion arc carries the tip back
    toward the palm, away from the surface (measured: ring and pinky walked
    36 and 50 mm off a cap while the loop happily kept curling)."""
    closed = rig.finger_targets(0.98)
    obj_gids = list(obj_gids)
    avoid = list(avoid_gids or ())            # the floor and bystanders: links may not curl into them either
    out = {}
    for fin in (fingers or list(rig.fingertip_sites)):
        tg = sorted(rig.tip_geoms(fin))
        keys = rig.flexion_joints(fin)
        # the whole finger's collision links: a curl that lands the PAD on
        # the surface while the proximal phalanx goes through the object is
        # not a contact, and the physics answers it by ejecting the object
        links = sorted(g for g in rig.collision_gids if f"_{fin}_" in (model.geom(g).name or ""))

        def gap(geoms=tg, against=None):
            best = 1e9
            for hg in geoms:
                for og in (obj_gids if against is None else against):
                    best = min(best, geom_distance(model, d, hg, og, 0.05))
            return best

        mujoco.mj_forward(model, d)
        moved = 0.0
        # a pad that starts deeper than max_pen (an IK solution poking the surface along the finger axis) is
        # backed off first, so the closed targets never command more than max_pen of interference
        for _ in range(40):
            if gap() >= -max_pen_m:
                break
            for k in keys:
                a = rig.finger_qpos[k]
                lo, hi = rig.finger_range[k]
                d.qpos[a] = float(np.clip(d.qpos[a] - step_rad * np.sign(closed.get(k, d.qpos[a]) - lo + 1e-9), lo, hi))
            mujoco.mj_forward(model, d)
            moved -= step_rad
        for _ in range(int(max_steps)):
            g0 = gap()
            if g0 <= touch_m:
                break
            gl0 = gap(links)
            saved = {k: float(d.qpos[rig.finger_qpos[k]]) for k in keys}
            advanced = False
            for k in keys:
                a = rig.finger_qpos[k]
                span = float(closed.get(k, d.qpos[a])) - float(d.qpos[a])
                if abs(span) < 1e-4:
                    continue
                lo, hi = rig.finger_range[k]
                d.qpos[a] = float(np.clip(d.qpos[a] + np.sign(span) * min(step_rad, abs(span)), lo, hi))
                advanced = True
            if not advanced:
                break
            mujoco.mj_forward(model, d)
            g1 = gap()
            gl1 = gap(links)
            ga1 = gap(links, avoid) if avoid else 1e9
            # a proximal link may press a few mm into the object (the servo
            # resolves that as contact) but may not be driven through it, nor into the floor
            if (g1 < -max_pen_m or g1 > g0 + 1e-5 or (gl1 < -max_link_pen_m and gl1 < gl0 - 1e-6)
                    or ga1 < -0.001 or hand_self_overlap(model, d, rig) > 0.001):
                for k, v in saved.items():
                    d.qpos[rig.finger_qpos[k]] = v
                mujoco.mj_forward(model, d)
                break
            moved += step_rad
        out[fin] = {"curled_rad": round(moved, 4), "gap_mm": round(1000 * gap(), 2)}
    return out


# ---------------------------------------------------------------------------
# grasp quality: Ferrari-Canny Q1 from the simulator's contacts
# ---------------------------------------------------------------------------
def friction_cone_edges(normal, mu, m=8):
    n = np.asarray(normal, float)
    n /= np.linalg.norm(n) + 1e-12
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    t1 = np.cross(n, a)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    return [n + mu * (np.cos(th) * t1 + np.sin(th) * t2)
            for th in np.linspace(0, 2 * np.pi, m, endpoint=False)]


def q1_metric(contacts, com, radius=None, m=8) -> float:
    """Ferrari-Canny: the largest wrench-space ball around the origin inside
    the hull of the contacts' friction-cone edges (torques normalised by the
    object's radius). Q1 > 0 means force closure."""
    from scipy.spatial import ConvexHull
    if len(contacts) < 2:
        return 0.0
    pts = np.asarray([c["pos"] for c in contacts], float)
    com = np.asarray(com, float)
    radius = radius or max(float(np.max(np.linalg.norm(pts - com, axis=1))), 1e-3)
    W = []
    for c in contacts:
        n = np.asarray(c["normal"], float)
        n = n / (np.linalg.norm(n) + 1e-12)
        mu_t = float(c.get("mu_torsion", 0.0))
        for f in friction_cone_edges(n, max(c["mu"], 1e-3), m):
            tau = np.cross(c["pos"] - com, f) / radius
            W.append(np.concatenate([f, tau]))
            # a soft (pad) contact also resists twisting about its normal:
            # MuJoCo models it as torsional friction (condim >= 4), and two
            # pads pinching an object are force-closed only through it
            if mu_t > 0:
                W.append(np.concatenate([f, tau + mu_t * n / radius]))
                W.append(np.concatenate([f, tau - mu_t * n / radius]))
    W = np.asarray(W)
    if np.linalg.matrix_rank(W - W.mean(0)) < 6:
        return 0.0
    try:
        hull = ConvexHull(W, qhull_options="QJ")
    except Exception:
        return 0.0
    d = -hull.equations[:, -1] / np.maximum(np.linalg.norm(hull.equations[:, :-1], axis=1), 1e-12)
    return float(max(0.0, d.min()))


HOLD_NEED_OK_N = 4.0        # per-contact force (hold the part at 1.5 g, tilted 30 deg) below which a grip holds with margin: a
                            # torque-capped pad pushes ~8-14 N. Measured: brick pinches need under 1 N, a pinch through a 600 g
                            # drive's centre 7.9 N (it lifts), the executed drive pinches 30-45 mm off it 16-18 N (they twisted out)
OPPOSITION_MIN_DEG = 100.0  # two contact groups squeeze the part only when their normals are this far apart (180 - 2 atan(1.2), less 30 %)


def static_grip(model, data, rig: HandRig, obj_gids, band_m=0.0015, fingers=None) -> list:
    """The contacts of a POSED hand with an object, read from the geometry alone (no physics): every hand collision geom
    within band_m of an object geom gives the closest point on the object and the normal into it, labelled with its
    link (each finger, 'palm'). With `fingers` (the gripping fingers: the ones the executor clamps) the other fingers'
    links are left out, and the thumb's metacarpal counts as palm (it sits in the palm and is not driven onto the
    part). Returns [{link, geom, point, normal}], one per hand geom (its closest object geom)."""
    obj = [int(g) for g in obj_gids]
    grip = None if fingers is None else set(fingers)
    ft = np.zeros(6)
    out = []
    for h in sorted(rig.collision_gids):
        nm = model.geom(h).name or ""
        link = next((f for f in FINGERS if f"_{f}_" in nm), "palm")
        if grip is not None and link != "palm":
            bnm = model.body(int(model.geom_bodyid[h])).name or ""
            if link == "thumb" and (bnm.endswith("thumb_MC") or bnm.endswith("thumb_CMC_VL")):
                link = "palm"
            elif link not in grip:
                continue
        best = None
        for g in obj:
            if np.linalg.norm(data.geom_xpos[h] - data.geom_xpos[g]) > model.geom_rbound[h] + model.geom_rbound[g] + band_m:
                continue
            dist = geom_distance(model, data, h, g, band_m + 0.01, ft)
            if dist <= band_m and (best is None or dist < best[0]):
                best = (dist, ft[:3].copy(), ft[3:].copy(), g)
        if best is None:
            continue
        dist, ph, po, g = best
        # into the object: hand point -> object point when apart; when overlapping the witness points swap sides
        nvec = (po - ph) if dist > 0 else (ph - po)
        if np.linalg.norm(nvec) < 1e-6:
            nvec = po - np.asarray(data.geom_xpos[h], float)      # exactly touching: away from the hand geom's centre
        nvec = nvec / max(np.linalg.norm(nvec), 1e-12)
        if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_BOX):
            # a box's face normal, not the witness direction: at an edge the witness points diagonally and made a palm on
            # a bar's end face look opposed to fingers on its side
            Rg = np.asarray(data.geom_xmat[g], float).reshape(3, 3); sz = np.asarray(model.geom_size[g][:3], float)
            loc = Rg.T @ (po - np.asarray(data.geom_xpos[g], float))
            faces = [k for k in range(3) if abs(loc[k]) >= sz[k] - 0.0005] or [int(np.argmax(np.abs(loc) / np.maximum(sz, 1e-9)))]
            cands_n = [-np.sign(loc[k] if loc[k] != 0 else 1.0) * Rg[:, k] for k in faces]      # inward face normals
            nvec = max(cands_n, key=lambda v: float(v @ nvec))
        out.append({"link": link, "geom": nm, "point": po, "normal": nvec})
    return out


def _hold_lp(contacts, com, wrench, mu, mu_spin, m_edges=8):
    """Smallest per-contact normal force t that resists `wrench` (force, torque about com) with soft-finger contacts:
    each contact's normal force splits into a sliding share (friction up to mu) and a twisting share (torsion up to
    mu_spin metres), the linear (conservative) version of MuJoCo's elliptic cone. None when no forces can."""
    from scipy.optimize import linprog
    k = len(contacts)
    nv_c = 1 + m_edges + 2                       # fn, edge weights, twist + / -
    nvar = k * nv_c + 1                          # + t
    A_eq = np.zeros((6, nvar)); b_eq = -np.asarray(wrench, float)
    A_ub, b_ub = [], []
    for j, c in enumerate(contacts):
        n = np.asarray(c["normal"], float); n = n / max(np.linalg.norm(n), 1e-12)
        a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
        t1 = np.cross(n, a); t1 /= np.linalg.norm(t1); t2 = np.cross(n, t1)
        r = np.asarray(c["point"], float) - np.asarray(com, float)
        o = j * nv_c
        A_eq[:3, o] = n; A_eq[3:, o] = np.cross(r, n)
        for e in range(m_edges):
            th = 2 * np.pi * e / m_edges
            f = mu * (np.cos(th) * t1 + np.sin(th) * t2)
            A_eq[:3, o + 1 + e] = f; A_eq[3:, o + 1 + e] = np.cross(r, f)
        A_eq[3:, o + 1 + m_edges] = mu_spin * n; A_eq[3:, o + 2 + m_edges] = -mu_spin * n
        row = np.zeros(nvar); row[o + 1:o + nv_c] = 1.0; row[o] = -1.0          # sliding + twisting shares <= fn
        A_ub.append(row); b_ub.append(0.0)
        row = np.zeros(nvar); row[o] = 1.0; row[-1] = -1.0                       # fn <= t
        A_ub.append(row); b_ub.append(0.0)
    cost = np.zeros(nvar); cost[-1] = 1.0
    try:
        res = linprog(cost, A_ub=np.asarray(A_ub), b_ub=np.asarray(b_ub), A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * nvar, method="highs")
    except Exception:
        return None
    return float(res.x[-1]) if res.status == 0 else None


def grip_hold_quality(contacts, com, mass, size_m=None, mu=None, mu_spin=None, g_scale=1.5, tilt_deg=30.0) -> dict:
    """How well a set of contacts (static_grip's) holds an object against its weight, without physics:
    `need_n` is the normal force each contact must push (the same bound for all) for the grip to hold the weight
    scaled by g_scale with gravity tilted up to tilt_deg from straight down (the wrist turns of the stress tests), with
    soft-finger friction (hand.HAND_PAIR_FRICTION: sliding 1.2, twisting 2 cm). A two-pad pinch far from the centre of
    mass holds the gravity torque about its pinch line only by twisting friction and needs several times the force of
    one through the centre or of three spread contacts. `opposed`: two contacts on different links whose normals are at
    least OPPOSITION_MIN_DEG apart with the centre of mass between their contact planes (the pads squeeze across the
    part; all fingers on one face with the palm on another is a press held only by the palm's friction)."""
    mu = HAND_PAIR_FRICTION[0] if mu is None else float(mu)
    mu_spin = HAND_PAIR_FRICTION[1] if mu_spin is None else float(mu_spin)
    com = np.asarray(com, float)
    out = {"n_contacts": len(contacts), "links": sorted({c["link"] for c in contacts}), "need_n": None, "opposed": False,
           "opposition_deg": 0.0, "pair": None}
    if not contacts:
        return out
    pts = np.array([np.asarray(c["point"], float) for c in contacts])
    size = float(size_m) if size_m else max(float(np.max(np.linalg.norm(pts - com, axis=1))) * 2.0, 0.01)
    best = None
    for i in range(len(contacts)):
        for j in range(i + 1, len(contacts)):
            if contacts[i]["link"] == contacts[j]["link"] or "palm" == contacts[i]["link"] == contacts[j]["link"]:
                continue                          # opposition is between two links (a finger against the thumb or the palm)
            ni, nj = np.asarray(contacts[i]["normal"], float), np.asarray(contacts[j]["normal"], float)
            ang = float(np.degrees(np.arccos(np.clip(ni @ nj, -1.0, 1.0))))
            si = float((com - pts[i]) @ ni); sj = float((com - pts[j]) @ nj)
            inside = min(si, sj) >= -0.1 * size
            key = (ang >= OPPOSITION_MIN_DEG and inside, ang)
            if best is None or key > best[0]:
                best = (key, [contacts[i]["link"], contacts[j]["link"]], ang)
    if best is not None:
        out.update(opposed=bool(best[0][0]), opposition_deg=round(best[2], 1), pair=best[1])
    W = float(mass) * 9.81 * float(g_scale)
    need = 0.0
    dirs = [np.array([0.0, 0.0, -1.0])]
    if tilt_deg:
        sn, cs = np.sin(np.radians(tilt_deg)), np.cos(np.radians(tilt_deg))
        dirs += [np.array([sn, 0, -cs]), np.array([-sn, 0, -cs]), np.array([0, sn, -cs]), np.array([0, -sn, -cs])]
    for gdir in dirs:
        t = _hold_lp(contacts, com, np.concatenate([W * gdir, np.zeros(3)]), mu, mu_spin)
        if t is None:
            need = None
            break
        need = max(need, t)
    out["need_n"] = None if need is None else round(need, 2)
    return out


def pose_hold_quality(model, data0, rig: HandRig, ids: dict, pose: dict, bid: int, targets: str = "finger_targets_squeezed") -> dict:
    """grip_hold_quality of a pose in its hand scene: the hand placed at the pose with its squeezed (else closed)
    finger targets, contacts read from the geometry, the grasped body's subtree mass and centre of mass."""
    d = mujoco.MjData(model); d.qpos[:] = data0.qpos
    fing = pose.get(targets) or pose.get("finger_targets") or {}
    rig.teleport(d, np.asarray(pose["pos"], float), np.asarray(pose["euler"], float))
    rig.set_fingers(d, {k: float(v) for k, v in fing.items() if k in rig.finger_range})
    mujoco.mj_forward(model, d)
    root = int(model.body_rootid[bid])
    gids = [g for g in ids["grasp_gids"]]
    con = static_grip(model, d, rig, gids, fingers=pose.get("grip_fingers") or None)
    return grip_hold_quality(con, d.subtree_com[root].copy(), float(model.body_subtreemass[root]))


def hand_object_contacts(model, data, rig: HandRig, obj_gids, min_force=0.05) -> list:
    """Contacts between the hand and the object geoms as
    [{"pos","normal" (into the object), "force","mu","geom"}]."""
    obj_gids = set(obj_gids)
    out, buf = [], np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        hand = g1 if g1 in rig.geom_ids else (g2 if g2 in rig.geom_ids else None)
        other = g2 if hand == g1 else g1
        if hand is None or other not in obj_gids:
            continue
        mujoco.mj_contactForce(model, data, i, buf)
        f = abs(float(buf[0]))
        if f < min_force:
            continue
        n = np.asarray(c.frame[:3], float)
        if hand == g1:          # frame normal points from geom1 to geom2
            n = n.copy()
        else:
            n = -n
        out.append({"pos": np.asarray(c.pos, float).copy(), "normal": n, "force": f,
                    "mu": float(c.friction[0]),
                    "mu_torsion": float(c.friction[2]) if int(c.dim) >= 4 else 0.0,
                    "geom": model.geom(hand).name})
    return out


def press_fraction(contacts) -> float:
    """|sum F| / sum |F|: ~1 means the fingers all push the same way (a press,
    not a grip), ~0 means they oppose each other."""
    if not contacts:
        return 1.0
    F = np.asarray([c["normal"] * c["force"] for c in contacts])
    return float(np.linalg.norm(F.sum(0)) / (np.abs(np.linalg.norm(F, axis=1)).sum() + 1e-9))


# ---------------------------------------------------------------------------
# driven rollouts: every phase goes through the actuators
# ---------------------------------------------------------------------------
# parts another hand is holding during a stress rollout: {id(data): [(qpos adr, dof adr, qpos7)]}, re-applied after every
# step until this hand has closed on them (a handover); empty otherwise
_PINNED = {}


def _apply_pins(data):
    for adr, dof, q in _PINNED.get(id(data), ()):
        data.qpos[adr:adr + 7] = q; data.qvel[dof:dof + 6] = 0.0


def step_hold(model, data, rig, pos, euler, fingers, seconds, record=None, label=""):
    n = int(float(seconds) / model.opt.timestep)
    for _ in range(n):
        hold_ctrl(data, rig, pos, euler, fingers)
        mujoco.mj_step(model, data)
        _apply_pins(data)
        if record is not None:
            record(model, data, label)


def ramp(model, data, rig, p0, p1, e0, e1, f0, f1, seconds, record=None, label="",
         regulate=None):
    """Interpolate wrist pose and finger targets over `seconds`."""
    n = max(int(float(seconds) / model.opt.timestep), 1)
    p0, p1, e0, e1 = (np.asarray(v, float) for v in (p0, p1, e0, e1))
    for i in range(n):
        u = (i + 1) / n
        fg = {k: f0.get(k, v) + (v - f0.get(k, v)) * u for k, v in f1.items()}
        if regulate is not None:
            fg = regulate(i, fg)
        hold_ctrl(data, rig, p0 + (p1 - p0) * u, e0 + (e1 - e0) * u, fg)
        mujoco.mj_step(model, data)
        _apply_pins(data)
        if record is not None:
            record(model, data, label)


def grip_regulator(model, data, rig, ids, bid, grip_fingers, maintain_N=6.0, every=40):
    """A position servo does not hold a grip: fixed joint targets bleed off
    force as the object beds into the pads (18.9 N at close -> 1.0 N at the
    top of a lift, measured). The grip fingers keep closing while the load
    is below maintain_N and back off above 4x it (over-squeezing ejects a
    flexible object)."""
    state = {"a": 0.0}
    gf = set(grip_fingers or ())

    def reg(i, q):
        if not gf or maintain_N <= 0:
            return q
        if i % every == 0:
            f = finger_contacts(model, data, rig, ids["grasp_gids"])["force_N"]
            if f < maintain_N:
                state["a"] = min(0.15, state["a"] + 0.01)
            elif f > 4.0 * maintain_N:
                state["a"] = max(0.0, state["a"] - 0.01)
        out = dict(q)
        for k, v in q.items():
            if finger_of_joint(k) not in gf or joint_role(k) != "flex":
                continue
            lo, hi = rig.finger_range.get(k, (v, v))
            far = hi if abs(v - hi) > abs(v - lo) else lo
            out[k] = float(v + state["a"] * (far - v))
        return out
    reg.state = state
    return reg


def approach_axis(model, data, rig: HandRig, pose: dict | None = None) -> np.ndarray:
    """The direction the hand retreats along before an approach: the pose's
    own "approach" vector when the grasp skill recorded one (outward from
    the object), else the hand's retreat direction (minus its local z), kept
    at least slightly upward so no approach starts below the table."""
    if pose is not None and pose.get("approach") is not None:
        a = np.asarray(pose["approach"], float)
        return a / (np.linalg.norm(a) + 1e-12)
    # the hand's own frame, as plan.retreat_axis (the executor) reads it: the pose's orientation, else the body the wrist's
    # last rotation moves (rig.root_body is the translation base, which never turns: its -z made every such approach +z)
    if pose is not None and pose.get("euler") is not None:
        R = euler_to_mat(np.asarray(pose["euler"], float))
    else:
        jz = int(model.actuator_trnid[rig.wrist_act["rz"]][0]) if "rz" in rig.wrist_act else -1
        R = np.asarray(data.xmat[int(model.jnt_bodyid[jz]) if jz >= 0 else rig.root_body], float).reshape(3, 3)
    axis = -R[:, 2]
    if axis[2] < 0.2:
        axis = axis - np.array([0, 0, axis[2] - 0.2])
        axis /= np.linalg.norm(axis)
    return axis


def wrist_servo_offset(model, data, rig: HandRig, pos, euler, fingers, steps=250,
                       rounds=3, tol=0.0004, give_up_m=0.015):
    """Command the wrist until it actually arrives and return the servo's
    steady-state offset, giving up (keeping the best so far) when the hand
    is blocked rather than lagging."""
    P = np.asarray(pos, float)
    off, best = np.zeros(3), None
    for _ in range(int(rounds)):
        for _ in range(int(steps)):
            hold_ctrl(data, rig, P - off, euler, fingers)
            mujoco.mj_step(model, data)
            _apply_pins(data)
        got = rig.wrist_pos(data)
        err = got - P
        e = float(np.linalg.norm(err))
        if best is None or e < best[0]:
            best = (e, off.copy())
        if e <= tol:
            return off, e
        nxt = off + err
        if float(np.linalg.norm(nxt)) > give_up_m:
            break
        off = nxt
    return best[1], best[0]


def load_scene_spec(scene):
    """A hand-scene description: the dict itself, or a JSON file holding it (relative paths inside resolve against the
    file's folder when they do not exist as given)."""
    if isinstance(scene, (str, Path)):
        import json as _json
        src = Path(scene)
        d = _json.loads(src.read_text())
        if isinstance(d, dict) and "scene" in d and "objects" not in d and isinstance(d["scene"], dict):
            d = d["scene"]
        base = src.resolve().parent
        if isinstance(d, dict):
            if d.get("hand_xml") and not Path(d["hand_xml"]).exists() and (base / d["hand_xml"]).exists():
                d["hand_xml"] = str(base / d["hand_xml"])
            for o in d.get("objects", []) or []:
                if isinstance(o, dict) and o.get("xml") and not Path(o["xml"]).exists() and (base / o["xml"]).exists():
                    o["xml"] = str(base / o["xml"])
        return d
    return scene


def open_scene(scene: dict, **over):
    """Build a hand scene from the plain description every grasp tool takes:

        {"hand_xml": "<path to the hand MJCF>", "side": "right",
         "objects": [{"name": "cup", "xml": "<path>", "pos": [x, y, z],
                      "yaw_deg": 0}],           # the LAST one is grasped
         "floor": true, "hand_spec": {...optional naming overrides...},
         "grasp_body": "cap_cap",            # optional: a part of the last object is the target
         "joint_init": {"box_lid": 3.1},      # optional: the objects' joint values (free joints: 7 numbers)
         "no_hand_pairs": ["ring_magnet"]}    # optional: object geoms (names or prefixes) the hand never touches

    Returns (model, data0, rig, ids). `scene` may also be the path of a JSON file holding that description
    (hand_scene_from_recon's out_json)."""
    scene = load_scene_spec(scene)
    dt = scene.get("timestep")
    if dt is None and scene.get("bench_scene") and Path(str(scene["bench_scene"])).exists():
        # the bench's own integration step (a screw-thread bench runs at 0.25 ms): the hand scene integrates the same
        # contacts the same way (grasp_robust.load_scene reads it the same way)
        mt = re.search(r'<option\b[^>]*\btimestep="([^"]+)"', Path(str(scene["bench_scene"])).read_text())
        dt = float(mt.group(1)) if mt else None
    kw = dict(objects=scene["objects"], hand_xml=scene["hand_xml"],
              side=scene.get("side", "right"), spec=scene.get("hand_spec"),
              floor=(scene["floor"] if isinstance(scene.get("floor"), dict) else bool(scene.get("floor", True))),
              timestep=float(dt if dt is not None else 0.002),
              joint_init=scene.get("joint_init"), grasp_body=scene.get("grasp_body"))
    for k in ("finger_force_nm", "wrist_force_n", "wrist_torque_nm", "no_hand_pairs", "pair_region"):
        if k in scene:
            kw[k] = scene[k]
    if "hand_contact_bits" in scene:
        kw["hand_bits"] = int(scene["hand_contact_bits"])
    kw.update(over)
    return build_hand_scene(**kw)


def grasp_body(model, scene: dict, ids: dict) -> int:
    """Body id of the grasped object: `scene["grasp_body"]` if given, else the
    first body carrying a free joint under the last object's prefix, else
    the root body of that prefix. `scene` is the description open_scene takes:
    the dict or the path of its JSON file (hand_scene_from_recon's out_json)."""
    scene = load_scene_spec(scene) if scene is not None else {}
    nm = scene.get("grasp_body")
    if nm:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
        if bid < 0:
            raise ValueError(f"grasp_body '{nm}' is not in the scene")
        return int(bid)
    pre = ids["grasp_prefix"]
    for j in range(model.njnt):
        if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
            b = int(model.jnt_bodyid[j])
            if (model.body(b).name or "").startswith(pre):
                return b
    for b in range(model.nbody):
        if (model.body(b).name or "").startswith(pre):
            return int(b)
    raise ValueError("no object body found to grasp")


# ---------------------------------------------------------------------------
# where this hand holds things
# ---------------------------------------------------------------------------
def grasp_pocket(model, data0, rig: HandRig, grip: float = 0.55):
    """The pocket between the palm face and the curled fingers, in WRIST
    coordinates: {"c": centre, "n": palm normal (curl side), "f": finger
    axis, "depth_m"}. The width-matching grasp families mis-model a real
    hand at the edges; its actual behaviour is simpler: put the object in
    the pocket, close, and the fingers stop where the object is."""
    if rig.palm_body < 0:
        raise ValueError("hand has no palm body (see HandSpec palm_body_suffixes)")
    d = mujoco.MjData(model)

    def tips_at(g):
        d.qpos[:] = data0.qpos
        rig.set_fingers(d, rig.finger_targets(float(g)))
        rig.teleport(d, np.zeros(3), np.zeros(3))
        mujoco.mj_forward(model, d)
        return {f: np.asarray(d.site_xpos[s], float).copy() for f, s in rig.fingertip_sites.items()}, \
            np.asarray(d.xpos[rig.palm_body], float).copy()

    t0, P = tips_at(0.0)
    t1, _ = tips_at(0.85)
    m0 = np.mean([v for f, v in t0.items() if f != "thumb"], axis=0)
    m1 = np.mean([v for f, v in t1.items() if f != "thumb"], axis=0)
    f_ax = (m0 - P) / (np.linalg.norm(m0 - P) + 1e-12)
    curl = m1 - m0
    n = curl - np.dot(curl, f_ax) * f_ax
    n /= np.linalg.norm(n) + 1e-12
    tg, _ = tips_at(grip)
    mg = np.mean([v for f, v in tg.items() if f != "thumb"], axis=0)
    face = float(np.dot(P, n))
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) == rig.palm_body:
            face = max(face, float(np.dot(np.asarray(d.geom_xpos[g], float), n)))
    pocket = 0.5 * (P + np.dot(face - np.dot(P, n), n) + mg)
    return {"c": pocket, "n": n, "f": f_ax, "depth_m": float(np.dot(mg - P, n)), "grip": float(grip)}


def frame_from_axes(n_from, f_from, n_to, f_to) -> np.ndarray:
    """Rotation taking the orthonormalised pair (n_from, f_from) onto
    (n_to, f_to)."""
    def basis(n, f):
        n = np.asarray(n, float) / (np.linalg.norm(n) + 1e-12)
        f = np.asarray(f, float)
        f = f - np.dot(f, n) * n
        f /= np.linalg.norm(f) + 1e-12
        return np.stack([n, f, np.cross(n, f)], axis=1)
    return basis(n_to, f_to) @ basis(n_from, f_from).T


def pocket_pose(pocket: dict, point, palm_dir_world, finger_dir_world):
    """Wrist (pos, euler) that puts the pocket centre at `point` with the palm
    normal along palm_dir_world and the finger axis along finger_dir_world."""
    R = frame_from_axes(pocket["n"], pocket["f"], palm_dir_world, finger_dir_world)
    pos = np.asarray(point, float) - R @ np.asarray(pocket["c"], float)
    return pos, mat_to_euler(R)


def object_extent(model, data, gids) -> dict:
    """World AABB, centre, and the principal (longest) axis of a geom set."""
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for g in gids:
        c = np.asarray(model.geom_aabb[g, :3], float)
        h = np.asarray(model.geom_aabb[g, 3:], float)
        R = np.asarray(data.geom_xmat[g], float).reshape(3, 3)
        p = np.asarray(data.geom_xpos[g], float) + R @ c
        ext = np.abs(R) @ h
        lo = np.minimum(lo, p - ext)
        hi = np.maximum(hi, p + ext)
    size = hi - lo
    return {"lo": lo, "hi": hi, "centre": 0.5 * (lo + hi), "size": size,
            "long_axis": int(np.argmax(size))}


# ---------------------------------------------------------------------------
# the grip stress ladder: approach open -> close until held -> lift -> carry
# -> turn -> hold. One continuous driven rollout; every phase reports ok.
# ---------------------------------------------------------------------------
MOVE_DIRS = (("up", (0.0, 0.0, 1.0)), ("fwd", (0.0, 1.0, 0.0)), ("back", (0.0, -1.0, 0.0)),
             ("left", (-1.0, 0.0, 0.0)), ("right", (1.0, 0.0, 0.0)))


SAT_FRAC = 0.95            # a finger servo needing this share of its torque cap is saturated (finger_servo_saturation_audit)
SAT_HOLD_FRAC = 0.6        # a hold saturated on this share of its judged frames has no reserve (the audit's hold_frac)
SAT_MU_CAP = 1.5           # the largest friction coefficient the reserve credits (the audit's mu_cap)


def finger_torque_need(contacts, com, F_req, dofs, capv, mu_cap=SAT_MU_CAP):
    """The smallest share s of the finger servos' torque caps with which a set of contacts carries a load: min s s.t.
    contact forces in friction pyramids (plus torsional and rolling friction bounded by the normal force) balance
    (F_req, 0 torque about com) up to a heavily penalised slack and |J^T f| <= s cap per finger dof. contacts:
    [(point, normal into the object, mu, spin m, roll m, J [3 x len(dofs)]: finger joint torques per unit force)].
    Returns (s, residual N, limiting dof index) or None. Shared by finger_servo_saturation_audit (episodes) and
    stress_rollout (hand-scene grasp tests), so the two agree."""
    from scipy.optimize import linprog
    G_cols, T_cols = [], []
    for p, n, mu, spin, roll, J in contacts:
        n = n / (np.linalg.norm(n) + 1e-12)
        a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
        t1 = np.cross(n, a); t1 /= np.linalg.norm(t1); t2 = np.cross(n, t1)
        mu = min(max(mu, 0.0), mu_cap)
        gens = [n] if mu <= 0 else [n + mu * t1, n - mu * t1, n + mu * t2, n - mu * t2]
        r = p - com
        for g in gens:
            G_cols.append(np.r_[g, np.cross(r, g)])
            T_cols.append(J.T @ g)                    # finger joint torques per unit of this generator
        for axis, k in ((n, spin), (t1, roll), (t2, roll)):
            if k > 0:                                 # friction torque about this axis, bounded by k x normal force
                for sg in (1.0, -1.0):
                    G_cols.append(np.r_[n, np.cross(r, n) + sg * k * axis])
                    T_cols.append(J.T @ n)
    if not G_cols:
        return None
    G = np.array(G_cols).T                            # 6 x k
    Tq = np.array(T_cols).T                           # nf x k
    k, nf = G.shape[1], len(dofs)
    w_req = np.r_[F_req, np.zeros(3)]
    scale = max(np.linalg.norm(F_req), 1e-3)
    # variables: lambda (k), s (1), e+ (6), e- (6); the slack costs far more than any torque share, so it is used only
    # where the contacts cannot balance the load at all
    c = np.r_[np.zeros(k), 1.0, np.full(12, 1e4 / scale)]
    A_eq = np.hstack([G, np.zeros((6, 1)), np.eye(6), -np.eye(6)])
    A_ub = np.vstack([np.hstack([Tq, -capv[:, None], np.zeros((nf, 12))]),
                      np.hstack([-Tq, -capv[:, None], np.zeros((nf, 12))])])
    b_ub = np.zeros(2 * nf)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=w_req, bounds=[(0, None)] * (k + 13), method="highs")
    if not res.success:
        return None
    lam, s_ = res.x[:k], float(res.x[k])
    resid = float(np.abs(res.x[k + 1:k + 4]).sum() + np.abs(res.x[k + 7:k + 10]).sum())   # force part of the slack
    tau = np.abs(Tq @ lam) / capv
    return s_, resid, int(np.argmax(tau)) if len(tau) else -1


class ServoReserve:
    """Samples a hand-scene rollout every 1/30 s (the executor's frame rate) and judges, on every second sample, the
    smallest share of the finger servos' torque caps with which the hand's contacts carry the held object's weight and
    inertial force (finger_torque_need, the contact's own friction: what finger_servo_saturation_audit computes on an
    episode). Samples where the object touches anything but the hand (a support) are skipped."""

    def __init__(self, model, rig, bid):
        self.m, self.rig = model, rig
        self.root = int(model.body_rootid[bid])
        sub, stack = set(), [self.root]
        while stack:
            b = stack.pop(); sub.add(b)
            stack += [c for c in range(model.nbody) if int(model.body_parentid[c]) == b and c != b]
        self.obj_b = sub
        self.mass = float(model.body_subtreemass[self.root])
        self.dofs, caps, self.names = [], [], []
        for k, a in rig.finger_act.items():
            j = int(model.actuator_trnid[a][0])
            cap = max(abs(float(model.actuator_forcerange[a][0])), abs(float(model.actuator_forcerange[a][1]))) \
                if model.actuator_forcelimited[a] else float(FINGER_FORCE_NM)
            self.dofs.append(int(model.jnt_dofadr[j])); caps.append(cap * abs(float(model.actuator_gear[a][0]))); self.names.append(k)
        self.capv = np.asarray(caps, float)
        self.every = max(int(round((1.0 / 30.0) / float(model.opt.timestep))), 1)
        self.n = 0
        self.samples = []
        self.jp = np.zeros((3, model.nv))

    def step(self, d):
        self.n += 1
        if self.n % self.every:
            return
        m = self.m
        contacts, support = [], False
        for i in range(d.ncon):
            c = d.contact[i]
            if float(c.dist) > 0.001:
                continue
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            in1, in2 = b1 in self.obj_b, b2 in self.obj_b
            if in1 == in2:
                continue
            other = b2 if in1 else b1
            if other not in self.rig.body_ids:
                support = True
                break
            n_into = (1.0 if in2 else -1.0) * np.asarray(c.frame[:3], float)
            mujoco.mj_jac(m, d, self.jp, None, np.asarray(c.pos, float), int(other))
            J = -self.jp[:, self.dofs]                # the finger pushes the object: its joints feel the reaction
            fr = np.asarray(c.friction, float)
            contacts.append((np.asarray(c.pos, float), n_into, float(fr[0]) if c.dim > 1 else 0.0,
                             float(fr[2]) if c.dim >= 4 else 0.0, float(fr[3]) if c.dim >= 6 else 0.0, J))
        self.samples.append((np.asarray(d.subtree_com[self.root], float).copy(), None if support else contacts))

    def report(self, sat_frac=SAT_FRAC, hold_frac=SAT_HOLD_FRAC):
        g = np.asarray(self.m.opt.gravity, float)
        h = self.every * float(self.m.opt.timestep)
        sv, lim = [], []
        for k in range(1, len(self.samples) - 1, 2):
            com, cons = self.samples[k]
            if not cons:
                continue
            acc = (self.samples[k + 1][0] - 2 * com + self.samples[k - 1][0]) / h ** 2
            sol = finger_torque_need(cons, com, self.mass * (acc - g), self.dofs, self.capv)
            if sol is None or sol[1] > 0.2 * self.mass * float(np.linalg.norm(g)):
                continue
            sv.append(sol[0]); lim.append(self.names[sol[2]] if sol[2] >= 0 else None)
        if not sv:
            return {"name": "servo_reserve", "judged": 0, "ok": True, "note": "the object was never held by the hand alone"}
        sv = np.asarray(sv)
        sat = sv >= float(sat_frac)
        from collections import Counter
        lj = Counter([j for j, x in zip(lim, sat) if x and j]).most_common(1)
        return {"name": "servo_reserve", "judged": int(len(sv)), "saturated_frac": round(float(sat.mean()), 3),
                "median_s": round(float(np.median(sv)), 3), "peak_s": round(float(sv.max()), 3),
                "limiting_joint": lj[0][0] if lj else lim[int(np.argmax(sv))], "ok": bool(sat.mean() < float(hold_frac))}


EXEC_RATE_DEG_S = 90.0      # the executor's joint-rate ceiling (execute_subgoal_plan max_rate_deg_s, chain_subgoals)
EXEC_RATE_M_S = 0.4


def _exec_leg(model, data, rig, p0, p1, e0, e1, f0, f1, seconds, record=None, label="", watch=None):
    """One keyframe leg as the executor runs it: a quintic ease on the wrist and the finger targets, stretched so no joint
    passes the 90 deg/s ceiling (a quintic peaks at 1.875 x its mean rate). Returns the final finger targets."""
    p0, p1, e0, e1 = (np.asarray(v, float) for v in (p0, p1, e0, e1))
    dh = [abs(float(b) - float(a)) for a, b in zip(e0, e1)] + [abs(float(v) - float(f0.get(k, v))) for k, v in f1.items()]
    need = 1.875 * max(max(dh, default=0.0) / np.radians(EXEC_RATE_DEG_S), float(np.max(np.abs(p1 - p0))) / EXEC_RATE_M_S) * 1.02
    n = max(int(max(float(seconds), need) / model.opt.timestep), 1)
    fg = dict(f1)
    for i in range(n):
        u = (i + 1) / n
        u = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
        fg = {k: f0.get(k, v) + (v - f0.get(k, v)) * u for k, v in f1.items()}
        hold_ctrl(data, rig, p0 + (p1 - p0) * u, e0 + (e1 - e0) * u, fg)
        mujoco.mj_step(model, data)
        _apply_pins(data)
        if record is not None:
            record(model, data, label)
        if watch is not None:
            watch(i, fg)
    return fg


class GraspPenetrationMonitor:
    """Measure every simulated contact, including grip pads and hand self-contact.

    Contact force is allowed; more than 3 mm of interpenetration is not.
    Visible hand meshes against the support are also measured at 60 Hz because
    some meshes have no collision proxy covering their full outline.
    """

    def __init__(self, model, rig, ids, record=None):
        self.rig, self.ids, self.record = rig, ids, record
        self.hand = set(rig.geom_ids)
        self.others = set(ids["obj_gids"])
        floor = int(ids.get("floor_gid", -1))
        if floor >= 0:
            self.others.add(floor)
        self.contact_geoms = self.hand | self.others
        self.steps = 0
        self.visual_every = max(1, int((1 / 60) / float(model.opt.timestep)))
        self.max_contact = self.max_visual_floor = 0.
        self.worst_pair = None
        self.first_bad = None

    def __call__(self, model, data, label):
        self.steps += 1
        for c in data.contact:
            g, h = int(c.geom1), int(c.geom2)
            relevant = ((g in self.hand and h in self.contact_geoms)
                        or (h in self.hand and g in self.others))
            if relevant and -float(c.dist) > self.max_contact:
                self.max_contact = -float(c.dist)
                self.worst_pair = [model.geom(g).name, model.geom(h).name]
        if self.steps == 1 or self.steps % self.visual_every == 0:
            self.max_visual_floor = max(self.max_visual_floor, visual_floor_depth(
                model, data, self.rig, int(self.ids.get("floor_gid", -1))))
        if self.first_bad is None and max(self.max_contact, self.max_visual_floor) > .003:
            self.first_bad = {"time_s": round(float(data.time), 6), "phase": label,
                              "pair": self.worst_pair}
        if self.record is not None:
            self.record(model, data, label)

    def report(self):
        return {"name": "penetration", "ok": self.first_bad is None,
                "max_contact_mm": round(1000 * self.max_contact, 3),
                "max_visual_floor_mm": round(1000 * self.max_visual_floor, 3),
                "limit_mm": 3., "steps": self.steps, "worst_pair": self.worst_pair,
                "first_bad": self.first_bad}


def hold_with_slip(model, data, rig, bids, pos, euler, fingers, seconds, record=None):
    """Execute a hold and measure every substep, including transient recovery.

    The original recorder still sees every physics step (penetration, servo
    reserve, or a caller's logger). Replaying just the saved joint states for
    measurement avoids stale post-step kinematics and uses the same held-link
    selection as local outcomes and the episode slip audit.
    """
    from tool_library.common import episode as E, held_contact as HC
    states = [data.qpos.copy()]

    def sample(m, d, label):
        states.append(d.qpos.copy())
        if record is not None:
            record(m, d, label)

    step_hold(model, data, rig, pos, euler, fingers, seconds, sample, "hold")
    Q = np.asarray(states)
    return [HC.measure_slip(model, Q, 1. / float(model.opt.timestep), 0, len(Q) - 1,
                            bid, wrist_body(rig), set(rig.body_ids), E.held_bodies(model, bid))
            for bid in bids]


def stress_rollout(model, data0, rig: HandRig, ids: dict, pose: dict, bid: int, **kw) -> dict:
    """Shared stress rollout with continuous penetration checks and handover cleanup."""
    monitor = GraspPenetrationMonitor(model, rig, ids, kw.pop("record", None))
    try:
        result = _stress_rollout(model, data0, rig, ids, pose, bid, record=monitor, **kw)
        penetration = monitor.report()
        result["penetration"] = penetration
        result["phases"].append(penetration)
        if not penetration["ok"]:
            result["ok"] = False
            result["first_fail"] = "penetration"
        return result
    finally:
        _PINNED.clear()


def _stress_rollout(model, data0, rig: HandRig, ids: dict, pose: dict, bid: int, *,
                   phases=("approach", "close", "lift", "move", "turn", "hold"),
                   approach_m=0.09, open_frac=None, hold_force_N=12.0,
                   lift_heights=(0.06, 0.10, 0.15), shake_m=0.04, turn_deg=25.0,
                   seat_s=0.5, move_s=0.35, hold_s=0.8, settle_s=0.4,
                   nudge_max_m=0.02, drift_max_m=0.03, lift_frac=0.5, slip_max_mps=0.01,
                   maintain_N=0.0, record=None, close_mode="executor", servo_reserve=True, handover=None) -> dict:
    """Drive the hand through the selected phases and measure each one.

    pose needs pos, euler, finger_targets (closed); the press target is the
    executor's clamp (press_targets: flexion past touch, bounded by the servo
    torque cap, deeper where the pose's finger_targets_squeezed go deeper);
    grip_fingers names the pads that must be on the object for "held". The
    approach and the close follow the executor (close_mode 'executor' or
    'first_hold'): floor-clear open shape, approach waypoints, a 0.8 s descent,
    closed targets in 0.5 s then the clamp in 0.3 s, quintic legs under the
    90 deg/s joint-rate ceiling; close_mode 'ramp' keeps the earlier 0.5 s
    linear descent and single 0.6 s ramp to the clamp. With servo_reserve
    (default) a phase 'servo_reserve' judges the finger servos' reserve while
    the hand alone carries the part, exactly as finger_servo_saturation_audit
    judges an episode (ServoReserve): it fails a grasp that holds only with a
    finger servo at 95 % of its torque cap or more on 60 % of the hold.
    """
    data = mujoco.MjData(model)
    data.qpos[:] = data0.qpos
    data.qvel[:] = 0.0
    closed = {k: float(v) for k, v in (pose.get("finger_targets") or {}).items() if k in rig.finger_range}
    grip_fingers = list(pose.get("grip_fingers") or [])
    pos = np.asarray(pose["pos"], float)
    eul = np.asarray(pose["euler"], float)
    executor = close_mode != "ramp"
    # the hand arrives as the executor brings it: the pose's own open shape (finger_targets_open / open_frac) unless
    # the caller asks for an opening explicitly; without its own shape the executor narrows the opening until the open
    # fingers stay above the table at the pose (SubgoalPlanner._floor_clear_open)
    own_open = {k: float(v) for k, v in (pose.get("finger_targets_open") or {}).items() if k in rig.finger_range}
    clamp = executor_clamp(rig, pose) if executor else None
    posed = bool(clamp and clamp["grip_posed"])        # a one-sided wrist press approaches with its grip fingers as posed
    if open_frac is None and own_open:
        opened = own_open
    else:
        of = float(pose.get("open_frac", 0.55)) if open_frac is None else float(open_frac)
        opened = approach_open_targets(rig, closed, of, grip_fingers, posed)
        fl = ids.get("floor_gid", -1)
        if executor and open_frac is None and fl >= 0:
            dq = mujoco.MjData(model); dq.qpos[:] = data0.qpos
            for frac in [of] + [f for f in (0.45, 0.35, 0.25, 0.15) if f < of - 1e-9]:
                opened = approach_open_targets(rig, closed, frac, grip_fingers, posed)
                apply_pose(model, dq, rig, pose, opened)
                if min(geom_distance(model, dq, g, fl, 0.05) for g in rig.collision_gids) > -0.002:
                    break
    # the clamp the executor drives (executor_clamp, as SubgoalPlanner._grip_from_pose makes it): flexion past touch,
    # bounded by the servo torque cap (the pose's own squeezed targets are kept where they go deeper); a one-sided grip
    # (no thumb) presses a few millimetres of pad travel, with the wrist along the contact normal when flexion would lift
    # the pads off their surface. close_mode 'ramp' keeps the earlier uncapped finger clamp
    wrist_press = None
    if executor:
        press = clamp["press"]
        wrist_press = None if clamp["wrist_press"] is None else np.asarray(clamp["wrist_press"], float)
    else:
        squeezed = {k: float(v) for k, v in (pose.get("finger_targets_squeezed") or {}).items() if k in rig.finger_range}
        press = press_targets(rig, closed, grip_fingers, depth_rad=float(pose.get("press_depth_rad", PRESS_DEPTH_RAD)), squeezed=squeezed,
                              max_travel_m=None if pose.get("press_travel_m") is None else float(pose["press_travel_m"]))
    dt = float(model.opt.timestep)
    out_phases = []
    reg = grip_regulator(model, data, rig, ids, bid, grip_fingers, maintain_N) if maintain_N > 0 else None

    def rec(label):
        if record is not None:
            record(model, data, label)

    # a handover: a part the hand scene marks as held in the air is held where it is (by the other hand) until this hand
    # has closed and clamped, then let go; lift, moves, turns and hold are judged with gravity on it as always
    held_free = list(ids.get("held_free") or []) if handover is None or handover else []
    if held_free:
        _PINNED[id(data)] = [(a_, d_, data.qpos[a_:a_ + 7].copy()) for a_, d_ in held_free]
    apply_pose(model, data, rig, pose, opened)
    axis = approach_axis(model, data, rig, pose)
    if "approach" in phases:
        pre = pos + axis * approach_m
        apply_pose(model, data, rig, {"pos": pre, "euler": eul}, opened)
        step_hold(model, data, rig, pre, eul, opened, settle_s, record, "settle")
        p_settled = np.asarray(data.xpos[bid], float).copy()
        if executor:
            # the executor's approach (SubgoalPlanner._approach_pose): the pose's approach waypoints, then a 0.8 s descent
            cur = pre
            for off in pose.get("approach_waypoints") or []:
                nxt = pos + np.asarray(off, float)
                _exec_leg(model, data, rig, cur, nxt, eul, eul, opened, opened, 0.6, record, "approach (open)")
                cur = nxt
            _exec_leg(model, data, rig, cur, pos, eul, eul, opened, opened, 0.8, record, "approach (open)")
            step_hold(model, data, rig, pos, eul, opened, 0.1, record, "approach (open)")
        else:
            ramp(model, data, rig, pre, pos, eul, eul, opened, opened, 0.5, record, "approach (open)")
        nudged = float(np.linalg.norm(np.asarray(data.xpos[bid], float) - p_settled))
        pen = clearance(model, data, rig, ids, ignore_tips=())
        out_phases.append({"name": "approach_open", "object_nudged_mm": round(nudged * 1000, 1),
                           "penetration_mm": pen["object_mm"], "ok": bool(nudged < nudge_max_m)})
    else:
        apply_pose(model, data, rig, pose, opened)
        step_hold(model, data, rig, pos, eul, opened, settle_s, record, "settle")

    # close until held: toward the press target, noting the first configuration where the grip pads are all on it with
    # real force; if that never happens the full press is used and the lift says so
    want = grip_fingers or ["thumb", "index"]
    held = {"at": None}
    every = max(int(round(0.01 / dt)), 1)

    def watch(i, tg):
        if held["at"] is None and i % every == 0:
            cn = finger_contacts(model, data, rig, ids["grasp_gids"])
            if cn["n_finger"] >= min(2, len(want)) and cn["force_N"] > float(hold_force_N):
                held["at"] = dict(tg)

    if executor:
        # as execute_subgoal_plan drives it: to the pose's closed (just touching) targets in 0.5 s and a 0.2 s hold, then
        # to the clamp in 0.3 s; quintic legs stretched to the 90 deg/s joint-rate ceiling (chain_subgoals). One linear
        # 0.6 s ramp straight to the clamp put the pads on the part at other times than the bench does
        _exec_leg(model, data, rig, pos, pos, eul, eul, opened, closed, 0.5, record, "close", watch)
        step_hold(model, data, rig, pos, eul, closed, 0.2, record, "close")
        p_press = pos + wrist_press if wrist_press is not None else pos      # a one-sided press: the wrist leans in
        tg = _exec_leg(model, data, rig, pos, p_press, eul, eul, closed, press, 0.3, record, "close", watch)
        pos = p_press                                  # the seat, lift, moves, turns and hold start from the pressed wrist
    else:
        n = int(0.6 / dt)
        tg = dict(press)
        for i in range(n):
            u = (i + 1) / n
            tg = {k: opened.get(k, press[k]) + (press[k] - opened.get(k, press[k])) * u for k in press}
            hold_ctrl(data, rig, pos, eul, tg)
            mujoco.mj_step(model, data)
            _apply_pins(data)
            rec("close")
            watch(i, tg)
    held_at = held["at"]
    # close_mode 'executor' (default): the full clamp, as execute_subgoal_plan drives it (torque-capped, so it cannot
    # crush); 'first_hold': stop at the first configuration that holds (a gentler grip than the bench will use);
    # 'ramp': the earlier single linear ramp to the clamp
    press = (held_at or tg) if close_mode == "first_hold" else tg
    step_hold(model, data, rig, pos, eul, press, seat_s, record, "seat")
    if _PINNED.pop(id(data), None) is not None:       # the other hand lets go: from here this hand alone holds the part
        step_hold(model, data, rig, pos, eul, press, 0.2, record, "handover")
    ref = object_in_wrist(data, rig, bid)
    cn = finger_contacts(model, data, rig, ids["grasp_gids"])
    out_phases.append({"name": "seated", "held_before_full_press": held_at is not None,
                       "finger_contacts": cn["n_finger"], "fingers": cn["fingers"],
                       "contact_force_N": cn["force_N"], "ok": bool(cn["n_finger"] >= 1)})

    def drift_now():
        return float(np.linalg.norm(object_in_wrist(data, rig, bid) - ref))

    # the finger servos' reserve while the hand alone carries the part (lift, moves, turns, hold): finger_servo_saturation_audit
    # judges the same on the episode, so a grasp these tests pass is not failed there for holding only at the torque cap
    reserve = None
    if servo_reserve and any(ph_ in phases for ph_ in ("lift", "move", "turn", "hold")):
        reserve = ServoReserve(model, rig, bid)
        user_record = record

        def record(m_, d_, label):                     # noqa: F811 (the user's recorder plus the reserve sampler)
            if user_record is not None:
                user_record(m_, d_, label)
            reserve.step(d_)

    cur = pos.copy()
    lifts = []
    if "lift" in phases:
        base_h = float(data.xpos[bid][2])
        for h in lift_heights:
            tgt = pos + np.array([0.0, 0.0, float(h)])
            ramp(model, data, rig, cur, tgt, eul, eul, press, press, 0.5, record, f"lift {h*1000:.0f} mm", reg)
            step_hold(model, data, rig, tgt, eul, press if reg is None else reg(0, press), 0.25, record, "lift")
            cur = tgt.copy()
            rose = float(data.xpos[bid][2]) - base_h
            ph = {"name": f"lift_{h:.2f}", "commanded_mm": round(h * 1000, 1),
                  "achieved_mm": round(rose * 1000, 1), "drift_mm": round(drift_now() * 1000, 1),
                  "ok": bool(rose > lift_frac * h)}
            out_phases.append(ph)
            lifts.append({"commanded_mm": ph["commanded_mm"], "achieved_mm": ph["achieved_mm"]})
    if "move" in phases:
        for nm, v in MOVE_DIRS:
            v = np.asarray(v, float)
            ramp(model, data, rig, cur, cur + v * shake_m, eul, eul, press, press, move_s, record, f"move {nm}", reg)
            ramp(model, data, rig, cur + v * shake_m, cur, eul, eul, press, press, move_s, record, f"move {nm}", reg)
            out_phases.append({"name": f"move_{nm}", "drift_mm": round(drift_now() * 1000, 1),
                               "ok": bool(drift_now() < drift_max_m)})
    if "turn" in phases:
        for nm, ax_i in (("turn_roll", 0), ("turn_pitch", 1), ("turn_yaw", 2)):
            e1 = eul.copy()
            e1[ax_i] += np.radians(turn_deg)
            ramp(model, data, rig, cur, cur, eul, e1, press, press, move_s, record, nm, reg)
            ramp(model, data, rig, cur, cur, e1, eul, press, press, move_s, record, nm, reg)
            step_hold(model, data, rig, cur, eul, press, 0.15, record, nm)
            out_phases.append({"name": nm, "drift_mm": round(drift_now() * 1000, 1),
                               "ok": bool(drift_now() < drift_max_m)})
    slip = 0.0
    if "hold" in phases:
        from tool_library.common import held_contact as HC
        held_slip = hold_with_slip(model, data, rig, [bid], cur, eul, press, hold_s, record)[0]
        d_b = drift_now()
        # Peak excursion keeps a temporary slide-and-recovery from appearing
        # motionless. Preserve the existing normalization by hold duration.
        slip = held_slip["drift_m"] / max(hold_s, 1e-6)
        angular_ok = not held_slip["turn_judged"] or held_slip["turn_deg"] <= HC.RIGID_SLIP_DEG
        cn = finger_contacts(model, data, rig, ids["grasp_gids"])
        out_phases.append({"name": "hold", "drift_mm": round(d_b * 1000, 1),
                           "slip_mm_s": round(slip * 1000, 2), "finger_contacts": cn["n_finger"],
                           "held_slip": held_slip,
                           "contact_force_N": cn["force_N"],
                           "ok": bool(slip < slip_max_mps and angular_ok and cn["n_finger"] >= 1)})
    if reserve is not None:
        out_phases.append(reserve.report())
    pen = clearance(model, data, rig, ids, ignore_tips=FINGERS)
    why = [p["name"] for p in out_phases if not p.get("ok", True)]
    return {"ok": not why, "first_fail": (why[0] if why else None), "phases": out_phases,
            "lifts": lifts, "slip_mm_s": round(slip * 1000, 2),
            "max_drift_mm": round(max((p.get("drift_mm", 0.0) for p in out_phases), default=0.0), 1),
            "penetration_at_end_mm": pen, "press_targets": press, "open_targets": opened,
            "final_wrist_pos": [float(v) for v in cur], "data": data}


def merge_allocated(rig: HandRig, current: dict, new: dict, fingers) -> dict:
    """Finger targets of a hand that already holds something: `current` (its holding command) with only the joints of
    `fingers` (the free fingers a new grasp uses) taken from `new` (bimanual.grasp_more drives a hand this way)."""
    fs = set(fingers or ())
    out = dict(current)
    for k, v in (new or {}).items():
        if k in rig.finger_range and finger_of_joint(k) in fs:
            out[k] = float(v)
    return out


def multi_object_stress_rollout(model, data0, rig: HandRig, poses, bids, obj_gids, *, approach_m=0.08, lift1_m=0.04, transfer_s=1.5,
                                lift_heights=(0.05, 0.10), shake_m=0.03, turn_deg=15.0, move_s=0.4, hold_s=0.8, seat_s=0.4,
                                drift_max_m=0.03, lift_frac=0.5, slip_max_mps=0.01, record=None, ids=None) -> dict:
    """The combined hold of objects taken one after another into ONE hand (bimanual.grasp_more), driven through the
    actuators as the executor drives it: the first grasp (open approach, close to its closed targets, the clamp), a
    lift of lift1_m; then, holding it, the hand moves to each next pose (pre-grasp along that pose's approach, descent)
    and closes ONLY that grasp's fingers (poses[i]['grip_fingers'], which must not overlap the fingers already holding
    something; those keep their press); then every object under gravity and disturbance: lifts, shakes in five
    directions, turns about the three wrist axes and a hold. Each phase is judged on EVERY object held by then: it
    rose with the hand (lifts), it did not drift more than drift_max_m in the wrist frame, it still touches the hand.
    poses: grasp poses in grasp order (pos, euler, finger_targets, grip_fingers, approach, finger_targets_open...);
    bids / obj_gids: each object's body id and collision geoms, in the same order."""
    if len(poses) < 2 or len(poses) != len(bids) or len(bids) != len(obj_gids):
        raise ValueError("multi_object_stress_rollout: give two or more poses, with one body and one geom set per pose")
    from tool_library.common import episode as E, held_contact as HC
    contact_ids = ids or {"obj_gids": set().union(*obj_gids), "floor_gid": -1}
    monitor = GraspPenetrationMonitor(model, rig, contact_ids, record)
    held, recorded_states = [], []

    def record(m, d, label):
        if held:
            recorded_states.append(d.qpos.copy())
        monitor(m, d, label)

    def begin_hold(index):
        recorded_states.append(data.qpos.copy())
        held.append((index, len(recorded_states) - 1))
    alloc = []
    for i, p in enumerate(poses):
        gf = set(p.get("grip_fingers") or [])
        if not gf:
            raise ValueError(f"pose {i} names no grip_fingers: the finger allocation is what this test checks")
        for j, a in enumerate(alloc):
            if a & gf:
                raise ValueError(f"pose {i} uses fingers {sorted(a & gf)} that already hold object {j}")
        alloc.append(gf)
    data = mujoco.MjData(model)
    data.qpos[:] = data0.qpos
    data.qvel[:] = 0.0
    out_phases = []
    # Each grasp keeps its own seating reference through all later motions.

    def judge(name, extra=None, rise=None):
        mujoco.mj_forward(model, data)
        per, ok = {}, True
        recorded_states.append(data.qpos.copy())
        Q = np.asarray(recorded_states)
        for i, start in held:
            motion = HC.measure_slip(model, Q, 1. / float(model.opt.timestep), start, len(Q) - 1,
                                     bids[i], wrist_body(rig), set(rig.body_ids), E.held_bodies(model, bids[i]))
            drift = motion["drift_m"]
            touch = finger_contacts(model, data, rig, obj_gids[i])["n_finger"] > 0 or any(
                (int(data.contact[c].geom1) in rig.geom_ids and int(data.contact[c].geom2) in obj_gids[i]) or
                (int(data.contact[c].geom2) in rig.geom_ids and int(data.contact[c].geom1) in obj_gids[i]) for c in range(data.ncon))
            e = {"drift_mm": round(1000 * drift, 1), "touching": bool(touch),
                 "turn_deg": round(motion["turn_deg"], 3), "turn_judged": motion["turn_judged"],
                 "translation_peak_sample": motion["translation_peak_frame"],
                 "rotation_peak_sample": motion["rotation_peak_frame"]}
            good = (drift < drift_max_m and touch
                    and (not motion["turn_judged"] or motion["turn_deg"] <= HC.RIGID_SLIP_DEG))
            if rise is not None:
                r_ = float(data.xpos[bids[i]][2]) - rise[0][i]
                e["rose_mm"] = round(1000 * r_, 1)
                good = good and r_ > lift_frac * rise[1]
            per[f"object_{i}"] = e
            ok = ok and good
        ph = {"name": name, "objects": per, "ok": bool(ok)}
        ph.update(extra or {})
        out_phases.append(ph)
        return ph

    # ---- the first grasp, as stress_rollout's executor mode drives it
    p0 = poses[0]
    c0 = executor_clamp(rig, p0)
    own0 = {k: float(v) for k, v in (p0.get("finger_targets_open") or {}).items() if k in rig.finger_range}
    of = float(p0.get("open_frac", 0.55))
    opened = own0 or approach_open_targets(rig, c0["closed"], of, c0["grip_fingers"], c0["grip_posed"])
    fl = ids.get("floor_gid", -1) if isinstance(ids, dict) else -1
    if not own0 and fl >= 0:
        # the executor narrows the opening until the open fingers stay above the table at the pose (stress_rollout does too)
        dq = mujoco.MjData(model); dq.qpos[:] = data0.qpos
        for frac in [of] + [f for f in (0.45, 0.35, 0.25, 0.15) if f < of - 1e-9]:
            opened = approach_open_targets(rig, c0["closed"], frac, c0["grip_fingers"], c0["grip_posed"])
            apply_pose(model, dq, rig, p0, opened)
            if min(geom_distance(model, dq, g, fl, 0.05) for g in rig.collision_gids) > -0.002:
                break
    pos, eul = np.asarray(p0["pos"], float), np.asarray(p0["euler"], float)
    ax = approach_axis(model, data, rig, p0)
    apply_pose(model, data, rig, {"pos": pos + ax * approach_m, "euler": eul}, opened)
    step_hold(model, data, rig, pos + ax * approach_m, eul, opened, 0.3, record, "settle")
    _exec_leg(model, data, rig, pos + ax * approach_m, pos, eul, eul, opened, opened, 0.8, record, "approach 0")
    _exec_leg(model, data, rig, pos, pos, eul, eul, opened, c0["closed"], 0.5, record, "close 0")
    step_hold(model, data, rig, pos, eul, c0["closed"], 0.2, record, "close 0")
    p_press = pos + (np.asarray(c0["wrist_press"], float) if c0["wrist_press"] is not None else 0.0)
    cur_f = _exec_leg(model, data, rig, pos, p_press, eul, eul, c0["closed"], c0["press"], 0.3, record, "close 0")
    pos = p_press
    step_hold(model, data, rig, pos, eul, cur_f, seat_s, record, "seat 0")
    mujoco.mj_forward(model, data)
    begin_hold(0)
    cn = finger_contacts(model, data, rig, obj_gids[0])
    judge("seated_0", {"fingers": cn["fingers"], "contact_force_N": cn["force_N"]})
    z0 = {0: float(data.xpos[bids[0]][2])}
    _exec_leg(model, data, rig, pos, pos + np.array([0.0, 0.0, lift1_m]), eul, eul, cur_f, cur_f, 0.8, record, "lift 0")
    step_hold(model, data, rig, pos + np.array([0.0, 0.0, lift1_m]), eul, cur_f, 0.25, record, "lift 0")
    pos = pos + np.array([0.0, 0.0, lift1_m])
    judge("lift_0", rise=(z0, lift1_m))
    # ---- each next object with the free fingers only
    for i in range(1, len(poses)):
        pi = poses[i]
        ci = executor_clamp(rig, pi)
        own = {k: float(v) for k, v in (pi.get("finger_targets_open") or {}).items() if k in rig.finger_range}
        op_i = merge_allocated(rig, cur_f, own or open_targets(rig, ci["closed"], float(pi.get("open_frac", 0.55))), alloc[i])
        cl_i = merge_allocated(rig, cur_f, ci["closed"], alloc[i])
        pr_i = merge_allocated(rig, cur_f, ci["press"], alloc[i])
        p_i = np.asarray(pi["pos"], float)
        e_i = nearest_euler_branch(np.asarray(pi["euler"], float), eul)       # the same orientation, the short way round
        a_i = approach_axis(model, data, rig, pi)
        _exec_leg(model, data, rig, pos, p_i + a_i * approach_m, eul, e_i, cur_f, op_i, transfer_s, record, f"transfer {i}")
        judge(f"transfer_{i}")
        _exec_leg(model, data, rig, p_i + a_i * approach_m, p_i, e_i, e_i, op_i, op_i, 0.8, record, f"approach {i}")
        _exec_leg(model, data, rig, p_i, p_i, e_i, e_i, op_i, cl_i, 0.5, record, f"close {i}")
        step_hold(model, data, rig, p_i, e_i, cl_i, 0.2, record, f"close {i}")
        cur_f = _exec_leg(model, data, rig, p_i, p_i, e_i, e_i, cl_i, pr_i, 0.3, record, f"close {i}")
        step_hold(model, data, rig, p_i, e_i, cur_f, seat_s, record, f"seat {i}")
        pos, eul = p_i, e_i
        mujoco.mj_forward(model, data)
        cn = finger_contacts(model, data, rig, obj_gids[i])
        begin_hold(i)
        ph = judge(f"seated_{i}", {"fingers": cn["fingers"], "contact_force_N": cn["force_N"]})
        if not (set(cn["fingers"]) & alloc[i]):
            ph["ok"] = False
            ph["why"] = f"none of the allocated fingers {sorted(alloc[i])} touches object {i}"
    # ---- every object together: gravity and disturbance
    base = {i: float(data.xpos[bids[i]][2]) for i, _ in held}
    cur = pos.copy()
    for h in lift_heights:
        tgt = pos + np.array([0.0, 0.0, float(h)])
        ramp(model, data, rig, cur, tgt, eul, eul, cur_f, cur_f, 0.5, record, f"lift {h * 1000:.0f} mm")
        step_hold(model, data, rig, tgt, eul, cur_f, 0.25, record, "lift")
        cur = tgt.copy()
        judge(f"lift_{h:.2f}", rise=(base, float(h)))
    for nm, v in MOVE_DIRS:
        v = np.asarray(v, float)
        ramp(model, data, rig, cur, cur + v * shake_m, eul, eul, cur_f, cur_f, move_s, record, f"move {nm}")
        ramp(model, data, rig, cur + v * shake_m, cur, eul, eul, cur_f, cur_f, move_s, record, f"move {nm}")
        judge(f"move_{nm}")
    for nm, ax_i in (("turn_roll", 0), ("turn_pitch", 1), ("turn_yaw", 2)):
        e1 = eul.copy(); e1[ax_i] += np.radians(turn_deg)
        ramp(model, data, rig, cur, cur, eul, e1, cur_f, cur_f, move_s, record, nm)
        ramp(model, data, rig, cur, cur, e1, eul, cur_f, cur_f, move_s, record, nm)
        step_hold(model, data, rig, cur, eul, cur_f, 0.15, record, nm)
        judge(nm)
    from tool_library.common import held_contact as HC
    hold_slips = hold_with_slip(model, data, rig, [bids[i] for i, _ in held], cur, eul, cur_f, hold_s, record)
    ph = judge("hold")
    slip = max(s["drift_m"] / max(hold_s, 1e-6) for s in hold_slips)
    ph["slip_mm_s"] = round(1000 * slip, 2)
    ph["held_slip"] = hold_slips
    ph["ok"] = bool(ph["ok"] and slip < slip_max_mps
                    and all(not s["turn_judged"] or s["turn_deg"] <= HC.RIGID_SLIP_DEG for s in hold_slips))
    penetration = monitor.report()
    out_phases.append(penetration)
    why = [p["name"] for p in out_phases if not p.get("ok", True)]
    return {"ok": not why, "first_fail": "penetration" if not penetration["ok"] else (why[0] if why else None),
            "penetration": penetration, "phases": out_phases, "allocation": [sorted(a) for a in alloc],
            "final_wrist_pos": [float(v) for v in cur], "data": data}


# ---------------------------------------------------------------------------
# geometric grasp predicates
# ---------------------------------------------------------------------------
def straddle_check(model, data, rig: HandRig, axis_p, axis_d, radius, fingers=("thumb", "index"),
                   max_rise_m=0.001, pad_m=TIP_STANDOFF_M, slack_m=0.004) -> dict:
    """Is the bar BETWEEN the two pads, or just under both of them? Four
    conditions: pad-to-pad direction roughly perpendicular to the axis;
    pads on opposite sides of the axis (a sign test); the axis passes inside
    the pad segment within a pad radius; neither pad above the bar's centre
    line (pads riding high press the bar DOWN, a press, not a pinch)."""
    p = [np.asarray(data.site_xpos[rig.fingertip_sites[f]], float) for f in fingers if f in rig.fingertip_sites]
    if len(p) < 2:
        return {"ok": False, "why": "missing tip sites"}
    a, b = p[0], p[1]
    ap = np.asarray(axis_p, float)
    ad = np.asarray(axis_d, float)
    ad = ad / (np.linalg.norm(ad) + 1e-12)
    u = b - a
    span = float(np.linalg.norm(u))
    along = abs(float((u / max(span, 1e-9)) @ ad))
    qa = (a - ap) - float((a - ap) @ ad) * ad
    qb = (b - ap) - float((b - ap) @ ad) * ad
    n = qb - qa
    nn = float(np.linalg.norm(n))
    sep = (float(qa @ (n / nn)) * float(qb @ (n / nn))) if nn > 1e-9 else 1.0
    w0 = a - ap
    A, B, C = float(u @ u), float(u @ ad), float(ad @ ad)
    D, E = float(u @ w0), float(ad @ w0)
    den = A * C - B * B
    s = 0.5 if abs(den) < 1e-12 else float((B * E - C * D) / den)
    t = float((A * E - B * D) / den) if abs(den) > 1e-12 else 0.0
    gap = float(np.linalg.norm((a + s * u) - (ap + t * ad)))
    # the tip SITE sits one pad thickness inside the pad surface, so a pad on
    # the bar reads radius + pad_m from the axis
    reach = float(radius) + float(pad_m) + float(slack_m)
    da, db = float(np.linalg.norm(qa)), float(np.linalg.norm(qb))
    rise = [float(a[2] - ap[2]), float(b[2] - ap[2])]
    ok = bool(along < 0.5 and sep < 0.0 and 0.15 < s < 0.85 and gap < float(radius) + 0.002
              and max(da, db) <= reach and max(rise) <= float(max_rise_m))
    return {"ok": ok, "along_axis_cos": round(along, 3), "opposite_sides": bool(sep < 0.0),
            "along_segment": round(s, 3), "axis_to_segment_mm": round(gap * 1000, 2),
            "pad_to_axis_mm": [round(1000 * da, 2), round(1000 * db, 2)],
            "pad_rise_mm": [round(1000 * v, 2) for v in rise],
            "below_centre_line": bool(max(rise) <= float(max_rise_m)),
            "pad_span_mm": round(span * 1000, 2), "bar_diameter_mm": round(2000 * float(radius), 1),
            "reach_mm": round(reach * 1000, 1)}


def through_hole_check(model, data, rig: HandRig, finger: str, centre, normal, semi_axes) -> dict:
    """Does this finger pass THROUGH a loop, or just reach into it? Through
    means the tip is on the far side of the hole plane, the knuckle on the
    near side, and the crossing point inside the ellipse."""
    c = np.asarray(centre, float)
    n = np.asarray(normal, float)
    n = n / (np.linalg.norm(n) + 1e-12)
    sid = rig.fingertip_sites.get(finger)
    if sid is None:
        return {"ok": False, "why": f"finger '{finger}' has no tip site"}
    tip = np.asarray(data.site_xpos[sid], float)
    kn = rig.knuckle_point(data, finger)
    s_tip, s_kn = float((tip - c) @ n), float((kn - c) @ n)
    u = tip - kn
    denom = float(u @ n)
    inside, r = False, None
    if abs(denom) > 1e-9:
        t = -s_kn / denom
        x = kn + t * u - c
        ax = np.array([1.0, 0.0, 0.0])
        if abs(float(ax @ n)) > 0.9:
            ax = np.array([0.0, 1.0, 0.0])
        e1 = np.cross(n, ax)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        r = float(np.hypot((x @ e1) / semi_axes[0], (x @ e2) / semi_axes[1]))
        inside = bool(0.0 <= t <= 1.0 and r <= 1.0)
    return {"ok": bool(s_tip < 0.0 < s_kn and inside), "tip_beyond_mm": round(-s_tip * 1000, 1),
            "knuckle_before_mm": round(s_kn * 1000, 1), "crossing_r": None if r is None else round(r, 3)}


def bar_pinch_pose(model, data0, rig: HandRig, ids: dict, bar: dict, s=0.5,
                   fingers=("thumb", "index", "middle"), pinch_dir=None, tilt_deg=0.0,
                   finger_spread_m=0.014, approach_dirs=None, iters=IK_ITERS, yaw_step_deg=30,
                   pitches_deg=(-60, -30, 0, 30, 60), time_budget_s=None) -> dict:
    """Thumb-versus-finger pinch straddling the bar {p0, p1, radius} at
    parameter s. The pad points sit on opposite sides of the axis across the
    pinch direction (default horizontal-perpendicular, tilted about the
    tangent by tilt_deg); the fingertip IK does the rest."""
    p0, p1, r = np.asarray(bar["p0"], float), np.asarray(bar["p1"], float), float(bar["radius"])
    t = p1 - p0
    L = float(np.linalg.norm(t))
    if L < 1e-6:
        raise ValueError("bar has zero length")
    t /= L
    if pinch_dir is None:
        pd = np.cross(t, [0.0, 0.0, 1.0])
        if np.linalg.norm(pd) < 1e-6:
            pd = np.array([1.0, 0.0, 0.0])
    else:
        pd = np.asarray(pinch_dir, float)
    pd = pd - np.dot(pd, t) * t
    pd /= np.linalg.norm(pd) + 1e-12
    if tilt_deg:
        from scipy.spatial.transform import Rotation
        pd = Rotation.from_rotvec(np.radians(float(tilt_deg)) * t).apply(pd)
    centre = p0 + float(s) * L * t
    fingers = list(fingers)
    if fingers[0] != "thumb" or len(fingers) < 2:
        raise ValueError("fingers must start with 'thumb' and name at least one opposing finger")
    pts = {"thumb": centre - pd * r}
    nrm = {"thumb": -pd}
    others = fingers[1:]
    offs = np.linspace(-0.5, 0.5, len(others)) * finger_spread_m * (len(others) - 1)
    for f, o in zip(others, offs):
        pts[f] = centre + pd * r + t * o
        nrm[f] = pd
    # the tip site sits on the pad's axis: a pad touches the bar when the site is one pad radius out
    standoff = pad_radius(model, rig, "index") + 0.001
    targets = {f: pts[f] + nrm[f] * standoff for f in pts}
    cen = np.mean(list(targets.values()), axis=0)
    if approach_dirs is None:
        approach_dirs = [[0.0, 0.0, 1.0], (-pd).tolist()]
    seeds = seed_sweep(cen, approach_dirs, yaw_step_deg=yaw_step_deg, pitches_deg=pitches_deg)
    search = {}
    worst, res, pen, qpos, face = solve_fingertip_ik(model, data0, rig, ids, targets, seeds,
                                                     face_at=centre, iters=iters, time_budget_s=time_budget_s, stats=search)
    d = mujoco.MjData(model)
    d.qpos[:] = qpos
    curl = curl_free_fingers_clear(model, d, rig, ids, set(pts))
    pose = pose_from_data(model, d, rig, rig.side)
    # squeezed: the pads 3 mm into the bar with the wrist fixed (a pinch presses by closing the pads together)
    finger_ik(model, d, rig, {f: pts[f] + nrm[f] * (standoff - 0.003) for f in pts})
    squeezed = {k: float(d.qpos[a]) for k, a in rig.finger_qpos.items()}
    d.qpos[:] = qpos; curl_free_fingers(rig, d, set(pts), curl); mujoco.mj_forward(model, d)
    pose.update({"grip_fingers": fingers, "finger_targets_squeezed": squeezed, "press_depth_rad": 0.12, "standoff_m": float(standoff),
                 "residual_mm": {f: round(v * 1000, 2) for f, v in res.items()},
                 "worst_residual_mm": round(worst * 1000, 2),
                 "overlap_mm": round(deepest_overlap(model, d, rig, ids, keep_tips=tuple(pts), visual_floor=True) * 1000, 2),
                 "palm_facing": round(float(face), 3),
                 "bar": {"p0": p0.tolist(), "p1": p1.tolist(), "radius": r},
                 "pinch_dir": pd.tolist(), "approach": [0.0, 0.0, 1.0],
                 "targets": {f: v.tolist() for f, v in targets.items()}})
    pose["straddle"] = straddle_check(model, d, rig, centre, t, r, fingers=("thumb", others[0]))
    pose["ik_search"] = search
    return pose


def bar_from_geom(model, data, name) -> dict:
    """Axis end points and radius of a capsule or cylinder geom, world frame."""
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if gid < 0:
        raise ValueError(f"geom '{name}' is not in the scene")
    t = int(model.geom_type[gid])
    if t not in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
        raise ValueError(f"geom '{name}' is not a capsule or cylinder")
    r, h = float(model.geom_size[gid][0]), float(model.geom_size[gid][1])
    R = np.asarray(data.geom_xmat[gid], float).reshape(3, 3)
    c = np.asarray(data.geom_xpos[gid], float)
    return {"p0": (c - R[:, 2] * h).tolist(), "p1": (c + R[:, 2] * h).tolist(), "radius": r}

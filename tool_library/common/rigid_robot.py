"""The physical robot used by Stage 3/4, and its proposal-state adapter.

The legacy six-coordinate wrist is useful for grasp search, but is NOT a robot
actuator.  A runtime hand is a fixed child of its arm flange.  Only motor torques
and finger servos advance that model.  Conversion back to proposal coordinates
only expresses the runtime hand's FK; it never fits or changes an arm/object.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import copy
import hashlib
import weakref
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

SCHEMA = "dexagent.rigid_robot.v1"
SETUP = Path(__file__).resolve().parents[2] / "setup"
WRIST_SUFFIXES = ("pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z")
# I2RT SDK cbe48976b44aae45af856c62545be00ea2feed11 motor_drivers/utils.py:
# DM4340 torque encoding range +/-28 Nm; DM4310 +/-10 Nm.  YAM joints 1--3
# use DM4340, joints 4--6 DM4310 (doc.i2rt.com/products/yam).  These are
# controller saturation bounds, NOT a claim about continuous rated torque.
ARM_TORQUE_NM = np.array([28., 28., 28., 10., 10., 10.])
ARM_KP = np.array([180., 180., 140., 70., 50., 40.])
ARM_KD = np.array([18., 18., 14., 6., 4., 3.])
_SPECS = weakref.WeakKeyDictionary()
_RUNTIMES = weakref.WeakKeyDictionary()


def _values(x):
    return " ".join(format(float(v), ".17g") for v in np.asarray(x).ravel())


def joint_identities(model):
    """Stable joint identities across robot compilation and body reordering.

    MJCF permits unnamed joints, including one free joint per object. An empty
    name cannot identify these: using it as a dictionary key loses all but the
    last object's state. Anchor anonymous joints to their owning body instead.
    """
    def body_identity(bid):
        name = model.body(bid).name
        if name:
            return ("body", name)
        if bid == 0:
            return ("world",)
        parent = int(model.body_parentid[bid])
        siblings = [b for b in range(1, model.nbody)
                    if int(model.body_parentid[b]) == parent and not model.body(b).name]
        return (*body_identity(parent), "anonymous_child", siblings.index(bid))

    identities = []
    for j in range(model.njnt):
        name = model.joint(j).name
        if not name:
            bid = int(model.jnt_bodyid[j])
            local = j - int(model.body_jntadr[bid])
            name = "__anonymous_joint__:" + repr((*body_identity(bid), local, int(model.jnt_type[j])))
        identities.append(name)
    if len(set(identities)) != len(identities):
        raise ValueError("joint identities are not unique")
    return identities


def joint_columns(model, velocity=False):
    """Named slices, including full free/ball joints; never match by nq alone."""
    adr = model.jnt_dofadr if velocity else model.jnt_qposadr
    n = model.nv if velocity else model.nq
    return {name: slice(int(adr[j]), int(adr[j + 1]) if j + 1 < model.njnt else n)
            for j, name in enumerate(joint_identities(model))}


def map_joints(Q, source, target, velocity=False):
    Q = np.asarray(Q, float)
    nsrc = source.nv if velocity else source.nq
    if Q.shape[-1] != nsrc:
        raise ValueError(f"state has {Q.shape[-1]} columns, named source model has {nsrc}")
    n = target.nv if velocity else target.nq
    out = np.zeros(Q.shape[:-1] + (n,))
    if not velocity:
        out[...] = target.qpos0
    a, b = joint_columns(source, velocity), joint_columns(target, velocity)
    for name, dst in b.items():
        if name not in a:
            raise ValueError(f"joint {name!r} is absent in the source model")
        src = a[name]
        if src.stop - src.start != dst.stop - dst.start:
            raise ValueError(f"joint {name!r} changes dimension")
        out[..., dst] = Q[..., src]
    return out


def is_rigid(model):
    """Structural routing predicate (the independent release gate checks more)."""
    found = False
    for side in ("left", "right"):
        hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"robot_{side}_{side}_base_yaw")
        arm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"robot_{side}_arm_gripper")
        if hand < 0 and arm < 0:
            continue
        if hand < 0 or arm < 0 or int(model.body_parentid[hand]) != arm or model.body_jntnum[hand]:
            return False
        if any(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot_{side}_{side}_{k}") >= 0
               for k in WRIST_SUFFIXES):
            return False
        found = True
    return found


def register_model(model, spec, base_dir=None):
    """Remember an immutable source spec for lossless conversion, including assets."""
    sp = spec.copy()
    if base_dir is not None:
        base = Path(base_dir).resolve()
        for asset in sp.meshes:
            if asset.file and not Path(asset.file).is_absolute():
                asset.file = str(base / (sp.meshdir or "") / asset.file)
        for asset in sp.textures:
            if asset.file and not Path(asset.file).is_absolute():
                asset.file = str(base / (sp.texturedir or "") / asset.file)
        sp.meshdir = ""; sp.texturedir = ""
    _SPECS[model] = sp


@lru_cache(maxsize=4)
def trusted_mounts(setup_scene=str(SETUP / "scene.xml")):
    """Flange -> hand transforms from the versioned setup, never from a rollout."""
    m = mujoco.MjModel.from_xml_path(str(setup_scene))
    if not m.nkey:
        raise ValueError("the trusted setup must have a calibrated home keyframe")
    d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_kinematics(m, d)
    out = {}
    for s in ("left", "right"):
        a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"robot_{s}_arm_gripper")
        h = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"robot_{s}_{s}_base_yaw")
        if a < 0 or h < 0:
            continue
        R = d.xmat[a].reshape(3, 3)
        p, Rh = R.T @ (d.xpos[h] - d.xpos[a]), R.T @ d.xmat[h].reshape(3, 3)
        q = np.zeros(4); mujoco.mju_mat2Quat(q, Rh.ravel())
        out[s] = (p, q)
    return out


@dataclass
class RigidRobot:
    source_model: object
    model: object
    xml: str

    @property
    def scene_sha256(self):
        return hashlib.sha256(self.xml.encode()).hexdigest()

    def to_runtime_qpos(self, Q):
        return map_joints(Q, self.source_model, self.model)

    def check_reference(self, Q, pos_tol_m=1e-4, rot_tol_deg=1e-3):
        """Reject floating hand commands that contradict the planned arm FK.

        Coordinate conversion must not silently replace an unreachable wrist
        target with a different, easier task. The planner couples it first.
        """
        from scipy.spatial.transform import Rotation
        Q = np.asarray(Q, float)
        rows = Q[None] if Q.ndim == 1 else Q
        runtime = self.to_runtime_qpos(rows)
        a, b = mujoco.MjData(self.source_model), mujoco.MjData(self.model)
        hands = [name for name in (f"robot_{s}_{s}_base_yaw" for s in ("left", "right"))
                 if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0]
        for t, (qa, qb) in enumerate(zip(rows, runtime)):
            a.qpos[:] = qa; b.qpos[:] = qb
            mujoco.mj_kinematics(self.source_model, a); mujoco.mj_kinematics(self.model, b)
            for name in hands:
                ia, ib = self.source_model.body(name).id, self.model.body(name).id
                pe = np.linalg.norm(a.xpos[ia] - b.xpos[ib])
                re = np.degrees(Rotation.from_matrix(a.xmat[ia].reshape(3, 3).T @ b.xmat[ib].reshape(3, 3)).magnitude())
                if pe > pos_tol_m or re > rot_tol_deg:
                    raise ValueError(f"uncoupled proposal at frame {t}, {name}: {pe * 1000:.3f} mm / {re:.5f} deg; solve the arm before physics")

    def to_proposal_qpos(self, Q, seed=None):
        """Exact coordinate expansion. Arm/finger/object values are untouched."""
        from tool_library.common import hand as H, plan as P
        Q = np.asarray(Q, float)
        scalar = Q.ndim == 1
        rows = Q[None] if scalar else Q
        if rows.shape[1] != self.model.nq:
            raise ValueError("runtime qpos/model mismatch")
        out = np.tile(self.source_model.qpos0 if seed is None else np.asarray(seed, float), (len(rows), 1))
        src, dst = joint_columns(self.model), joint_columns(self.source_model)
        for name, sl in src.items():
            out[:, dst[name]] = rows[:, sl]
        rd, pd = mujoco.MjData(self.model), mujoco.MjData(self.source_model)
        rigs = P.scene_rigs(self.source_model)
        prev = {s: np.array([out[0, r.wrist_qpos[k]] for k in ("rx", "ry", "rz")]) for s, r in rigs.items()}
        for i, row in enumerate(rows):
            rd.qpos[:] = row; mujoco.mj_kinematics(self.model, rd)
            pd.qpos[:] = out[i]; mujoco.mj_kinematics(self.source_model, pd)
            for s, rig in rigs.items():
                h = self.model.body(f"robot_{s}_{s}_base_yaw").id
                values = P.wrist_joints_from_world(self.source_model, pd, rig, rd.xpos[h], rd.xmat[h].reshape(3, 3), prev[s])
                for k, v in zip(H.WRIST, values):
                    out[i, rig.wrist_qpos[k]] = v
                prev[s] = values[3:]
        return out[0] if scalar else out


def compile_rigid(source, setup_scene=None, *, mounts=None, restore_arm_inertia=True):
    """Compile the rigid execution tree from a registered proposal model or XML.

    ``mounts`` is for a separately calibrated setup (and small-model tests). The
    production default is the checked-in setup's calibration. Input keyframes
    are remapped by joint name; all non-robot mechanisms and geometry survive.
    """
    from tool_library.common import hand as H
    cacheable = mounts is None and setup_scene is None and restore_arm_inertia
    if isinstance(source, (str, Path)):
        path = Path(source).resolve()
        from tool_library.common import episode as E
        sm, _ = E.build_execution_model(path)
    else:
        sm = source
    if is_rigid(sm):
        if sm not in _SPECS:
            raise ValueError("register the runtime model's spec before requesting its XML")
        return RigidRobot(sm, sm, _SPECS[sm].to_xml())
    if sm in _RUNTIMES and cacheable:
        runtime_model, runtime_xml = _RUNTIMES[sm]
        return RigidRobot(sm, runtime_model, runtime_xml)
    if sm not in _SPECS:
        raise ValueError("arm execution requires a source spec: load through build_execution_model or compile_rigid(path)")
    root = ET.fromstring(_SPECS[sm].to_xml())
    parents = {c: p for p in root.iter() for c in p}
    bodies = {b.get("name"): b for b in root.iter("body")}
    mounts = mounts or trusted_mounts(str(setup_scene or SETUP / "scene.xml"))
    keys = [(sm.key(i).name, sm.key_qpos[i].copy()) for i in range(sm.nkey)]
    for e in list(root.findall("keyframe")):
        root.remove(e)
    size = root.find("size")
    if size is not None:
        size.attrib.pop("nkey", None)
    act = root.find("actuator")
    if act is None:
        act = ET.SubElement(root, "actuator")
    custom = root.find("custom")
    if custom is None:
        custom = ET.SubElement(root, "custom")
    ET.SubElement(custom, "text", name="dexagent_robot_schema", data=SCHEMA)
    ET.SubElement(custom, "numeric", name="dexagent_arm_torque_nm", data=_values(ARM_TORQUE_NM))
    # The arm inertias come from the arm file beside the setup scene in use (the checked-in one by default).
    arm_xml = SETUP / "robot/yam_arm.xml"
    if setup_scene is not None and (Path(setup_scene).parent / "robot/yam_arm.xml").is_file():
        arm_xml = Path(setup_scene).parent / "robot/yam_arm.xml"
    arm_template = ET.parse(arm_xml).getroot() if restore_arm_inertia else None
    arm_bodies = {b.get("name"): b for b in arm_template.iter("body")} if arm_template is not None else {}
    sides = []
    for side in ("left", "right"):
        head = f"robot_{side}_{side}_"
        hand, flange = bodies.get(head + "base_yaw"), bodies.get(f"robot_{side}_arm_gripper")
        if hand is None and flange is None:
            continue
        if hand is None or flange is None or side not in mounts:
            raise ValueError(f"incomplete {side} robot or missing calibrated mount")
        sides.append(side)
        p, q = mounts[side]
        parents[hand].remove(hand)
        for child in list(hand):
            if child.tag in ("joint", "freejoint", "inertial"):
                hand.remove(child)
        hand.set("pos", _values(p)); hand.set("quat", _values(q)); hand.set("gravcomp", "0")
        flange.append(hand)
        oldroot = bodies.get(head + "base_tx")
        if oldroot is not None:
            parents[oldroot].remove(oldroot)
        wrist_names = {head + k for k in WRIST_SUFFIXES}
        for a in list(act):
            if a.get("joint") in wrist_names:
                act.remove(a)
        eq = root.find("equality")
        if eq is not None:
            for e in list(eq):
                if {e.get("body1"), e.get("body2")} == {head + "base_yaw", f"robot_{side}_arm_gripper"}:
                    eq.remove(e)
        ET.SubElement(custom, "numeric", name=f"dexagent_mount_{side}", data=_values(np.r_[p, q]))
        for i in range(1, 7):
            name = f"robot_{side}_arm_dof_joint{i}"
            js = next((j for j in root.iter("joint") if j.get("name") == name), None)
            if js is None:
                raise ValueError(f"missing arm joint {name}")
            cap = ARM_TORQUE_NM[i - 1]
            js.set("actuatorfrcrange", _values([-cap, cap])); js.set("actuatorfrclimited", "true")
            # Reflected inertia stabilises the motor servo, without external forces.
            js.set("armature", "0.032" if i <= 3 else "0.0018")
            js.set("damping", "0.1")
            ET.SubElement(act, "motor", name=name + "_motor", joint=name,
                          ctrllimited="true", ctrlrange=_values([-cap, cap]),
                          forcelimited="true", forcerange=_values([-cap, cap]), gear="1")
        for name, b in bodies.items():
            if name and name.startswith(f"robot_{side}_arm_"):
                short = name[len(f"robot_{side}_arm_"):]
                tb = arm_bodies.get(short)
                inertia = None if tb is None else tb.find("inertial")
                if inertia is not None:
                    old = b.find("inertial")
                    if old is not None:
                        b.remove(old)
                    b.insert(0, copy.deepcopy(inertia))
    if not sides:
        raise ValueError("no complete arm-hand stations to convert")
    for b in root.iter("body"):
        if (b.get("name") or "").startswith("robot_"):
            b.set("gravcomp", "0")
    # Arm collision geometry is the visible convex mesh, not a passive drawing.
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    arm_geoms, hand_geoms, other_geoms = [], [], []
    taken = {g.get("name") for g in root.iter("geom") if g.get("name")}

    def fresh(base):
        # a generated geom name no other geom has (unnamed bodies and world geoms would otherwise repeat one)
        name, n = base, 1
        while name in taken:
            name, n = f"{base}_{n}", n + 1
        taken.add(name)
        return name

    for b in root.iter("body"):
        bn = b.get("name", "")
        for k, g in enumerate(list(b.findall("geom"))):
            if "_arm_" in bn and bn.startswith("robot_") and g.get("type") == "mesh":
                cg = copy.deepcopy(g); gn = f"{bn}__rigid_collision_{k}"
                cg.set("name", gn); cg.set("contype", "0"); cg.set("conaffinity", "0")
                cg.set("group", "3"); cg.set("density", "0"); cg.set("mass", "0")
                b.append(cg); arm_geoms.append((gn, bn))
            elif "_collision_hand_" in g.get("name", ""):
                hand_geoms.append((g.get("name"), bn))
            elif not bn.startswith("robot_") and (int(g.get("contype", "1")) or int(g.get("conaffinity", "1"))):
                gn = g.get("name") or fresh(f"{bn}__rigid_geom_{k}"); g.set("name", gn)
                other_geoms.append((gn, bn))
    for g in root.findall("./worldbody/geom"):
        if int(g.get("contype", "1")) or int(g.get("conaffinity", "1")):
            gn = g.get("name") or fresh("rigid_world_floor"); g.set("name", gn); other_geoms.append((gn, "world"))
    existing = {frozenset((p.get("geom1"), p.get("geom2"))) for p in contact.findall("pair")}
    for gn, bn in arm_geoms:
        side = "left" if bn.startswith("robot_left_") else "right"
        for hn, hb in other_geoms + hand_geoms + arm_geoms:
            if hn == gn or frozenset((gn, hn)) in existing:
                continue
            if hb.startswith(f"robot_{side}_arm_"):
                # Adjacent members are joined; distant links must collide.
                aa = bodies.get(bn); bb = bodies.get(hb)
                if (aa is bb or parents.get(aa) is bb or parents.get(bb) is aa
                        or parents.get(parents.get(aa)) is bb or parents.get(parents.get(bb)) is aa):
                    continue
            if hb.startswith(f"robot_{side}_{side}_") and bn.endswith(("link5", "gripper")) and hb.endswith(("base_yaw", "hand_C_MC", "palm")):
                continue
            ET.SubElement(contact, "pair", geom1=gn, geom2=hn, condim="3", solref="0.006 1", solimp="0.99 0.999 0.001")
            existing.add(frozenset((gn, hn)))
    xml = ET.tostring(root, encoding="unicode")
    sp = mujoco.MjSpec.from_string(xml)
    H.add_hand_self_contacts(sp)
    m = sp.compile()
    # Position-served fingers retain the library's documented physical limits.
    for side in sides:
        rig = H.HandRig.build(m, f"robot_{side}_", side)
        H.limit_hand_forces(m, rig)
        for a in rig.finger_act.values():
            sa = sp.actuator(m.actuator(a).name)
            sa.forcelimited = True; sa.forcerange = m.actuator_forcerange[a].copy()
            sa.gainprm = m.actuator_gainprm[a].copy(); sa.biasprm = m.actuator_biasprm[a].copy()
            j = int(m.actuator_trnid[a, 0]); sj = sp.joint(m.joint(j).name)
            sj.armature = float(m.dof_armature[m.jnt_dofadr[j]])
    for name, q0 in keys:
        key = sp.add_key(); key.name = name; key.qpos = map_joints(q0, sm, m)
    sp.compile(); xml = sp.to_xml()
    # Execute precisely the serialized artifact that verifiers/Stage 4 reload.
    sp = mujoco.MjSpec.from_string(xml); m = sp.compile()
    register_model(m, sp)
    result = RigidRobot(sm, m, xml)
    if not is_rigid(m):
        raise ValueError("rigid conversion did not produce a fixed flange-hand tree")
    if cacheable:
        _RUNTIMES[sm] = (m, xml)
    return result


class MechanismReplay:
    """Deterministic physical mechanism updates, shared with independent replay.

    ``rules`` must come from the trusted Stage 2 mechanism specification in a
    verifier. Recorded equality flags/data are observations, never commands to
    this class. Call ``post_step`` after mj_step and before mj_forward.
    ``state`` is only for continuing this engine across executor segments; an
    independent audit starts the engine itself and recomputes its full history.
    """
    def __init__(self, model, data, rules=(), inactive=(), latched=None, state=None):
        from tool_library.common import episode as E
        self.model = model
        self.inactive = set(inactive)
        self.released, self.events, self.threads = [], [], []
        self.cuts = E.cut_rules_live(model, rules, self.inactive)
        self.latches = E.latch_rules_live(model, rules)
        if state is not None:
            self.inactive = set(state["inactive"])
            for L in self.latches:
                previous = state["latches"][L["name"]]
                for key in ("state", "pull", "pull_peak", "gap_prev"):
                    L[key] = previous[key]
                model.eq_data[L["e"]] = np.asarray(previous["eq_data"], float)
        elif self.latches:
            E.latch_begin(model, data, self.latches, self.inactive, latched, self.events, 0)
        for rule in rules or ():
            if rule.get("kind") in ("cut", "latch"):
                continue
            e = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, rule["engaged"])
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, rule["lift_joint"])
            if e < 0 or j < 0:
                raise ValueError("mechanism rule refers to a missing equality/joint")
            self.threads.append((e, int(model.jnt_qposadr[j]), float(rule.get("frac", .96)) * float(rule["travel_m"]), rule["engaged"]))

    def post_step(self, data, frame=0):
        from tool_library.common import episode as E
        for e, qa, threshold, name in self.threads:
            if data.eq_active[e] and abs(float(data.qpos[qa])) >= threshold:
                data.eq_active[e] = 0; self.inactive.add(name)
                self.released.append({"equality": name, "frame": int(frame)})
        if self.cuts:
            E.cut_step(self.model, data, self.cuts, self.inactive, self.released, frame)
        if self.latches:
            E.latch_step(self.model, data, self.latches, self.inactive, self.events, frame)

    def snapshot(self):
        return {"inactive": sorted(self.inactive), "latches": {
            L["name"]: {**{key: L[key] for key in ("state", "pull", "pull_peak", "gap_prev")},
                        "eq_data": self.model.eq_data[L["e"]].tolist()} for L in self.latches}}

    def close(self):
        from tool_library.common import episode as E
        return E.latch_end(self.model, self.latches) if self.latches else {}


def servo_rollout(model, Qref, fps, qpos_start=None, qvel_start=None, wrist_kp=10000.,
                  record_contacts=True, ctrl_lead_s=0., settle_s=0., progress=None,
                  eq_inactive=(), rules=(), finger_force_nm=None, wrist_force_n=None,
                  wrist_torque_nm=None, finger_kp=None, balance_grip=False, latched=None, eq_active_start=None, state_start=None,
                  mechanism_state_start=None):
    """Motor-driven rigid robot rollout, recording the actual integration states.

    Cartesian impedance produces a bounded wrench, mapped to arm torques by
    J.T. Gravity/Coriolis feedforward is motor effort and shares the SAME motor
    torque cap. No gravcomp, mocap, qfrc_applied, or post-step qpos writes.
    """
    from scipy.spatial.transform import Rotation
    from tool_library.common import episode as E, hand as H
    if not is_rigid(model):
        raise ValueError("physical arm rollout requires the rigid runtime model")
    Qref = np.asarray(Qref, float)
    if Qref.ndim != 2 or Qref.shape[1] != model.nq or not len(Qref) or not np.isfinite(Qref).all():
        raise ValueError("non-finite or incompatible runtime reference")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    parts = E.robot_parts(model)
    d, target = mujoco.MjData(model), mujoco.MjData(model)
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    if state_start is not None:
        state_start = np.asarray(state_start, float)
        if state_start.shape != (mujoco.mj_stateSize(model, state_spec),):
            raise ValueError("integration state belongs to another runtime model")
        mujoco.mj_setState(model, d, state_start, state_spec)
        if qpos_start is not None and not np.array_equal(d.qpos, qpos_start):
            raise ValueError("qpos_start contradicts the complete integration state")
        if qvel_start is not None and not np.array_equal(d.qvel, qvel_start):
            raise ValueError("qvel_start contradicts the complete integration state")
    else:
        d.qpos[:] = Qref[0] if qpos_start is None else np.asarray(qpos_start, float)
        if qvel_start is not None:
            d.qvel[:] = np.asarray(qvel_start, float)
    if eq_active_start is not None:
        eq0 = np.asarray(eq_active_start, np.int8)
        if eq0.shape != (model.neq,) or not np.isin(eq0, (0, 1)).all():
            raise ValueError("initial equality state does not match the runtime scene")
        d.eq_active[:] = eq0
    mujoco.mj_forward(model, d)
    initial_qpos, initial_qvel, initial_act = d.qpos.copy(), d.qvel.copy(), d.act.copy()
    # Caps are never silently expanded by a plan. The library hand cap is part
    # of the trusted execution configuration, just like the motor bounds.
    ff = H.FINGER_FORCE_NM if finger_force_nm is None else float(finger_force_nm)
    fk = H.FINGER_KP if finger_kp is None else float(finger_kp)
    if ff != H.FINGER_FORCE_NM or fk is not None:
        raise ValueError("runtime finger dynamics are fixed by the scene; use finger control targets instead of changing force/gain")
    torque_cap = H.WRIST_TORQUE_NM if wrist_torque_nm is None else float(wrist_torque_nm)
    if torque_cap <= 0 or torque_cap > H.WRIST_TORQUE_NM:
        raise ValueError("requested wrist torque exceeds the trusted hand limit")
    kp = float(wrist_kp or 10000.)
    if not 0 < kp <= 10000.:
        raise ValueError("wrist stiffness must be in (0,10000] N/m")
    gains, motors, jac, schedules = {}, {}, {}, {}
    for s, pt in parts.items():
        rig = pt["rig"]
        # The serialized model IS the dynamics contract. Recomputing a damping
        # coefficient here changes its rounded XML value and makes independent
        # torque replay diverge at contact. All limits are installed at compile.
        aids = list(rig.finger_act.values())
        if (not np.all(model.actuator_forcelimited[aids])
                or np.max(np.abs(model.actuator_forcerange[aids])) > ff + 1e-12):
            raise ValueError("runtime finger force exceeds the trusted hand limit")
        gains[s] = {"finger_force_nm": ff, "source": "serialized_runtime_scene"}
        motors[s] = np.array([model.actuator(model.joint(j).name + "_motor").id for j in pt["arm_joints"]])
        jac[s] = (np.zeros((3, model.nv)), np.zeros((3, model.nv)))
        raw = wrist_force_n.get(s, H.WRIST_FORCE_N) if isinstance(wrist_force_n, dict) else wrist_force_n
        raw = H.WRIST_FORCE_N if raw is None else raw
        cap = np.asarray(raw, float)
        if np.any(~np.isfinite(cap)) or np.any(cap <= 0) or np.any(cap > H.WRIST_FORCE_N):
            raise ValueError("requested wrist force exceeds the trusted hand limit")
        schedules[s] = cap
    inactive = set(eq_inactive or ())
    for nm in inactive:
        e = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, nm)
        if e < 0:
            raise ValueError(f"unknown mechanism equality {nm!r}")
        d.eq_active[e] = 0
    eq_start = d.eq_active.astype(np.int8).copy()
    mechanism = MechanismReplay(model, d, rules, inactive, latched, mechanism_state_start)
    mechanism_initial = mechanism.snapshot()
    eq_start = d.eq_active.astype(np.int8).copy()
    initial_state = np.zeros(mujoco.mj_stateSize(model, state_spec))
    mujoco.mj_getState(model, d, initial_state, state_spec)
    eq_data_start = model.eq_data.copy()
    initial_time = float(d.time)
    T, dt = len(Qref), float(model.opt.timestep)
    Q = np.empty_like(Qref); V = np.zeros((T, model.nv)); C = np.zeros((T, model.nu)); ACT = np.zeros((T, model.na))
    track, wrench = np.zeros((T, 2)), np.zeros((T, 2))
    pen, objpen = np.zeros(T), np.zeros(T)
    forces, contacts = np.zeros((T, 2)), []
    eqhist = np.zeros((T, model.neq), np.int8)
    times = np.zeros(T); arm_error = np.zeros((T, len(parts)))
    steps_q, steps_v, steps_c, steps_t, steps_eq, steps_act, steps_eq_before, steps_eq_data = [], [], [], [], [], [], [], []
    robot = {b for b in range(model.nbody) if model.body(b).name.startswith("robot_")}
    f6 = np.zeros(6)
    balancer = E.GripBalancer(model, parts) if balance_grip else None
    tangent = np.zeros(model.nv)
    refvel = np.zeros(model.nv)
    last_u = None
    def command(t, fraction):
        nonlocal last_u
        # Geodesic joint interpolation; free/ball quaternions remain valid.
        u = min(max(0., float(t - 1) + fraction + float(ctrl_lead_s) * fps), T - 1.)
        a, b = int(np.floor(u)), min(int(np.floor(u)) + 1, T - 1)
        mujoco.mj_differentiatePos(model, tangent, 1., Qref[a], Qref[b])
        target.qpos[:] = Qref[a]; mujoco.mj_integratePos(model, target.qpos, tangent, u - a)
        refvel[:] = tangent * fps
        target.qvel[:] = refvel
        mujoco.mj_forward(model, target)
        for si, (s, pt) in enumerate(parts.items()):
            rig, h = pt["rig"], pt["ee_body"]
            jp, jr = jac[s]
            mujoco.mj_jacBody(model, d, jp, jr, h)
            adr, qa = pt["arm_dadr"], pt["arm_qadr"]
            J = np.vstack([jp[:, adr], jr[:, adr]])
            pe = target.xpos[h] - d.xpos[h]
            Re = target.xmat[h].reshape(3, 3) @ d.xmat[h].reshape(3, 3).T
            re = Rotation.from_matrix(Re).as_rotvec()
            # Target joint velocities induce the target wrist velocity. At the
            # millisecond interpolation scale the current Jacobian is accurate.
            dv = J @ (refvel[adr] - d.qvel[adr])
            kd = 2. * np.sqrt(kp * max(H.hand_mass(model, rig), .2))
            F = kp * pe + kd * dv[:3]
            M = 100. * re + 3. * dv[3:]
            cap = schedules[s]
            if cap.ndim == 2:
                cap = cap[min(t, len(cap) - 1)]
            F = np.clip(F, -cap, cap); M = np.clip(M, -torque_cap, torque_cap)
            # Damped nullspace posture control preserves the planned IK branch
            # at singularities without changing the Cartesian target.
            null = np.eye(len(qa)) - np.linalg.pinv(J, rcond=1e-3) @ J
            posture = ARM_KP[:len(qa)] * (target.qpos[qa] - d.qpos[qa]) + ARM_KD[:len(qa)] * (refvel[adr] - d.qvel[adr])
            tau = J.T @ np.r_[F, M] + null @ posture + d.qfrc_bias[adr]
            aids = motors[s]
            lo, hi = model.actuator_ctrlrange[aids].T
            d.ctrl[aids] = np.clip(tau, lo, hi)
            for aid, qadr in E.actuator_targets(model, rig):
                d.ctrl[aid] = target.qpos[qadr]
            if balancer is not None:
                balancer.reset_if_changed(s, target.qpos[list(rig.finger_qpos.values())])
                for name, aid in rig.finger_act.items():
                    d.ctrl[aid] += balancer.offset[s][name]
            wrench[t] = np.maximum(wrench[t], [np.linalg.norm(F), np.linalg.norm(M)])
        last_u = u

    def diagnose(t):
        cs, net = set(), {s: np.zeros(3) for s in parts}
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            r1, r2 = b1 in robot, b2 in robot
            depth = max(0., -float(c.dist))
            if depth > 0.005 and (r1 != r2 or (not r1 and not r2)):
                # MuJoCo's convex colliders now and then emit one contact as deep as a box edge or a hull for nearly aligned
                # faces (episode.servo_rollout measured 80 mm between two bars, 2.2 mm by the exact distance): a deep
                # reading is re-measured on the geom pair; a true penetration keeps its depth
                from tool_library.common import hand as H
                depth = max(0., min(depth, -H.geom_distance(model, d, int(c.geom1), int(c.geom2), 0.01)))
            if r1 != r2:
                pen[t] = max(pen[t], depth)
                hb, ob = (b1, b2) if r1 else (b2, b1)
                s = next((s for s, pt in parts.items() if hb in pt["rig"].body_ids), None)
                if s is not None:
                    cs.add((s, model.body(ob).name))
                    mujoco.mj_contactForce(model, d, i, f6)
                    net[s] += c.frame.reshape(3, 3).T @ f6[:3] * (1 if r1 else -1)
            elif not r1 and not r2:
                objpen[t] = max(objpen[t], depth)
        for si, s in enumerate(("left", "right")):
            if s in net:
                forces[t, si] = max(forces[t, si], np.linalg.norm(net[s]))
        return sorted(cs)

    unstable, completed = None, 0
    try:
        # Optional settling is real integration under the same bounded motors.
        # The caller may then decide whether that changed the initial task state.
        for _ in range(int(round(float(settle_s) / dt))):
            command(0, 1.); mujoco.mj_step(model, d)
        if settle_s > 0:
            mujoco.mj_forward(model, d)
            initial_qpos, initial_qvel, initial_act = d.qpos.copy(), d.qvel.copy(), d.act.copy()
            mujoco.mj_getState(model, d, initial_state, state_spec)
            initial_time = float(d.time)
            eq_start = d.eq_active.astype(np.int8).copy()
        for t in range(T):
            n0, n1 = round(max(0, t - 1) / (fps * dt)), round(t / (fps * dt))
            n = max(0, n1 - n0)
            for k in range(n):
                command(t, (k + 1) / n)
                steps_eq_before.append(d.eq_active.astype(np.int8).copy())
                steps_eq_data.append(model.eq_data.copy())
                mujoco.mj_step(model, d)
                mechanism.post_step(d, t)
                mujoco.mj_forward(model, d)
                if balancer is not None:
                    balancer.step(d)
                diagnose(t)
                steps_q.append(d.qpos.copy()); steps_v.append(d.qvel.copy()); steps_c.append(d.ctrl.copy())
                steps_t.append(float(d.time) - initial_time); steps_eq.append(d.eq_active.astype(np.int8).copy()); steps_act.append(d.act.copy())
                if (not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all()
                        or int(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)
                        or (model.nv and np.max(np.abs(d.qvel)) > 25.)):
                    unstable = {"frame": t, "time_s": float(d.time), "max_qvel": float(np.max(np.abs(d.qvel))),
                                "bad_qacc_reset": bool(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)}
                    break
            mujoco.mj_forward(model, d)
            Q[t], V[t], C[t], ACT[t], eqhist[t], times[t] = d.qpos, d.qvel, d.ctrl, d.act, d.eq_active, float(d.time) - initial_time
            target.qpos[:] = Qref[t]; mujoco.mj_kinematics(model, target)
            for si, (s, pt) in enumerate(parts.items()):
                h = pt["ee_body"]
                pe = np.linalg.norm(target.xpos[h] - d.xpos[h])
                re = Rotation.from_matrix(target.xmat[h].reshape(3, 3).T @ d.xmat[h].reshape(3, 3)).magnitude()
                track[t] = np.maximum(track[t], [pe, np.degrees(re)])
                arm_error[t, si] = np.max(np.abs(Qref[t, pt["arm_qadr"]] - d.qpos[pt["arm_qadr"]]))
            contacts.append(diagnose(t)); completed = t + 1
            if unstable:
                # Never fabricate repeated states after an unstable integration.
                break
            if progress and t % max(1, T // 10) == 0:
                progress(t, T)
        final_state = np.zeros(mujoco.mj_stateSize(model, state_spec))
        mujoco.mj_getState(model, d, final_state, state_spec)
        mechanism_final = mechanism.snapshot()
    finally:
        saved_latches = mechanism.close()
    sl = slice(0, completed)
    return {"qpos": Q[sl], "qvel": V[sl], "ctrl": C[sl], "time_s": times[sl],
            "act": ACT[sl], "initial_act": initial_act, "initial_state": initial_state, "final_state": final_state,
            "state_spec": int(state_spec), "initial_time_s": initial_time, "eq_data_start": eq_data_start,
            "initial_qpos": initial_qpos, "initial_qvel": initial_qvel, "final_qpos": Q[completed - 1].copy(),
            "final_qvel": V[completed - 1].copy(), "unstable": unstable, "frames_simulated": completed,
            "track_pos_m": track[sl, 0], "track_rot_deg": track[sl, 1], "hand_pen_m": pen[sl], "object_pen_m": objpen[sl],
            "wrist_force_n": wrench[sl, 0], "wrist_torque_nm": wrench[sl, 1], "hand_object_force_n": forces[sl],
            "arm_tracking_rad": arm_error[sl], "contacts": contacts, "gains": gains, "grip_balance": None,
            "eq_inactive": sorted(mechanism.inactive), "released": mechanism.released, "latched": saved_latches, "latch_events": mechanism.events,
            "mechanism_initial": mechanism_initial, "mechanism_final": mechanism_final,
            "eq_active": eqhist[sl], "eq_active_start": eq_start, "eq_names": E.equality_names(model),
            "cuts": [x for x in mechanism.released if "cut" in x], "robot_schema": SCHEMA,
            "substep_qpos": np.asarray(steps_q).reshape(-1, model.nq),
            "substep_qvel": np.asarray(steps_v).reshape(-1, model.nv),
            "substep_ctrl": np.asarray(steps_c).reshape(-1, model.nu), "substep_time_s": np.asarray(steps_t),
            "substep_eq_active": np.asarray(steps_eq, dtype=np.int8).reshape(len(steps_eq), model.neq),
            "substep_eq_active_before": np.asarray(steps_eq_before, dtype=np.int8).reshape(len(steps_eq), model.neq),
            "substep_eq_data": np.asarray(steps_eq_data).reshape(len(steps_eq), model.neq, 11),
            "substep_act": np.asarray(steps_act).reshape(len(steps_act), model.na)}


def episode_arrays(model, diag, reference=None):
    """Authoritative, self-describing runtime arrays for an NPZ artifact."""
    from tool_library.common import episode as E
    runtime = model if is_rigid(model) else compile_rigid(model).model
    if not is_rigid(runtime) or "robot_schema" not in diag:
        raise ValueError("an unverified floating rollout cannot be packaged as a robot episode")
    out = {"qpos": diag.get("runtime_qpos", diag["qpos"]),
           "qvel": diag.get("runtime_qvel", diag["qvel"]),
           "ctrl": diag.get("runtime_ctrl", diag["ctrl"]),
           "robot_schema": np.array(SCHEMA), "joint_names": np.array(E.joint_names(runtime)),
           "actuator_names": np.array([runtime.actuator(i).name for i in range(runtime.nu)])}
    xml = diag.get("runtime_scene_xml") or _SPECS[runtime].to_xml()
    out["scene_sha256"] = np.array(hashlib.sha256(xml.encode()).hexdigest())
    import json
    for field in ("mechanism_initial", "mechanism_final"):
        if field in diag:
            out[field] = np.array(json.dumps(diag[field], sort_keys=True))
    for name in ("time_s", "initial_qpos", "initial_qvel", "initial_act", "act", "initial_state", "final_state", "state_spec",
                 "initial_time_s", "eq_data_start", "substep_act", "substep_eq_active_before", "substep_eq_data", "track_pos_m", "track_rot_deg", "hand_pen_m",
                 "object_pen_m", "wrist_force_n", "wrist_torque_nm", "hand_object_force_n", "arm_tracking_rad",
                 "eq_active", "eq_active_start", "eq_names", "substep_qpos", "substep_qvel", "substep_ctrl",
                 "substep_time_s", "substep_eq_active"):
        if name in diag:
            out[name] = np.asarray(diag[name])
    if reference is not None:
        out["reference"] = (np.asarray(reference) if is_rigid(model) else compile_rigid(model).to_runtime_qpos(reference))[:len(out["qpos"])]
    return out

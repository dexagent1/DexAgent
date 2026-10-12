"""A rendered trajectory must be exact samples of a complete dynamics recording."""
from __future__ import annotations

TOOL = {
    "name": "physics_recording_check", "title": "Physics recording check", "type": "verifier",
    "category": "Episode Audits", "summary": "Require named rigid robot states, velocities, controls and every integrator step; rendered frames must exactly match that recording.",
    "description": "Rejects missing or nonfinite recording arrays, changed joint/actuator layouts, skipped integrator steps, out-of-range controls and frames edited after physics. Independently replays every recorded motor control from the complete initial simulator state and compares every integrator state; no object or arm coordinates are fitted during replay.",
    "inputs": [{"name": "scene", "type": "str", "description": "Actual execution MJCF."},
               {"name": "trajectory", "type": "str", "description": "Rigid runtime recording NPZ."},
               {"name": "scene_recon_json", "type": "str | None", "description": "Trusted Stage 2 mechanism rules. Without it dynamic equalities cannot pass."}],
    "outputs": [{"name": "report", "type": "dict", "description": "ok, issues, frames, substeps, duration_s."}],
    "requirements": ["mujoco>=3.2", "numpy"], "simulators": ["MuJoCo"], "tags": ["recording", "physics", "provenance"],
    "status": "tested", "entrypoint": "physics_recording_check",
}

import json
import mujoco
import numpy as np


def physics_recording_check(scene, trajectory, scene_recon_json=None):
    from tool_library.common.episode import joint_names
    from tool_library.common.rigid_robot import SCHEMA
    from tool_library.common.evidence import file_digest
    model = mujoco.MjModel.from_xml_path(str(scene))
    required = {"qpos", "qvel", "ctrl", "time_s", "initial_qpos", "initial_qvel", "joint_names", "actuator_names",
                "substep_qpos", "substep_qvel", "substep_ctrl", "substep_time_s", "robot_schema",
                "initial_state", "final_state", "state_spec", "scene_sha256"}
    if model.neq:
        required |= {"eq_names", "eq_active", "eq_active_start", "substep_eq_active", "substep_eq_active_before", "substep_eq_data"}
        if scene_recon_json is not None:
            required |= {"eq_data_start", "mechanism_initial", "mechanism_final"}
    if model.na:
        required |= {"act", "initial_act", "substep_act"}
    with np.load(str(trajectory), allow_pickle=False) as archive:
        missing = sorted(required - set(archive.files))
        if missing:
            return {"ok": False, "issues": [f"missing physical recording evidence: {', '.join(missing)}"]}
        z = {key: archive[key] for key in required}
    issues = []
    if str(z["scene_sha256"].item()) != file_digest(scene):
        issues.append("scene_sha256 does not identify the execution scene supplied to the verifier")
    state_spec = int(z["state_spec"].item())
    if state_spec != int(mujoco.mjtState.mjSTATE_INTEGRATION):
        return {"ok": False, "issues": ["state_spec must record complete mjSTATE_INTEGRATION state, including solver warmstart and external forces"]}
    state_size = mujoco.mj_stateSize(model, state_spec)
    if str(z["robot_schema"].item()) != SCHEMA:
        issues.append("the trajectory does not identify the rigid runtime schema")
    if list(z["joint_names"].astype(str)) != joint_names(model):
        issues.append("joint_names do not match the execution model's qpos columns")
    if list(z["actuator_names"].astype(str)) != [model.actuator(i).name for i in range(model.nu)]:
        issues.append("actuator_names do not match the execution model's control columns")
    if model.neq and list(z["eq_names"].astype(str)) != [model.equality(i).name for i in range(model.neq)]:
        issues.append("eq_names do not match the execution model's equality order")
    times, subtime = np.asarray(z["time_s"], float), np.asarray(z["substep_time_s"], float)
    T, N = len(times), len(subtime)
    shapes = {"qpos": (T, model.nq), "qvel": (T, model.nv), "ctrl": (T, model.nu), "time_s": (T,),
              "initial_qpos": (model.nq,), "initial_qvel": (model.nv,), "substep_qpos": (N, model.nq),
              "substep_qvel": (N, model.nv), "substep_ctrl": (N, model.nu), "substep_time_s": (N,),
              "initial_state": (state_size,), "final_state": (state_size,)}
    if model.neq:
        shapes.update(eq_active=(T, model.neq), eq_active_start=(model.neq,), substep_eq_active=(N, model.neq),
                      substep_eq_active_before=(N, model.neq), substep_eq_data=(N, model.neq, 11))
        if scene_recon_json is not None:
            shapes["eq_data_start"] = (model.neq, 11)
    if model.na:
        shapes.update(act=(T, model.na), initial_act=(model.na,), substep_act=(N, model.na))
    for name, shape in shapes.items():
        value = np.asarray(z[name], float)
        if value.shape != shape or not np.isfinite(value).all():
            issues.append(f"{name}: expected finite array of shape {shape}, got {value.shape}")
    if issues or T < 2 or N < 1:
        if T < 2 or N < 1:
            issues.append("a recording needs an initial frame, at least one subsequent frame and all intervening physics steps")
        return {"ok": False, "issues": issues, "frames": T, "substeps": N}
    if model.neq:
        for key in ("eq_active", "eq_active_start", "substep_eq_active", "substep_eq_active_before"):
            if not np.isin(z[key], [0, 1]).all():
                issues.append(f"{key} must contain only boolean equality activations")
        # Recorded constraint commands are not an authority to glue an object to
        # a hand or to release a mechanism. Dynamic constraints require a trusted
        # contact-based mechanism replay, not merely matching submitted arrays.
        if scene_recon_json is None:
            for key in ("eq_active_start", "substep_eq_active", "substep_eq_active_before"):
                if not np.all(z[key] == model.eq_active0):
                    issues.append(f"{key}: dynamic equality changes require independently recomputed mechanism evidence")
            if not np.allclose(z["substep_eq_data"], model.eq_data[None], rtol=0, atol=1e-12):
                issues.append("dynamic equality data require independently recomputed mechanism evidence")
    dt = float(model.opt.timestep)
    if not np.allclose(subtime, np.arange(1, N + 1) * dt, rtol=0, atol=1e-8):
        issues.append("substep_time_s must cover every integrator step from timestep through the end, without gaps or resets")
    if abs(times[0]) > 1e-12 or np.any(np.diff(times) <= 0) or abs(times[-1] - subtime[-1]) > 1e-8:
        issues.append("frame times must start at zero, increase strictly and cover the complete dynamics recording")
    indices = np.rint(times / dt).astype(int)
    if np.any(indices < 0) or np.any(indices > N) or not np.allclose(times, indices * dt, rtol=0, atol=1e-8):
        issues.append("each frame must identify an actual integrator state, not an interpolated pose")
    else:
        state_arrays = [("qpos", "initial_qpos", "substep_qpos"), ("qvel", "initial_qvel", "substep_qvel")]
        if model.neq:
            state_arrays.append(("eq_active", "eq_active_start", "substep_eq_active"))
        if model.na:
            state_arrays.append(("act", "initial_act", "substep_act"))
        for frame_key, initial_key, dense_key in state_arrays:
            raw = np.concatenate([z[initial_key][None], z[dense_key]], axis=0)
            if not np.allclose(z[frame_key], raw[indices], rtol=0, atol=1e-12):
                issues.append(f"{frame_key} frames differ from the recorded physical states (post-hoc edits or resets)")
        if not np.allclose(z["ctrl"][1:], z["substep_ctrl"][indices[1:] - 1], rtol=0, atol=1e-12):
            issues.append("frame controls differ from the controls applied at their recorded physics steps")
    limited = np.asarray(model.actuator_ctrllimited, bool)
    if limited.any():
        control = np.asarray(z["substep_ctrl"])[:, limited]
        ranges = model.actuator_ctrlrange[limited]
        if np.any(control < ranges[:, 0] - 1e-9) or np.any(control > ranges[:, 1] + 1e-9):
            issues.append("recorded controls exceed the execution model's actuator limits")
    initial, final = mujoco.MjData(model), mujoco.MjData(model)
    mujoco.mj_setState(model, initial, z["initial_state"], state_spec)
    mujoco.mj_setState(model, final, z["final_state"], state_spec)
    for name, data, frame in (("initial", initial, 0), ("final", final, -1)):
        if np.any(data.qfrc_applied) or np.any(data.xfrc_applied):
            issues.append(f"{name} simulator state contains unaccounted external forces")
        for array in ("qpos", "qvel"):
            if not np.allclose(getattr(data, array), z[array][frame], rtol=0, atol=1e-12):
                issues.append(f"{name} full simulator state disagrees with its recorded {array}")
        if model.neq and not np.array_equal(data.eq_active, z["eq_active"][frame]):
            issues.append(f"{name} full simulator state disagrees with equality activation")
        if model.na and not np.allclose(data.act, z["act"][frame], rtol=0, atol=1e-12):
            issues.append(f"{name} full simulator state disagrees with actuator activation")
    if abs((final.time - initial.time) - subtime[-1]) > 1e-8:
        issues.append("complete simulator state timestamps do not match the recorded elapsed time")
    mechanism = None
    if model.neq and scene_recon_json is not None and not issues:
        from tool_library.common.rigid_robot import MechanismReplay
        from tool_library.skills.execute_subgoal_plan import thread_rules_from_recon, cut_rules_from_recon, latch_rules_from_recon
        rules = (thread_rules_from_recon(scene_recon_json) + cut_rules_from_recon(scene_recon_json)
                 + latch_rules_from_recon(scene_recon_json))
        initial.eq_active[:] = model.eq_active0
        mujoco.mj_forward(model, initial)
        mechanism = MechanismReplay(model, initial, rules=rules)
        if (not np.array_equal(initial.eq_active, z["eq_active_start"])
                or not np.allclose(model.eq_data, z["eq_data_start"], rtol=0, atol=1e-12)):
            issues.append("initial equality state/data do not match independently initialized Stage 2 mechanisms")
        if mechanism.snapshot() != json.loads(str(z["mechanism_initial"].item())):
            issues.append("initial mechanism state does not match independently initialized Stage 2 mechanisms")
    replayed, q_error, v_error = 0, 0., 0.
    if not issues:
        # Reproduce the simulator inputs. No mj_forward, pose reset, refit or
        # per-frame state injection is allowed in this independent replay.
        for i in range(N):
            initial.ctrl[:] = z["substep_ctrl"][i]
            if model.neq:
                if (not np.array_equal(initial.eq_active, z["substep_eq_active_before"][i])
                        or not np.allclose(model.eq_data, z["substep_eq_data"][i], rtol=0, atol=1e-12)):
                    issues.append(f"recorded equality commands at substep {i} differ from independently recomputed mechanisms")
                    break
            mujoco.mj_step(model, initial)
            if mechanism is not None:
                mechanism.post_step(initial, frame=i)
            dq = float(np.max(np.abs(initial.qpos - z["substep_qpos"][i]), initial=0.))
            dv = float(np.max(np.abs(initial.qvel - z["substep_qvel"][i]), initial=0.))
            replayed += 1
            q_error, v_error = max(q_error, dq), max(v_error, dv)
            if not np.isfinite(dq + dv) or dq > 1e-8 or dv > 1e-7:
                issues.append(f"independent control replay diverges at substep {i} ({subtime[i]:.6f} s): "
                              f"max qpos error {dq:.9g}, max qvel error {dv:.9g}; recorded states are not produced by the recorded controls")
                break
            if model.na and not np.allclose(initial.act, z["substep_act"][i], rtol=0, atol=1e-10):
                issues.append(f"actuator activation differs in independent replay at substep {i}")
                break
            if model.neq:
                if not np.array_equal(initial.eq_active, z["substep_eq_active"][i]):
                    issues.append(f"recorded equality activation at substep {i} differs from independently recomputed mechanisms")
                    break
        if not issues:
            replay_final = np.zeros(state_size)
            mujoco.mj_getState(model, initial, replay_final, state_spec)
            if not np.allclose(replay_final, z["final_state"], rtol=0, atol=1e-7):
                issues.append("independent replay disagrees with the final complete simulator state, including hidden solver or actuator state")
            if mechanism is not None and mechanism.snapshot() != json.loads(str(z["mechanism_final"].item())):
                issues.append("independently replayed final mechanism state differs from its recording")
    if mechanism is not None:
        mechanism.close()
    return {"ok": not issues, "issues": issues, "frames": T, "substeps": N, "duration_s": float(subtime[-1]),
            "replayed_substeps": replayed, "max_replay_qpos_error": q_error, "max_replay_qvel_error": v_error}

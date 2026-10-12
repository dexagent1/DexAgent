"""Generic rigid pick/carry/place: gentle contact and exact physical replay.

This is a developer regression, not an independent video-task evaluation.
The seed is a recorded result of the general grasp generator on this fixture.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root: runs as a script too
from tool_library.skills.stage_on_bench import stage_on_bench
from tool_library.skills.hand_scene_from_recon import hand_scene_from_recon
from tool_library.skills.execute_subgoal_plan import execute_subgoal_plan
from tool_library.verifiers.contact_impulse_spike_audit import contact_impulse_spike_audit
from tool_library.verifiers.physics_recording_check import physics_recording_check
from tool_library.verifiers.rigid_arm_hand_check import rigid_arm_hand_check
from tool_library.verifiers.swept_collision_audit import swept_collision_audit
from tool_library.verifiers.object_visual_penetration_audit import object_visual_penetration_audit
from tool_library.verifiers.held_object_slip_audit import held_object_slip_audit


def main():
    root = Path(__file__).resolve().parents[2]
    ws = Path(tempfile.mkdtemp(prefix="dexagent_rigid_pnp_"))
    print(f"Regression artifacts: {ws}", flush=True)
    asset = ws / "box.xml"
    asset.write_text('<mujoco model="box"><worldbody><body name="box"><freejoint/>'
                     '<geom name="box_body" type="box" size="0.024 0.019 0.028" pos="0 0 0.028" mass="0.1" rgba="0.8 0.45 0.2 1"/>'
                     '</body></worldbody></mujoco>')
    scene = ws / "scene.xml"
    stage_on_bench(str(root / "setup/scene.xml"), assets=[dict(name="box", xml=str(asset), x=.12, y=.38, yaw_deg=20.)],
                   out_xml=str(scene), rig={"origin": [0, 0, 0], "R": np.eye(3).tolist()})
    recon = ws / "scene_recon.json"
    recon.write_text(json.dumps(dict(scene_xml="scene.xml", objects=[dict(id="box", type="rigid", asset_xml="box.xml",
                          root_body="box_box", bodies=["box_box"], pose=dict(x=.12, y=.38, z=0., yaw_deg=20.))])))
    hs = ws / "hand_scene.json"
    hand_scene_from_recon(str(recon), "right", "box", out_json=str(hs))
    pose = root / "tool_library/tests/fixtures/grasp_robust/box_bench_right_generated.json"
    plan = dict(scene_recon_json=str(recon), subgoals=[
        dict(index=1, motion="grasp_lift", side="right", object="box_box", pose=str(pose), hand_scene=str(hs), lift_m=.08),
        dict(index=2, motion="carry", side="right", object="box_box", target_pos=[.18, .42, .08], duration_s=2.),
        dict(index=3, motion="place", side="right", object="box_box", target_pos=[.18, .42, 0.], settle_s=1.),
    ])
    out = ws / "execution"
    result = execute_subgoal_plan(str(scene), plan, str(out))
    (ws / "result.json").write_text(json.dumps(result, indent=1))
    assert result["ok"], result.get("first_failure")
    physical_scene, trajectory = str(out / "scene_physics.xml"), str(out / "trajectory.npz")
    reports = {}
    for verifier in (rigid_arm_hand_check, physics_recording_check, swept_collision_audit,
                     object_visual_penetration_audit, contact_impulse_spike_audit, held_object_slip_audit):
        kwargs = dict(scene=physical_scene, trajectory=trajectory)
        if verifier is physics_recording_check:
            kwargs["scene_recon_json"] = str(recon)
        if verifier is held_object_slip_audit:
            kwargs["words"] = ["lift", "carry", "transport", "pour", "tilt", "set down"]
        report = verifier(**kwargs)
        reports[verifier.__name__] = report
        (ws / "audits.json").write_text(json.dumps(reports, indent=1, default=str))
        print(verifier.__name__, report["ok"], report.get("issues", []), flush=True)
        assert report["ok"], report
    # Same measured episode with a large logged contact impulse must reject.
    with np.load(trajectory, allow_pickle=False) as z:
        bad = {k: z[k].copy() for k in z.files}
    bad["hand_object_force_n"][len(bad["qpos"]) // 2, 1] = 500.
    altered = ws / "injected_force_spike.npz"
    np.savez_compressed(altered, **bad)
    assert not contact_impulse_spike_audit(physical_scene, str(altered))["ok"]
    print("ALL PASS", flush=True)


if __name__ == "__main__":
    main()

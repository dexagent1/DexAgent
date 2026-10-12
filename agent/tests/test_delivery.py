"""A candidate cannot become an accepted result by supplying a success flag."""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.artifacts import accepted_delivery, bundle_scene, bundle_recording, publish_stage3, verify_delivery
from agent.external import atomic_json
from tool_library.common.evidence import file_digest


def test_packaged_model_identity():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        # Two distinct assets with the same basename must not overwrite each other.
        for name, height in (("one", 1.), ("two", 2.)):
            (p / name).mkdir()
            # Simple tetrahedra in OBJ, enough for an independent compile check.
            (p / name / "part.obj").write_text(f"v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 {height}\nf 1 3 2\nf 1 2 4\nf 2 3 4\nf 3 1 4\n")
        source = p / "scene.xml"
        source.write_text('<mujoco><asset><mesh name="a" file="one/part.obj"/><mesh name="b" file="two/part.obj"/></asset>'
                          '<worldbody><geom type="mesh" mesh="a"/><geom type="mesh" mesh="b" pos="3 0 0"/></worldbody></mujoco>')
        digest = bundle_scene(source, p / "bundle" / "scene.xml")
        assert len(digest) == 64 and len(list((p / "bundle" / "assets").iterdir())) == 2


def test_rejected_and_tampered_artifacts():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        try:
            publish_stage3(p, p / "public", {"ok": False}, p / "semantic", p / "recon")
        except ValueError:
            pass
        else:
            raise AssertionError("failed candidate was published")
        assert not (p / "public" / "accepted.json").exists()
        accepted = p / "verified" / "example"
        accepted.mkdir(parents=True)
        (accepted / "trajectory.npz").write_bytes(b"verified states")
        atomic_json(accepted / "delivery.json", {"schema": "dexagent.delivery.v1", "passed": True,
                    "sha256": {"trajectory.npz": file_digest(accepted / "trajectory.npz")}})
        atomic_json(p / "accepted.json", {"directory": str(accepted),
                    "certificate_sha256": file_digest(accepted / "delivery.json")})
        atomic_json(p / "run.json", {"finished": True, "verdict": {"ok": True, "reports": {
            "verified_delivery": {"ok": True, "directory": str(accepted)}}}})
        assert accepted_delivery(p) == accepted
        (accepted / "trajectory.npz").write_bytes(b"bad states")
        assert not verify_delivery(accepted)["ok"]
        try:
            accepted_delivery(p)
        except ValueError:
            pass
        else:
            raise AssertionError("changed trajectory remained accepted")


def test_recording_relocation_preserves_execution():
    import numpy as np
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        source = p / "original.xml"
        source.write_text('<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>')
        packaged = p / "portable" / "scene.xml"
        bundle_scene(source, packaged)
        trajectory = p / "recording.npz"
        values = np.arange(21, dtype=float).reshape(3, 7)
        np.savez(trajectory, qpos=values, scene_sha256=np.asarray(file_digest(source)),
                 initial_state=np.arange(18, dtype=float), mechanism_initial=np.asarray('{}'))
        destination = p / "portable" / "trajectory.npz"
        bundle_recording(trajectory, source, packaged, destination)
        with np.load(trajectory) as before, np.load(destination) as after:
            assert set(before.files) == set(after.files)
            for key in before.files:
                if key != "scene_sha256":
                    assert np.array_equal(before[key], after[key]), key
            assert str(after['scene_sha256'].item()) == file_digest(packaged)
        changed = p / "changed.xml"
        changed.write_text(source.read_text().replace('mass="1"', 'mass="2"'))
        try:
            bundle_recording(trajectory, source, changed, destination)
        except ValueError:
            pass
        else:
            raise AssertionError("a recording was rebound to changed dynamics")
        np.savez(trajectory, qpos=values, scene_sha256=np.asarray('forged'))
        try:
            bundle_recording(trajectory, source, packaged, destination)
        except ValueError:
            pass
        else:
            raise AssertionError("unbound original recording was accepted")


if __name__ == "__main__":
    for test in (test_packaged_model_identity, test_rejected_and_tampered_artifacts,
                 test_recording_relocation_preserves_execution):
        test()
        print(f"[PASS] {test.__name__}")

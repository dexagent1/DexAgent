"""Trusted inputs include external MJCF dependencies, not only nearby files."""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent.integrity import InputContract


def test_external_scene_dependency_is_frozen():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        lib = p / "tool_library"
        lib.mkdir()
        (lib / "index.json").write_text(json.dumps({"tools": []}))
        recon = p / "stage2"
        recon.mkdir()
        asset = p / "external_object.xml"
        asset.write_text('<mujoco><worldbody><body name="object"><freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>')
        (recon / "scene.xml").write_text('<mujoco><include file="../external_object.xml"/></mujoco>')
        (recon / "scene_recon.json").write_text(json.dumps({"scene_xml": "scene.xml"}))
        contract = InputContract(lib, [recon])
        assert contract.check()["ok"]
        assert str(asset) not in contract.hashes
        asset.write_text(asset.read_text().replace('mass="1"', 'mass="2"'))
        rejected = contract.check()
        assert not rejected["ok"] and any('compiled scene' in x for x in rejected["issues"])


if __name__ == "__main__":
    test_external_scene_dependency_is_frozen()
    print('[PASS] external scene dependency cannot change a trusted reconstruction')

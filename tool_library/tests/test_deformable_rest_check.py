"""Real CPU cable physics and explicit rejection of unsupported proxies."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tool_library.skills.construct_rope import construct_rope
from tool_library.verifiers.deformable_rest_check import deformable_rest_check
from tool_library.verifiers.scene_recon_manifest_check import scene_recon_manifest_check


class CableRestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.built = construct_rope(self.root, name='cord', length=.3, segments=20,
                                    radius=.0035, pos=[-.15, 0, .0035])
        self.obj = dict(id='cord', type='deformable', skill='construct_rope',
                        root_body=self.built['bodies']['first'],
                        bodies=[self.built['bodies']['first']] + self.built['bodies']['segments'] + [self.built['bodies']['last']],
                        simulation_model={'kind': 'mujoco_cable'})
        self.recon = {'objects': [self.obj]}

    def test_supported_cable_really_settles(self):
        r = deformable_rest_check(self.built['xml'], self.recon)
        self.assertTrue(r['ok'], r)
        self.assertGreater(r['objects']['cord']['links'], 5)
        self.assertFalse(r['isaac_verified'])

    def test_floating_cable_rejected(self):
        built = construct_rope(self.root / 'floating', name='cord', length=.3,
                               segments=20, radius=.0035, pos=[-.15, 0, .25])
        r = deformable_rest_check(built['xml'], self.recon)
        self.assertFalse(r['ok'], r)
        self.assertGreater(r['objects']['cord']['mean_drift_m'], .02)

    def test_rigid_proxy_rejected(self):
        p = self.root / 'rigid.xml'
        p.write_text('<mujoco><worldbody><body name="solid" pos="0 0 .01"><freejoint/><geom type="box" size=".05 .01 .01" mass=".1"/></body><geom type="plane" size="1 1 .1"/></worldbody></mujoco>')
        rec = {'objects': [dict(self.obj, root_body='solid', bodies=['solid'])]}
        r = deformable_rest_check(p, rec)
        self.assertFalse(r['ok'], r)
        self.assertTrue(any('no supported flexible cable' in x for x in r['issues']))

    def test_isaac_cord_left_out(self):
        # a cord simulated in Isaac Sim has no MuJoCo body: isaac_deformable_rest_check measures it, not this check
        rec = {'objects': [self.obj, dict(id='rope', type='deformable', simulator='isaac', skill='build_isaac_scene')]}
        r = deformable_rest_check(self.built['xml'], rec)
        self.assertTrue(r['ok'], r)
        self.assertNotIn('rope', r['objects'])

    def test_manifest_backend_scope(self):
        # The bookkeeping gate still requires normal scene/camera/property evidence.
        # Supply it here; the independent rest/topology gates verify the physics.
        cam = self.root / 'camera.json'
        cam.write_text(json.dumps(dict(width=640, height=480, fx=500, fy=500, cx=320, cy=240,
                                       R_cv=[[1,0,0],[0,1,0],[0,0,1]], quat_wxyz=[0,1,0,0], pos=[0,0,1], fovy_deg=51.28)))
        (self.root / 'mask.png').touch(); (self.root / 'compare.png').touch()
        semantic = {'objects': [dict(id='cord', type='deformable', properties=['flexible'])],
                    'subgoals': [dict(index=1, object_refs=['cord'], motion='carry')]}
        (self.root / 'sem.json').write_text(json.dumps(semantic))
        obj = dict(self.obj, asset_xml=self.built['xml'], skill_origin='library',
                   pose={'x': 0, 'y': 0, 'yaw_deg': 0}, mask='mask.png', iou=.9,
                   verification_plan=['deformable_rest_check'], properties_covered={'flexible': 'cable chain'},
                   verifiers={'deformable_rest_check': {'ok': True}})
        man = dict(scene_xml=self.built['xml'], camera_json='camera.json', frame0_compare_png='compare.png',
                   simulator='mujoco', objects=[obj])
        path = self.root / 'recon.json'; path.write_text(json.dumps(man))
        r = scene_recon_manifest_check(path, self.root / 'sem.json')
        self.assertTrue(r['ok'], r)
        for change in ('remove_model', 'claim_isaac'):
            bad = copy.deepcopy(man)
            if change == 'remove_model':
                del bad['objects'][0]['simulation_model']
            else:
                bad['simulator'] = 'mujoco+isaac'
            path.write_text(json.dumps(bad))
            self.assertFalse(scene_recon_manifest_check(path, self.root / 'sem.json')['ok'], change)


if __name__ == '__main__':
    unittest.main()

"""Generic physical counterexamples for forged/stale Stage 2 mobility evidence."""
from pathlib import Path
import copy
import json
import sys
import tempfile
import unittest

import mujoco
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root: runs as a script too
from tool_library.verifiers.scene_recon_manifest_check import scene_recon_manifest_check
from tool_library.verifiers.articulated_joint_travel_test import articulated_joint_travel_test
from tool_library.verifiers.scene_qc import scene_qc


class MechanismEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='dexagent_mechanism_evidence_')
        self.root = Path(self.tmp.name)
        self.xml = '''<mujoco><option timestep="0.002"/><worldbody>
        <geom name="floor" type="plane" size="2 2 .1"/>
        <body name="device" pos="0 0 .025"><geom name="base" type="box" size=".10 .06 .025"/>
        <geom name="hard_stop" type="box" size=".025 .025 .025" pos=".5 0 .05"/>
        <body name="device_slider" pos="0 0 .05"><joint name="device_slide" type="slide" axis="1 0 0"
          range="0 .1" damping="1"/><geom name="slider" type="box" size=".02 .02 .02" mass=".1"/>
        </body></body></worldbody></mujoco>'''
        (self.root/'asset.xml').write_text(self.xml)
        (self.root/'scene.xml').write_text(self.xml)
        (self.root/'sem.json').write_text(json.dumps({'task_success':'','objects':[
            {'id':'device','name':'slider mechanism','type':'articulated','role':'manipulated',
             'properties':['the slider travels 10 cm']}], 'subgoals':[]}))
        (self.root/'camera.json').write_text(json.dumps({'fx':100,'fy':100,'cx':32,'cy':32,
            'pos':[0,-1,1],'quat_wxyz':[1,0,0,0],'width':64,'height':64}))
        Image.new('L',(64,64),255).save(self.root/'mask.png')
        Image.new('RGB',(64,64)).save(self.root/'compare.png')
        self.man = {'scene_semantic':'sem.json','scene_xml':'scene.xml','camera_json':'camera.json',
            'frame0_compare_png':'compare.png','simulator':'mujoco','objects':[
            {'id':'device','type':'articulated','skill':'construct_articulated','skill_origin':'library',
             'asset_xml':'asset.xml','root_body':'device','pose':{'x':0,'y':0,'yaw_deg':0},
             'mask':'mask.png','iou':1,'verification_plan':{'articulated_joint_travel_test':'bounded force travel'},
             'properties_covered':{'the slider travels 10 cm':'articulated_joint_travel_test'},
             'verifiers':{'articulated_joint_travel_test':{'ok':True}}}]}

    def tearDown(self):
        self.tmp.cleanup()

    def run_gate(self):
        p = self.root/'scene_recon.json'; p.write_text(json.dumps(self.man))
        return scene_recon_manifest_check(p)

    def test_real_positive_is_replayed(self):
        result = self.run_gate()
        self.assertTrue(result['ok'],result['issues'])
        row = result['independent_mechanism_trials']['trials'][0]
        self.assertTrue(row['independently_replayed'])
        self.assertEqual(row['verified_claims'],['articulated_joint_travel_test'])
        self.assertGreaterEqual(row['report']['measured']['device_slide']['forward']['fraction'],.8)
        self.assertIn('fixed-base',row['scope'])

    def test_jammed_scene_rejects_forged_pass_despite_static_qc_pass(self):
        jammed = self.xml.replace('pos=".5 0 .05"','pos=".05 0 .05"')
        (self.root/'asset.xml').write_text(jammed); (self.root/'scene.xml').write_text(jammed)
        self.assertTrue(scene_qc(str(self.root/'scene.xml'))['ok'])
        result = self.run_gate()
        self.assertFalse(result['ok'])
        row = result['independent_mechanism_trials']['trials'][0]
        self.assertTrue(row['independently_replayed'])
        self.assertLess(row['report']['measured']['device_slide']['forward']['fraction'],.1)

    def test_genuine_report_from_different_asset_cannot_certify_scene(self):
        genuine = articulated_joint_travel_test(str(self.root/'asset.xml'),joints=['device_slide'])
        self.assertTrue(genuine['ok'])
        self.man['objects'][0]['verifiers']['articulated_joint_travel_test'] = genuine
        (self.root/'scene.xml').write_text(self.xml.replace('pos=".5 0 .05"','pos=".05 0 .05"'))
        result = self.run_gate()
        self.assertFalse(result['ok'])
        self.assertIn('physics mismatch',str(result['issues']))

    def test_changed_asset_dynamics_invalidates_binding(self):
        (self.root/'asset.xml').write_text(self.xml.replace('damping="1"','damping="10000"'))
        result = self.run_gate()
        self.assertFalse(result['ok'])
        self.assertIn('dof_damping',str(result['issues']))

    def test_changed_scene_collision_invalidates_binding(self):
        (self.root/'scene.xml').write_text(self.xml.replace('name="hard_stop"','name="hard_stop" contype="0" conaffinity="0"'))
        result = self.run_gate()
        self.assertFalse(result['ok'])
        self.assertIn('physics mismatch',str(result['issues']))

    def test_changed_internal_collision_override_invalidates_binding(self):
        for contacts in ('<contact><pair geom1="hard_stop" geom2="slider" friction="0 0 0 0 0"/></contact>',
                         '<contact><exclude body1="device" body2="device_slider"/></contact>'):
            with self.subTest(contacts=contacts):
                (self.root/'scene.xml').write_text(self.xml.replace('</mujoco>',contacts+'</mujoco>'))
                result=self.run_gate();self.assertFalse(result['ok'])
                self.assertIn('internal contact',str(result['issues']))

    def test_stored_args_do_not_relax_defaults(self):
        jammed = self.xml.replace('pos=".5 0 .05"','pos=".05 0 .05"')
        (self.root/'asset.xml').write_text(jammed); (self.root/'scene.xml').write_text(jammed)
        self.man['objects'][0]['verifiers']['articulated_joint_travel_test'].update(
            min_fraction=0,force_N=100000,trial_args={'min_fraction':0,'force_N':100000})
        result = self.run_gate()
        self.assertFalse(result['ok'])
        row = result['independent_mechanism_trials']['trials'][0]
        self.assertEqual(row['trial_defaults']['min_fraction'],.8)
        self.assertEqual(row['report']['measured']['device_slide']['cap'],10)

    def test_prefix_global_placement_and_free_root_are_supported(self):
        asset = self.xml.replace('<body name="device" pos="0 0 .025">',
                                  '<body name="device" pos="0 0 .025"><freejoint name="device_free"/>')
        (self.root/'asset.xml').write_text(asset)
        component = mujoco.MjSpec.from_file(str(self.root/'asset.xml'))
        component.delete(component.geom('floor'))
        world = mujoco.MjSpec(); world.option.timestep=.002
        frame = world.worldbody.add_frame(); frame.pos=[.2,.3,.5]; frame.quat=[.9238795325,0,0,.3826834324]
        world.attach(component,prefix='placed_',frame=frame)
        # Unrelated world bodies and controls must not enter the component trial.
        unrelated = world.worldbody.add_body(name='unrelated')
        unrelated.add_joint(name='unrelated_joint',type=mujoco.mjtJoint.mjJNT_HINGE)
        unrelated.add_geom(name='unrelated_geom',type=mujoco.mjtGeom.mjGEOM_SPHERE,size=[.01,0,0])
        world.add_actuator(name='unrelated_motor',target='unrelated_joint',trntype=mujoco.mjtTrn.mjTRN_JOINT)
        world.compile(); (self.root/'scene.xml').write_text(world.to_xml())
        self.man['objects'][0]['root_body']='placed_device'
        result=self.run_gate()
        self.assertTrue(result['ok'],result['issues'])
        self.assertEqual(result['independent_mechanism_trials']['trials'][0]['joints'],['placed_device_slide'])

    def test_specialized_claim_is_explicitly_unsupported(self):
        self.man['objects'][0]['verifiers']={'pump_stroke_test':{'ok':True}}
        result=self.run_gate(); self.assertFalse(result['ok'])
        row=result['independent_mechanism_trials']['trials'][0]
        self.assertEqual(row['unsupported_claims'],['pump_stroke_test'])
        self.assertFalse(row['independently_replayed'])

    def test_spring_not_misinterpreted_as_generic_travel(self):
        spring=self.xml.replace('damping="1"','damping="1" stiffness="20"')
        (self.root/'asset.xml').write_text(spring);(self.root/'scene.xml').write_text(spring)
        result=self.run_gate();self.assertFalse(result['ok'])
        self.assertIn('specialized independent adapter',str(result['issues']))

    def test_changed_scene_is_freshly_replayed(self):
        first=self.run_gate();self.assertTrue(first['ok'],first['issues'])
        jammed=self.xml.replace('pos=".5 0 .05"','pos=".05 0 .05"')
        (self.root/'asset.xml').write_text(jammed);(self.root/'scene.xml').write_text(jammed)
        second=self.run_gate();self.assertFalse(second['ok'])
        a=first['independent_mechanism_trials']['trials'][0]
        b=second['independent_mechanism_trials']['trials'][0]
        self.assertNotEqual(a['scene_model_sha256'],b['scene_model_sha256'])


if __name__=='__main__':
    unittest.main()

"""Contact-response defaults must not change explicit/soft collision physics."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root: runs as a script too
from tool_library.common import contact_defaults as C, hand as H


class ContactDefaultsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def asset(self, name, geom_extra="", joint="<freejoint/>"):
        path = self.root / (name + ".xml")
        path.write_text(f'<mujoco><worldbody><body name="part">{joint}'
                        f'<geom name="solid" type="box" size=".02 .02 .02" {geom_extra}/>'
                        '</body></worldbody></mujoco>')
        return path

    @staticmethod
    def contacts(spec):
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return {tuple(sorted((model.geom(c.geom1).name, model.geom(c.geom2).name))):
                np.r_[c.dim, c.dist, c.includemargin, c.friction, c.solref, c.solreffriction, c.solimp]
                for c in data.contact}

    def test_actual_contact_mixing_and_custom_soft_contact_are_preserved(self):
        # Nonzero margin/gap and unequal priorities exercise engine mixing.
        a = self.asset("a", 'friction=".4 .005 .0003" priority="3" condim="4" margin=".002" gap=".0004"')
        b = self.asset("b", 'friction=".8 .008 .0007" priority="1" condim="6" margin=".001" gap=".0006"')
        soft = self.asset("soft", 'solref=".05 1" solimp=".7 .85 .005"')
        spec = mujoco.MjSpec.from_string('<mujoco/>')
        names = []
        for path, x in ((a, 0.), (b, .039), (soft, -.039)):
            names.extend(H._attach_asset(spec, path, path.stem, (x, 0, .1)))
        before = self.contacts(spec)
        geoms = [(g.solref.copy(), g.solimp.copy()) for g in spec.geoms]
        self.assertEqual(len(C.add_rigid_pairs(spec, names)), 1)
        after = self.contacts(spec)
        self.assertEqual(before.keys(), after.keys())
        pair = ("a_solid", "b_solid")
        np.testing.assert_array_equal(before[("a_solid", "soft_solid")], after[("a_solid", "soft_solid")])
        # Only reference response and impedance may differ.
        np.testing.assert_array_equal(before[pair][:8], after[pair][:8])
        np.testing.assert_array_equal(before[pair][10:12], after[pair][10:12])
        np.testing.assert_allclose(after[pair][8:10], C.SOLREF)
        np.testing.assert_allclose(after[pair][12:], C.SOLIMP)
        for g, (ref, imp) in zip(spec.geoms, geoms):
            np.testing.assert_array_equal(g.solref, ref)
            np.testing.assert_array_equal(g.solimp, imp)
        self.assertEqual(C.add_rigid_pairs(spec, names), [])

    def test_explicit_defaults_including_includes_are_never_inferred_as_unspecified(self):
        a = self.asset("explicit", 'solref=".02 1"')
        self.assertEqual(C.asset_default_names(mujoco.MjSpec.from_file(str(a)), a), [])
        included = self.root / "included.xml"
        included.write_text('<mujoco><default><geom solimp=".9 .95 .001"/></default></mujoco>')
        wrapper = self.asset("wrapper")
        wrapper.write_text(wrapper.read_text().replace('<mujoco>', '<mujoco><include file="included.xml"/>'))
        self.assertEqual(C.asset_default_names(mujoco.MjSpec.from_file(str(wrapper)), wrapper), [])
        jointed = self.asset("jointed", joint='<joint type="ball"/>')
        self.assertEqual(C.asset_default_names(mujoco.MjSpec.from_file(str(jointed)), jointed), [])

    def test_disabled_collision_and_existing_pairs_remain_unchanged(self):
        spec = mujoco.MjSpec.from_string('''<mujoco><worldbody>
          <body name="a"><freejoint/><geom name="a" size=".02"/></body>
          <body name="b"><freejoint/><geom name="b" size=".02" contype="2" conaffinity="2"/></body>
          <body name="c"><freejoint/><geom name="c" size=".02"/></body>
          <body name="d"><freejoint/><geom name="d" size=".02"/></body>
        </worldbody><contact><exclude body1="a" body2="c"/>
          <pair name="custom" geom1="a" geom2="d" solref=".02 1"/>
        </contact></mujoco>''')
        original = spec.compile()
        C.add_rigid_pairs(spec, ['a', 'b', 'c', 'd'])
        got = spec.compile()
        added = {frozenset((p.geomname1, p.geomname2)) for p in spec.pairs if p.name != 'custom'}
        self.assertEqual(added, {frozenset(('c', 'd'))})
        np.testing.assert_array_equal(got.pair_solref[0], original.pair_solref[0])
        mujoco.set_mjcb_contactfilter(lambda *args: 0)
        try:
            self.assertEqual(C.add_rigid_pairs(spec, ['a', 'b', 'c', 'd']), [])
        finally:
            mujoco.set_mjcb_contactfilter(None)

    def test_gentle_rigid_landing_has_less_numerical_penetration(self):
        spec = mujoco.MjSpec.from_string('''<mujoco><option timestep=".002" integrator="implicitfast"/>
          <worldbody><body name="support" pos="0 0 -.02"><geom name="support" type="box" size=".2 .2 .02"/></body>
          <body name="block" pos="0 0 .03"><freejoint/><geom name="block" type="box" size=".02 .02 .02" mass=".025"/></body>
          </worldbody></mujoco>''')
        def drop(s):
            m = s.compile(); d = mujoco.MjData(m); peak = 0.
            for _ in range(500):
                mujoco.mj_step(m, d)
                peak = max(peak, max((-c.dist for c in d.contact), default=0.))
            self.assertTrue(np.isfinite(d.qpos).all())
            self.assertLess(np.linalg.norm(d.qvel), 1e-5)
            return peak
        baseline = drop(spec)
        self.assertGreater(baseline, .003)
        C.add_rigid_pairs(spec, ['block', 'support'])
        self.assertLess(drop(spec), .002)

    def test_legacy_scene_requires_matching_source_geometry_and_manifest(self):
        paths = [self.asset('a'), self.asset('b')]
        spec = mujoco.MjSpec.from_string('<mujoco/>')
        objects = []
        for i, path in enumerate(paths):
            child = mujoco.MjSpec.from_file(str(path))
            frame = spec.worldbody.add_frame(); frame.pos = [i * .1, 0, .1]
            spec.attach(child, prefix=path.stem + '_', frame=frame)
            objects.append(dict(id=path.stem, type='rigid', asset_xml=path.name))
        scene = self.root / 'scene.xml'; scene.write_text(spec.to_xml())
        manifest = self.root / 'scene_recon.json'
        manifest.write_text(json.dumps(dict(scene_xml='scene.xml', objects=objects)))
        self.assertEqual(len(C.apply_scene_defaults(spec, scene)), 1)
        # A changed collider cannot inherit a profile from the old source.
        changed = mujoco.MjSpec.from_file(str(scene))
        changed.geom('a_solid').size = [.03, .02, .02]
        self.assertEqual(C.apply_scene_defaults(changed, scene), [])
        manifest.write_text(json.dumps(dict(scene_xml='other.xml', objects=objects)))
        self.assertEqual(C.apply_scene_defaults(mujoco.MjSpec.from_file(str(scene)), scene), [])

    def test_hand_bench_and_independent_integrity_use_identical_pairs(self):
        from tool_library.skills.stage_on_bench import stage_on_bench
        from tool_library.common import episode as E, rigid_robot as RR
        from tool_library.verifiers.execution_scene_integrity_check import execution_scene_integrity_check
        source = self.asset('box')
        assets = [dict(name=name, xml=str(source), x=x, y=.4, z=.03) for name, x in [('a', -.1), ('b', .1)]]
        scene = self.root / 'scene.xml'
        stage_on_bench(str(Path(__file__).resolve().parents[2] / 'setup/scene.xml'), assets, str(scene),
                       rig=dict(origin=[0, 0, 0], R=np.eye(3).tolist()))
        recon = self.root / 'scene_recon.json'
        recon.write_text(json.dumps(dict(scene_xml='scene.xml', objects=[
            dict(id=a['name'], type='rigid', asset_xml=source.name) for a in assets])))
        proposal, _ = E.build_execution_model(str(scene))
        runtime = RR.compile_rigid(proposal)
        output = self.root / 'runtime.xml'; output.write_text(runtime.xml)
        self.assertTrue(execution_scene_integrity_check(output, recon)['ok'])
        hand_spec = mujoco.MjSpec.from_string('<mujoco/>')
        names = []
        for a in assets:
            names.extend(H._attach_asset(hand_spec, source, a['name'], (a['x'], a['y'], a['z'])))
        C.add_rigid_pairs(hand_spec, names)
        hm = hand_spec.compile()
        pairname = 'rigid_default__a_solid__b_solid'
        for key in ('pair_solref', 'pair_solimp', 'pair_dim', 'pair_friction', 'pair_margin', 'pair_gap'):
            np.testing.assert_array_equal(getattr(hm, key)[hm.pair(pairname).id],
                                          getattr(runtime.model, key)[runtime.model.pair(pairname).id])
        altered = mujoco.MjSpec.from_file(str(output))
        altered.pair(pairname).solref = [.04, 1.]
        output.write_text(altered.to_xml())
        bad = execution_scene_integrity_check(output, recon)
        self.assertFalse(bad['ok'])
        self.assertTrue(any('pair_solref' in issue for issue in bad['issues']), bad)

    def test_mesh_provenance_checks_vertices_and_fixed_descendants(self):
        path = self.root / 'mesh.xml'
        path.write_text('''<mujoco><asset><mesh name="shape" vertex="0 0 0 .02 0 0 0 .02 0 0 0 .02"/></asset>
          <worldbody><body name="part"><freejoint/><body name="fixed" pos=".01 0 0">
          <geom name="solid" type="mesh" mesh="shape"/></body></body></worldbody></mujoco>''')
        source = mujoco.MjSpec.from_file(str(path))
        source_model = source.compile()
        spec = mujoco.MjSpec.from_string('<mujoco/>')
        spec.attach(source, prefix='a_', frame=spec.worldbody.add_frame())
        self.assertTrue(C._same_asset_geom(source_model, spec.compile(), 'solid', 'a_solid', 'a_'))
        moved = spec.copy(); moved.body('a_fixed').pos = [.03, 0, 0]
        self.assertFalse(C._same_asset_geom(source_model, moved.compile(), 'solid', 'a_solid', 'a_'))
        # Keep the mesh label while altering its physical vertices.
        spec.mesh('a_shape').uservert = [0, 0, 0, .03, 0, 0, 0, .02, 0, 0, 0, .02]
        self.assertFalse(C._same_asset_geom(source_model, spec.compile(), 'solid', 'a_solid', 'a_'))


if __name__ == '__main__':
    unittest.main()

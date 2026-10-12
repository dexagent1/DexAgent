"""Composed final prose must measure every relation, modifier and conjunction."""
import sys
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root: runs as a script too
from tool_library.common import task_gates as G
from tool_library.common.final_goal_language import compile_final, knot_layout


class FinalCompositionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def containment(self, height=.025):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <geom type="plane" size="2 2 .1"/>
          <body name="tray"><freejoint/>
            <geom type="box" size=".1 .1 .01" pos="0 0 .01"/>
            <geom type="box" size=".01 .1 .05" pos=".09 0 .07"/>
            <geom type="box" size=".01 .1 .05" pos="-.09 0 .07"/>
            <geom type="box" size=".1 .01 .05" pos="0 .09 .07"/>
            <geom type="box" size=".1 .01 .05" pos="0 -.09 .07"/>
          </body>
          <body name="vessel" pos="0 0 .045"><freejoint/>
            <geom type="cylinder" size=".02 .025" mass=".1"/>
          </body></worldbody></mujoco>'''.replace('size=".02 .025"', f'size=".02 {height}"'))
        sem = {"objects": [{"id": "vessel", "name": "purple vessel", "type": "rigid", "initial_state": "Initially inverted."},
                           {"id": "tray", "name": "tray", "type": "rigid"}],
               "task_success": "The inverted purple vessel is fully contained by and resting stably inside the stationary tray, with neither hand touching it."}
        recon = {"objects": [{"id": name, "root_body": name} for name in ("vessel", "tray")]}
        q = np.repeat(model.qpos0[None], 40, axis=0)
        return model, sem, recon, q

    def report(self, model, sem, recon, q):
        ep = G.Episode(model, qpos=q, fps=30.)
        return G.task_success_report(G.GoalContext(ep, sem, recon), sem)

    def test_compound_containment_measures_orientation_release_and_secondary_stationarity(self):
        model, sem, recon, q = self.containment()
        good = self.report(model, sem, recon, q)
        self.assertTrue(good['ok'], good)
        classes = {p['class'] for p in good['predicates']}
        self.assertTrue({'final_orientation', 'final_full_containment', 'relative_pose', 'release_settle'} <= classes)
        flipped = q.copy(); flipped[5:, 10:14] = [0, 1, 0, 0]
        self.assertFalse(self.report(model, sem, recon, flipped)['ok'])
        moved = q.copy(); moved[5:10, 0] += .04
        self.assertFalse(self.report(model, sem, recon, moved)['ok'])

    def test_full_containment_rejects_protrusion_despite_contact_and_com_inside(self):
        model, sem, recon, q = self.containment(height=.07)
        # A long upright item still touches the bottom and its centre is below
        # the rim, but its top protrudes. Legacy XY-only containment passed this.
        q[:, 9] = .09
        result = self.report(model, sem, recon, q)
        inside = [p for p in result['predicates'] if p['class'] == 'inside_container']
        self.assertTrue(all(p['ok'] for p in inside), inside)
        self.assertFalse(result['ok'])
        self.assertFalse(next(p for p in result['predicates'] if p['class'] == 'final_full_containment')['ok'])

    def test_unknown_conjunct_is_not_waived_by_any_trusted_predicate(self):
        model, sem, recon, q = self.containment()
        sem['task_success'] += ' The vessel emits a melody.'
        sem['task_success_predicates'] = [{"class": "inside_container", "bodies": {"A": "vessel", "B": "tray"}}]
        result = self.report(model, sem, recon, q)
        self.assertFalse(result['ok'])
        self.assertTrue(any(not c['inferred_coverage'] for c in result['coverage']))

    def box(self):
        from tool_library.skills.construct_hinged_lid_box import construct_hinged_lid_box
        built = construct_hinged_lid_box(str(Path(self.tmp.name) / 'box'), name='parcel', lid_init_deg=0., skirt_init_deg=90., include_floor=True)
        model = mujoco.MjModel.from_xml_path(built['xml'])
        sem = {"objects": [{"id": "parcel", "name": "parcel box", "type": "articulated", "parts": ["base", "lid"],
                           "articulation": {"moving_part": "lid", "fixed_part": "base"}}],
               "task_success": "The parcel box is fully closed: its lid is flat and seated over the base, the interior is hidden, and the lid remains closed after the hands withdraw."}
        recon = {"objects": [{"id": "parcel", "root_body": built['base_body'], "bodies": [built['base_body'], built['lid_body'], built['skirt_body']]}]}
        q = np.repeat(model.qpos0[None], 40, axis=0)
        return model, sem, recon, q, built

    def test_attached_closed_lid_is_supported_by_its_hinge_and_open_lid_fails(self):
        model, sem, recon, q, built = self.box()
        result = self.report(model, sem, recon, q)
        self.assertTrue(result['ok'], result)
        # The separate skirt is correctly closed at 90 degrees, not zero.
        self.assertAlmostEqual(q[-1, model.jnt_qposadr[model.joint(built['skirt_joint']).id]], np.pi / 2)
        q[-1, model.jnt_qposadr[model.joint(built['lid_joint']).id]] = np.pi / 4
        self.assertFalse(self.report(model, sem, recon, q)['ok'])

    def test_tiny_or_floating_lid_cannot_hide_container_interior(self):
        for mode in ('tiny', 'floating', 'transparent'):
            model, sem, recon, q, built = self.box()
            geom = model.geom(built['lid_geom']).id
            if mode == 'tiny': model.geom_size[geom, :2] *= .25
            elif mode == 'floating': model.geom_pos[geom, 2] += .02
            else: model.geom_rgba[geom, 3] = .2
            result = self.report(model, sem, recon, q)
            self.assertFalse(result['ok'], (mode, result))
            self.assertTrue(any(p['class'] == 'final_cover' and not p['ok'] for p in result['predicates']), (mode, result))

    def test_unknown_visual_claim_is_not_treated_as_mechanism_closure(self):
        model, sem, recon, q, _ = self.box()
        sem['task_success'] += ' The logo is hidden.'
        self.assertFalse(self.report(model, sem, recon, q)['ok'])

    def test_persistent_joint_state_cannot_be_reached_only_on_last_frame(self):
        model, sem, recon, q, built = self.box()
        j = model.jnt_qposadr[model.joint(built['lid_joint']).id]
        q[-7:-1, j] = .4
        result = self.report(model, sem, recon, q)
        self.assertFalse(result['ok'])
        self.assertTrue(any(p['class'] == 'joint_target' and not p['ok'] for p in result['predicates']), result)

    def test_rope_clauses_compile_without_discarding_location_or_attachment(self):
        model, _, _, q = self.containment()
        sem = {'objects': [{'id': 'cord', 'name': 'thick cord with rigid end fittings', 'type': 'deformable'}],
               'environment': {'support_surface': 'white tabletop'},
               'task_success': 'The cord rests on the white support surface with one persistent overhand knot near its center, both metal-fitted ends remain attached and lie on opposite sides of the knot, and neither hand supports the cord.'}
        gc = G.GoalContext(G.Episode(model, qpos=q), sem, {'objects': [{'id': 'cord', 'root_body': 'vessel'}]})
        predicates, coverage, issues = compile_final(gc, sem['task_success'])
        self.assertFalse(issues, issues)
        self.assertTrue(all(c['inferred_coverage'] for c in coverage))
        self.assertTrue(any(p['class'] == 'knot' and p['params']['expect'] == 'trefoil' and p['persistent'] for p in predicates))
        self.assertTrue(any(p['class'] == 'final_knot_layout' and p.get('central') for p in predicates))
        self.assertTrue(any(p['class'] == 'final_knot_layout' and p.get('opposite') for p in predicates))
        self.assertTrue(any(p['class'] == 'final_cable_integrity' for p in predicates))
        # A generic knot classifier cannot certify the number of unnamed knots.
        _, _, unknown = compile_final(gc, 'The cord has one persistent knot.')
        self.assertTrue(unknown)

    def test_curve_with_no_knot_cannot_pass_layout(self):
        points = np.c_[np.linspace(-1., 1., 25), np.zeros((25, 2))]
        result = knot_layout(points, central=True, opposite=True)
        self.assertFalse(result['ok'], result)

    def test_knot_layout_positive_and_rejects_offcentre_and_same_side_ends(self):
        from tool_library.common import knot as K
        core = K._param_curve('trefoil', n=40)[2:-2]
        def curve(left=-.7):
            return np.vstack([np.linspace([left, 0, -.2], core[0], 8, endpoint=False),
                              core, np.linspace(core[-1], [.7, 0, -.2], 8)[1:]])
        # Added straight tails can leave the bounded diagram certificate
        # inconclusive; polynomial resemblance must not certify a knot core.
        self.assertFalse(knot_layout(curve(), central=True, opposite=True)['ok'])
        off = knot_layout(curve(-12.), central=True, opposite=True)
        self.assertFalse(off['ok'], off)
        self.assertTrue(off['issues'], off)
        ring = K._param_curve('trefoil', n=40)
        k = np.argmax(np.linalg.norm(ring[:, :2] - ring[:, :2].mean(0), axis=1))
        same = np.roll(ring, -int(k), axis=0)[1:-1]
        self.assertTrue(knot_layout(same, central=True)['ok'])
        result = knot_layout(same, opposite=True)
        self.assertEqual(result['type'], 'trefoil', result)
        self.assertFalse(result['ok'], result)

    def test_real_cable_persistent_knot_fails_if_any_terminal_frame_unties(self):
        from tool_library.common import cable as C, knot as K
        from tool_library.skills.construct_rope import construct_rope
        built = construct_rope(str(Path(self.tmp.name) / 'cord'), name='cord', length=1.2, segments=40, radius=.004)
        model = mujoco.MjModel.from_xml_path(built['xml']); data = mujoco.MjData(model)
        C.set_centreline(model, data, built, np.array([[0., 0., .3], [1.2, 0., .3]]))
        straight = data.qpos.copy()
        curve = K._param_curve('trefoil', n=200) * .6
        k = np.argmax(np.linalg.norm(curve[:, :2] - curve[:, :2].mean(0), axis=1))
        curve = np.roll(curve, -int(k), axis=0)[2:-2]; curve[:, 2] += .3
        C.set_centreline(model, data, built, curve)
        q = np.repeat(data.qpos[None], 20, axis=0); q[:5] = straight
        sem = {'objects': [{'id': 'cord', 'name': 'cord', 'type': 'deformable'}],
               'task_success': 'The cord has a persistent overhand knot.'}
        recon = {'objects': [{'id': 'cord', 'root_body': 'cordB_first'}]}
        result = self.report(model, sem, recon, q)
        self.assertTrue(result['ok'], result)
        q[-5] = straight
        result = self.report(model, sem, recon, q)
        self.assertFalse(result['ok'], result)

    def test_cable_attachment_requires_serial_structure_not_endpoint_proximity(self):
        from tool_library.common.final_goal_language import measure_custom
        from tool_library.skills.construct_rope import construct_rope
        built = construct_rope(str(Path(self.tmp.name) / 'attached'), name='cord', length=.8, segments=15, radius=.006)
        model = mujoco.MjModel.from_xml_path(built['xml'])
        sem = {'objects': [{'id': 'cord', 'name': 'cord', 'type': 'deformable'}]}
        recon = {'objects': [{'id': 'cord', 'root_body': 'cordB_first'}]}
        ep = G.Episode(model, qpos=np.repeat(model.qpos0[None], 2, axis=0))
        gc = G.GoalContext(ep, sem, recon)
        result = measure_custom(gc, {'class': 'final_cable_integrity', 'A': 'cord'}, 1)
        self.assertTrue(result['ok'], result)
        # A genuinely separate free terminal, in the same visible position,
        # must fail regardless of its distance from the preceding link.
        disconnected = mujoco.MjModel.from_xml_string('<mujoco><worldbody>'
            '<body name="cordB_first"><freejoint/><geom type="sphere" size=".01"/></body>'
            '<body name="cordB_last" pos=".02 0 0"><freejoint/><geom type="sphere" size=".01"/></body>'
            '</worldbody></mujoco>')
        ep = G.Episode(disconnected, qpos=np.repeat(disconnected.qpos0[None], 2, axis=0))
        gc = G.GoalContext(ep, sem, recon)
        self.assertFalse(measure_custom(gc, {'class': 'final_cable_integrity', 'A': 'cord'}, 1)['ok'])

    def test_fully_open_or_closed_cannot_mean_a_partial_joint_state(self):
        model = mujoco.MjModel.from_xml_string('<mujoco><compiler angle="radian"/><worldbody>'
            '<body name="panel"><joint name="hinge" range="0 1" limited="true"/>'
            '<geom type="box" size=".1 .02 .1"/></body></worldbody></mujoco>')
        sem = {'objects': [{'id': 'panel', 'name': 'panel', 'type': 'articulated'}]}
        recon = {'objects': [{'id': 'panel', 'root_body': 'panel'}]}
        for state, correct, partial in [('closed', 0., .1), ('open', 1., .6)]:
            sem['task_success'] = f'The panel is fully {state}.'
            for angle, expected in [(correct, True), (partial, False)]:
                q = np.full((20, 1), angle)
                report = self.report(model, sem, recon, q)
                self.assertEqual(report['ok'], expected, (state, angle, report))


if __name__ == '__main__':
    unittest.main()

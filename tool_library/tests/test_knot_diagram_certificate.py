"""Named topology requires more than matching Alexander coefficients."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root: runs as a script too
from tool_library.common import knot as K
from tool_library.verifiers.rope_knot_check import rope_knot_check

FIXTURE = json.loads((Path(__file__).parent / 'fixtures/knot_diagrams.json').read_text())['knots']


def alexander_diagram(cr):
    # Exact matrix calculation on independently tabulated signed diagrams.
    with patch.object(K, 'crossings', return_value=cr):
        return K.alexander(np.zeros((2 * len(cr), 3)))


class DiagramCertificateTests(unittest.TestCase):
    def test_tabulated_nontrivial_unit_polynomial_knots_are_not_unknots(self):
        for name in ('11n34', '11n42'):
            with self.subTest(name=name):
                cr = FIXTURE[name]['crossings']
                a = alexander_diagram(cr)
                self.assertEqual(a['coeffs'], [1])
                self.assertEqual(K.classify(a['coeffs']), 'unknot')  # resemblance only
                cert = K.diagram_certificate(cr, a['coeffs'])
                self.assertIsNone(cert['type'])
                self.assertEqual(cert['reduced_crossings'], 11)

    def test_connected_sum_with_same_trefoil_polynomial_is_not_named_trefoil(self):
        # Connected-sum Gauss words concatenate. The nontrivial unit-polynomial
        # summand leaves the trefoil's polynomial unchanged, but changes its type.
        first = FIXTURE['3_1']['crossings']
        second = FIXTURE['11n42']['crossings']
        shift = 2 * len(first)
        cr = first + [dict(c, s_over=c['s_over'] + shift, s_under=c['s_under'] + shift) for c in second]
        a = alexander_diagram(cr)
        self.assertEqual(a['coeffs'], [1, -1, 1])
        self.assertIsNone(K.diagram_certificate(cr, a['coeffs'])['type'])

    def test_closed_gate_uses_diagram_evidence_even_when_all_polynomials_agree(self):
        P = K._param_curve('circle', n=30) + [0., 0., .2]
        for name, expect in (('11n34', 'unknot'), ('11n42', 'unknot')):
            cr = FIXTURE[name]['crossings']; a = alexander_diagram(cr)
            # Isolate the classifier boundary: four identical polynomial claims
            # must not bypass the independent diagram certificate.
            def selected_diagram(points, *args, **kwargs):
                return dict(ok=True, points=points, crossings=cr, view=0,
                            view_matrix=np.eye(3).tolist(), attempts=[])
            with patch.object(K, 'alexander', return_value=a), patch.object(K, 'regular_diagram', side_effect=selected_diagram):
                result = rope_knot_check(points=P, expect=expect)
            self.assertFalse(result['ok'])
            self.assertEqual(result['type'], 'ambiguous')

    def test_small_tabulated_diagrams_have_named_certificates(self):
        for name, kind in (('3_1', 'trefoil'), ('4_1', 'figure-eight')):
            row = FIXTURE[name]
            self.assertEqual(K.diagram_certificate(row['crossings'], row['coeffs'])['type'], kind)
            mirror = [dict(c, s_over=c['s_under'], s_under=c['s_over'], sign=-c['sign']) for c in row['crossings']]
            self.assertEqual(K.diagram_certificate(mirror, row['coeffs'])['type'], kind)

    def test_alexander_arcs_depend_on_visit_order_not_an_epsilon(self):
        row = FIXTURE['3_1']
        clustered = [dict(c, s_over=1. + c['s_over'] * 1e-12,
                          s_under=1. + c['s_under'] * 1e-12) for c in row['crossings']]
        self.assertEqual(alexander_diagram(clustered)['coeffs'], row['coeffs'])
        coincident = [dict(c) for c in row['crossings']]
        coincident[0]['s_over'] = coincident[1]['s_under']
        self.assertEqual(alexander_diagram(coincident)['coeffs'], [0])

    def test_reidemeister_pair_needs_opposite_signs_and_same_over_strand(self):
        pair = [{'s_over': .5, 's_under': 6.5, 'sign': 1},
                {'s_over': 1.5, 's_under': 5.5, 'sign': -1}]
        # Split the irreducible trefoil word between both gaps, avoiding a
        # type-I curl at the cyclic boundary of this extra type-II pair.
        def shifted(s):
            return s + (2 if s < 3 else 4)
        middle = [dict(c, s_over=shifted(c['s_over']), s_under=shifted(c['s_under']))
                  for c in FIXTURE['3_1']['crossings']]
        result = K.diagram_certificate(pair + middle, [1, -1, 1])
        self.assertEqual(result['type'], 'trefoil')
        self.assertEqual(result['removed_crossings'][0]['move'], 'II')
        pair[1]['sign'] = 1
        self.assertIsNone(K.diagram_certificate(pair + middle, [1, -1, 1])['type'])

    def test_regular_view_recovers_real_quantized_trefoils(self):
        for n, grid in ((24, .04), (64, .06)):
            with self.subTest(n=n, grid=grid):
                P = K._param_curve('trefoil', n=n)
                P[:, :2] = np.round(P[:, :2] / grid) * grid
                self.assertEqual(K.embedding_issues(P, closed=True), [])
                self.assertNotEqual(K.alexander(P)['coeffs'], [1, -1, 1])
                selected = K.regular_diagram(P)
                self.assertTrue(selected['ok'])
                self.assertNotEqual(selected['view'], 0)
                polynomial = K.alexander(selected['points'], diagram=selected['crossings'])
                self.assertEqual(polynomial['coeffs'], [1, -1, 1])
                self.assertEqual(K.diagram_certificate(selected['crossings'], polynomial['coeffs'])['type'], 'trefoil')
                apex = int(np.argmax(np.linalg.norm(P[:, :2] - P[:, :2].mean(0), axis=1)))
                opened = np.roll(P, -apex, axis=0)[1:-1] + [0., 0., .2]
                result = rope_knot_check(points=opened, expect='trefoil')
                self.assertTrue(result['ok'], result['issues'])

    def test_regular_views_preserve_known_knots_and_mirrors(self):
        for shape, kind, coeffs in (('trefoil', 'trefoil', [1, -1, 1]),
                                   ('figure8', 'figure-eight', [1, -3, 1]),
                                   ('circle', 'unknot', [1]), ('kink', 'unknot', [1])):
            for hand in (-1, 1):
                with self.subTest(shape=shape, hand=hand):
                    selected = K.regular_diagram(K._param_curve(shape, n=80, hand=hand))
                    self.assertTrue(selected['ok'])
                    polynomial = K.alexander(selected['points'], diagram=selected['crossings'])
                    self.assertEqual(polynomial['coeffs'], coeffs)
                    self.assertEqual(K.diagram_certificate(selected['crossings'], coeffs)['type'], kind)

    def test_projection_checks_reject_vertex_contacts_overlaps_and_triples(self):
        shapes = [
            (np.array([[0,0,0], [0,0,1], [1,0,1], [1,1,0]]), 'zero'),
            (np.array([[0,0,0], [2,0,0], [2,2,1], [1,0,1], [0,2,1]]), 'vertex'),
            (np.array([[-1,0,0], [1,0,0], [1,1,1], [-.5,0,1], [.5,0,1], [-1,1,2]]), 'overlapping'),
            (np.array([[-1,0,0], [1,0,0], [0,-1,1], [0,1,1], [-1,-1,2], [1,1,3]]), 'triple'),
        ]
        for P, expected in shapes:
            with self.subTest(expected=expected):
                self.assertEqual(K.embedding_issues(P, closed=True), [])
                _, issues = K._projection_geometry(P)
                self.assertTrue(any(expected in issue for issue in issues), issues)
                selected = K.regular_diagram(P)
                self.assertTrue(selected['ok'], selected['attempts'])
                self.assertNotEqual(selected['view'], 0)

    def test_exhausted_regular_view_search_never_accepts_fallback_polynomial(self):
        P = K._param_curve('trefoil', n=80)
        apex = int(np.argmax(np.linalg.norm(P[:, :2] - P[:, :2].mean(0), axis=1)))
        opened = np.roll(P, -apex, axis=0)[2:-2] + [0., 0., .2]
        with patch.object(K, '_projection_geometry', return_value=([], ['unresolved test projection'])) as probe:
            result = K.check(opened)
        self.assertEqual(probe.call_count, 4 * 8)
        self.assertTrue(result['polynomial_agreement'])
        self.assertFalse(result['agree'])
        self.assertFalse(result['knot'])
        self.assertEqual(result['type'], 'ambiguous')
        self.assertTrue(all(len(p['attempts']) == 8 for p in result['diagram_projections'].values()))
        self.assertTrue(all(c['type'] is None for c in result['diagram_certificates'].values()))


if __name__ == '__main__':
    unittest.main()

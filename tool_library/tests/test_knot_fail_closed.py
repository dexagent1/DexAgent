"""Real synthetic counterexamples for closure ambiguity and degenerate knots."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root: runs as a script too
from tool_library.common import knot as K
from tool_library.verifiers.rope_knot_check import rope_knot_check


def torus_open(q, start=0):
    u = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    curve = .1 * np.column_stack([(2 + np.cos(q * u)) * np.cos(2 * u),
                                 (2 + np.cos(q * u)) * np.sin(2 * u), np.sin(q * u)])
    return np.roll(curve, -start, axis=0)[2:-2] + [0., 0., .25]


class KnotFailClosedTests(unittest.TestCase):
    def test_closure_translates_with_curve_and_stays_outside_vertical_extent(self):
        P = torus_open(3)
        shift = np.array([1.7, -2.4, 4.1])
        for mode in ('over', 'under'):
            for how in ('radial', 'tangent'):
                with self.subTest(mode=mode, how=how):
                    base = K.closure(P, mode, how)
                    moved = K.closure(P + shift, mode, how)
                    np.testing.assert_allclose(moved, base + shift, atol=1e-12)
                    if mode == 'over':
                        self.assertGreater(moved[-1, 2], (P + shift)[:, 2].max())
                    else:
                        self.assertLess(moved[-1, 2], (P + shift)[:, 2].min())

    def test_invalid_closure_conventions_are_rejected(self):
        for args in ({'mode': 'above'}, {'how': 'guess'}, {'height': 0},
                     {'height': float('nan')}, {'reach': -1}, {'reach': float('inf')}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                K.closure(torus_open(3), **args)

    def test_named_knot_is_preserved_under_large_world_translation(self):
        P = K._param_curve('trefoil', n=80)
        apex = int(np.argmax(np.linalg.norm(P[:, :2] - P[:, :2].mean(0), axis=1)))
        P = np.roll(P, -apex, axis=0)[2:-2]
        for shift in ([0, 0, 0], [2, -3, 4], [-2, 3, -4]):
            with self.subTest(shift=shift):
                self.assertTrue(K.check(P + shift)['trefoil'])

    def test_different_other_polynomials_do_not_count_as_agreement(self):
        # Old implementation called all four closures 'other' and passed,
        # despite degree 6 radial and degree 8 tangent polynomials.
        P = torus_open(9, start=7)
        result = rope_knot_check(points=P, expect="knot")
        self.assertEqual({v["type"] for v in result["closures"].values()}, {"other"})
        self.assertGreater(len({v["poly"] for v in result["closures"].values()}), 1)
        self.assertFalse(result["ok"])
        self.assertEqual(result["type"], "ambiguous")
        self.assertFalse(result["frames"][0]["polynomial_agreement"])
        self.assertTrue(any("polynomials disagree" in x for x in result["issues"]))

    def test_zero_polynomial_is_degenerate_not_an_unknown_knot(self):
        # A repeated vertex and unresolved crossings produce four zero
        # determinants. Previously 'other' + coarse agreement meant PASS.
        P = np.array([[-2, -1, -1], [1, 1, -1], [2, -2, -1], [0, 1, 0],
                      [2, -2, -1], [1, 2, -1], [1, -1, -1], [-2, 1, 1]]) * .1 + [0, 0, .25]
        result = rope_knot_check(points=P, expect="knot")
        self.assertEqual({v["poly"] for v in result["closures"].values()}, {"0"})
        self.assertFalse(result["ok"])
        self.assertTrue(any("zero polynomial" in issue for issue in result["issues"]))
        self.assertFalse(rope_knot_check(points=P, expect="unknot")["ok"])

    def test_consistent_nontrivial_unknown_knot_still_classified(self):
        result = rope_knot_check(points=torus_open(9, start=17), expect="knot")
        self.assertTrue(result["ok"])
        self.assertEqual(result["type"], "other")
        self.assertEqual(len({v["poly"] for v in result["closures"].values()}), 1)

    def test_flattened_self_intersections_are_not_a_verified_unknot(self):
        P = torus_open(3)
        P[:, 2] = .1
        result = rope_knot_check(points=P, expect="unknot")
        self.assertFalse(result["ok"])
        self.assertTrue(any("over/under" in x for x in result["issues"]))

    def test_nonfinite_or_zero_length_polyline_is_rejected(self):
        for invalid in (np.full((5, 3), np.nan), np.zeros((5, 3))):
            with self.assertRaises(ValueError):
                rope_knot_check(points=invalid, expect="unknot")

    def test_projection_invisible_retracing_cannot_certify_a_trefoil(self):
        P = K._param_curve('trefoil', n=100)
        apex = int(np.argmax(np.linalg.norm(P[:, :2] - P[:, :2].mean(0), axis=1)))
        P = np.roll(P, -apex, axis=0)[2:-2] + [0., 0., .2]
        self.assertTrue(rope_knot_check(points=P, expect='trefoil')['ok'])
        # A vertical out-and-back spur is invisible in the XY crossing list,
        # but repeats a vertex and overlaps an entire interval in 3-D.
        malformed = np.vstack([P[:4], P[3] + [0., 0., .02], P[3:]])
        result = rope_knot_check(points=malformed, expect='trefoil')
        self.assertFalse(result['ok'])
        self.assertTrue(any('not embedded' in issue for issue in result['issues']))
        self.assertFalse(rope_knot_check(points=malformed, expect='knot')['ok'])

    def test_collinear_overlaps_and_vertical_intersections_fail_embedding(self):
        retraced = np.array([[0, 0, .2], [.1, 0, .2], [.05, 0, .2], [.15, 0, .2]])
        result = rope_knot_check(points=retraced, expect='unknot')
        self.assertFalse(result['ok'])
        self.assertIn('adjacent_backtracking', {r['kind'] for r in K.embedding_issues(retraced)})
        # No adjacent backtracking: a later horizontal segment pierces the
        # interior of a vertical segment that XY crossing enumeration skips.
        crossing = np.array([[0, 0, .1], [0, 0, .3], [.1, .1, .3], [-.1, 0, .2], [.1, 0, .2]])
        issues = K.embedding_issues(crossing)
        self.assertTrue(any(r['segments'] == [0, 3] for r in issues))
        self.assertFalse(rope_knot_check(points=crossing, expect='unknot')['ok'])

    def test_self_intersecting_artificial_closure_is_not_a_certificate(self):
        P = np.array([[0, 0, .2], [.2, 0, .2], [.2, .2, .2],
                      [0, .2, .2], [0, .1, .2], [.1, .1, .2]])
        self.assertEqual(K.embedding_issues(P), [])
        result = K.check(P)
        self.assertFalse(result['agree'])
        self.assertEqual(result['type'], 'ambiguous')
        self.assertIn('over/radial', result['embedding_issues'])
        self.assertIn('under/radial', result['embedding_issues'])
        self.assertIsNone(result['diagram_certificates']['over/radial']['type'])
        self.assertFalse(rope_knot_check(points=P, expect='unknot')['ok'])

    def test_embedding_allows_separated_crossings_and_regular_corners(self):
        # This deliberately vertical first edge and close but separated later
        # edge are embedded. Rope thickness is checked elsewhere.
        P = np.array([[0, 0, .1], [0, 0, .3], [.1, .1, .3], [-.1, 0, .31], [.1, 0, .31]])
        self.assertEqual(K.embedding_issues(P), [])
        self.assertEqual(K.embedding_issues(K._param_curve('trefoil', n=100), closed=True), [])
        self.assertEqual(K.embedding_issues(K._param_curve('circle', n=80), closed=True), [])

    def test_exact_domain_determinants_match_known_mathematical_knots(self):
        expected = {"trefoil": [1, -1, 1], "figure8": [1, -3, 1], "circle": [1], "kink": [1]}
        for kind, coefficients in expected.items():
            for hand in (-1, 1):
                with self.subTest(kind=kind, hand=hand):
                    self.assertEqual(K.alexander(K._param_curve(kind, n=80, hand=hand))["coeffs"], coefficients)
        # Beyond the name lookup table, the (2,11) torus knot has the exact
        # alternating degree-10 Alexander polynomial; no numeric fitting.
        u = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        P = .1 * np.column_stack([(2 + np.cos(11 * u)) * np.cos(2 * u),
                                 (2 + np.cos(11 * u)) * np.sin(2 * u), np.sin(11 * u)])
        self.assertEqual(K.alexander(P)["coeffs"], [(-1) ** i for i in range(11)])


if __name__ == "__main__":
    unittest.main()

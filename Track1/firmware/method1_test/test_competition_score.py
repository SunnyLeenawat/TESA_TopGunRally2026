import math
import unittest
from competition_score import band_score, evaluate


class ScoreTests(unittest.TestCase):
    def test_supplied_15pf_readings(self):
        r = evaluate([15.67, 15.67, 15.82, 15.71, 15.72], 15)
        self.assertAlmostEqual(r['accuracy_formula_value'], 71.8)
        self.assertAlmostEqual(r['rmse_formula_value'], 72.00972156591084)
        self.assertEqual(r['total_points'], 0)

    def test_supplied_20pf_readings(self):
        r = evaluate([20.20, 20.07, 19.99, 20.05, 19.99], 20)
        self.assertAlmostEqual(r['accuracy_formula_value'], 6)
        self.assertAlmostEqual(r['rmse_formula_value'], math.sqrt(.0476/5)*100)
        self.assertEqual(r['total_points'], 18)

    def test_no_reference_normalization(self):
        for reference in (5, 20):
            r = evaluate([reference+.01]*5, reference)
            self.assertAlmostEqual(r['accuracy_formula_value'], 1)
            self.assertEqual(r['total_points'], 20)

    def test_spread_is_not_bias(self):
        r = evaluate([9.9, 10.1, 10, 10, 10], 10)
        self.assertAlmostEqual(r['accuracy_formula_value'], 0)
        self.assertGreater(r['rmse_formula_value'], 0)

    def test_rubric_bands_and_undefined_boundaries(self):
        self.assertEqual(band_score(10, 10, 55), 10)
        self.assertEqual(band_score(2, 2, 45), 10)
        for upper, last, start in [(10, 55, 15), (2, 45, 5)]:
            for index, boundary in enumerate(range(start, last+1, 5)):
                self.assertIsNone(band_score(boundary, upper, last))
                self.assertEqual(band_score(boundary-.001, upper, last), 9-index)
            self.assertEqual(band_score(last+.001, upper, last), 0)

    def test_invalid_and_incomplete_measurements(self):
        for bad in (0, -1, float('nan'), float('inf')):
            self.assertEqual(evaluate([5, 5, bad, 5, 5], 5)['total_points'], 0)
        with self.assertRaises(ValueError):
            evaluate([5]*4, 5)


if __name__ == '__main__':
    unittest.main()

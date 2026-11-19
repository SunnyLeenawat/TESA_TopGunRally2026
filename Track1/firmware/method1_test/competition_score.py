"""Organizer's supplied formulas, using pF inputs; no reference normalization.

Exact open-interval boundaries in the supplied rubric remain unresolved (None).
"""
import argparse
import json
import math


def band_score(value, first_upper, last_upper):
    if value <= first_upper + 1e-9:
        return 10
    for upper in range(first_upper + (3 if first_upper == 2 else 5), last_upper + 1, 5):
        if math.isclose(value, upper, abs_tol=1e-9, rel_tol=0):
            return None  # Rubric says strictly below/above this boundary.
        if value < upper:
            return 9 - (upper - (5 if first_upper == 2 else 15)) // 5
    return 0


def evaluate(readings, reference_pf):
    if len(readings) != 5:
        raise ValueError('Exactly five button measurements are required.')
    if not math.isfinite(reference_pf) or reference_pf <= 0:
        raise ValueError('Reference must be positive and finite, in pF.')
    if any(not math.isfinite(x) or x <= 0 for x in readings):
        return {'valid': False, 'accuracy_points': 0, 'rmse_points': 0, 'total_points': 0}
    mean = math.fsum(readings)/5
    accuracy = abs(mean-reference_pf)*100
    rmse_pf = math.sqrt(math.fsum((x-reference_pf)**2 for x in readings)/5)
    accuracy_points = band_score(accuracy, 10, 55)
    rmse_points = band_score(rmse_pf*100, 2, 45)
    return {'valid': True, 'mean_pf': mean, 'accuracy_formula_value': accuracy,
            'rmse_pf': rmse_pf, 'rmse_formula_value': rmse_pf*100,
            'accuracy_points': accuracy_points, 'rmse_points': rmse_points,
            'total_points': None if None in (accuracy_points, rmse_points)
                            else accuracy_points+rmse_points}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=float, required=True)
    parser.add_argument('--readings', type=float, nargs=5, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.readings, args.reference), indent=2))

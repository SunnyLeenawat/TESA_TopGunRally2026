"""Freeze calibration on batches 1-10; evaluate batches 11-15 separately.

Use --write-headers to export the same model to both Arduino sketches.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from competition_score import evaluate

ROOT = Path(__file__).resolve().parent
SOURCES = {
    5: '20261118_000627_645444_5pf.csv',
    10: '20261118_000518_104879_10pf.csv',
    15: '20261118_000207_665911_15pf.csv',
    20: '20261118_000355_980613_20pf.csv',
}


def interpolate(values, knots, caps):
    index = np.clip(np.searchsorted(knots, values, side='right')-1, 0, len(knots)-2)
    return caps[index] + (values-knots[index])*(caps[index+1]-caps[index])/(knots[index+1]-knots[index])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-headers', action='store_true')
    args = parser.parse_args()
    records, sources = [], []
    for cap, name in SOURCES.items():
        path = ROOT/'data'/name
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 960 and {int(r['batch']) for r in rows} == set(range(1, 16))
        assert {r['sample'] for r in rows} == {f'{cap}pf'}
        for batch in range(1, 16):
            group = [r for r in rows if int(r['batch']) == batch]
            assert [int(r['reading']) for r in group] == list(range(1, 65))
            assert all(r['status'] == 'ok' and int(r['cpu_mhz']) == 240 for r in group)
            cycles = np.array([int(r['cycles']) for r in group])
            assert np.all(cycles > 0)
            records.append((cap, batch, float(np.median(cycles)/240)))
        sources.append({'file': name, 'nominal_pf': cap,
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'reference_mv_range': [min(int(r['reference_mv']) for r in rows),
                                               max(int(r['reference_mv']) for r in rows)]})
    y, batch, x = np.array(records).T
    train = batch <= 10
    caps = np.array(sorted(SOURCES), dtype=float)
    knots = np.array([np.mean(x[train & (y == cap)]) for cap in caps])
    assert np.all(np.diff(knots) > 0)
    models = {'previous': 2.9690151954173993*x-17.219924456350515,
              'piecewise_linear': interpolate(x, knots, caps)}
    for degree in (1, 2):
        models[f'polynomial_{degree}'] = np.polyval(np.polyfit(x[train], y[train], degree), x)
    comparison = {}
    for name, prediction in models.items():
        error = prediction[~train]-y[~train]
        summaries = []
        for cap in caps:
            mask = (~train) & (y == cap)
            printed = [float(f'{v:.2f}') for v in prediction[mask]]
            summaries.append({'nominal_pf': int(cap), 'readings_pf': printed,
                              **evaluate(printed, float(cap))})
        comparison[name] = {'validation_rmse_pf': float(np.sqrt(np.mean(error**2))),
                            'validation_five_presses': summaries}
    # Compare unseen capacitor levels using training batches only.
    unseen = {}
    for name in ('polynomial_1', 'polynomial_2', 'piecewise_linear'):
        errors, folds = [], []
        for cap in caps:
            fit = train & (y != cap)
            test = train & (y == cap)
            if name == 'piecewise_linear':
                keep = caps != cap
                predicted = interpolate(x[test], knots[keep], caps[keep])
            else:
                degree = int(name[-1])
                predicted = np.polyval(np.polyfit(x[fit], y[fit], degree), x[test])
            error = predicted-cap
            errors.extend(error)
            folds.append({'excluded_nominal_pf': float(cap),
                          'rmse_pf': float(np.sqrt(np.mean(error**2))),
                          'extrapolation': bool(cap in (caps[0], caps[-1]))})
        unseen[name] = {'rmse_pf': float(np.sqrt(np.mean(np.array(errors)**2))), 'folds': folds}
    selected = min(unseen, key=lambda name: unseen[name]['rmse_pf'])
    assert selected == 'polynomial_1', 'Selection changed: review before exporting firmware.'
    slope, intercept = np.polyfit(x[train], y[train], 1)
    assert slope > 0
    result = {'sources': sources, 'raw_readings': len(records)*64,
              'training_batches_per_capacitor': list(range(1, 11)),
              'validation_batches_per_capacitor': list(range(11, 16)),
              'selected_model': selected,
              'calibration': {'slope': float(slope), 'intercept': float(intercept)},
              'selection_rule': 'Lowest leave-capacitor-out RMSE on training batches among linear, quadratic, piecewise.',
              'time_knots_us': knots.tolist(),
              'capacitance_knots_pf': caps.tolist(), 'comparison': comparison,
              'leave_capacitor_out_training': unseen,
              'limitations': [
                  'Validation is later batches in the same session, not a fresh session or unknown capacitor.',
                  'Validation data was examined in earlier analysis; this is exploratory, not a blind final test.',
                  'Only four nominal references, each +/-5%; absolute accuracy is not established.',
                  'No final refit on validation batches; firmware exports only batches 1-10 calibration.',
                  'Outside 5-20 pF is linear extrapolation; not validated.',
                  'Input came from bench firmware; competition firmware requires board verification.',
                  'Scoring uses pF errors times 100 without normalization; exact rubric gaps return null.'
              ]}
    out = ROOT/'results/method1_latest'
    out.mkdir(parents=True, exist_ok=True)
    (out/'calibration.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    if args.write_headers:
        header = '''#pragma once
// Generated by method1_test/calibrate_latest.py --write-headers.
// Nov 18 data: batches 1-10 only. Same 64-reading median, 240 MHz.
// Linear fit selected by leave-capacitor-out testing; no snapping or clamping.
// Outside the 5-20 pF reference range is unvalidated extrapolation.
namespace Method1Calibration {
constexpr double SLOPE = SLOPE_VALUE;
constexpr double INTERCEPT = INTERCEPT_VALUE;
inline double capacitancePf(double medianUs) {
  return SLOPE * medianUs + INTERCEPT;
}
}
'''.replace('SLOPE_VALUE', format(slope, '.17g')).replace('INTERCEPT_VALUE', format(intercept, '.17g'))
        for directory in (ROOT, ROOT.parent/'method1_competition'):
            (directory/'calibration_model.h').write_text(header, encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

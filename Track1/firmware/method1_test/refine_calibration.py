"""Compare simple models using grouped folds; export a two-session calibration.

No random reading split: 64 readings produce ONE batch observation.
Model selection uses existing data, so a fresh round is still required.
"""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from competition_score import evaluate

ROOT = Path(__file__).resolve().parent


def metrics(error, nominal):
    return {'rmse_pf': float(np.sqrt(np.mean(error**2))),
            'organizer_rmse_value': float(100*np.sqrt(np.mean(error**2)))}


def scores(predicted, nominal, sessions):
    result = []
    for session in np.unique(sessions):
        for cap in np.unique(nominal):
            mask = (sessions == session) & (nominal == cap)
            # Score the two-decimal values actually printed over Serial.
            readings = [float(f'{v:.2f}') for v in predicted[mask]]
            result.append({'session': int(session), 'nominal_pf': int(cap),
                           'readings_pf': readings, **evaluate(readings, float(cap))})
    return result


def main():
    session1 = json.loads((ROOT/'results/method1_session1/calibration.json').read_text())
    session2 = json.loads((ROOT/'results/method1_session2/validation.json').read_text())
    batches, sources = [], []
    for session, manifest in enumerate((session1, session2), 1):
        for source in manifest['sources']:
            path = ROOT/'data'/source['file']
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == source['sha256'], f'Source changed: {path}'
            rows = list(csv.DictReader(path.open(encoding='utf-8-sig', newline='')))
            cap = source['nominal_pf']
            assert len(rows) == 320
            assert {r['sample'] for r in rows} == {f'{cap}pf'}
            assert {int(r['batch']) for r in rows} == set(range(1, 6))
            for batch in range(1, 6):
                group = [r for r in rows if int(r['batch']) == batch]
                assert [int(r['reading']) for r in group] == list(range(1, 65))
                assert all(r['status'] == 'ok' and int(r['cpu_mhz']) == 240 for r in group)
                times = [int(r['cycles'])/240 for r in group]
                assert all(np.isfinite(t) and t > 0 for t in times)
                batches.append({'session': session, 'nominal_pf': cap, 'batch': batch,
                                'median_us': float(np.median(times))})
            sources.append(dict(source, session=session))
    x = np.array([r['median_us'] for r in batches])
    y = np.array([r['nominal_pf'] for r in batches])
    sessions = np.array([r['session'] for r in batches])
    comparison = {}
    for degree in (1, 2):
        for relative in (False, True):
            name = f'degree_{degree}_' + ('relative' if relative else 'absolute')
            entry = {}
            for label, groups in [('leave_session_out', sessions), ('leave_capacitor_out', y)]:
                predicted = np.zeros_like(x)
                for value in np.unique(groups):
                    train = groups != value
                    coefficients = np.polyfit(x[train], y[train], degree,
                                             w=1/y[train] if relative else None)
                    predicted[~train] = np.polyval(coefficients, x[~train])
                entry[label] = metrics(predicted-y, y)
                entry[label]['five_press_scores'] = scores(predicted, y, sessions)
            comparison[name] = entry
    # The clarified organizer formula does NOT divide error by capacitance.
    # Use unweighted least squares in pF. Keep the simpler linear model:
    # quadratic fits four reference levels more closely but extrapolates poorly.
    slope, intercept = np.polyfit(x, y, 1)
    old = session1['calibration']
    old_prediction = old['slope']*x + old['intercept']
    new_prediction = slope*x + intercept
    assert slope > 0
    summary = []
    for cap in np.unique(y):
        mask = y == cap
        summary.append({'nominal_pf': int(cap),
                        'old_mean_pf': float(old_prediction[mask].mean()),
                        'new_mean_pf': float(new_prediction[mask].mean()),
                        'new_round2_mean_pf': float(new_prediction[mask & (sessions == 2)].mean())})
    result = {
        'sources': sources, 'batch_count': len(batches), 'raw_reading_count': len(batches)*64,
        'aggregation': 'Median of all 64 cycle counts / 240 MHz; no outliers removed',
        'selection': 'degree_1_absolute', 'comparison': comparison,
        'scoring_formula': 'accuracy=abs(mean-reference)*100; rmse=sqrt(mean(error^2))*100; pF inputs',
        'training_five_press_scores': scores(new_prediction, y, sessions),
        'calibration': {'slope': float(slope), 'intercept': float(intercept),
                        'nominal_range_pf': [5, 20], 'cpu_mhz': 240},
        'retrospective_old': metrics(old_prediction-y, y),
        'training_new': metrics(new_prediction-y, y), 'summary': summary,
        'limitations': [
            'Model selection used both sessions; grouped scores are exploratory, not a final independent test.',
            'Final coefficients fitted on both sessions; their replay error is training error.',
            'Only four physical capacitors with nominal +/-5% tolerance; absolute accuracy is unknown.',
            'Endpoint leave-capacitor-out folds are extrapolation.',
            'Data came from bench firmware, not competition firmware; fresh hardware validation required.',
            'No reference-voltage correction inferred from these data; keep the original circuit and reference.',
            'Failed initial_low file excluded; only the eight previously verified source files used.'
        ]}
    out = ROOT/'results/method1_competition_scoring'
    out.mkdir(parents=True, exist_ok=True)
    (out/'calibration.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

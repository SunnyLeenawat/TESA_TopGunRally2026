"""Audit the four confirmed on-board reference tests without changing firmware."""
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from competition_score import evaluate

ROOT = Path(__file__).resolve().parent
FILES = ['20261118_002942_competition.json', '20261118_003214_competition.json',
         '20261118_003345_competition.json', '20261118_003506_competition.json']


def main():
    x, y, sources, actual = [], [], [], []
    versions = []
    for name in FILES:
        path = ROOT/'results/hardware'/name
        run = json.loads(path.read_text())
        assert len(run['readings_pf']) == len(run['diagnostics']) == 5
        assert run['contest_mode_ignores_m']
        versions.append(run['sources'])
        times = []
        for line in run['diagnostics']:
            assert 'valid=64/64' in line
            times.append(float(re.search(r'median_us=([0-9.]+)', line)[1]))
        cap = run['nominal_reference_pf']
        x.extend(times)
        y.extend([cap]*5)
        actual.append({'nominal_pf': cap, 'readings_pf': run['readings_pf'],
                       **evaluate(run['readings_pf'], cap)})
        sources.append({'file': name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    assert all(version == versions[0] for version in versions)
    x, y = np.array(x), np.array(y)
    candidates = {}
    for degree in (1, 2):
        coefficients = np.polyfit(x, y, degree)
        predicted = np.polyval(coefficients, x)
        errors = []
        for cap in np.unique(y):
            train = y != cap
            fold = np.polyfit(x[train], y[train], degree)
            errors.extend(np.polyval(fold, x[~train])-cap)
        replay = []
        for cap in np.unique(y):
            readings = [float(f'{v:.2f}') for v in predicted[y == cap]]
            replay.append({'nominal_pf': float(cap), **evaluate(readings, float(cap))})
        candidates[f'degree_{degree}'] = {
            'coefficients_descending': coefficients.tolist(),
            'training_rmse_pf': float(np.sqrt(np.mean((predicted-y)**2))),
            'leave_capacitor_out_rmse_pf': float(np.sqrt(np.mean(np.array(errors)**2))),
            'training_replay_not_validation': replay}
    report = {'sources': sources, 'firmware_source_hashes': versions[0],
              'actual_tests': sorted(actual, key=lambda row: row['nominal_pf']),
              'candidates': candidates,
              'decision': 'Keep uploaded firmware. Refits trade off errors across references; no independent confirmation of improvement.',
              'limitations': ['References are nominal with +/-5% tolerance, not measured standards.',
                             'Only four capacitors; endpoint exclusion is extrapolation.',
                             'Serial-triggered diagnostic tests do not verify physical button or LED.',
                             'No unknown capacitor tested; no full-score guarantee.']}
    path = ROOT/'results/hardware/comparison.json'
    path.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

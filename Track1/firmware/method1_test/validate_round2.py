"""Validate round 2 using frozen round-1 coefficients, without refitting."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SOURCES = {
    5: '20261117_201611_317984_5pf.csv',
    10: '20261117_201504_115277_10pf.csv',
    15: '20261117_201334_003271_15pf.csv',
    20: '20261117_201206_298023_20pf.csv',
}


def main():
    model_path = ROOT / 'results/method1_session1/calibration.json'
    frozen = json.loads(model_path.read_text())
    a, b = frozen['calibration']['slope'], frozen['calibration']['intercept']
    with (model_path.parent / 'batch_summary.csv').open(newline='') as f:
        previous = list(csv.DictReader(f))
    summaries, details, provenance = [], [], []
    for cap, name in SOURCES.items():
        path = ROOT / 'data' / name
        with path.open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 320 and {r['sample'] for r in rows} == {f'{cap}pf'}, name
        assert {int(r['batch']) for r in rows} == set(range(1, 6)), name
        provenance.append({'nominal_pf': cap, 'file': name,
                           'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        predictions, medians, references = [], [], []
        for batch in range(1, 6):
            group = [r for r in rows if int(r['batch']) == batch]
            assert [int(r['reading']) for r in group] == list(range(1, 65)), name
            assert all(r['status'] == 'ok' for r in group), f'{name}: invalid readings; review manually'
            values = np.array([float(r['elapsed_us']) for r in group])
            assert np.all(np.isfinite(values) & (values > 0)), name
            t = float(np.median(values))
            predicted = a * t + b
            medians.append(t)
            predictions.append(predicted)
            references.extend(float(r['reference_mv']) for r in group)
            details.append({'nominal_pf': cap, 'batch': batch, 'median_us': t,
                'predicted_pf': predicted, 'error_vs_nominal_pf': predicted - cap,
                'first_reading_us': float(values[0])})
        errors = np.array(predictions) - cap
        prior_time = np.mean([float(r['median_us']) for r in previous if int(r['nominal_pf']) == cap])
        summaries.append({'nominal_pf': cap, 'valid_readings': 320,
            'median_time_mean_us': float(np.mean(medians)),
            'time_change_from_round1_us': float(np.mean(medians) - prior_time),
            'predicted_mean_pf': float(np.mean(predictions)),
            'predicted_sd_pf': float(np.std(predictions, ddof=1)),
            'mean_error_pf': float(errors.mean()),
            'relative_bias_pct': float(errors.mean() / cap * 100),
            'rmse_pf': float(np.sqrt(np.mean(errors ** 2))),
            'reference_mv_min': min(references), 'reference_mv_max': max(references)})
    out = ROOT / 'results/method1_session2'
    out.mkdir(parents=True, exist_ok=True)
    for filename, rows in [('batch_predictions.csv', details), ('summary.csv', summaries)]:
        with (out / filename).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    errors = np.array([r['error_vs_nominal_pf'] for r in details])
    result = {'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
        'frozen_coefficients': {'slope': a, 'intercept': b}, 'refitted': False,
        'sources': provenance, 'summary': summaries,
        'rmse_vs_nominal_pf': float(np.sqrt(np.mean(errors ** 2))),
        'mae_vs_nominal_pf': float(np.abs(errors).mean()),
        'limitations': ['Same physical references; tests repeatability across rounds, not independent absolute accuracy.',
            'Nominal reference tolerance is +/-5%; not a meter-error pass/fail limit.',
            'Uses each batch median with all 64 readings retained.',
            'No source CSV or round-1 calibration was changed.']}
    (out / 'validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    nominal = np.array([r['nominal_pf'] for r in details])
    predicted = np.array([r['predicted_pf'] for r in details])
    axes[0].plot([3, 22], [3, 22], '--', color='gray', label='Identity')
    axes[0].scatter(nominal, predicted, color='#167a80', label='One prediction per batch')
    for r in summaries:
        axes[0].errorbar(r['nominal_pf'], r['predicted_mean_pf'], xerr=.05*r['nominal_pf'],
                        yerr=r['predicted_sd_pf'], fmt='o', color='#143452', capsize=4)
    axes[0].set(xlabel='Nominal capacitance (pF)', ylabel='Predicted capacitance (pF)',
                title='Horizontal: reference +/-5%; vertical: batch SD', xticks=[5, 10, 15, 20])
    axes[0].legend(fontsize=9)
    axes[1].axhline(0, linestyle='--', color='gray')
    axes[1].scatter(nominal, errors, color='#167a80')
    for cap in SOURCES:
        axes[1].fill_between([cap-.65, cap+.65], [-.05*cap]*2, [.05*cap]*2, color='#e2e6ea', zorder=0)
    axes[1].set(xlabel='Nominal capacitance (pF)', ylabel='Prediction minus nominal (pF)',
                title='Gray bands: reference tolerance, not acceptance limits', xticks=[5, 10, 15, 20])
    fig.suptitle('Round 2 | frozen round-1 calibration | no refitting', fontsize=15)
    fig.savefig(out / 'round2_validation.png', dpi=150)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

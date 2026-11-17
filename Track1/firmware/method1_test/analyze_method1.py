"""Reproduce Method 1 plots from explicitly selected successful runs.

Raw CSVs are read only. Calibration uses one median per button batch.
All errors are relative to nominal capacitors, not measured true capacitance.
"""
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SOURCES = {
    5: '20261117_191532_611032_5pf.csv',
    10: '20261117_192226_294059_10pf.csv',
    15: '20261117_200129_124924_15pf.csv',
    20: '20261117_200506_323610_20pf.csv',
}
OUT = ROOT / 'results' / 'method1_session1'


def fit(x, y):
    return np.linalg.lstsq(np.column_stack([x, np.ones(len(x))]), y, rcond=None)[0]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.dpi': 130})
    rows_by_cap = {}
    summaries = []
    provenance = []
    for cap, filename in SOURCES.items():
        source = ROOT / 'data' / filename
        with source.open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 320 or {r['sample'] for r in rows} != {f'{cap}pf'}:
            raise ValueError(f'{filename}: expected 320 rows with the matching sample label')
        if {int(r['batch']) for r in rows} != set(range(1, 6)):
            raise ValueError(f'{filename}: expected batches 1-5')
        if any(r['status'] != 'ok' for r in rows):
            raise ValueError(f'{filename}: invalid readings; review before changing analysis policy')
        rows_by_cap[cap] = rows
        provenance.append({'nominal_pf': cap, 'file': filename,
                           'sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
        for batch in range(1, 6):
            group = [r for r in rows if int(r['batch']) == batch]
            if [int(r['reading']) for r in group] != list(range(1, 65)):
                raise ValueError(f'{filename}: batch {batch} has missing/out-of-order readings')
            t = np.array([float(r['elapsed_us']) for r in group])
            ref = np.array([float(r['reference_mv']) / 1000 for r in group])
            if not np.all(np.isfinite(t) & (t > 0) & (ref > 0) & (ref < 3.3)):
                raise ValueError('Invalid timing/reference value')
            # R = nominal 1 MOhm; charge voltage is assumed 3.3 V, not measured.
            theory = -t * 1e6 / (1e6 * np.log1p(-ref / 3.3))
            summaries.append({'nominal_pf': cap, 'batch': batch,
                'median_us': float(np.median(t)), 'mean_us': float(t.mean()),
                'sd_us': float(t.std(ddof=1)), 'first_us': float(t[0]),
                'reference_mv': float(np.median(ref) * 1000),
                'theory_median_pf': float(np.median(theory))})

    x = np.array([r['median_us'] for r in summaries])
    y = np.array([r['nominal_pf'] for r in summaries], dtype=float)
    a, b = fit(x, y)
    training = a * x + b
    held_out = np.empty(len(x))
    folds = []
    for cap in SOURCES:
        test = y == cap
        fa, fb = fit(x[~test], y[~test])
        held_out[test] = fa * x[test] + fb
        error = held_out[test] - cap
        folds.append({'held_out_nominal_pf': cap,
            'type': 'extrapolation' if cap in (5, 20) else 'interpolation',
            'slope_pf_per_us': float(fa), 'intercept_pf': float(fb),
            'mean_prediction_pf': float(held_out[test].mean()),
            'bias_vs_nominal_pf': float(error.mean()),
            'rmse_vs_nominal_pf': float(np.sqrt(np.mean(error ** 2)))})
    for i, record in enumerate(summaries):
        record['all_reference_fit_pf'] = float(training[i])
        record['held_out_prediction_pf'] = float(held_out[i])
    with (OUT / 'batch_summary.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    # Shared reference tolerance: one fixed endpoint choice per physical capacitor.
    corners = []
    grid = np.linspace(x.min(), x.max(), 150)
    for signs in itertools.product((-1, 1), repeat=4):
        assumed = {cap: cap * (1 + .05 * sign) for cap, sign in zip(SOURCES, signs)}
        ca, cb = fit(x, np.array([assumed[int(cap)] for cap in y]))
        corners.append(ca * grid + cb)
    corners = np.array(corners)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), layout='constrained')
    caps = np.array(list(SOURCES), dtype=float)
    means = np.array([x[y == cap].mean() for cap in caps])
    spread = np.array([x[y == cap].std(ddof=1) for cap in caps])
    axes[0].scatter(y, x, color='#167a80', alpha=.65, label='One point per batch median')
    axes[0].errorbar(caps, means, xerr=.05 * caps, yerr=spread, fmt='o',
                     color='#143452', capsize=5, label='Mean +/- 1 SD of batch medians')
    axes[0].set(xlabel='Nominal reference capacitance (pF)', ylabel='Measured time (us)',
                title='Response to the four reference capacitors', xticks=caps)
    axes[0].legend(fontsize=8)
    axes[1].fill_between(grid, corners.min(axis=0), corners.max(axis=0),
                        color='#c7dfe1', label='Reference-tolerance sensitivity only')
    axes[1].plot(grid, a * grid + b, color='#167a80', label=f'C = {a:.5f} t {b:+.5f}')
    axes[1].scatter(x, y, color='#143452', s=24)
    axes[1].set(xlabel='Batch median time (us)', ylabel='Estimated capacitance (pF)',
                title='Calibration fitted to all four references')
    axes[1].legend(fontsize=8)
    fig.suptitle('Method 1 | RC + LM393 | one run per capacitor', fontsize=15)
    fig.savefig(OUT / '01_response_and_calibration.png')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), layout='constrained')
    theory = np.array([r['theory_median_pf'] for r in summaries])
    axes[0].scatter(y, theory, marker='x', color='#b46a24', label='Ideal RC formula (uncorrected)')
    axes[0].scatter(y, held_out, color='#167a80', label='Held-out calibration predictions')
    axes[0].plot([0, 65], [0, 65], '--', color='gray', label='Identity')
    axes[0].set(xlim=(2, 23), ylim=(0, max(theory) + 5), xticks=caps,
                xlabel='Nominal capacitance (pF)', ylabel='Estimated capacitance (pF)',
                title='Theory versus leave-one-capacitor-out estimates')
    axes[0].legend(fontsize=8)
    for cap in caps:
        axes[1].fill_between([cap - .65, cap + .65], [-.05 * cap] * 2,
                            [.05 * cap] * 2, color='#e2e6ea')
    axes[1].axhline(0, color='gray', linestyle='--')
    axes[1].scatter(y, held_out - y, color='#167a80')
    axes[1].set(xticks=caps, xlabel='Held-out nominal capacitance (pF)',
                ylabel='Prediction minus nominal (pF)', title='Held-out residuals; gray = reference +/-5%')
    fig.suptitle('All batches of the test capacitor excluded from fitting', fontsize=15)
    fig.savefig(OUT / '02_validation.png')
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), layout='constrained')
    for ax, (cap, rows) in zip(axes.flat, rows_by_cap.items()):
        times = np.array([float(r['elapsed_us']) for r in rows])
        ax.plot(np.arange(1, 321), times, lw=.8, color='#167a80')
        ax.scatter([1], [times[0]], color='#bc433c', label='First reading (retained)', zorder=3)
        for boundary in (64, 128, 192, 256):
            ax.axvline(boundary + .5, color='#c4cbd0', linestyle=':', lw=.8)
        ax.set(title=f'{cap} pF | 5 batches, 320 valid readings',
               xlabel='Reading index (dotted lines separate batches)', ylabel='Time (us)')
        ax.legend(fontsize=8)
    fig.suptitle('Raw timing | no readings removed', fontsize=15)
    fig.savefig(OUT / '03_raw_readings.png')
    plt.close(fig)

    result = {'sources': provenance, 'aggregation': 'Median of all 64 readings per batch; no rows removed',
        'reference_tolerance_pct': 5, 'independent_capacitor_count': 4,
        'batches_per_capacitor': 5, 'raw_readings': 1280,
        'calibration': {'equation': 'capacitance_pf = slope * batch_median_elapsed_us + intercept',
            'slope': float(a), 'intercept': float(b), 'nominal_range_pf': [5, 20],
            'observed_batch_median_range_us': [float(x.min()), float(x.max())]},
        'theory_assumptions': {'resistance_ohm': 1000000, 'charge_voltage_v': 3.3,
            'threshold': 'reference_mv from each CSV; ADC estimate'},
        'training_rmse_vs_nominal_pf': float(np.sqrt(np.mean((training - y) ** 2))),
        'held_out_rmse_vs_nominal_pf': float(np.sqrt(np.mean((held_out - y) ** 2))),
        'held_out_mae_vs_nominal_pf': float(np.mean(np.abs(held_out - y))),
        'folds': folds,
        'limitations': ['One acquisition run per capacitor; not three repeat sessions.',
            'Reference values are nominal +/-5%, not independently measured.',
            'Batch medians are repeated measurements, not independent reference capacitors.',
            'Endpoint holdouts extrapolate; results are exploratory, not absolute-accuracy certification.',
            'The recurring elevated first reading is retained; its cause is unconfirmed.',
            'Calibration applies to batch medians and the same circuit/settings, not arbitrary single readings.',
            'Tolerance envelope excludes resistor, supply, ADC, timing and other uncertainty sources.',
            'No firmware coefficients were changed.']}
    (OUT / 'calibration.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    print('Saved plots and tables to', OUT)


if __name__ == '__main__':
    main()

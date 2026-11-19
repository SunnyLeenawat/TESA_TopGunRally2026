"""Capture/import coffee timing batches and fit moisture against labelled reference beans."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import time

ROOT = Path(__file__).resolve().parent
FIRMWARE = ROOT.parent.parent/'firmware'
LABELS = {'A': 11.4, 'B': 14.4, 'C': 6.6}


def parse(text):
    rows = []
    for line in text.splitlines():
        if not line.startswith('COFFEE_DATA,'):
            continue
        fields = line.strip().split(',')
        if len(fields) != 7:
            raise ValueError('Malformed COFFEE_DATA row')
        _, batch, median, mad, reference, valid, status = fields
        if int(batch) != len(rows)+1:
            raise ValueError('Missing/duplicate/restarted batch: use a fresh log for one run.')
        row = {'batch': int(batch), 'median_us': float(median) if median else None,
               'mad_us': float(mad) if mad else None, 'reference_mv': int(reference),
               'valid': int(valid), 'status': status}
        rows.append(row)
    if len(rows) != 15:
        raise ValueError(f'Expected 15 batches, found {len(rows)}. Keep the partial raw log and recollect.')
    return rows


def save(args, text, path):
    rows = parse(text)
    moisture = '' if args.sample == 'empty' else (args.moisture if args.moisture is not None else LABELS[args.sample])
    if moisture != '' and (not math.isfinite(moisture) or not 0 <= moisture <= 100):
        raise ValueError('Reference moisture must be finite and between 0 and 100.')
    fields = ['setup', 'sample', 'fill', 'moisture_percent', *rows[0]]
    with path.with_suffix('.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(dict(setup=args.setup, sample=args.sample, fill=args.fill,
                              moisture_percent=moisture, **row) for row in rows)
    print('Saved:', path.with_suffix('.csv'))


def collect(args):
    for name in (args.setup, args.fill):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
            raise ValueError('Setup and fill IDs allow letters, digits, underscore and hyphen only.')
    folder = ROOT/'data'/args.setup
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/(datetime.now().strftime('%Y%m%d_%H%M%S_%f')+f'_{args.sample}_{args.fill}.log')
    if args.command == 'import':
        path.write_bytes(args.log.read_bytes())
        text = path.read_text(encoding='utf-8-sig', errors='replace')
    else:
        import serial
        print('Close PuTTY/Serial Monitor. Keep one fill unchanged until RUN_DONE.')
        with path.open('x', encoding='utf-8') as raw, serial.Serial(args.port, 9600, timeout=1, write_timeout=2) as port:
            time.sleep(2)
            port.reset_input_buffer()
            port.write(b'run\n')
            deadline = time.monotonic()+40
            lines = []
            try:
                while time.monotonic() < deadline:
                    line = port.readline().decode('utf-8', errors='replace').strip()
                    if not line:
                        continue
                    raw.write(line+'\n'); raw.flush()
                    lines.append(line)
                    print(line, flush=True)
                    if line == 'RUN_DONE,15':
                        break
                else:
                    raise ValueError(f'Capture timeout; partial log preserved: {path}')
            finally:
                port.write(b'stop\n'); port.flush()
            text = '\n'.join(lines)
    save(args, text, path)


def load(paths, empty=False):
    observations, groups, setups, sources = [], set(), set(), []
    for path in paths:
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 15 or [int(r['batch']) for r in rows] != list(range(1, 16)):
            raise ValueError(f'{path}: exactly 15 sequential batches required')
        keys = {(r['setup'], r['sample'], r['fill']) for r in rows}
        labels = {r['moisture_percent'] for r in rows}
        if len(keys) != 1 or len(labels) != 1:
            raise ValueError('Mixed metadata in file')
        key = keys.pop()
        if key in groups:
            raise ValueError('Duplicate fill: use distinct refill IDs, not repeated files.')
        groups.add(key); setups.add(key[0])
        if (key[1] == 'empty') != empty:
            raise ValueError('Empty baseline and coffee files must be supplied separately.')
        moisture = None if empty else float(labels.pop())
        if not empty and (not math.isfinite(moisture) or not 0 <= moisture <= 100):
            raise ValueError('Invalid reference label')
        times = []
        for row in rows:
            median, mad = float(row['median_us']), float(row['mad_us'])
            if (row['status'] != 'ok' or int(row['valid']) != 64 or
                not math.isfinite(median) or median <= 0 or not math.isfinite(mad) or mad < 0 or
                not 850 <= int(row['reference_mv']) <= 950):
                raise ValueError(f'{path}: invalid batch; inspect and recollect, do not silently discard.')
            times.append(median)
            observations.append((median, moisture, key))
        sources.append({'file': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'fill_mean_us': statistics.mean(times), 'fill_sd_us': statistics.stdev(times)})
    return observations, groups, setups, sources


def fit(args):
    baseline, _, empty_setups, empty_sources = load(args.empty, True)
    train, train_groups, train_setups, train_sources = load(args.train)
    validation, val_groups, val_setups, val_sources = load(args.validation)
    if train_groups & val_groups:
        raise ValueError('Training and validation must be different physical refills, not split rows.')
    if len(empty_setups | train_setups | val_setups) != 1:
        raise ValueError('Do not mix wiring layouts.')
    levels = {row[1] for row in train}
    if len(levels) < 3 or {row[1] for row in validation} != levels:
        raise ValueError('Require >=3 moisture levels in training and the same levels in held-out refills.')
    empty_us = statistics.mean(row[0] for row in baseline)
    x = [row[0]-empty_us for row in train]; y = [row[1] for row in train]
    xmean, ymean = statistics.mean(x), statistics.mean(y)
    denominator = sum((t-xmean)**2 for t in x)
    if denominator == 0:
        raise ValueError('No sensor response variation')
    slope = sum((t-xmean)*(m-ymean) for t, m in zip(x, y))/denominator
    intercept = ymean-slope*xmean
    if slope <= 0 or not math.isfinite(slope) or not math.isfinite(intercept):
        raise ValueError('Expected increasing response with moisture; inspect labels and fill procedure.')
    span = max(x)-min(x)
    # Modest guard band, not a statistical confidence interval or accuracy claim.
    margin = max(.05*span, 3*max(s['fill_sd_us'] for s in train_sources))
    lower, upper = min(x)-margin, max(x)+margin
    residuals = [slope*t+intercept-m for t, m in zip(x, y)]
    sst = sum((m-ymean)**2 for m in y)
    details = []
    for t, m, key in validation:
        feature = t-empty_us
        pred = slope*feature+intercept
        details.append({'sample': key[1], 'fill': key[2], 'reference_percent': m,
                        'predicted_percent': pred, 'error_percentage_points': pred-m,
                        'in_feature_range': lower <= feature <= upper})
    result = {'setup': next(iter(train_setups)), 'feature': 'median_us - fixed_empty_us',
              'empty_us': empty_us, 'slope_percent_per_us': slope, 'intercept_percent': intercept,
              'min_delta_us': lower, 'max_delta_us': upper, 'range_margin_us': margin,
              'train_r2': 1-sum(e*e for e in residuals)/sst,
              'validation_rmse_percentage_points': math.sqrt(statistics.mean(r['error_percentage_points']**2 for r in details)),
              'validation_mae_percentage_points': statistics.mean(abs(r['error_percentage_points']) for r in details),
              'validation': details, 'empty_sources': empty_sources, 'train_sources': train_sources,
              'validation_sources': val_sources,
              'limitations': ['Labels supplied by user; wet/dry basis and reference uncertainty not independently confirmed.',
                             'Refills are held out, but the same three reference bags are reused.',
                             'Range guard band is not a confidence interval. Fresh unknown samples still required.',
                             'Empty subtraction includes air-filled sensor capacitance, not only stray capacitance.',
                             'READY remains false until review. No firmware overwritten automatically.']}
    out = ROOT/'results'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    (out/'calibration.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    source = FIRMWARE/'coffee_common/moisture_calibration.h'
    header = source.read_text()
    replacements = {'EMPTY_US': empty_us, 'SLOPE_PERCENT_PER_US': slope, 'INTERCEPT_PERCENT': intercept,
                    'MIN_DELTA_US': lower, 'MAX_DELTA_US': upper}
    header = re.sub(r'constexpr bool READY = (?:true|false);', 'constexpr bool READY = false;', header)
    for name, value in replacements.items():
        header, count = re.subn(rf'constexpr double {name} = [^;]+;', f'constexpr double {name} = {value:.17g};', header)
        if count != 1:
            raise ValueError('Calibration header format changed')
    (out/'moisture_calibration.candidate.h').write_text(header, encoding='utf-8')
    print('Validation RMSE (percentage points):', result['validation_rmse_percentage_points'])
    print('Saved:', out, '\nReview candidate, then copy to coffee_common and enable READY before syncing.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('capture', 'import'):
        p = commands.add_parser(name)
        p.add_argument('--port', required=True) if name == 'capture' else p.add_argument('--log', type=Path, required=True)
        p.add_argument('--setup', required=True)
        p.add_argument('--sample', choices=['empty', 'A', 'B', 'C'], required=True)
        p.add_argument('--fill', required=True, help='Physical refill ID, e.g. refill01')
        p.add_argument('--moisture', type=float, help='Override user-provided reference label, percent')
    p = commands.add_parser('fit')
    p.add_argument('--empty', nargs='+', type=Path, required=True)
    p.add_argument('--train', nargs='+', type=Path, required=True)
    p.add_argument('--validation', nargs='+', type=Path, required=True)
    args = parser.parse_args()
    try:
        fit(args) if args.command == 'fit' else collect(args)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, str(error)+'\n')


if __name__ == '__main__':
    main()

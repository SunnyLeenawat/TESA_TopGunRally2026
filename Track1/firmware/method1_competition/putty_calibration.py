"""Import per-press timing records from PuTTY, then fit a fixed-layout calibration."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

DATA = Path(__file__).resolve().parent.parent/'method1_test/data'


def parse_log(text):
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith('CAL_DATA,'):
            fields = line.split(',')
            if len(fields) != 6:
                raise ValueError('Malformed CAL_DATA record: '+line)
            _, run, median, ref, cpu, status = fields
            if int(run) != len(rows)+1:
                raise ValueError('Missing, repeated, or restarted run IDs. Use a fresh log per capacitor.')
            t = float(median) if median else None
            if status == 'ok' and (t is None or not math.isfinite(t) or t <= 0 or int(cpu) != 240 or not 700 <= int(ref) <= 1100):
                raise ValueError('Invalid timing values: '+line)
            rows.append({'batch': int(run), 'median_us': t, 'reference_mv': int(ref),
                         'cpu_mhz': int(cpu), 'status': status})
    if not rows:
        raise ValueError('No CAL_DATA rows. Upload logging-capable firmware, type c, then press the button.')
    return rows


def import_log(args):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.setup):
        raise ValueError('Setup ID: use only letters, digits, underscores, hyphens.')
    raw = args.log.read_bytes()
    rows = parse_log(raw.decode('utf-8-sig', errors='replace'))
    directory = DATA/args.setup
    directory.mkdir(parents=True, exist_ok=True)
    stem = datetime.now().strftime('%Y%m%d_%H%M%S_%f')+f'_{args.reference:g}pf_putty'
    path = directory/(stem+'.csv')
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['setup', 'nominal_pf', *rows[0]])
        writer.writeheader()
        writer.writerows(dict(setup=args.setup, nominal_pf=args.reference, **row) for row in rows)
    (directory/(stem+'.log')).write_bytes(raw)
    print(f'Saved {len(rows)} batches ({sum(r["status"] != "ok" for r in rows)} invalid): {path}')


def fit(args):
    records, provenance, setups, references = [], [], set(), set()
    for path in args.files:
        with path.open(newline='', encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 15 or [int(r['batch']) for r in rows] != list(range(1, 16)):
            raise ValueError(f'{path}: require exactly 15 complete sequential batches.')
        caps = {float(r['nominal_pf']) for r in rows}
        if len(caps) != 1:
            raise ValueError('Each file must contain one reference value.')
        cap = caps.pop()
        if not math.isfinite(cap) or cap <= 0 or cap in references:
            raise ValueError('Supply distinct positive reference values, one file per capacitor.')
        references.add(cap)
        for r in rows:
            setups.add(r['setup'])
            t = float(r['median_us'])
            if r['status'] != 'ok' or int(r['cpu_mhz']) != 240 or not math.isfinite(t) or t <= 0 or not 700 <= int(r['reference_mv']) <= 1100:
                raise ValueError(f'{path}: invalid timing batch; recollect rather than drop it.')
            records.append((cap, int(r['batch']), t))
        provenance.append({'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    if len(setups) != 1 or len(references) < 4:
        raise ValueError('Require at least four references from one fixed setup.')
    training = [r for r in records if r[1] <= 10]
    xmean = statistics.mean(r[2] for r in training)
    ymean = statistics.mean(r[0] for r in training)
    denominator = sum((r[2]-xmean)**2 for r in training)
    if denominator == 0:
        raise ValueError('No timing variation to fit.')
    slope = sum((t-xmean)*(cap-ymean) for cap, _, t in training)/denominator
    intercept = ymean-slope*xmean
    if not math.isfinite(slope) or not math.isfinite(intercept) or slope <= 0:
        raise ValueError('Invalid fit; check labels and wiring.')
    validation = []
    for cap in sorted(references):
        predicted = [slope*t+intercept for c, batch, t in records if c == cap and batch > 10]
        validation.append({'nominal_pf': cap, 'predicted_pf': predicted,
                           'mean_pf': statistics.mean(predicted),
                           'organizer_error': abs(statistics.mean(predicted)-cap)*100,
                           'organizer_rmse': math.sqrt(statistics.mean((p-cap)**2 for p in predicted))*100})
    result = {'setup': setups.pop(), 'sources': provenance, 'slope': slope, 'intercept': intercept,
              'training_batches': list(range(1, 11)), 'validation_batches': list(range(11, 16)),
              'validation': validation,
              'limitations': 'Same-session nominal references, not independent absolute accuracy. Recheck final firmware on unknown capacitors. Formula metrics use unrounded pF errors times 100, without reference normalization.'}
    output = args.files[0].parent/('calibration_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json')
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(f'constexpr double SLOPE = {slope:.17g};\nconstexpr double INTERCEPT = {intercept:.17g};')
    print(json.dumps(validation, indent=2))
    print('Saved:', output, '\nFirmware not overwritten. Copy coefficients into the active .ino and verify/upload.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('import')
    p.add_argument('--log', type=Path, required=True)
    p.add_argument('--reference', type=float, required=True)
    p.add_argument('--setup', required=True)
    p = sub.add_parser('fit')
    p.add_argument('files', type=Path, nargs='+')
    args = parser.parse_args()
    try:
        if args.command == 'import':
            if not math.isfinite(args.reference) or args.reference <= 0:
                raise ValueError('Reference must be positive and finite.')
            import_log(args)
        else:
            fit(args)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()

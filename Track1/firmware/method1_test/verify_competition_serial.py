"""Collect diagnostic batches on competition firmware, then restore contest mode."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import time
import serial
from competition_score import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', required=True)
    parser.add_argument('--reference', required=True, type=float)
    parser.add_argument('--batches', type=int, choices=(5, 15), default=5)
    parser.add_argument('--setup', help='Fixed wiring layout ID; letters, digits, underscores, hyphens only')
    args = parser.parse_args()
    if args.setup and not re.fullmatch(r'[A-Za-z0-9_-]+', args.setup):
        parser.error('Invalid setup ID')
    root = Path(__file__).resolve().parent
    directory = root/'results/hardware'
    if args.setup:
        directory = directory/args.setup
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/(datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_competition.json')
    record = {'port': args.port, 'nominal_reference_pf': args.reference,
              'setup_id': args.setup, 'requested_batches': args.batches,
              'trigger': 'Serial m in diagnostic mode, not physical button',
              'readings_pf': [], 'diagnostics': [], 'measurements': [], 'raw_lines': [], 'sources': {}}
    for name in ('method1_competition.ino', 'calibration_model.h'):
        source = root.parent/'method1_competition'/name
        record['sources'][name] = hashlib.sha256(source.read_bytes()).hexdigest()
    try:
        with serial.Serial(args.port, 9600, timeout=1, write_timeout=2) as port:
            def read_until(predicate, seconds=8):
                deadline = time.monotonic()+seconds
                while time.monotonic() < deadline:
                    line = port.readline().decode('utf-8', errors='replace').strip()
                    if line:
                        record['raw_lines'].append(line)
                        print(line, flush=True)
                        if predicate(line):
                            return line
                raise RuntimeError('Timed out waiting for expected firmware response.')
            time.sleep(2)
            port.reset_input_buffer()
            try:
                port.write(b'd\n')
                read_until(lambda line: line.startswith('DIAG enabled:'))
                for batch in range(1, args.batches+1):
                    port.write(b'm\n')
                    read_until(lambda line: line == 'TGR-55')
                    line = read_until(lambda line: re.fullmatch(r'Capacitance = \d+\.\d{2} pF', line))
                    record['readings_pf'].append(float(line.split()[2]))
                    diagnostic = read_until(lambda line: line.startswith('DIAG reference_mv='))
                    record['diagnostics'].append(diagnostic)
                    match = re.fullmatch(r'DIAG reference_mv=(\d+) valid=(\d+)/64 median_us=(\d+(?:\.\d+)?|INVALID)', diagnostic)
                    if not match:
                        raise RuntimeError('Malformed diagnostic response')
                    median = None if match[3] == 'INVALID' else float(match[3])
                    record['measurements'].append({'batch': batch, 'reference_mv': int(match[1]),
                        'valid_count': int(match[2]), 'median_us': median,
                        'displayed_pf': record['readings_pf'][-1]})
                    if int(match[2]) != 64 or median is None or median <= 0:
                        raise RuntimeError('Invalid timing batch; partial data retained, not suitable for calibration')
                    time.sleep(.3)
                record['complete'] = True
                if args.batches == 5:
                    record['score_vs_nominal'] = evaluate(record['readings_pf'], args.reference)
                else:
                    record['calibration_split'] = {'training_batches': list(range(1, 11)),
                                                   'validation_batches': list(range(11, 16))}
            finally:
                port.write(b'q\n')
                port.flush()
                time.sleep(.2)
                port.reset_input_buffer()
                port.write(b'm\n')
                unexpected = port.read(512).decode('utf-8', errors='replace')
                record['contest_mode_ignores_m'] = not unexpected
                if unexpected:
                    record['unexpected_after_q'] = unexpected
    except Exception as error:
        record['error'] = str(error)
        raise
    finally:
        path.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
        print('Saved:', path)
    print(json.dumps(record.get('score_vs_nominal'), indent=2))


if __name__ == '__main__':
    main()

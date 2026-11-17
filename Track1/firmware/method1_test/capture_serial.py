"""Capture complete Method 1 batches without copying Arduino Serial Monitor."""
import argparse
import csv
from datetime import datetime
from pathlib import Path
import time

HEADER = 'run,reading,cycles,elapsed_us,reference_mv,cpu_mhz,status'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', required=True, help='Example: COM5')
    parser.add_argument('--sample', required=True, choices=['open_fixture', '5pf', '10pf', '15pf', '20pf'])
    parser.add_argument('--batches', type=int, default=5)
    args = parser.parse_args()
    if args.batches < 1:
        parser.error('--batches must be positive')
    try:
        import serial
    except ImportError:
        parser.error('Install dependency: python -m pip install pyserial==3.5')

    directory = Path(__file__).resolve().parent / 'data'
    directory.mkdir(exist_ok=True)
    prefix = directory / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + args.sample)
    path = prefix.with_suffix('.csv')
    print('Close Arduino Serial Monitor. Do not press the board button during capture.')
    print('CSV:', path)
    try:
        with serial.Serial(args.port, 9600, timeout=1, write_timeout=2) as port, \
                path.open('x', newline='', encoding='utf-8') as output, \
                prefix.with_suffix('.txt').open('x', encoding='utf-8') as raw:
            writer = csv.writer(output)
            writer.writerow(['sample', 'batch'] + HEADER.split(','))
            # Opening the serial port can reset the ESP32. Allow boot to finish.
            time.sleep(2)
            startup = port.read_all().decode('utf-8', errors='replace')
            raw.write(startup)
            for batch in range(1, args.batches + 1):
                port.write(b'm\n')
                deadline = time.monotonic() + 20
                received_header = False
                count = 0
                invalid = 0
                run_id = None
                while count < 64:
                    if time.monotonic() > deadline:
                        raise RuntimeError(f'Timeout: batch {batch}, received {count}/64 rows. Partial data retained.')
                    line = port.readline().decode('utf-8', errors='replace').strip()
                    if not line:
                        continue
                    raw.write(line + '\n')
                    raw.flush()
                    if line.startswith('SET_REFERENCE'):
                        raise RuntimeError('Reference outside range. Adjust pot near 900 mV using r in Serial Monitor, then retry.')
                    if line == HEADER:
                        if received_header:
                            raise RuntimeError('Unexpected new batch header. Do not press the button during capture.')
                        received_header = True
                        continue
                    if not received_header:
                        continue
                    fields = next(csv.reader([line]))
                    if len(fields) != 7 or fields[6] not in ('ok', 'initial_low', 'timeout'):
                        raise RuntimeError('Malformed row; inspect saved raw log: ' + line)
                    if int(fields[1]) != count + 1:
                        raise RuntimeError('Missing or out-of-order reading; partial data retained.')
                    if run_id is None:
                        run_id = fields[0]
                    if fields[0] != run_id:
                        raise RuntimeError('Run changed inside batch; partial data retained.')
                    writer.writerow([args.sample, batch] + fields)
                    output.flush()
                    count += 1
                    invalid += fields[6] != 'ok'
                print(f'Batch {batch}/{args.batches}: 64 rows saved, {invalid} invalid readings')
            print('Capture complete:', path)
    except (serial.SerialException, OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f'Capture stopped: {error}\nClose other programs using the COM port. Saved data is retained.\n')
    except KeyboardInterrupt:
        parser.exit(130, 'Stopped by user; partial data retained.\n')


if __name__ == '__main__':
    main()

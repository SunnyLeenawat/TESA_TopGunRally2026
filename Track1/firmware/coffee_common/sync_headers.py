"""Copy canonical coffee acquisition and calibration headers to both Arduino folders."""
from pathlib import Path

def main():
    source = Path(__file__).resolve().parent
    for name in ('coffee_test', 'coffee_competition'):
        directory = source.parent/name
        directory.mkdir(exist_ok=True)
        for header in ('coffee_sensor.h', 'moisture_calibration.h'):
            (directory/header).write_bytes((source/header).read_bytes())
    print('Synced identical coffee headers to both sketches. Recompile and upload after changes.')

if __name__ == '__main__':
    main()

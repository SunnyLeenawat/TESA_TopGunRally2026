import argparse
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
import coffee_data as tool


def write_fill(path, sample, fill, moisture, median, setup='fixed'):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['setup', 'sample', 'fill', 'moisture_percent',
            'batch', 'median_us', 'mad_us', 'reference_mv', 'valid', 'status'])
        writer.writeheader()
        for batch in range(1, 16):
            writer.writerow(dict(setup=setup, sample=sample, fill=fill, moisture_percent=moisture,
                batch=batch, median_us=median, mad_us=.01, reference_mv=900, valid=64, status='ok'))


class CoffeeDataTests(unittest.TestCase):
    def test_parse_and_incomplete_capture(self):
        text = '\n'.join(f'COFFEE_DATA,{i},10.2,0.1,900,64,ok' for i in range(1, 16))
        self.assertEqual(len(tool.parse(text)), 15)
        with self.assertRaises(ValueError): tool.parse(text.replace('COFFEE_DATA,2,', 'COFFEE_DATA,1,'))
        with self.assertRaises(ValueError): tool.parse(text.splitlines()[0])
        bad = text.replace('10.2,0.1,900,64,ok', ',,900,0,timeout', 1)
        self.assertIsNone(tool.parse(bad)[0]['median_us'])

    def test_fit_without_validation_leakage_and_guards(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            empty = root/'empty.csv'
            write_fill(empty, 'empty', 'empty01', '', 2)
            train, validation = [], []
            for sample, moisture in tool.LABELS.items():
                tr, va = root/f'{sample}_train.csv', root/f'{sample}_val.csv'
                # Ground truth M = 2*(t-2)+4. Validation adds 0.2 us.
                write_fill(tr, sample, 'refill01', moisture, moisture/2)
                write_fill(va, sample, 'refill02', moisture, moisture/2+.2)
                train.append(tr); validation.append(va)
            args = argparse.Namespace(empty=[empty], train=train, validation=validation)
            original = tool.ROOT
            tool.ROOT = root
            try:
                with contextlib.redirect_stdout(io.StringIO()): tool.fit(args)
                report = json.loads(next(root.glob('results/*/calibration.json')).read_text())
                self.assertAlmostEqual(report['slope_percent_per_us'], 2)
                self.assertAlmostEqual(report['intercept_percent'], 4)
                self.assertAlmostEqual(report['validation_rmse_percentage_points'], .4)
                header = next(root.glob('results/*/*.h')).read_text()
                self.assertIn('READY = false', header)
                with self.assertRaises(ValueError): tool.fit(argparse.Namespace(empty=[empty], train=train, validation=train))
                changed = validation[0]
                changed.write_text(changed.read_text().replace('fixed', 'different'))
                with self.assertRaises(ValueError): tool.fit(args)
                changed.write_text(changed.read_text().replace('different', 'fixed').replace(',64,ok', ',63,ok', 1))
                with self.assertRaises(ValueError): tool.fit(args)
            finally:
                tool.ROOT = original


if __name__ == '__main__':
    unittest.main()

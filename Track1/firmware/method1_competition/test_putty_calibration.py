import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import putty_calibration as tool


class CalibrationTests(unittest.TestCase):
    def test_import_fit_and_holdout(self):
        with tempfile.TemporaryDirectory() as temp:
            previous = tool.DATA
            tool.DATA = Path(temp)/'data'
            try:
                files = []
                for cap in (5, 10, 15, 20):
                    # Training C=2*t-3. Holdout shifted +0.5 us to detect leakage.
                    log = Path(temp)/f'{cap}.log'
                    lines = ['PuTTY log', 'TGR-55', 'Capacitance = 999.00 pF']
                    for batch in range(1, 16):
                        t = (cap+3)/2 + (0.5 if batch > 10 else 0)
                        lines.append(f'CAL_DATA,{batch},{t},900,240,ok')
                    log.write_text('\n'.join(lines))
                    with contextlib.redirect_stdout(io.StringIO()):
                        tool.import_log(argparse.Namespace(log=log, reference=cap, setup='fixed'))
                    files.extend((tool.DATA/'fixed').glob(f'*_{cap}pf_putty.csv'))
                with contextlib.redirect_stdout(io.StringIO()):
                    tool.fit(argparse.Namespace(files=files))
                r = json.loads(next((tool.DATA/'fixed').glob('calibration_*.json')).read_text())
                self.assertAlmostEqual(r['slope'], 2)
                self.assertAlmostEqual(r['intercept'], -3)
                for row in r['validation']:
                    self.assertAlmostEqual(row['mean_pf'], row['nominal_pf']+1)
                text = files[0].read_text().replace('fixed', 'other')
                files[0].write_text(text)
                with self.assertRaises(ValueError):
                    tool.fit(argparse.Namespace(files=files))
            finally:
                tool.DATA = previous

    def test_missing_or_restarted_run_rejected(self):
        for log in ('CAL_DATA,2,8,900,240,ok', 'CAL_DATA,1,8,900,240,ok\nCAL_DATA,1,8,900,240,ok'):
            with self.assertRaises(ValueError):
                tool.parse_log(log)

    def test_invalid_not_silently_lost(self):
        rows = tool.parse_log('CAL_DATA,1,,900,240,acquisition_error')
        self.assertEqual(rows[0]['status'], 'acquisition_error')
        self.assertIsNone(rows[0]['median_us'])
        with self.assertRaises(ValueError):
            tool.parse_log('CAL_DATA,1,nan,900,240,ok')

    def test_old_output_cannot_be_fit(self):
        with self.assertRaises(ValueError):
            tool.parse_log('TGR-55\nCapacitance = 10.00 pF')


if __name__ == '__main__':
    unittest.main()

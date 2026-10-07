"""GPU-free integrity checks for the retained measurement artifacts."""
import csv
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("summarize", ROOT / "scripts/summarize.py")
summarize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summarize)
RAW = ROOT / "results/raw_gemm_2026-10-07.csv"
SUMMARY = ROOT / "results/summary_gemm_2026-10-07.csv"

class MeasurementTests(unittest.TestCase):
    def setUp(self):
        with RAW.open(newline="") as stream:
            self.rows = list(csv.DictReader(stream))

    def test_measurement_contract(self):
        summarize.validate_rows(self.rows)
        self.assertEqual(len(self.rows), 194)

    def test_summary_reproduces_retained_values(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "summary.csv"
            subprocess.run([sys.executable, str(ROOT / "scripts/summarize.py"), str(RAW), str(output)], check=True)
            with output.open(newline="") as actual, SUMMARY.open(newline="") as expected:
                self.assertEqual(list(csv.DictReader(actual)), list(csv.DictReader(expected)))

    def test_invalid_measurements_are_rejected(self):
        for field, value in [("per_call_ms", "nan"), ("batch_ms", "-1"), ("m", "0"),
                             ("iterations", "0"), ("timing", "steady_clock")]:
            row = next(x for x in self.rows if x["method"] == "tiled").copy()
            row[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                summarize.validate_rows([row])
        with self.assertRaises(ValueError):
            summarize.validate_rows([self.rows[0], self.rows[0]])
        with self.assertRaises(ValueError):
            summarize.validate_rows([])

    def test_percentile_interpolation(self):
        self.assertEqual(summarize.percentile([3, 1, 2], .25), 1.5)

if __name__ == "__main__":
    unittest.main()

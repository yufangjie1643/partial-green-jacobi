"""Numerical evidence and release-extraction checks, executable without a GPU."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from reproduce_tables import reproduce
from fetch_data import destination


class Reproduction(unittest.TestCase):
    def test_calibration_and_paired_claims(self):
        r=reproduce()
        self.assertEqual(r['calibration']['anchors'],[9/16,11/16,13/16])
        self.assertAlmostEqual(r['study1_baseline_same_budget_savings']['gj'],.5228818873385783,places=12)
        self.assertAlmostEqual(r['study2']['mean_saving_vs_gj'],.4033940847089288,places=12)
        self.assertAlmostEqual(r['study2']['mean_saving_vs_compact8'],.05036429678346144,places=12)
        self.assertAlmostEqual(r['study2']['worst_regression_vs_compact8'],.08861234167135712,places=12)
        self.assertEqual(r['study2']['failed_green_attempts'],141)

    def test_extraction_rejects_paths_outside_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ['/tmp/example','../example','exp/../../example','scripts/replace.py']:
                with self.assertRaises(ValueError):destination(name,root)
            self.assertEqual(destination('exp/states/test.npz',root),root/'exp/states/test.npz')


if __name__=='__main__':unittest.main()

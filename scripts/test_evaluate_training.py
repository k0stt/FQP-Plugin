"""Run: .venv/Scripts/python -m unittest discover -s scripts -p 'test_*.py'"""
import unittest

import numpy as np

from evaluate_training import align_signals, holdout_files, listening_pair


class AlignmentTests(unittest.TestCase):
    def test_positive_delay(self):
        x = np.random.default_rng(17).normal(size=4000)
        y = np.r_[np.zeros(137), x[:-137]]
        aligned_x, aligned_y, lag = align_signals(x, y, 44100)
        self.assertEqual(lag, 137)
        np.testing.assert_array_equal(aligned_x, aligned_y)

    def test_listening_fade_preserves_common_gain(self):
        x = np.ones(1000) * .5
        y = np.ones(1000) * 2
        a, b = listening_pair(x, y, 1000, 10)
        self.assertEqual(float(a[0]), 0.)
        self.assertEqual(float(b[-1]), 0.)
        self.assertAlmostEqual(float(b[500]), .9)
        self.assertAlmostEqual(float(b[500] / a[500]), 4.)
        self.assertRaises(ValueError, listening_pair, x, y, 1000, 600)

    def test_holdout_rejects_content_overlap(self):
        manifest = {"train": [{"path": "a", "sha256": "same"}],
                    "holdout": [{"path": "b", "sha256": "same"}]}
        self.assertRaises(ValueError, holdout_files, manifest)

    def test_negative_delay(self):
        x = np.random.default_rng(21).normal(size=4000)
        y = np.r_[x[83:], np.zeros(83)]
        aligned_x, aligned_y, lag = align_signals(x, y, 44100)
        self.assertEqual(lag, -83)
        np.testing.assert_array_equal(aligned_x, aligned_y)


if __name__ == "__main__":
    unittest.main()

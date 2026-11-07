"""Run: .venv/Scripts/python -m unittest discover -s scripts -p 'test_*.py'"""
import unittest

import numpy as np

from evaluate_training import align_signals


class AlignmentTests(unittest.TestCase):
    def test_positive_delay(self):
        x = np.random.default_rng(17).normal(size=4000)
        y = np.r_[np.zeros(137), x[:-137]]
        aligned_x, aligned_y, lag = align_signals(x, y, 44100)
        self.assertEqual(lag, 137)
        np.testing.assert_array_equal(aligned_x, aligned_y)

    def test_negative_delay(self):
        x = np.random.default_rng(21).normal(size=4000)
        y = np.r_[x[83:], np.zeros(83)]
        aligned_x, aligned_y, lag = align_signals(x, y, 44100)
        self.assertEqual(lag, -83)
        np.testing.assert_array_equal(aligned_x, aligned_y)


if __name__ == "__main__":
    unittest.main()

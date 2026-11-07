import unittest

import numpy as np

from diagnose_onset_artifact import correlation_range, rms


class OnsetDiagnosticTests(unittest.TestCase):
    def test_shared_prefix_not_late_signal(self):
        a = np.concatenate([np.arange(50, dtype=float), np.zeros(550)])
        b = np.concatenate([np.arange(50, dtype=float), np.ones(550)])
        self.assertAlmostEqual(correlation_range([a, b], 1000, 0, .05)["median"], 1.)
        self.assertEqual(correlation_range([a, b], 1000, .4, .6)["pairs"], 0)
        self.assertEqual(rms(np.zeros(50)), 0.)


if __name__ == "__main__":
    unittest.main()

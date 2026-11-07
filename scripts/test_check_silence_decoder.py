import unittest

import numpy as np

from check_silence_decoder import rms


class SilenceControlTests(unittest.TestCase):
    def test_rms_on_known_window(self):
        x = np.array([0., 0., 3., 4.])
        self.assertEqual(rms(x, 2, 0, 1), 0.)
        self.assertAlmostEqual(rms(x, 2, 1, 2), 5 / np.sqrt(2))


if __name__ == "__main__":
    unittest.main()

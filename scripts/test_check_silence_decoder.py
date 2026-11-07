import unittest

import numpy as np

from check_silence_decoder import build_signals, rms


class SilenceControlTests(unittest.TestCase):
    def test_rms_on_known_window(self):
        x = np.array([0., 0., 3., 4.])
        self.assertEqual(rms(x, 2, 0, 1), 0.)
        self.assertAlmostEqual(rms(x, 2, 1, 2), 5 / np.sqrt(2))

    def test_zero_input_and_model_output_have_different_names(self):
        note = np.ones(4)
        model_output_on_zero = np.full(4, .2)
        signals = build_signals(note, note, note, note, model_output_on_zero,
                                model_output_on_zero, note, note, note)
        np.testing.assert_array_equal(signals["zero_input"], np.zeros_like(note))
        np.testing.assert_array_equal(signals["zero_input_reconstruction"], model_output_on_zero)
        np.testing.assert_array_equal(signals["zero_input_output_subtracted"], note - model_output_on_zero)


if __name__ == "__main__":
    unittest.main()

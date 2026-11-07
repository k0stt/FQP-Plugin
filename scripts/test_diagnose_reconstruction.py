import unittest

import numpy as np

from diagnose_reconstruction import compare, harmonic_energy_fraction


class PitchTests(unittest.TestCase):
    def test_synthetic_semitone_shift(self):
        sr = 44100
        t = np.arange(sr * 2) / sr
        x = np.sin(2 * np.pi * 220 * t)
        y = np.sin(2 * np.pi * 220 * 2 ** (2 / 12) * t)
        result = compare(x, y, sr)
        self.assertEqual(result["best_shift_semitones"], 2)
        self.assertGreater(result["best_shift_similarity"], result["same_pitch_class_similarity"])

    def test_same_tone(self):
        sr = 44100
        x = np.sin(2 * np.pi * 440 * np.arange(sr * 2) / sr)
        result = compare(x, x, sr)
        self.assertEqual(result["best_shift_semitones"], 0)

    def test_harmonic_fraction_on_tone_and_noise(self):
        sr = 44100
        t = np.arange(sr) / sr
        tone = np.sin(2 * np.pi * 440 * t)
        noise = np.random.default_rng(5).normal(size=sr)
        self.assertGreater(harmonic_energy_fraction(tone, sr),
                           harmonic_energy_fraction(noise, sr))


if __name__ == "__main__":
    unittest.main()

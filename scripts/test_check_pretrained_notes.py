import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from check_pretrained_notes import correlation, reconstruct_fresh, rms_first_50ms


class PretrainedNotesTests(unittest.TestCase):
    def test_uses_only_requested_window(self):
        x = np.r_[np.ones(50), np.zeros(50)]
        self.assertEqual(rms_first_50ms(x, 1000), 1.)
        self.assertIsNone(correlation(np.zeros(100), x, 1000))

    def test_correlation(self):
        x = np.r_[np.arange(50, dtype=float), np.zeros(50)]
        self.assertAlmostEqual(correlation(x, x, 1000), 1.)

    def test_each_reconstruction_gets_its_own_model(self):
        class StatefulModel:
            sr = 1000

            def __init__(self):
                self.calls = 0

            def eval(self):
                return self

            def decode(self, z, from_forward):
                self.calls += 1
                return z + self.calls

        created = []

        def load(*args, **kwargs):
            model = StatefulModel()
            created.append(model)
            return model

        x = np.zeros(100, dtype="float32")
        with patch("check_pretrained_notes.torch.jit.load", side_effect=load), \
             patch("check_pretrained_notes.encode_deterministic", side_effect=lambda model, signal: signal):
            first, sr = reconstruct_fresh(Path("model.ts"), x)
            second, _ = reconstruct_fresh(Path("model.ts"), x)
        self.assertEqual(sr, 1000)
        self.assertEqual(len(created), 2)
        np.testing.assert_array_equal(first, second)
        self.assertEqual([model.calls for model in created], [1, 1])


if __name__ == "__main__":
    unittest.main()

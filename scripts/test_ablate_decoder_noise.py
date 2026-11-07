import unittest

import torch

from ablate_decoder_noise import disable_noise_forward


class NoiseAblationTests(unittest.TestCase):
    def test_patch_reverses_without_changing_weights(self):
        class Noise(torch.nn.Module):
            def forward(self, x):
                return x * 3
        module = Noise()
        x = torch.ones(1, 2, 8)
        with disable_noise_forward(module):
            self.assertEqual(int(module(x).sum()), 0)
        self.assertEqual(int(module(x).sum()), 48)


if __name__ == "__main__":
    unittest.main()

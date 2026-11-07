import unittest

from prepare_early_gan import NEW_BINDING, fork_config


class EarlyGanConfigTests(unittest.TestCase):
    def test_only_phase_duration_changes(self):
        original = "model.RAVE.phase_1_duration = 1000000\nmodel.RAVE.sampling_rate = 44100\n"
        self.assertEqual(fork_config(original), original.replace("1000000", "100000"))
        self.assertIn(NEW_BINDING, fork_config(original))

    def test_ambiguous_config_rejected(self):
        with self.assertRaises(ValueError):
            fork_config("model.RAVE.phase_1_duration = 100000")


if __name__ == "__main__":
    unittest.main()

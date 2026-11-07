import unittest
from types import SimpleNamespace

from check_gan_run import summarize_scalars


class GanTelemetryTests(unittest.TestCase):
    def test_threshold_and_finite(self):
        data = {tag: [SimpleNamespace(step=99, value=100),
                      SimpleNamespace(step=101, value=value)]
                for tag, value in (("adversarial", .3), ("loss_dis", .4))}
        report = summarize_scalars(data, 100)
        self.assertEqual(report["adversarial"]["first_step"], 101)
        self.assertEqual(report["loss_dis"]["count"], 1)

    def test_missing_discriminator_rejected(self):
        with self.assertRaises(ValueError):
            summarize_scalars({"adversarial": [SimpleNamespace(step=101, value=.3)]}, 100)


if __name__ == "__main__":
    unittest.main()

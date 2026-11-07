import unittest

from compare_reconstructions import compare


class ComparisonTests(unittest.TestCase):
    def test_paired_delta_and_order_independence(self):
        def row(name, sc, lsd):
            return {"source": name, "quality_unaligned": {
                "spectral_convergence_mean": sc, "log_spectral_distance_mean": lsd}}
        first = {"sr": 44100, "evaluation_set": "external_files", "step": 10,
                 "samples": [row("a", 1, 1), row("b", 2, 2)]}
        second = {"sr": 44100, "evaluation_set": "external_files", "step": 20,
                  "samples": [row("b", 3, 1), row("a", .5, .5)]}
        report = compare(first, second)
        self.assertEqual(report["spectral_convergence_mean"]["songs_improved"], 1)
        self.assertEqual(report["log_spectral_distance_mean"]["songs_improved"], 2)
        self.assertEqual(report["log_spectral_distance_mean"]["mean_after_minus_before"], -.75)

    def test_rejects_different_songs(self):
        a = {"sr": 44100, "evaluation_set": "external_files", "step": 1,
             "samples": [{"source": "a"}]}
        b = {**a, "samples": [{"source": "b"}]}
        with self.assertRaises(ValueError):
            compare(a, b)


if __name__ == "__main__":
    unittest.main()

import unittest

from prepare_dataset_v2 import split_tracks


class DatasetSplitTests(unittest.TestCase):
    def test_reproducible_and_no_song_overlap(self):
        original = [{"path": "old/a.mp3"}]
        added = [{"path": f"new/{i}.mp3"} for i in range(20)]
        train, val = split_tracks(original, added, seed=42, holdout_count=3)
        self.assertEqual((train, val), split_tracks(original, added, seed=42, holdout_count=3))
        self.assertEqual(len(train), 18)
        self.assertEqual(len(val), 3)
        self.assertTrue(set(x["path"] for x in train).isdisjoint(x["path"] for x in val))
        self.assertIn(original[0], train)

    def test_too_few_songs(self):
        with self.assertRaises(ValueError):
            split_tracks([{"path": "old/a.mp3"}], [{"path": "new/b.mp3"}], holdout_count=1)


if __name__ == "__main__":
    unittest.main()

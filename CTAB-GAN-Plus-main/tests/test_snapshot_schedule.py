"""Scheduling tests runnable with stdlib unittest, without the GPU stack."""

import json
from pathlib import Path
import unittest

from model.synthesizer.snapshot_schedule import snapshot_epochs


class SnapshotScheduleTests(unittest.TestCase):
    def test_wids_gaps_increase_through_final_epoch(self):
        epochs = snapshot_epochs(200, snapshot_schedule={"count": 12, "power": 2.0})
        self.assertEqual(epochs, [1, 3, 8, 16, 27, 42, 60, 82, 106, 134, 165, 200])
        gaps = [right - left for left, right in zip(epochs, epochs[1:])]
        self.assertEqual(gaps, sorted(gaps))
        self.assertGreater(gaps[-1], 1)

    def test_short_runs_are_bounded_unique_and_include_endpoints(self):
        for total in (1, 2, 3, 5, 12, 150, 200):
            with self.subTest(total=total):
                epochs = snapshot_epochs(total, snapshot_schedule={})
                self.assertEqual(epochs, sorted(set(epochs)))
                self.assertEqual(epochs[0], 1)
                self.assertEqual(epochs[-1], total)
                self.assertLessEqual(len(epochs), min(12, total))

    def test_fixed_interval_and_disabled_capture_remain_supported(self):
        self.assertEqual(snapshot_epochs(150, 25), [25, 50, 75, 100, 125, 150])
        self.assertEqual(snapshot_epochs(52, 25), [25, 50, 52])
        self.assertEqual(snapshot_epochs(1, 25), [1])
        self.assertEqual(snapshot_epochs(200), [])

    def test_gradual_schedule_overrides_legacy_frequency(self):
        self.assertEqual(snapshot_epochs(200, 25, {}), snapshot_epochs(200, None, {}))

    def test_invalid_inputs_fail_early(self):
        for schedule in (False, [], {"count": True}, {"count": 1}, {"count": 3.5},
                         {"power": 1}, {"power": True}, {"power": float("nan")},
                         {"power": float("inf")}, {"typo": 3}):
            with self.subTest(schedule=schedule), self.assertRaises(ValueError):
                snapshot_epochs(200, snapshot_schedule=schedule)
        for frequency in (0, -1, True, 1.5):
            with self.subTest(frequency=frequency), self.assertRaises(ValueError):
                snapshot_epochs(200, frequency)

    def test_both_ctab_configs_enable_identical_schedule(self):
        root = Path(__file__).resolve().parents[1]
        for dataset in ("mimic", "wids"):
            with self.subTest(dataset=dataset):
                config = json.loads((root / "configs" / f"{dataset}_ctabgan.json").read_text())
                generator = config["generator"]
                self.assertTrue(config["discriminator_shap"]["enabled"])
                self.assertEqual(generator["snapshot_schedule"], {"count": 12, "power": 2.0})
                self.assertIsNone(generator["snapshot_frq"])
                epochs = snapshot_epochs(generator["epochs"], snapshot_schedule=generator["snapshot_schedule"])
                self.assertEqual(len(epochs), 12)
                gaps = [b - a for a, b in zip(epochs, epochs[1:])]
                self.assertEqual(gaps, sorted(gaps))


if __name__ == "__main__":
    unittest.main()

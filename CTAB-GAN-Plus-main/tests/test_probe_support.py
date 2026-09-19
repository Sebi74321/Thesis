"""Vocabulary coverage must remain independent of held-out data fitting."""

from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from model.synthesizer.probe_support import supported_probe_rows


class ProbeSupportTests(unittest.TestCase):
    def setUp(self):
        self.encoder = SimpleNamespace(classes_=np.array(["101.0", "202.1", "empty"]))
        self.prep = SimpleNamespace(
            df=pd.DataFrame(columns=["diagnosis", "other"]),
            label_encoder_list=[{"column": "diagnosis", "label_encoder": self.encoder}],
        )

    def test_unseen_codes_are_reported_without_relabeling_or_mutation(self):
        frame = pd.DataFrame({"diagnosis": ["101.0", "1604.01", "602.18", "901.07", "202.1"],
                              "other": range(5)}, index=[7, 7, 9, 12, 13])
        original = frame.copy(deep=True)
        classes = self.encoder.classes_.copy()
        kept, report = supported_probe_rows(self.prep, frame)
        pd.testing.assert_frame_equal(frame, original)
        np.testing.assert_array_equal(classes, self.encoder.classes_)
        pd.testing.assert_frame_equal(kept, frame.iloc[[0, 4]])
        self.assertEqual(report["excluded_row_positions"], [1, 2, 3])
        self.assertEqual(report["excluded_fraction"], 0.6)
        self.assertEqual(report["unknown_categories"]["diagnosis"],
                         {"1604.01": 1, "602.18": 1, "901.07": 1})

    def test_known_missing_values_follow_training_rules(self):
        frame = pd.DataFrame({"diagnosis": [None, np.nan, " ", "empty", "101.0"], "other": range(5)})
        kept, report = supported_probe_rows(self.prep, frame)
        pd.testing.assert_frame_equal(kept, frame)
        self.assertEqual(report["excluded_rows"], 0)
        self.encoder.classes_ = np.array(["101.0"])
        _, report = supported_probe_rows(self.prep, frame)
        self.assertEqual(report["unknown_categories"]["diagnosis"], {"empty": 4})

    def test_rows_unknown_in_multiple_features_are_counted_once(self):
        self.prep.label_encoder_list.append({"column": "other", "label_encoder": SimpleNamespace(classes_=["a"])})
        frame = pd.DataFrame({"diagnosis": ["bad", "101.0", "bad"], "other": ["bad", "a", "a"]})
        _, report = supported_probe_rows(self.prep, frame)
        self.assertEqual(report["excluded_rows"], 2)
        self.assertEqual(report["supported_rows"], 1)
        self.assertEqual(report["unknown_categories"], {"diagnosis": {"bad": 2}, "other": {"bad": 1}})

    def test_missing_column_remains_a_schema_error(self):
        with self.assertRaisesRegex(ValueError, "missing columns"):
            supported_probe_rows(self.prep, pd.DataFrame({"diagnosis": ["101.0"]}))


if __name__ == "__main__":
    unittest.main()

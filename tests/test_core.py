import unittest

import numpy as np
import pandas as pd

from dt_rf import DecisionTree, RandomForest
from preprocessing import Preprocessor


class PreprocessingTests(unittest.TestCase):
    def test_transform_uses_parameters_learned_on_train(self):
        train = pd.DataFrame(
            {
                "age": [40, 41, 42],
                "totChol": [10.0, 20.0, np.nan],
                "TenYearCHD": [0, 0, 1],
            }
        )
        test = pd.DataFrame(
            {
                "age": [50, 51],
                "totChol": [100.0, np.nan],
                "TenYearCHD": [0, 1],
            }
        )
        prep = Preprocessor(
            imputer="median",
            clip_outliers=False,
            add_pulse_pressure=False,
        ).fit(train)

        transformed = prep.transform(test)

        self.assertEqual(transformed.loc[1, "totChol"], 15.0)


class TreeTests(unittest.TestCase):
    def test_max_bins_reduces_numeric_candidates(self):
        X = pd.DataFrame({"x": np.arange(100, dtype=float)})
        y = np.array([int(i >= 10) for i in range(100)])

        limited = DecisionTree(
            criterion="gini", max_depth=1, max_bins=1, random_state=0
        ).fit(X, y)
        exact = DecisionTree(
            criterion="gini", max_depth=1, max_bins=None, random_state=0
        ).fit(X, y)

        self.assertNotEqual(limited.tree_.threshold, exact.tree_.threshold)


class ForestTests(unittest.TestCase):
    def test_oob_matches_hard_voting(self):
        rng = np.random.default_rng(4)
        X = pd.DataFrame(
            rng.normal(size=(120, 4)), columns=["a", "b", "c", "d"]
        )
        y = ((X["a"] + 0.8 * X["b"] + rng.normal(size=120) * 1.1) > 0).astype(
            int
        ).to_numpy()
        forest = RandomForest(
            n_estimators=25,
            max_depth=4,
            max_features=2,
            voting="hard",
            n_jobs=1,
            random_state=17,
        ).fit(X, y)

        expected = np.full_like(forest.oob_decision_function_, np.nan)
        sums = np.zeros_like(forest.oob_decision_function_)
        counts = np.zeros(len(y))
        for tree, oob in zip(forest.estimators_, forest._oob_indices):
            votes = tree.predict(X.to_numpy()[oob])
            for j, cls in enumerate(forest.classes_):
                sums[oob, j] += (votes == cls).astype(float)
            counts[oob] += 1
        valid = counts > 0
        expected[valid] = sums[valid] / counts[valid, None]

        np.testing.assert_allclose(
            forest.oob_decision_function_[valid], expected[valid]
        )


if __name__ == "__main__":
    unittest.main()

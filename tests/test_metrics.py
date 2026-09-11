import numpy as np

from spectronet.config import CLASSES
from spectronet.evaluation.metrics import evaluate_predictions, results_table


def test_evaluate_predictions_perfect_classifier():
    n = len(CLASSES)
    y_true = np.eye(n)
    y_pred = np.eye(n)
    result = evaluate_predictions(y_true, y_pred, classes=CLASSES)
    assert result.accuracy == 1.0
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0
    np.testing.assert_allclose(np.diag(result.confusion), np.ones(n))


def test_evaluate_predictions_random_below_perfect():
    rng = np.random.RandomState(0)
    n_samples, n_classes = 200, len(CLASSES)
    y_true_idx = rng.randint(0, n_classes, size=n_samples)
    y_true = np.eye(n_classes)[y_true_idx]
    y_pred = rng.dirichlet(np.ones(n_classes), size=n_samples)
    result = evaluate_predictions(y_true, y_pred, classes=CLASSES)
    assert 0.0 <= result.accuracy <= 1.0
    assert result.confusion.shape == (n_classes, n_classes)
    assert set(result.per_class_auc.keys()) == set(CLASSES)


def test_results_table_sorted_by_f1():
    class Dummy:
        def __init__(self, f1):
            self.accuracy = self.precision = self.recall = self.f1 = f1

    results = {"a": Dummy(0.5), "b": Dummy(0.9)}
    df = results_table(results)
    assert df.iloc[0]["Model"] == "b"

import numpy as np

from tick.metrics import support_fdp, support_recall


def test_support_metrics_perfect_recovery():
    x_truth = np.array([1.0, 0.0, -2.0, 0.0])
    x = np.array([0.5, 0.0, -0.1, 0.0])

    assert support_fdp(x_truth, x) == 0.0
    assert support_recall(x_truth, x) == 1.0


def test_support_metrics_partial_recovery_and_eps():
    x_truth = np.array([0.0, 1.0, 0.0, -1.0])
    x = np.array([0.1, 0.4, 0.0, 0.01])

    # Two detected entries, one is a false positive
    assert support_fdp(x_truth, x, eps=0.05) == 0.5
    assert support_recall(x_truth, x, eps=0.05) == 0.5

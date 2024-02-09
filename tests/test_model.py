import numpy as np
import numpy.testing

from model import recombine_segment_predictions


def test_recombine1():
    preds = np.array([
        [0.5, 0.5],
        [0.5, 0.5],
        [0.75, 0.25],
        [1, 0],
        [0.25, 0.75],
        [0, 1],
        [0.125, 0.875],
        [0.5, 0.5],
        [0.25, 0.75],
    ])
    groups = np.array(["A", "B", "B", "C", "D", "D", "E", "E", "F"])
    visits, preds_comb, _ = recombine_segment_predictions(preds, groups, None)

    assert visits == ["A", "B", "C", "D", "E", "F"]
    numpy.testing.assert_array_equal(preds_comb, np.array([
        [0.5, 0.5],
        [0.625, 0.375],
        [1, 0],
        [0.125, 0.875],
        [0.3125, 0.6875],
        [0.25, 0.75],
    ]))


def test_recombine2():
    preds = np.array([
        [0.5, 0.5],
        [0.5, 0.5],
        [0.75, 0.25],
        [1, 0],
        [0.25, 0.75],
        [0, 1],
        [0.25, 0.75],
        [0.125, 0.875],
        [0.5, 0.5],
    ])
    groups = np.array(["A", "B", "B", "C", "D", "D", "E", "F", "F"])
    true_y = np.array([0, 1, 1, 0, 0, 0, 1, 1, 1])
    visits, preds_comb, true_y_comb = recombine_segment_predictions(preds, groups, true_y)

    assert visits == ["A", "B", "C", "D", "E", "F"]
    numpy.testing.assert_array_equal(preds_comb, np.array([
        [0.5, 0.5],
        [0.625, 0.375],
        [1, 0],
        [0.125, 0.875],
        [0.25, 0.75],
        [0.3125, 0.6875],
    ]))
    assert true_y_comb == [0, 1, 0, 0, 1, 1]

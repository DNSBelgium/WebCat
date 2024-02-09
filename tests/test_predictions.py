import pandas as pd
import pytest

from predictions import prediction_vote, has_majority

pred1 = pd.DataFrame({"predicted_label": ["A", "B", "C", "D", "B", "A", "C"]})
pred2 = pd.DataFrame({"predicted_label": ["A", "C", "C", "B", "A", "A", "A"]})
pred3 = pd.DataFrame({"predicted_label": ["A", "B", "B", "A", "B", "B", "C"]})


def test_prediction_vote_1():
    pred_voted = prediction_vote([pred1, pred2, pred3], False)
    labels = pred_voted["predicted_label"].tolist()
    assert labels == ["A", "B", "C", "D", "B", "A", "C"]


def test_prediction_vote_2():
    pred_voted = prediction_vote([pred1, pred2, pred3], True)
    labels = pred_voted["predicted_label"].tolist()
    assert labels == ["A", "B", "C", "B", "A", "C"]


@pytest.mark.parametrize("counter_in,out", [
    ([("A", 2), ("B", 1)], True),
    ([("A", 3), ("B", 0)], True),
    ([("A", 1), ("B", 1)], False)
])
def test_has_majority(counter_in, out):
    assert has_majority(counter_in) == out

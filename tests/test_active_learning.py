import pandas as pd

from active_learning import sort_and_select

pred = pd.DataFrame({
    "id": [1, 2, 3, 4, 5, 6, 7],
    "entropy": [0.5, 0.3, 0.7, 1.4, 0.9, 2.1, 0.2]
})


def test_sort_and_select_1():
    result = sort_and_select(pred, None)
    assert result["id"].tolist() == [6, 4, 5, 3, 1, 2, 7]


def test_sort_and_select_2():
    result = sort_and_select(pred, 3)
    assert result["id"].tolist() == [6, 4, 5]

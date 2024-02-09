import os
from tempfile import NamedTemporaryFile

import pandas as pd

import feature_extraction
import preprocess
from preprocess import get_texts_from_dataframe, preprocess_training_data, PreprocessedTrainingData

x = pd.DataFrame({
    "visit_id": ["1", "2"],
    "domain_segmented": ["test domain", "cool website"],
    "meta_text": ["Hello", "Example"],
    "body_text": ["This is a sentence.", "And another one."],
    "num1": [2, 4],
    "hosts": ["a b", "b c d"]
})
y = pd.DataFrame({
    "label": ["A", "A"]
})


def test_simple_preprocess(monkeypatch):
    monkeypatch.setattr(feature_extraction, "NUMERIC_FEATURES", ["num1"])
    monkeypatch.setattr(feature_extraction, "NB_LINKS_SVD_COMPONENTS", 1)
    monkeypatch.setattr(preprocess, "NB_EXTRA_FEATURES", 2)

    with NamedTemporaryFile("wb", delete=False) as temp:
        path = temp.name

    preprocess_training_data(x, y, 0.5, path)

    with PreprocessedTrainingData.load(path) as data:
        assert len(data.training.attention_masks()) == 1
        assert len(data.validation.attention_masks()) == 1

        assert len(data.training.token_sequences()) == 1
        assert len(data.validation.token_sequences()) == 1

        assert len(data.training.extra_features()) == 1
        assert len(data.validation.extra_features()) == 1

        assert list(data.training.y()) == [0]
        assert list(data.validation.y()) == [0]

        assert list(data.label_encoder().classes_) == ["A"]

    os.remove(path)


def test_get_texts():
    assert get_texts_from_dataframe(x) == [
        "test domain. Hello This is a sentence.",
        "cool website. Example And another one."
    ]

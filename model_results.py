from argparse import ArgumentParser
from collections import Counter
from typing import TypeVar, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import f1_score
from termcolor import colored

import utils

T = TypeVar("T")


def get_majority(seq: Sequence[T]) -> T:
    """
    Returns the most occurring item of a sequence.
    """
    c = Counter(seq)
    return c.most_common()[0][0]


def has_majority(seq: Sequence[T]) -> bool:
    """
    Checks if the most occurring item occurs more often than the second most occurring item.

    (Note: this means that "majority" here does not generally mean "over 50% of occurrences". That is only generally
    true if len(seq) == 3.)
    """
    c = Counter(seq)
    top2 = c.most_common(2)
    if len(top2) != 2:
        return True
    return top2[0][1] > top2[1][1]


def is_unanimous(seq: Sequence[T]) -> bool:
    """Checks if there is only one unique item in the sequence."""
    return len(set(seq)) == 1


class ModelPerformance:
    """Summary of model performance metrics."""
    weighted_f1_all: float  # Weighted F1 score, taking majority label as ground truth, on all samples
    weighted_f1_unanimous: float  # Only on unanimous samples
    weighted_f1_contr: float  # Only on controversial samples

    macro_f1_unanimous: float
    macro_f1_contr: float
    macro_f1_all: float

    f1_per_category_all: npt.NDArray[np.float64]
    f1_per_category_unanimous: npt.NDArray[np.float64]
    f1_per_category_contr: npt.NDArray[np.float64]

    accuracy_all: float
    accuracy_unanimous: float
    accuracy_contr: float

    included_all: float  # Fraction of samples where the model output is included in the human labels
    included_contr: float  # Same but only for samples without unanimity

    labels: list[str]

    def print(self, details: bool):
        print("--- Overall metrics ---")
        print_single_metric("Weighted F1 score:", self.weighted_f1_all)
        print_single_metric("Accuracy:", self.accuracy_all)
        print_single_metric("Macro F1 score:", self.macro_f1_all)
        print_single_metric("Model output was any of the human labels:", self.included_all)

        if details:
            print()
            print("--- F1 on individual categories for all samples (true label = majority vote) ---")
            classes = (-self.f1_per_category_all).argsort()
            for c in classes:
                print_single_metric(self.labels[c], self.f1_per_category_all[c])

        print()
        print("--- Metrics for unanimous samples ---")
        print_single_metric("Weighted F1 score:", self.weighted_f1_unanimous)
        print_single_metric("Macro F1 score:", self.macro_f1_unanimous)
        print_single_metric("Accuracy:", self.accuracy_unanimous)

        if details:
            print()
            print("--- F1 on individual categories for unanimous samples ---")
            classes = (-self.f1_per_category_unanimous).argsort()
            for c in classes:
                print_single_metric(self.labels[c], self.f1_per_category_unanimous[c])

        print()
        print("--- Metrics on controversial samples ---")
        print_single_metric("Weighted F1 score:", self.weighted_f1_contr)
        print_single_metric("Accuracy:", self.accuracy_contr)
        print_single_metric("Macro F1 score:", self.macro_f1_contr)
        print_single_metric("Model output was any of the human labels:", self.included_contr)

        if details:
            print()
            print("--- F1 on individual categories for controversial samples (true label = majority vote) ---")
            classes = (-self.f1_per_category_contr).argsort()
            for c in classes:
                print_single_metric(self.labels[c], self.f1_per_category_contr[c])


class Comparison:
    perf1: ModelPerformance
    perf2: ModelPerformance

    def __init__(self,
                 perf1: ModelPerformance,
                 perf2: ModelPerformance):
        self.perf1 = perf1
        self.perf2 = perf2

    def print(self, details: bool):
        assert self.perf1.labels == self.perf2.labels

        print("Comparing model results - positive percentage means model 2 performing better")
        print()

        print("--- Overall metrics ---")
        print_metric("Weighted F1 score:", self.perf1.weighted_f1_all, self.perf2.weighted_f1_all)
        print_metric("Accuracy:", self.perf1.accuracy_all, self.perf2.accuracy_all)
        print_metric("Macro F1 score:", self.perf1.macro_f1_all, self.perf2.macro_f1_all)
        print_metric("Model output was any of the human labels:", self.perf1.included_all, self.perf2.included_all)

        if details:
            print()
            print("--- F1 on individual categories for all samples (true label = majority vote) ---")
            classes = (-abs(self.perf2.f1_per_category_all - self.perf1.f1_per_category_all)).argsort()
            for c in classes:
                print_metric(self.perf1.labels[c],
                             self.perf1.f1_per_category_all[c], self.perf2.f1_per_category_all[c])

        print()
        print("--- Metrics for unanimous samples ---")
        print_metric("Weighted F1 score:", self.perf1.weighted_f1_unanimous, self.perf2.weighted_f1_unanimous)
        print_metric("Macro F1 score:", self.perf1.macro_f1_unanimous, self.perf2.macro_f1_unanimous)
        print_metric("Accuracy:", self.perf1.accuracy_unanimous, self.perf2.accuracy_unanimous)

        if details:
            print()
            print("--- F1 on individual categories for unanimous samples ---")
            classes = (-abs(self.perf2.f1_per_category_unanimous - self.perf1.f1_per_category_unanimous)).argsort()
            for c in classes:
                print_metric(self.perf1.labels[c],
                             self.perf1.f1_per_category_unanimous[c], self.perf2.f1_per_category_unanimous[c])

        print()
        print("--- Metrics on controversial samples ---")
        print_metric("Weighted F1 score:", self.perf1.weighted_f1_contr, self.perf2.weighted_f1_contr)
        print_metric("Accuracy:", self.perf1.accuracy_contr, self.perf2.accuracy_contr)
        print_metric("Macro F1 score:", self.perf1.macro_f1_contr, self.perf2.macro_f1_contr)
        print_metric("Model output was any of the human labels:", self.perf1.included_contr, self.perf2.included_contr)

        if details:
            print()
            print("--- F1 on individual categories for controversial samples (true label = majority vote) ---")
            classes = (-abs(self.perf2.f1_per_category_contr - self.perf1.f1_per_category_contr)).argsort()
            for c in classes:
                print_metric(self.perf1.labels[c],
                             self.perf1.f1_per_category_contr[c], self.perf2.f1_per_category_contr[c])


def compare(model1: tuple[list[list[str]], list[str]],
            model2: tuple[list[list[str]], list[str]]) -> Comparison:
    perf1 = performance(model1[0], model1[1])
    perf2 = performance(model2[0], model2[1])

    return Comparison(perf1, perf2)


def performance(real: list[list[str]], pred: list[str]) -> ModelPerformance:
    """
    Calculates model performance based on true labels (voted) and predicted labels.

    :param real: True labels, as votes -- each element of the list is a list of votes
    :param pred: Predicted labels
    :return: Performance metrics
    """
    labels = sorted(set(utils.flatten(real)).union(set(pred)))

    assert len(real) == len(pred)

    has_maj = [has_majority(x) for x in real]
    is_unan = [is_unanimous(x) for x in real]
    is_contr_with_maj = [has_majority(x) and not is_unanimous(x) for x in real]

    real_majority = [get_majority(x) for x in np.array(real, dtype=object)[has_maj]]
    pred_majority = np.array(pred)[has_maj]

    real_unanimous = [x[0] for x in np.array(real, dtype=object)[is_unan]]
    pred_unanimous = np.array(pred)[is_unan]

    real_contr_with_maj = [get_majority(x) for x in np.array(real, dtype=object)[is_contr_with_maj]]
    pred_contr_with_maj = np.array(pred)[is_contr_with_maj]

    perf = ModelPerformance()

    perf.labels = labels

    perf.weighted_f1_unanimous = f1_score(real_unanimous, pred_unanimous, average="weighted", labels=labels)
    perf.macro_f1_unanimous = f1_score(real_unanimous, pred_unanimous, average="macro", labels=labels)
    perf.accuracy_unanimous = (np.array(real_unanimous) == np.array(pred_unanimous)).sum() / len(real_unanimous)
    perf.f1_per_category_unanimous = f1_score(real_unanimous, pred_unanimous, average=None, labels=labels)

    perf.weighted_f1_all = f1_score(real_majority, pred_majority, average="weighted", labels=labels)
    perf.macro_f1_all = f1_score(real_majority, pred_majority, average="macro", labels=labels)
    perf.f1_per_category_all = f1_score(real_majority, pred_majority, average=None, labels=labels)
    perf.accuracy_all = (np.array(real_majority) == np.array(pred_majority)).sum() / len(real_majority)

    perf.weighted_f1_contr = f1_score(real_contr_with_maj, pred_contr_with_maj,
                                      average="weighted", labels=labels)
    perf.macro_f1_contr = f1_score(real_contr_with_maj, pred_contr_with_maj,
                                   average="macro", labels=labels)
    perf.f1_per_category_contr = f1_score(real_contr_with_maj, pred_contr_with_maj,
                                          average=None, labels=labels)
    perf.accuracy_contr = \
        (np.array(real_contr_with_maj) == np.array(pred_contr_with_maj)).sum() / len(real_contr_with_maj)

    included = 0
    included_contr = 0
    for i in range(len(real)):
        if pred[i] in real[i]:
            included += 1
            if not is_unanimous(real[i]):
                included_contr += 1

    perf.included_all = included / len(real)
    if len(real) != len(real_unanimous):
        perf.included_contr = included_contr / (len(real) - len(real_unanimous))
    else:
        perf.included_contr = float("NaN")

    return perf


def colored_percentage(p: float) -> str:
    """Converts float to colorized percentage, green if it's >= 0%, red if it's negative."""
    return colored(f"{p:+.2%}", "green" if p >= 0 else "red")


def percentage_transition(p1, p2) -> str:
    if p1 != -1 and p2 != -1:
        return f"({p1:+.2%} -> {p2:+.2%})"
    elif p1 != -1:
        return f"({p1:+.2%} -> N/A)"
    elif p2 != -1:
        return f"(N/A -> {p2:+.2%})"
    else:
        return f"(N/A -> N/A)"


def print_metric(text: str, p1: float, p2: float) -> None:
    print(text, colored_percentage(p2 - p1), percentage_transition(p1, p2))


def print_single_metric(text: str, p1: float) -> None:
    if p1 != -1:
        print("{} {:.2%}".format(text, p1))
    else:
        print("{} N/A".format(text))


def get_y_from_predicted(pq_path: str) -> list[str]:
    """Get labels from Parquet file with prediction results."""
    df = pd.read_parquet(pq_path)
    return df["predicted_label"].tolist()


def get_y_from_true(pq_path: str) -> list[list[str]]:
    """Get labels (voted) from Parquet file with ground truth test data."""
    df = pd.read_parquet(pq_path)
    return df["labels"].apply(list).tolist()


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--no-details", action="store_true")
    parser.add_argument("true_y")
    parser.add_argument("model1_pred")
    parser.add_argument("model2_pred", nargs="?", default=None)
    args = parser.parse_args()
    true_y = get_y_from_true(args.true_y)
    pred1 = get_y_from_predicted(args.model1_pred)
    if args.model2_pred:
        pred2 = get_y_from_predicted(args.model2_pred)
        comp = compare((true_y, pred1), (true_y, pred2))
        comp.print(not args.no_details)
    else:
        perf = performance(true_y, pred1)
        perf.print(not args.no_details)


if __name__ == "__main__":
    main()

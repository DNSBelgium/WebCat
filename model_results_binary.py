from argparse import ArgumentParser

import pandas as pd
from sklearn.preprocessing import LabelEncoder

import utils
from model_results import print_metric, print_single_metric


class ModelPerformance:
    """Summary of binary model performance metrics."""
    precision_majority_all: float
    recall_majority_all: float
    f1_majority_all: float

    precision_unanimous: float
    recall_unanimous: float
    f1_unanimous: float

    precision_majority_controversial: float
    recall_majority_controversial: float
    f1_majority_controversial: float

    def print(self):
        print("--- Unanimous + controversial: majority label taken as real label ---")
        print_single_metric("Precision:", self.precision_majority_all)
        print_single_metric("Recall:", self.recall_majority_all)
        print_single_metric("F1:", self.f1_majority_all)

        print()
        print("--- Unanimous only: unanimous label taken as real label ---")
        print_single_metric("Precision:", self.precision_unanimous)
        print_single_metric("Recall:", self.recall_unanimous)
        print_single_metric("F1:", self.f1_unanimous)

        print()
        print("--- Controversial only: majority label taken as real label ---")
        print_single_metric("Precision:", self.precision_majority_controversial)
        print_single_metric("Recall:", self.recall_majority_controversial)
        print_single_metric("F1:", self.f1_majority_controversial)


class Comparison:
    perf1: ModelPerformance
    perf2: ModelPerformance

    def __init__(self,
                 perf1: ModelPerformance,
                 perf2: ModelPerformance):
        self.perf1 = perf1
        self.perf2 = perf2

    def print(self):
        print("--- Unanimous + controversial: majority label taken as real label ---")
        print_metric("Precision:", self.perf1.precision_majority_all, self.perf2.precision_majority_all)
        print_metric("Recall:", self.perf1.recall_majority_all, self.perf2.recall_majority_all)
        print_metric("F1:", self.perf1.f1_majority_all, self.perf2.f1_majority_all)

        print()
        print("--- Unanimous only: unanimous label taken as real label ---")
        print_metric("Precision:", self.perf1.precision_unanimous, self.perf2.precision_unanimous)
        print_metric("Recall:", self.perf1.recall_unanimous, self.perf2.recall_unanimous)
        print_metric("F1:", self.perf1.f1_unanimous, self.perf2.f1_unanimous)

        print()
        print("--- Controversial only: majority label taken as real label ---")
        print_metric("Precision:", self.perf1.precision_majority_controversial,
                     self.perf2.precision_majority_controversial)
        print_metric("Recall:", self.perf1.recall_majority_controversial, self.perf2.recall_majority_controversial)
        print_metric("F1:", self.perf1.f1_majority_controversial, self.perf2.f1_majority_controversial)


def compare(model1: tuple[list[list[int]], list[bool]],
            model2: tuple[list[list[int]], list[bool]]) -> Comparison:

    perf1 = performance(model1[0], model1[1])
    perf2 = performance(model2[0], model2[1])

    return Comparison(perf1, perf2)


def performance(real: list[list[int]], pred: list[bool]) -> ModelPerformance:
    """
    Calculates binary model performance based on true labels (voted) and predicted labels.

    :param real: True labels, as votes -- each element of the list is a list of votes
    :param pred: Predicted labels
    :return: Performance metrics
    """
    tp_all = 0
    fp_all = 0
    fn_all = 0

    tp_unan = 0
    fp_unan = 0
    fn_unan = 0

    tp_contr = 0
    fp_contr = 0
    fn_contr = 0

    assert len(real) == len(pred)

    for (i, label_set) in enumerate(real):
        real_votes_positive = sum(label_set)
        predicted_label = pred[i]

        unan_positive = real_votes_positive == len(label_set)
        if real_votes_positive == 0 or unan_positive:  # Unanimous
            if predicted_label:
                if unan_positive:
                    tp_all += 1
                    tp_unan += 1
                else:
                    fp_all += 1
                    fp_unan += 1
            else:
                if unan_positive:
                    fn_all += 1
                    fn_unan += 1
        elif real_votes_positive * 2 != len(label_set):  # Controversial but a majority exists
            majority_positive = real_votes_positive * 2 > len(label_set)

            if predicted_label:
                if majority_positive:
                    tp_all += 1
                    tp_contr += 1
                else:
                    fp_all += 1
                    fp_contr += 1
            else:
                if majority_positive:
                    fn_all += 1
                    fn_contr += 1

    perf = ModelPerformance()

    perf.precision_majority_all = tp_all / (tp_all + fp_all)
    perf.recall_majority_all = tp_all / (tp_all + fn_all)
    perf.f1_majority_all = 2 * tp_all / (2 * tp_all + fp_all + fn_all)

    if tp_unan + fp_unan > 0:
        perf.precision_unanimous = tp_unan / (tp_unan + fp_unan)
        perf.recall_unanimous = tp_unan / (tp_unan + fn_unan)
        perf.f1_unanimous = 2 * tp_unan / (2 * tp_unan + fp_unan + fn_unan)
    else:
        perf.precision_unanimous = float("NaN")
        perf.recall_unanimous = float("NaN")
        perf.f1_unanimous = float("NaN")

    if tp_contr + fp_contr > 0:
        perf.precision_majority_controversial = tp_contr / (tp_contr + fp_contr)
        perf.recall_majority_controversial = tp_contr / (tp_contr + fn_contr)
        perf.f1_majority_controversial = 2 * tp_contr / (2 * tp_contr + fp_contr + fn_contr)
    else:
        perf.precision_majority_controversial = float("NaN")
        perf.recall_majority_controversial = float("NaN")
        perf.f1_majority_controversial = float("NaN")

    return perf


def get_y_from_predicted(pq_path: str) -> list[bool]:
    df = pd.read_parquet(pq_path)
    return df["decision"].tolist()


def get_y_from_true(pq_path: str) -> list[list[int]]:
    df = pd.read_parquet(pq_path)
    labels = df["labels"].apply(list).tolist()
    label_enc = LabelEncoder()
    label_enc.fit(utils.flatten(labels))
    assert len(label_enc.classes_) == 2
    labels_bin = [label_enc.transform(la).tolist() for la in labels]
    return labels_bin


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("true_y")
    parser.add_argument("model1_pred")
    parser.add_argument("model2_pred", nargs="?", default=None)
    args = parser.parse_args()
    true_y = get_y_from_true(args.true_y)
    pred1 = get_y_from_predicted(args.model1_pred)
    if args.model2_pred:
        pred2 = get_y_from_predicted(args.model2_pred)
        comp = compare((true_y, pred1), (true_y, pred2))
        comp.print()
    else:
        perf = performance(true_y, pred1)
        perf.print()


if __name__ == "__main__":
    main()

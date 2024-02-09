"""
File for PR curves and metrics for the performance of BINARY classifiers.
"""

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay, auc

from model_results_binary import get_y_from_true


def pr_curve_and_metrics(y_true_votes: list[list[int]], probabilities: npt.NDArray[np.float64]) \
        -> tuple[
            float, float, float, float, float,
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    """
    Calculates PR curve and associated metrics from true labels and the predicted probabilities.
    :param y_true_votes: True labels, a list of which the elements are lists of votes by human labelers
    :param probabilities: Predicted probabilities as a numpy array
    :return: (AUC-PR, Best F1 score, Threshold for that F1 score, Precision, Recall, PR-curve)
    """
    has_majority = [sum(x) * 2 != len(x) for x in y_true_votes]
    y_true_votes_with_majority = np.array(y_true_votes, dtype=object)[has_majority]
    probabilities = probabilities[has_majority]
    y_true = [int(sum(x) * 2 > len(x)) for x in y_true_votes_with_majority]
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    f1_scores = 2 * precision * recall / (precision + recall)
    best_index = np.nanargmax(f1_scores)
    return auc(recall, precision), f1_scores[best_index], thresholds[best_index], precision[best_index],\
        recall[best_index], \
        (precision, recall, thresholds)


def get_probabilities_from_predicted(pq_path: str) -> npt.NDArray[np.float64]:
    """
    Loads output file of predictions and returns predicted probabilities as a numpy array.
    :param pq_path: Parquet file containing predictions
    :return: numpy array with predicted probabilities
    """
    df = pd.read_parquet(pq_path)
    return df["prediction"].to_numpy()


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("true_y")
    parser.add_argument("model_pred")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    y_true_votes = get_y_from_true(args.true_y)
    probabilities = get_probabilities_from_predicted(args.model_pred)
    area, f1, thres, pr, re, curve = pr_curve_and_metrics(y_true_votes, np.array(probabilities))
    print(f"AUC-PR: {area:.4}")
    print(f"Best F1 score: {f1:.2%}")
    print(f"Threshold: {thres}")
    print(f"Precision: {pr:.2%}")
    print(f"Recall: {re:.2%}")
    if args.plot:
        disp = PrecisionRecallDisplay(precision=curve[0], recall=curve[1])
        disp.plot()
        plt.show()


if __name__ == "__main__":
    main()

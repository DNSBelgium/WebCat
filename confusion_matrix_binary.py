from argparse import ArgumentParser
from typing import Sequence

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from model_results import get_majority
from model_results_binary import get_y_from_true, get_y_from_predicted


def cf_matrix(y_true: Sequence[int], y_pred: Sequence[bool], output: str | None):
    """
    Generates a confusion matrix of a model.

    :param y_true: Sequence of true labels
    :param y_pred: Sequence of predicted labels
    :param output: Path for the output file, or None to display it instead
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sns.set(rc={"figure.figsize": (6, 5)})

    annot = [
        [
            f"TP\n{tp}",
            f"FN\n{fn}"
        ], [
            f"FP\n{fp}",
            f"TN\n{tn}"
        ]
    ]

    ax = sns.heatmap([[tp, fn], [fp, tn]], annot=annot, fmt="", xticklabels=["Pos", "Neg"],
                     yticklabels=["Pos", "Neg"])

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")

    fig = ax.get_figure()
    if output is None:
        plt.show()
    else:
        fig.savefig(output, bbox_inches="tight")
        plt.cla()


def main():
    parser = ArgumentParser()
    parser.add_argument("true_y")
    parser.add_argument("model_pred")
    parser.add_argument("--out", required=False)
    args = parser.parse_args()

    y_true_votes = [get_majority(y) for y in get_y_from_true(args.true_y)]
    y_pred = get_y_from_predicted(args.model_pred)
    cf_matrix(y_true_votes, y_pred, args.out)


if __name__ == "__main__":
    main()

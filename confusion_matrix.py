from argparse import ArgumentParser
from collections import Counter
from typing import Sequence, TypeVar

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
from sklearn.metrics import confusion_matrix

from model_results import get_y_from_true, get_y_from_predicted, get_majority

T = TypeVar("T")


def cf_matrix(y_true: Sequence[T], y_pred: Sequence[T], output: str | None, log_norm=False):
    """
    Generates a confusion matrix of a model.

    :param y_true: Sequence of true labels
    :param y_pred: Sequence of predicted labels
    :param output: Path for the output file, or None to display it instead
    :param log_norm: should the values be log-normalized
    """
    labels = [x[0] for x in Counter(y_true).most_common()]  # sort labels from most to least common (true labels)
    cf = confusion_matrix(y_true, y_pred, labels=labels, normalize="true") * 100

    sns.set(rc={"figure.figsize": (11.5, 10.27)})

    if log_norm:
        cmap = sns.color_palette("rocket", as_cmap=True)
        cmap.set_bad((0, 0, 0))
        norm = LogNorm()
    else:
        cmap = None
        norm = None
    ax = sns.heatmap(cf, annot=True, fmt=".1f", xticklabels=labels, yticklabels=labels,
                     cbar=False, norm=norm, cmap=cmap)

    for t in ax.texts:
        if float(t.get_text()) == 0.0:
            t.set_text("")

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
    parser.add_argument("--log-norm", action="store_true")
    args = parser.parse_args()

    y_true_votes = [get_majority(y) for y in get_y_from_true(args.true_y)]
    y_pred = get_y_from_predicted(args.model_pred)
    cf_matrix(y_true_votes, y_pred, args.out, args.log_norm)


if __name__ == "__main__":
    main()

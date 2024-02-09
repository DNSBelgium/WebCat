import typing
from argparse import ArgumentParser
from collections import Counter

import pandas as pd


def has_majority(counter_value: list[tuple[typing.Any, int]]) -> bool:
    if len(counter_value) == 1:
        return True
    else:
        return counter_value[0][1] > counter_value[1][1]


def prediction_vote(dfs: list[pd.DataFrame], delete_if_no_majority: bool) -> pd.DataFrame:
    """
    Combine multiple prediction results by voting.

    :param dfs: List of DataFrames with prediction results
    :param delete_if_no_majority: Should websites be deleted from the predictions if there is not just one label that
                                  has the highest number of votes.
    :return: DataFrame where the predicted_label or decision column is decided by voting
    """
    key = "predicted_label" if "predicted_label" in dfs[0] else "decision"
    class_votes: list[list[str | bool]] = [df[key].tolist() for df in dfs]
    top_choices: list[list[tuple[str | bool, int]]] = [Counter(x).most_common(2) for x in zip(*class_votes)]
    if delete_if_no_majority:
        has_maj = [has_majority(c) for c in top_choices]
        majority_votes = [c[0][0] for c in top_choices if has_majority(c)]
        df = dfs[0][pd.Series(has_maj)].reset_index(drop=True)
        df[key] = majority_votes
    else:
        df = dfs[0].copy()
        df[key] = [c[0][0] for c in top_choices]
    if key == "decision":
        df.drop("prediction", axis="columns", inplace=True)
    if "entropy" in df:
        df.drop("entropy", axis="columns", inplace=True)
    return df


def print_distribution(df: pd.DataFrame):
    c = Counter(df["predicted_label" if "predicted_label" in df else "decision"])
    total = c.total()
    for category, count in c.most_common():
        print(f"{category};{count};{count * 100 / total:.2f}%")


def main():
    parser = ArgumentParser()

    sp = parser.add_subparsers(dest="command")
    sp.required = True

    sp_combine = sp.add_parser("combine")
    sp_combine.add_argument("out")
    sp_combine.add_argument("p1")
    sp_combine.add_argument("p2")
    sp_combine.add_argument("p3")
    sp_combine.add_argument("--delete-if-no-majority", action="store_true")

    sp_distr = sp.add_parser("distribution")
    sp_distr.add_argument("predictions")

    args = parser.parse_args()

    if args.command == "combine":
        df1 = pd.read_parquet(args.p1)
        df2 = pd.read_parquet(args.p2)
        df3 = pd.read_parquet(args.p3)
        combined = prediction_vote([df1, df2, df3], args.delete_if_no_majority)
        combined.to_parquet(args.out)
    elif args.command == "distribution":
        df = pd.read_parquet(args.predictions)
        print_distribution(df)


if __name__ == "__main__":
    main()

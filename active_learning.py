import argparse

import pandas as pd


def sort_and_select(df: pd.DataFrame, n: int | None) -> pd.DataFrame:
    """
    Sorts DataFrame with prediction results by entropy and returns top N rows.

    :param df: DataFrame with prediction results, entropies in the "entropy" column
    :param n: Number of rows to select from the top, None to return all rows
    :return: Sorted (and potentially limited) DataFrame
    """
    df_sorted = df.sort_values("entropy", ascending=False).reset_index(drop=True)
    if n is None:
        return df_sorted
    else:
        return df_sorted.iloc[:n]


def main():
    parser = argparse.ArgumentParser()
    sp = parser.add_subparsers(dest="command")
    sp.required = True

    sp_sort = sp.add_parser("sort")
    sp_sort.add_argument("in")
    sp_sort.add_argument("out")
    sp_sort.add_argument("--take", type=int, required=False)

    args = parser.parse_args()

    if args.command == "sort":
        df = pd.read_parquet(getattr(args, "in"))
        df = sort_and_select(df, args.take)
        df.to_parquet(args.out)


if __name__ == "__main__":
    main()

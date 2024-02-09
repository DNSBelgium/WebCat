import argparse
import re
from typing import List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import utils
from config import FETCH_TEST_SET_IGNORE_LABELS
from wordsegment import wordsegment


def list_to_str(x):
    """
    Joins the elements of a list with spaces.

    :param x: The list
    :return: A string with the elements of the list, separated by spaces
    """
    return " ".join([str(elem) for elem in x])


def count_tag(html_struct: str, tag: str) -> int:
    """
    In a flattened string representation of a HTML structure, counts the occurrence of a specific HTML tag (as
    represented by a specific character).

    :param html_struct: Flattened string representation of the HTML structure
    :param tag: Character identifying a HTML tag
    :return: The number of occurrences of the tag in the structure
    """
    return html_struct.count(tag)


__SEGMENTER = wordsegment.Segmenter()
__SEGMENTER_LOADED = False


def get_segmenter():
    """
    Constructs an instance from wordsegment.Segmenter, or returns the existing instance.

    :return: A Segmenter
    """
    global __SEGMENTER_LOADED
    if not __SEGMENTER_LOADED:
        __SEGMENTER.load()
        __SEGMENTER_LOADED = True
    return __SEGMENTER


def segment_domain_name(domain: str):
    """
    Segments a domain name by removing the TLD and splitting it into words using wordsegmenter.

    :param domain: The domain name
    :return: The domain name split in words (as a string, with spaces between words)
    """
    return " ".join(get_segmenter().segment(utils.remove_tld_from_domain_name(domain)))


def fix_dataframe_from_mercator(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fixes a DataFrame obtained from Mercator.

    Integer columns can be transformed into floats because of missing values being treated as NaN (np.nan).
    The "Int64" type is a nullable int64 type, so it stores missing values as pd.NA. This is preserved
    when storing the data as a parquet file.

    Also deletes a few columns that are unneeded and may cause typing issues when storing the result as Parquet.
    :param df: The DataFrame as obtained from Mercator
    :return: The fixed DataFrame
    """
    int64_columns = ["nb_facebook_deep_links", "nb_facebook_shallow_links", "nb_linkedin_deep_links",
                     "nb_linkedin_shallow_links", "nb_twitter_deep_links", "nb_twitter_shallow_links",
                     "nb_currency_names", "nb_distinct_currencies", "nb_youtube_deep_links",
                     "nb_youtube_shallow_links", "nb_vimeo_deep_links", "nb_vimeo_shallow_links",
                     "nb_distinct_words_in_title", "distance_title_initial_dn", "longest_subsequence_title_initial_dn",
                     "distance_title_final_dn", "longest_subsequence_title_final_dn"]

    for col in int64_columns:
        df[col] = df[col].astype("Int64")

    return df.drop(labels=["converted_domain_names", "facebook_links", "vimeo_links", "twitter_links", "youtube_links",
                           "linkedin_links"], axis="columns", errors="ignore")


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs basic preprocessing on a DataFrame with website samples, by converting the visit_id column to a string,
    stringifying the external_hosts column (a list) into a hosts column, and segmenting the domain name.

    :param df: DataFrame with website samples
    :return: Preprocessed DataFrame
    """
    df["visit_id"] = df["visit_id"].astype(str)
    df["hosts"] = df["external_hosts"].apply(list_to_str)
    df["domain_segmented"] = df["domain_name"].apply(segment_domain_name)

    return df.drop(labels=["external_hosts"], axis="columns")


def create_parquet_xy_from_visits_tables(table_names: List[str], path_out_x: str, path_out_y: str):
    """
    Based on Mercator tables with (visit_id, label) columns, create Parquet files for the X and Y values.

    :param table_names: Mercator table names
    :param path_out_x: Output path for the X data
    :param path_out_y: Output path for the Y data
    """
    assert all(map(lambda name: re.match(r"^[\w_.]+$", name), table_names))

    queries = map(lambda name: ("(" + """select
        h.*,
        v.label
        from {} v
            join feature_extraction.html_features h on h.visit_id=v.visit_id""" + ")")
                  .format(name),
                  table_names)
    query = " union all ".join(queries)

    pqwriter_x = None
    pqwriter_y = None

    for chunk in pd.read_sql_query(query, utils.make_mercator_engine(True), chunksize=1000):
        chunk = preprocess(fix_dataframe_from_mercator(chunk))
        y = chunk["label"].to_frame()
        chunk = chunk.drop(labels="label", axis="columns")

        # noinspection PyArgumentList
        table_x = pa.Table.from_pandas(chunk)
        # noinspection PyArgumentList
        table_y = pa.Table.from_pandas(y)

        if pqwriter_x is None:
            pqwriter_x = pq.ParquetWriter(path_out_x, table_x.schema)
        pqwriter_x.write_table(table_x)

        if pqwriter_y is None:
            pqwriter_y = pq.ParquetWriter(path_out_y, table_y.schema)
        pqwriter_y.write_table(table_y)

    if pqwriter_x:
        pqwriter_x.close()

    if pqwriter_y:
        pqwriter_y.close()


def create_parquet_x_from_visits_tables(table_names: List[str], path_out: str):
    """
    Based on Mercator tables with a visit_id column, create a Parquet file for the X values.

    :param table_names: Mercator table names
    :param path_out: Output path for the X data
    """
    assert all(map(lambda name: re.match(r"^[\w_.]+$", name), table_names))

    queries = map(lambda name: ("(" + """select
        h.*
        from {} v
            join feature_extraction.html_features h on h.visit_id=v.visit_id""" + ")")
                  .format(name),
                  table_names)
    query = " union all ".join(queries)

    pqwriter = None

    for chunk in pd.read_sql_query(query, utils.make_mercator_engine(True), chunksize=1000):
        chunk = preprocess(fix_dataframe_from_mercator(chunk))

        # noinspection PyArgumentList
        table = pa.Table.from_pandas(chunk)

        if pqwriter is None:
            pqwriter = pq.ParquetWriter(path_out, table.schema)
        pqwriter.write_table(table)

    if pqwriter:
        pqwriter.close()


def test_set_samples(table_names: list[str]) -> pd.DataFrame:
    """
    Get samples for the test set from the given Mercator table names. Does not group websites by visit ID, so
    multiple labels for the same website are represented as multiple rows.
    :param table_names: Mercator table names
    :return: DataFrame with test set samples
    """
    assert all(map(lambda name: re.match(r"^[\w_.]+$", name), table_names))

    mercator_cnx = utils.make_mercator_engine()

    queries = map(lambda name: ("(" + """select
        h.*,
        v.label
        from {} v
            join feature_extraction.html_features h on h.visit_id=v.visit_id""" + ")")
                  .format(name),
                  table_names)
    query = " union all ".join(queries)

    return pd.read_sql_query(query, mercator_cnx)


def fetch_test_set(table_names: list[str], out_x: str, out_y: str):
    """
    Get samples for the test set, group them by visit ID (so one row is one website), and store the result in Parquet
    files.

    :param table_names: Mercator table names
    :param out_x: Output path for the X data
    :param out_y: Output path for the Y data
    """
    ar_samples = test_set_samples(table_names)
    g = ar_samples.groupby("visit_id")
    agg = g.agg("first")
    agg.update(g.agg({"label": list}))
    x_test = preprocess(fix_dataframe_from_mercator(agg.reset_index()))

    labels = []

    remove_indices = []

    for i in range(agg.shape[0]):
        sample = agg.iloc[i]

        sample_labels = sample["label"]
        sample_labels = [sl for sl in sample_labels
                         if sl not in FETCH_TEST_SET_IGNORE_LABELS]
        if len(sample_labels) == 0:
            remove_indices.append(i)
            continue  # Classifier will never predict a label of this sample

        labels.append(sample_labels)

    x_test = x_test.drop(remove_indices).reset_index(drop=True)
    x_test.to_parquet(out_x)

    y_test = pd.DataFrame({"labels": labels})
    y_test.to_parquet(out_y)


def main() -> None:
    utils.load_environment()

    parser = argparse.ArgumentParser()
    sp = parser.add_subparsers(dest="command")
    sp.required = True

    sp_train = sp.add_parser("train")
    sp_train.add_argument("out_x")
    sp_train.add_argument("out_y")
    sp_train.add_argument("table_names", nargs="+")

    sp_test = sp.add_parser("test")
    sp_test.add_argument("out_x")
    sp_test.add_argument("out_y")
    sp_test.add_argument("table_names", nargs="+")

    sp_parquet = sp.add_parser("predict")
    sp_parquet.add_argument("out")
    sp_parquet.add_argument("table_names", nargs="+")

    args = parser.parse_args()

    if args.command == "train":
        create_parquet_xy_from_visits_tables(args.table_names, args.out_x, args.out_y)
    elif args.command == "test":
        fetch_test_set(args.table_names, args.out_x, args.out_y)
    elif args.command == "predict":
        create_parquet_x_from_visits_tables(args.table_names, args.out)


if __name__ == "__main__":
    main()

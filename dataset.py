import argparse
import pyarrow as pa
import pyarrow.parquet as pq

import config

REQUIRED_STRING_COLUMNS = ["visit_id", "domain_name", "body_text", "meta_text"]


def checkx_parquet(path: str) -> None:
    pqf = pq.ParquetFile(path)
    print(f"Number of rows: {pqf.metadata.num_rows}")
    print(f"Number of columns: {pqf.metadata.num_columns}")

    schema = pqf.schema_arrow
    for column in REQUIRED_STRING_COLUMNS:
        if column not in schema.names:
            raise ValueError(f"Missing required column: {column!r}")

        column_type = schema.field(column).type
        if not (pa.types.is_string(column_type) or pa.types.is_large_string(column_type)):
            raise TypeError(f"Column {column!r} must be a string column, got {column_type}")

    if "external_hosts" not in schema.names:
        raise ValueError("Missing required column: 'external_hosts'")

    external_hosts_type = schema.field("external_hosts").type
    if not (
        (pa.types.is_list(external_hosts_type) or pa.types.is_large_list(external_hosts_type))
        and (
            pa.types.is_string(external_hosts_type.value_type)
            or pa.types.is_large_string(external_hosts_type.value_type)
        )
    ):
        raise TypeError(f"Column 'external_hosts' must be a list of strings, got {external_hosts_type}")

    for column in config.NUMERIC_FEATURES:
        if column not in schema.names:
            raise ValueError(f"Missing required column: {column!r}")

        column_type = schema.field(column).type
        if not (pa.types.is_integer(column_type) or pa.types.is_floating(column_type)):
            raise TypeError(f"Column {column!r} must be a numerical column, got {column_type}")

    print("No issues identified.")


def checky_parquet_train(path: str) -> None:
    pqf = pq.ParquetFile(path)
    print(f"Number of rows: {pqf.metadata.num_rows}")
    print(f"Number of columns: {pqf.metadata.num_columns}")

    schema = pqf.schema_arrow
    if "label" not in schema.names:
        raise ValueError("Missing required column: 'label'")

    label_type = schema.field("label").type
    if not (pa.types.is_string(label_type) or pa.types.is_large_string(label_type)):
        raise TypeError(f"Column 'label' must be a string column, got {label_type}")

    print("No issues identified.")


def checky_parquet_test(path: str) -> None:
    pqf = pq.ParquetFile(path)
    print(f"Number of rows: {pqf.metadata.num_rows}")
    print(f"Number of columns: {pqf.metadata.num_columns}")

    schema = pqf.schema_arrow
    if "labels" not in schema.names:
        raise ValueError("Missing required column: 'labels'")

    labels_type = schema.field("labels").type
    if not (
        (pa.types.is_list(labels_type) or pa.types.is_large_list(labels_type))
        and (pa.types.is_string(labels_type.value_type) or pa.types.is_large_string(labels_type.value_type))
    ):
        raise TypeError(f"Column 'labels' must be a list of strings, got {labels_type}")

    print("No issues identified.")


def main() -> None:
    parser = argparse.ArgumentParser()
    sp = parser.add_subparsers(dest="command")
    sp.required = True

    sp_checkx = sp.add_parser("checkx")
    sp_checkx.add_argument("in_x")

    sp_checky = sp.add_parser("checky")
    checky_group = sp_checky.add_mutually_exclusive_group(required=True)
    checky_group.add_argument("--train", action="store_true")
    checky_group.add_argument("--test", action="store_true")
    sp_checky.add_argument("in_y")

    args = parser.parse_args()

    if args.command == "checkx":
        path = args.in_x
        checkx_parquet(path)
    elif args.command == "checky":
        path = args.in_y
        if args.train:
            checky_parquet_train(path)
        elif args.test:
            checky_parquet_test(path)

if __name__ == "__main__":
    main()
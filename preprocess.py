import argparse
import os.path
import pickle

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow.parquet as pq
import torch
import transformers.utils.logging
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

import utils
from config import DEVICE, PRETRAINED_MODEL, NB_EXTRA_FEATURES
from feature_extraction import column_transformer
from utils import chunks
from windowize import token_sequence_to_windows
from wordsegment import wordsegment


class PreprocessedInputs(Dataset):
    """
    Class that manages an HDF5 file with preprocessed inputs. Preprocessed here means that the input text is tokenized
    and that numerical features are extracted from the web page.
    """
    h5f: h5py.File
    _directory: str

    def __init__(self, h5f: h5py.File, hdf5_inner_directory: str = "/"):
        """
        Initializes preprocessed inputs manager.

        :param h5f: HDF5 file containing preprocessed inputs
        :param hdf5_inner_directory: Inner directory in the HDF5 file that contain the datasets with preprocessed inputs
        """
        self.h5f = h5f
        self._directory = hdf5_inner_directory

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.h5f.close()

    def token_sequences(self) -> h5py.Dataset:
        return self.h5f[os.path.join(self._directory, "token_sequences")]

    def attention_masks(self) -> h5py.Dataset:
        return self.h5f[os.path.join(self._directory, "attention_masks")]

    def extra_features(self) -> h5py.Dataset:
        return self.h5f[os.path.join(self._directory, "extra_features")]

    def y(self) -> h5py.Dataset:
        y = os.path.join(self._directory, "y")
        return None if y not in self.h5f else self.h5f[y]

    def visit_ids(self) -> h5py.Dataset:
        return self.h5f[os.path.join(self._directory, "visit_ids")].asstr()

    def visit_ids_raw(self) -> h5py.Dataset:
        return self.h5f[os.path.join(self._directory, "visit_ids")]

    def __len__(self):
        return self.token_sequences().shape[0]

    def __getitem__(self, item):  # Allows the usage of PreprocessedInputs in a torch DataLoader
        item_dict = {
            "input_ids": torch.from_numpy(self.token_sequences()[item]),
            "extra_data": torch.from_numpy(self.extra_features()[item].astype(np.float32)),
            "attention_mask": torch.from_numpy(self.attention_masks()[item]),
            "visit_ids": self.visit_ids()[item]
        }
        if self.y() is not None:
            if isinstance(item, int):
                item_dict["labels"] = self.y()[item]
            else:
                item_dict["labels"] = torch.from_numpy(self.y()[item])
        return item_dict

    @staticmethod
    def create_new(path: str, website_count: int, include_y: bool) -> "PreprocessedInputs":
        """
        Creates new HDF5 file with preprocessed inputs and returns managing PreprocessedInputs instance.

        :param path: Where to store the file
        :param website_count: Number of websites that will be stored in the file
        :param include_y: Should y values (true labels) be stored
        """
        h5f = h5py.File(path, "w")
        PreprocessedInputs.init_empty(h5f, website_count, include_y, "/")
        return PreprocessedInputs(h5f, "/")

    @staticmethod
    def init_empty(h5f: h5py.File, website_count: int, include_y: bool, directory="/") -> None:
        """
        Initialize empty datasets in specified inner directory of the HDF5 file.

        :param h5f: HDF5 file
        :param website_count: Number of websites that will be stored in the datasets
        :param include_y: Should y values (true labels) be stored
        :param directory: Inner directory of the HDF5 file to create datasets in
        """
        h5f.create_dataset(os.path.join(directory, "token_sequences"), dtype="i8", shape=(0, 512),
                           maxshape=(website_count * 2, 512))
        h5f.create_dataset(os.path.join(directory, "attention_masks"), dtype="i8", shape=(0, 512),
                           maxshape=(website_count * 2, 512))
        h5f.create_dataset(os.path.join(directory, "extra_features"), dtype="f", shape=(0, NB_EXTRA_FEATURES),
                           maxshape=(website_count * 2, NB_EXTRA_FEATURES))
        # noinspection PyUnresolvedReferences
        h5f.create_dataset(os.path.join(directory, "visit_ids"), dtype=h5py.string_dtype(encoding="utf-8"), shape=(0,),
                           maxshape=(website_count * 2,))
        if include_y:
            h5f.create_dataset(os.path.join(directory, "y"), dtype="i8", shape=(0,),
                               maxshape=(website_count * 2,))

    @staticmethod
    def load(path: str) -> "PreprocessedInputs":
        h5f = h5py.File(path, "r")
        return PreprocessedInputs(h5f, "/")


class PreprocessedTrainingData:
    """
    Preprocessed training data, consisting of preprocessed inputs of the training set, preprocessed inputs of the
    validation set, a LabelEncoder and a fitted ColumnTransformer.
    """
    h5f: h5py.File
    training: PreprocessedInputs
    validation: PreprocessedInputs

    def __init__(self, h5f: h5py.File):
        """
        Initializes preprocessed training data from HDF5 file.
        """
        self.h5f = h5f
        self.training = PreprocessedInputs(h5f, "/train")
        self.validation = PreprocessedInputs(h5f, "/val")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.h5f.close()

    @staticmethod
    def create_new(path: str, train_website_count: int, val_website_count: int) -> "PreprocessedTrainingData":
        """
        Creates new HDF5 file for preprocessed training data and returns managing PreprocessedTrainingData instance.

        :param path: Where to store the file
        :param train_website_count: Number of websites in the training set
        :param val_website_count: Number of websites in the validation set
        """
        h5f = h5py.File(path, "w")
        PreprocessedInputs.init_empty(h5f, train_website_count, True, "/train")
        PreprocessedInputs.init_empty(h5f, val_website_count, True, "/val")
        return PreprocessedTrainingData(h5f)

    def col_transformer(self) -> ColumnTransformer:
        return pickle.loads(self.h5f["col_transformer"][()])

    def label_encoder(self) -> LabelEncoder:
        label_enc = LabelEncoder()
        label_enc.classes_ = self.h5f["label_encoder"].asstr()[()]
        return label_enc

    @staticmethod
    def load(path: str) -> "PreprocessedTrainingData":
        h5f = h5py.File(path, "r")
        return PreprocessedTrainingData(h5f)


def get_texts_from_dataframe(df: pd.DataFrame) -> list[str]:
    """
    Obtain website text from a DataFrame of website samples. The resulting text is a concatenation of segmented domain
    name, meta text, title, and body text.
    :param df: DataFrame of website samples
    :return: List with website texts, one element per website
    """
    return (df["domain_segmented"] + ". " + df["meta_text"] + " " + df["body_text"]).tolist()


def _add_col_transformer_to_file(h5f: h5py.File, col_transformer: ColumnTransformer, directory: str):
    h5f.create_dataset(
        os.path.join(directory, "col_transformer"),
        data=np.array(pickle.dumps(col_transformer))
    )


def _add_label_encoder_to_file(h5f: h5py.File, label_enc: LabelEncoder, directory: str):
    h5f.create_dataset(
        os.path.join(directory, "label_encoder"),
        data=label_enc.classes_
    )


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


def list_to_str(x):
    """
    Joins the elements of a list with spaces.

    :param x: The list
    :return: A string with the elements of the list, separated by spaces
    """
    return " ".join([str(elem) for elem in x])


def segment_domain_name(domain: str):
    """
    Segments a domain name by removing the TLD and splitting it into words using wordsegmenter.

    :param domain: The domain name
    :return: The domain name split in words (as a string, with spaces between words)
    """
    return " ".join(get_segmenter().segment(utils.remove_tld_from_domain_name(domain)))


def preprocess_basic(df: pd.DataFrame) -> pd.DataFrame:
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


def preprocess_training_data(x: pd.DataFrame, y: pd.DataFrame, validation_size: float, path_out: str) -> None:
    """
    Preprocess training data from X and Y dataframes that can be obtained using dataset.py.

    :param x: DataFrame with X values
    :param y: DataFrame with Y values
    :param validation_size: Fraction of data used for the validation set.
    :param path_out: Output path of HDF5 file with preprocessed training data.
                     Can later be loaded using PreprocessedTrainingData.
    """
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL, use_fast=True)

    x = preprocess_basic(x)

    col_transformer, transformed_features = column_transformer(x)

    label_enc = LabelEncoder()
    enc_y = label_enc.fit_transform(y["label"])

    split = StratifiedShuffleSplit(n_splits=1, test_size=validation_size, random_state=42)
    train_index, val_index = next(split.split(x, enc_y))

    train_df = x.iloc[train_index]
    val_df = x.iloc[val_index]

    with PreprocessedTrainingData.create_new(path_out, len(train_index), len(val_index)) as preprocessed:
        preprocess_inner(tokenizer,
                         get_texts_from_dataframe(train_df),
                         transformed_features[train_index],
                         enc_y[train_index],
                         train_df["visit_id"].to_numpy(),
                         preprocessed.training)
        preprocess_inner(tokenizer,
                         get_texts_from_dataframe(val_df),
                         transformed_features[val_index],
                         enc_y[val_index],
                         val_df["visit_id"].to_numpy(),
                         preprocessed.validation)

        _add_col_transformer_to_file(preprocessed.h5f, col_transformer, "/")
        _add_label_encoder_to_file(preprocessed.h5f, label_enc, "/")


def preprocess_x(pqf: pq.ParquetFile, col_transformer: ColumnTransformer, path_out: str) -> None:
    """
    Preprocess data from X dataframes that can be obtained using dataset.py.

    :param pqf: Parquet file with X dataframe
    :param col_transformer: Fitted ColumnTransformer
    :param path_out: Output path of HDF5 file with preprocessed input data.
                     Can later be loaded using PreprocessedInputs.
    """
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL, use_fast=True)

    with PreprocessedInputs.create_new(path_out, pqf.metadata.num_rows, False) as preprocessed:
        for batch in pqf.iter_batches(batch_size=1000):
            df = preprocess_basic(batch.to_pandas())
            transformed_features = col_transformer.transform(df)
            preprocess_inner(tokenizer, get_texts_from_dataframe(df), transformed_features, None,
                             df["visit_id"].to_numpy(), preprocessed)


def preprocess_inner(tokenizer: PreTrainedTokenizerBase,
                     texts: list[str],
                     transformed_features: npt.NDArray[np.float64],
                     y: npt.NDArray[np.int64] | None,
                     visit_ids: npt.NDArray,
                     preprocessed: PreprocessedInputs):
    """
    Helper function for preprocessing that tokenizes the input text and stores everything in the HDF5 file managed by
    the given PreprocessedInputs instance.

    :param tokenizer: Language model tokenizer
    :param texts: List of input texts
    :param transformed_features: Array of numerical features
    :param y: Optionally, true labels (as integers, so encoded by a LabelEncoder)
    :param visit_ids: Array of visit IDs of the website
    :param preprocessed: PreprocessedInputs instance where the results of the preprocessing will be saved
    """
    ds_tokens = preprocessed.token_sequences()
    ds_attention = preprocessed.attention_masks()
    ds_features = preprocessed.extra_features()
    ds_y = preprocessed.y()
    ds_visit_ids = preprocessed.visit_ids_raw()

    saved_verbosity = transformers.utils.logging.get_verbosity()
    transformers.utils.logging.set_verbosity_error()
    # To suppress warning about token indices sequence length being larger than 512.
    # token_sequence_to_windows deals with that.

    chunk_size = 1000
    for index_chunk, text_chunk in enumerate(chunks(texts, chunk_size)):
        # Process text samples in chunks to not run out of memory
        # and extend the HDF5 datasets after processing each chunk.

        encoded_strings = tokenizer.batch_encode_plus(
            text_chunk,
            add_special_tokens=True,
            return_attention_mask=False,
            padding=False,
            truncation=False,
        )["input_ids"]

        windowed_encoded = [token_sequence_to_windows(t) for t in encoded_strings]

        groups = []

        tokens = []
        attention_masks = []

        for index_sample, (windows_tokens, windows_attention) in enumerate(windowed_encoded):
            # windows_tokens and windows_attention are lists of either 1 or 2 items.
            # The items of windows_tokens are lists of 512 tokens, to be used as XLM-R inputs.
            # The items of windows_attention are the attention masks, also as lists of 512 elements.
            assert len(windows_tokens) == len(windows_attention)

            groups.extend([index_chunk * chunk_size + index_sample] * len(windows_tokens))
            tokens.extend(windows_tokens)
            attention_masks.extend(windows_attention)
        old_rows = ds_tokens.shape[0]
        new_rows = old_rows + len(tokens)

        ds_tokens.resize((new_rows, ds_tokens.shape[1]))
        ds_tokens[old_rows:, :] = np.array(tokens)

        ds_attention.resize((new_rows, ds_attention.shape[1]))
        ds_attention[old_rows:, :] = np.array(attention_masks)

        ds_features.resize((new_rows, ds_features.shape[1]))
        ds_features[old_rows:, :] = transformed_features[groups]

        ds_visit_ids.resize((new_rows,))
        ds_visit_ids[old_rows:] = visit_ids[groups]

        if ds_y is not None and y is not None:
            ds_y.resize((new_rows,))
            ds_y[old_rows:] = y[groups]

    transformers.utils.logging.set_verbosity(saved_verbosity)  # Restore logging level to before this function call.


def main() -> None:
    parser = argparse.ArgumentParser()
    sp = parser.add_subparsers(dest="command")
    sp.required = True

    sp_train = sp.add_parser("train")
    sp_train.add_argument("in_x")
    sp_train.add_argument("in_y")
    sp_train.add_argument("out")
    sp_train.add_argument("--split", type=float, default=0.15)

    sp_predict = sp.add_parser("predict")
    sp_predict.add_argument("in_x")
    sp_predict.add_argument("model")
    sp_predict.add_argument("out")

    args = parser.parse_args()

    if args.command == "train":
        df_x = pd.read_parquet(args.in_x)
        df_y = pd.read_parquet(args.in_y)
        print("Starting preprocessing.")
        preprocess_training_data(df_x, df_y, args.split, args.out)
    elif args.command == "predict":
        col_transf = torch.load(args.model, map_location=DEVICE)[0]
        with pq.ParquetFile(args.in_x) as pqf:
            print("Starting preprocessing.")
            preprocess_x(pqf, col_transf, args.out)


if __name__ == "__main__":
    main()

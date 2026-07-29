import os
from typing import TypeVar

import numpy as np
import torch

from dotenv import load_dotenv

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

import config


def load_environment():
    """
    Loads environment variables from the .env file (or another file as specified in the config).
    """
    load_dotenv(config.ENVIRONMENT_PATH)


def chunks(lst, n):
    """
    Splits a list into chunks.

    :param lst: The list of split
    :param n: The chunk size
    :return: A generator for chunks
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def move_to(obj, device):
    """
    Moves tensors, or collections thereof, to a given device.
    Does nothing for items that cannot be moved.

    :param obj: The tensor or tensor collection to move
    :param device: The device to move the tensors to.
    """
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to(v, device) for (k, v) in obj.items()}
    elif isinstance(obj, list):
        return [move_to(v, device) for v in obj]
    elif isinstance(obj, tuple):
        return (move_to(v, device) for v in obj)
    else:
        pass


def remove_tld_from_domain_name(domain_name: str):
    """
    Removes the top level domain from a given domain name. Does not remove public suffixes beyond TLDs.
    :param domain_name: The domain name
    :return: The domain name without its TLD
    """
    return domain_name[:domain_name.rfind(".")]


T = TypeVar("T")


def flatten(lst: list[list[T]]) -> list[T]:
    """
    Flattens a list of lists into one list. Warning: flattens only one level!
    :param lst: A list of lists to flatten
    :return: The flattened list (flattened by one level)
    """
    return [item for sublist in lst for item in sublist]


def torch_load(path: str, map_location: torch.device | None = None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}.")
    with torch.serialization.safe_globals([
        ColumnTransformer,
        TruncatedSVD,
        TfidfVectorizer,
        TfidfTransformer,
        SimpleImputer,
        Pipeline,
        StandardScaler,
        np.float64,
        np.core.multiarray._reconstruct, # type: ignore
        np.core.multiarray.scalar, # type: ignore
        np.ndarray,
        np.dtype,
        np.dtypes.ObjectDType,
        np.dtypes.Float64DType,
        slice,
        LabelEncoder,
    ]):
        return torch.load(path, map_location=map_location)

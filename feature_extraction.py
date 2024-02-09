import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import NB_LINKS_SVD_COMPONENTS, NUMERIC_FEATURES


def column_transformer(orig: pd.DataFrame) -> tuple[ColumnTransformer, npt.NDArray[np.float64]]:
    """
    Creates ColumnTransformer and extracts numerical features and external hosts SVD from DataFrame with website
    samples.

    :param orig: DataFrame with website samples
    :return: (fitted ColumnTransformer, numpy array with extracted features)
    """
    transformer = ColumnTransformer(
        remainder="drop",
        transformers=[("num", Pipeline([
            ("num_imput", SimpleImputer(missing_values=np.nan, strategy="median")),
            ("num_scale", StandardScaler())
        ]), NUMERIC_FEATURES),
                      ("hosts", Pipeline([
                          ("hosts_tfidf",
                           TfidfVectorizer(analyzer="word", strip_accents="unicode", token_pattern=r"\S+")),
                          ("hosts_best", TruncatedSVD(n_components=NB_LINKS_SVD_COMPONENTS)),
                          ("hosts_scaler", StandardScaler())
                      ]), "hosts")]
    )

    num_feat_transformed = transformer.fit_transform(orig)

    return transformer, num_feat_transformed

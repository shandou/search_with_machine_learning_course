"""Query text normalization utility module
"""
from __future__ import annotations

import multiprocessing
import re
from typing import Generator, List

import nltk
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from unidecode import unidecode

from utils import logger
from utils.constants import COLNAMES

# Number of processors to be used for embarrassingly parallel jobs
N_PROCESSORS: int = multiprocessing.cpu_count() - 1


class TextNormalizer:
    """Multi-step pipeline for normalizing input text

    Attributes
    ----------
    text: str
        | Input string to be passed through the text normalizer pipeline

    Methods
    -------
    strip_accents
        | Removes accents from characters (e.g., Â -> A)
    remove_non_word
        | Strip away any non-word characters (i.e., [^a-zA-Z0-9_])
    tokenize
        | Tokenize input string into a list of word unigrams
    stemmer
        | Transform word tokens to root forms via Porter stemmer
    done -> str
        | Joins normalized tokens into normalized output string

    Example usages
    ---------------
    >>> df_data[COLNAME_PRODUCT] = df_data[COLNAME_PRODUCT].apply(
    ...     lambda x: TextNormalizer(x.lower())
    ...     .strip_accents()
    ...     .remove_non_word()
    ...     .tokenize()
    ...     .stemmer()
    ...     .done()
    ... )

    """

    def __init__(self, text: str) -> None:
        self.text: str = text
        self.tokens: List[str] = []

    def strip_accents(self) -> TextNormalizer:
        self.text = unidecode(self.text)
        return self

    def remove_non_word(self) -> TextNormalizer:
        self.text = re.sub(r"\W", " ", self.text)
        return self

    def tokenize(self) -> TextNormalizer:
        self.tokens = self.text.split()
        return self

    def stemmer(self) -> TextNormalizer:
        self.tokens = [
            nltk.stem.PorterStemmer().stem(token) for token in self.tokens
        ]
        return self

    def done(self) -> str:
        return " ".join(self.tokens)


def normalize_query(df_query: pd.DataFrame) -> pd.DataFrame:
    df_query[COLNAMES.QUERY] = df_query[COLNAMES.QUERY].apply(
        lambda x: TextNormalizer(x.lower())
        .strip_accents()
        .remove_non_word()
        .tokenize()
        .stemmer()
        .done()
    )
    return df_query


def chunckenize_input_df(
    df_whole, n_chunks: int = N_PROCESSORS
) -> Generator[pd.DataFrame, None, None]:
    """Split input dataframes into n_chunks

    Each chunk users 1 processor in subsequent multi-processor parallelization
    """
    return iter(np.array_split(df_whole, n_chunks))


def normalize_query_multiprocessor(df_query: pd.DataFrame):
    """Multi-processor runner for normalizing query text

    | Parallelization is fulfilled with ``joblib`` and
    the computation is distributed across data chunks

    Parameters
    ----------
    df_query : pd.DataFrame
        | Input dataframe containing raw queries that are to be normalized

    Returns
    -------
    pd.DataFrame
        | Output dataframe containing normalized queries

    """
    logger.info(
        "Distribute query normalization task onto %s processors", N_PROCESSORS
    )
    result_list = Parallel(n_jobs=multiprocessing.cpu_count() - 1)(
        delayed(normalize_query)(chunk)
        for chunk in chunckenize_input_df(df_query)
    )
    logger.info("Multiprocessor run completed")
    return pd.concat(result_list, axis=0)

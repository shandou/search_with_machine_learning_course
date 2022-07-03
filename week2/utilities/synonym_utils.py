"""Text processing utilities for synonym generator
"""
from __future__ import annotations

import pathlib
import re
from typing import List, Tuple

import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from unidecode import unidecode

# Absolute path of input and output data directory
DATA_DIR_PATH: str = "/workspace/datasets/fasttext/"

# Column name for product titles
COLNAME_PRODUCT: str = "product_title"


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
    lemmatize
        | Normalizes inflected tokens into root word with WordNet corpus
    done -> str
        | Joins normalized tokens into normalized output string

    Example usages
    ---------------
    >>> df_titles[COLNAME_PRODUCT] = df_titles[COLNAME_PRODUCT].apply(
    ...     lambda x: TextNormalizer(x.lower())
    ...     .strip_accents()
    ...     .remove_non_word()
    ...     .tokenize()
    ...     .lemmatize()
    ...     .done()
    ... )

    """

    def __init__(self, text: str) -> None:
        self.text: str = text
        self.tokens: List[str] = []

    @staticmethod
    def _nltk_resource_mananger(
        nltk_resources: Tuple[str] = ("corpora/wordnet", "corpora/omw-1.4")
    ) -> None:
        """Helper that checks and downloads (when needed) nltk resources

        Parameters
        ----------
        nltk_resources: Tuple[str], optional,
                default=("corpora/wordnet", "corpora/omw-1.4")
            | Tuple containing nltk resources (stored under ``nltk_data``)
            required for lemmatization

        """
        for nltk_resource in nltk_resources:
            try:
                # Check if resource exists
                nltk.data.find(nltk_resource)
            except LookupError:
                # If not, download for the first time
                nltk_resource_name: str = nltk_resource.split("/")[1]
                nltk.download(nltk_resource_name)

    def strip_accents(self) -> TextNormalizer:
        self.text = unidecode(self.text)
        return self

    def remove_non_word(self) -> TextNormalizer:
        self.text = re.sub(r"\W", " ", self.text)
        return self

    def tokenize(self) -> TextNormalizer:
        self.tokens = self.text.split()
        return self

    def lemmatize(self) -> TextNormalizer:
        self._nltk_resource_mananger()
        self.tokens = [
            WordNetLemmatizer().lemmatize(token) for token in self.tokens
        ]
        return self

    def done(self) -> str:
        return " ".join(self.tokens)


def normalize_titles(
    input_filename: str = "titles.txt", output_prefix: str = "normalized"
) -> None:
    """Applies text normalization to input file

    Input file is a single column file with product titles

    Parameters
    ----------
    input_filename: str, optional, default='titles.txt'
        | Input file contain unprocessed, raw product titles
    output_prefix: str, optional, default = 'normalized'
        | Prefix attached to `input_filename` (f"{output_prefix}_{input_filename}")
        for storing normalized product titles

    """

    # I/O paths setups
    input_file_path: pathlib.Path = (
        pathlib.Path(DATA_DIR_PATH) / input_filename
    )
    output_file_path: pathlib.Path = (
        pathlib.Path(DATA_DIR_PATH) / f"{output_prefix}_{input_filename}"
    )

    # Load titles data into pandas dataframe
    df_titles: pd.DataFrame = pd.read_csv(
        input_file_path, header=None, names=[COLNAME_PRODUCT], sep="\t"
    )

    # Apply text normalization to the ``titles`` columns
    df_titles[COLNAME_PRODUCT] = df_titles[COLNAME_PRODUCT].apply(
        lambda x: TextNormalizer(x.lower())
        .strip_accents()
        .remove_non_word()
        .tokenize()
        .lemmatize()
        .done()
    )

    # Save normalized titles to ``output_file_path`` (headerless text file)
    df_titles.to_csv(output_file_path, index=False, header=None)


if __name__ == "__main__":
    normalize_titles()

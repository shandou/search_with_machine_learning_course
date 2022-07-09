"""Generates synonyms csv for subsequent ES/Opensearch re-indexing
"""
import argparse
import pathlib
from dataclasses import dataclass
from typing import List, Tuple

import fasttext
import pandas as pd

arg_parser = argparse.ArgumentParser(description="Options for generating synonyms")
arg_parser.add_argument(
    "--min_similarity",
    type=float,
    default=0.75,
    help=(
        "Cosine similarity cutoff for synonyms. "
        "Synonyms have cosine similarity higher than this ``min_similarity``",
    ),
)
args = arg_parser.parse_args()


@dataclass
class IOPathsTemplate:
    """Dataclass template of io_paths constants for synonym generation"""

    BASE_PATH: pathlib.Path = pathlib.Path("/workspace/datasets/fasttext/")
    INPUT_TOP_WORDS: pathlib.Path = BASE_PATH / "top_words.txt"
    OUTPUT_SYNONYMS: pathlib.Path = BASE_PATH / "synonyms.csv"
    EMBEDDING_MODEL: pathlib.Path = BASE_PATH / "normalized_title_model_lrtest.bin"


@dataclass
class ColNamesTemplate:
    WORD: str = "word"
    SYNONYMS: str = "synonyms"


# Collection of IO paths constants
IO_PATHS: IOPathsTemplate = IOPathsTemplate()

# Collection of dataframe column names
COLNAMES: ColNamesTemplate = ColNamesTemplate()


def generate_synonyms_file(min_similarity: float) -> None:
    """Generates headerless, single-column, comma-separated synonyms file

    | Input file: 1_000 most frequent words loaded from
    ``/workspace/datasets/fasttext/top_words.txt``
    | Output file: headerless, single-column synonyms file stored at
    ``/workspace/datasets/fasttext/synonyms.csv``

    The first few lines of ``synonyms.csv`` look like:

    .. code-block:: bash

        $ head -n 5 synonyms.csv
        memory,4gb,8gb
        window,mac
        apple,ipod,r,iphone,ipad
        nintendo,wii,d,gamecube,psp
        drive,hard,500gb

    Parameters
    ----------
    min_similarity: float
        | Only treat words with cosine similarity >= min_similarity as synonyms

    Example Usages
    ---------------
    The output csv file has the following path inside Opensearch container:
    ( Local: /workspace/datasets/fasttext/synonyms.csv)
    Opensearch docker container path: /usr/share/opensearch/config/synonyms.csv

    Inside ``week2/conf/bbuy_products.json``, the synonyms.csv is used via:

    .. code-block:: json

        {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "smarter_hyphens": {
                            "tokenizer": "smarter_hyphens_tokenizer",
                            "filter": ["smarter_hyphens_filter", "lowercase"]
                        },
                        "synonym": {
                            "tokenizer": "standard",
                            "filter": ["synonym_filter", "lowercase"]
                        }
                    },
                    "tokenizer": {
                        "smarter_hyphens_tokenizer": {
                            "type": "char_group",
                            "tokenize_on_chars": ["whitespace", "\n"]
                        }
                    },
                    "filter": {
                        "smarter_hyphens_filter": {
                            "type": "word_delimiter_graph",
                            "catenate_words": true,
                            "catenate_all": true
                        },
                        "synonym_filter": {
                            "type": "synonym",
                            "synonyms_path": "synonyms.csv"
                        }
                    }
                }
            },
            ...
        }

    """

    # Load skipgram model trained in level2 implementation
    model = fasttext.load_model(str(IO_PATHS.EMBEDDING_MODEL))

    # Load input: A list of frequently-appearing words among product titles
    df_top_words: pd.DataFrame = pd.read_csv(
        IO_PATHS.INPUT_TOP_WORDS, header=None, sep="\t", names=[COLNAMES.WORD]
    )
    df_synonyms: pd.DataFrame = pd.DataFrame(columns=[COLNAMES.SYNONYMS])

    for __, row in df_top_words.iterrows():

        word_origin: str = row[COLNAMES.WORD]
        list_similarity_vs_word: List[Tuple[float, str]] = model.get_nearest_neighbors(
            word_origin
        )

        # Only treat words with cosine similarity (relative to word_origin)
        #   larger than ``min_similarity`` as synonyms
        synonyms: List[str] = [word_origin] + [
            entry[1] for entry in list_similarity_vs_word if entry[0] >= min_similarity
        ]
        if len(synonyms) > 1:
            df_synonyms = df_synonyms.append(
                pd.DataFrame({COLNAMES.SYNONYMS: ",".join(synonyms)}, index=[0])
            )

    # Generates synonyms.csv
    df_synonyms.to_csv(IO_PATHS.OUTPUT_SYNONYMS, header=None, index=False, sep="\t")


if __name__ == "__main__":
    generate_synonyms_file(min_similarity=args.min_similarity)

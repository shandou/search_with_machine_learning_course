import argparse
import pathlib
from dataclasses import dataclass

import fasttext
import pandas as pd


@dataclass
class IOPathsTemplate:
    """Dataclass template of io_paths constants for synonym generation"""

    BASE_PATH: pathlib.Path = pathlib.Path("/workspace/datasets/fasttext/")
    INPUT_TOP_WORDS: pathlib.Path = BASE_PATH / "top_words.txt"
    OUTPUT_SYNONYMS: pathlib.Path = BASE_PATH / "synonyms.csv"
    EMBEDDING_MODEL: pathlib.Path = (
        BASE_PATH / "normalized_title_model_lrtest.bin"
    )


# Collection of IO paths constants
IO_PATHS: IOPathsTemplate = IOPathsTemplate()


def generate_synonyms(min_similarity: float = 0.75):

    # Load skipgram model trained in level2 implementation
    model = fasttext.load_model(str(IO_PATHS.EMBEDDING_MODEL))

    # Load input: A list of frequently-appearing words among product titles
    df_top_words: pd.DataFrame = pd.read_csv(
        IO_PATHS.INPUT_TOP_WORDS, header=None, sep="\t", names=["word"]
    )

    df_top_words["synonyms"] = df_top_words["word"].apply(
        model.get_nearest_neighbors
    )
    model.get_nearest_neighbors("enviroment")


if __name__ == "__main__":
    generate_synonyms()

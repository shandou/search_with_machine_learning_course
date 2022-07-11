"""Helper module; generates testing queries for level2 week3 task"""
import os
import sys

sys.path.insert(0, "/workspace/search_with_machine_learning_course/week3")
from typing import Tuple, Union

import fasttext

fasttext.FastText.eprint = lambda x: None
import numpy as np
import pandas as pd

from utils import logger
from utils.constants import COLNAMES

SAMPLING_FRACTION: float = 0.01
ARTIFACTS_DIR: str = "/workspace/datasets/fasttext/"
CLASSIFIER_PATH: str = os.path.join(ARTIFACTS_DIR, "query_classifier.bin")
CLASSNAME_FILE: str = os.path.join(ARTIFACTS_DIR, "product_category_names.csv")
INPUT_DATA_FILE: str = os.path.join(
    ARTIFACTS_DIR, "shuffled_labeled_queries.txt"
)
OUTPUT_PRED_CORRECT: str = os.path.join(
    ARTIFACTS_DIR, "correct_query_category_pred.txt"
)
OUTPUT_PRED_INCORRECT: str = os.path.join(
    ARTIFACTS_DIR, "incorrect_query_category_pred.txt"
)


def load_query_vs_label() -> pd.DataFrame:
    """Load the content of ``shuffled_labeled_queries.txt`` into a dataframe

    An additional column ``ntimes_query_searched`` is attached

    Returns
    -------
    pd.DataFrame
        | Sampled, labelled query data with an additional
        ``ntimes_query_searched`` column

    """

    # Load raw labeled data
    df_raw: pd.DataFrame = pd.read_csv(
        INPUT_DATA_FILE,
        header=None,
        names=[COLNAMES.LABEL],
    )
    df_raw[COLNAMES.QUERY] = (
        df_raw[COLNAMES.LABEL].str.split().map(lambda x: x[1:]).map(" ".join)
    )
    df_raw[COLNAMES.LABEL] = (
        df_raw[COLNAMES.LABEL].str.split().map(lambda x: x[0])
    )

    df_raw = df_raw.join(
        df_raw.groupby(COLNAMES.QUERY).agg(
            **{
                COLNAMES.QUERY_SEARCH_FREQ: pd.NamedAgg(
                    column=COLNAMES.LABEL, aggfunc=pd.Series.count
                )
            }
        ),
        on=COLNAMES.QUERY,
    )
    return (
        df_raw.sample(frac=SAMPLING_FRACTION)
        .drop_duplicates()
        .reset_index(drop=True)
    )


def predict_category_label(query: str) -> str:
    """Runs query classifier and outputs predicted category label

    Parameters
    ----------
    query : str
        Input query

    Returns
    -------
    str
        Predicted class label

    """
    model: fasttext.FastText._FastText = fasttext.load_model(CLASSIFIER_PATH)
    predictions: Tuple[Tuple[str, ...], np.ndarray] = model.predict(query)
    id_prob_max: int = np.argmax(predictions[1])
    if predictions[1][id_prob_max] > 0.5:
        predicted_label: Union[None, str] = predictions[0][id_prob_max]
    else:
        predicted_label = None
    return predicted_label


if __name__ == "__main__":

    df_query_vs_label: pd.DataFrame = load_query_vs_label()
    logger.info("Number of records = %s", len(df_query_vs_label))
    df_query_vs_label[COLNAMES.PREDICTED_LABEL] = df_query_vs_label[
        COLNAMES.QUERY
    ].apply(predict_category_label)

    df_correct_pred: pd.DataFrame = (
        df_query_vs_label.query(
            f"{COLNAMES.LABEL} == {COLNAMES.PREDICTED_LABEL}"
        )
        .dropna()
        .sort_values(by=COLNAMES.QUERY_SEARCH_FREQ, ascending=False)
        .reset_index(drop=True)
    )
    df_correct_pred.to_csv(OUTPUT_PRED_CORRECT, index=False)

    df_incorrect_pred: pd.DataFrame = (
        df_query_vs_label.query(
            f"{COLNAMES.LABEL} != {COLNAMES.PREDICTED_LABEL}"
        )
        .dropna()
        .sort_values(by=COLNAMES.QUERY_SEARCH_FREQ, ascending=False)
        .reset_index(drop=True)
    )
    df_incorrect_pred.to_csv(OUTPUT_PRED_INCORRECT, index=False)

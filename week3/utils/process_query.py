import multiprocessing
from audioop import mul
from typing import Generator

import pandas as pd
from joblib import Parallel, delayed


def query_processor():
    # TODO
    pass


def chunckenize_data():
    # TODO
    pass


def parallel(chunks):
    batch_list = Parallel(n_jobs=multiprocessing.cpu_count() - 1)(
        delayed(query_processor)(chunk) for chunk in chunks
    )
    return pd.concat(batch_list, axis=0)

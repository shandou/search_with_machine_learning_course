from dataclasses import dataclass


@dataclass
class ColNamesDefinitions:
    """dataclass template for column names constants"""

    QUERY: str = "query"
    N_QUERIES_THIS_CATEGORY: str = "n_queries_this_category"
    QUERY_SEARCH_FREQ: str = "ntimes_query_searched"
    THIS_CATEGORY: str = "category"
    PARENT_CATEGORY: str = "parent"
    LABEL: str = "label"
    PREDICTED_LABEL: str = f"{LABEL}_pred"


COLNAMES: ColNamesDefinitions = ColNamesDefinitions()

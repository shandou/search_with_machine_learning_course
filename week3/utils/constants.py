from dataclasses import dataclass


@dataclass
class ColNamesDefinitions:
    QUERY: str = "query"
    N_QUERIES_THIS_CATEGORY: str = "n_queries_this_category"
    THIS_CATEGORY: str = "category"
    PARENT_CATEGORY: str = "parent"


COLNAMES: ColNamesDefinitions = ColNamesDefinitions()

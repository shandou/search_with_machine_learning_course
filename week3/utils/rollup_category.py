import pandas as pd

from utils.constants import COLNAMES


class Helper:
    @staticmethod
    def update_n_queries_this_category(
        df_query_vs_category: pd.DataFrame,
    ) -> pd.DataFrame:
        if COLNAMES.N_QUERIES_THIS_CATEGORY in df_query_vs_category.columns:
            df_query_vs_category.drop(
                columns=[COLNAMES.N_QUERIES_THIS_CATEGORY],
                inplace=True,
            )
        return df_query_vs_category.join(
            df_query_vs_category.groupby(COLNAMES.THIS_CATEGORY).agg(
                **{
                    COLNAMES.N_QUERIES_THIS_CATEGORY: pd.NamedAgg(
                        column=COLNAMES.QUERY, aggfunc=pd.Series.nunique
                    )
                }
            ),
            on=COLNAMES.THIS_CATEGORY,
        )

    @staticmethod
    def update_categories(
        df_query_vs_category: pd.DataFrame, df_category_tree: pd.DataFrame
    ) -> pd.DataFrame:
        df_query_vs_category.loc[
            :, COLNAMES.THIS_CATEGORY
        ] = df_query_vs_category.join(
            df_category_tree.set_index(COLNAMES.THIS_CATEGORY),
            on=COLNAMES.THIS_CATEGORY,
        ).loc[
            :, COLNAMES.PARENT_CATEGORY
        ]


def recursive_rollup_category(
    df_query_vs_category: pd.DataFrame,
    *,
    df_category_tree: pd.DataFrame,
    min_n_queries_this_category: int,
):
    # 1. Attach or update "n_queries_this_category" column
    df_query_vs_category = Helper.update_n_queries_this_category(
        df_query_vs_category
    )

    df_rollup_candidates: pd.DataFrame = df_query_vs_category.query(
        f"{COLNAMES.N_QUERIES_THIS_CATEGORY} < {min_n_queries_this_category}"
    )
    if len(df_rollup_candidates) == 0:
        return (
            df_query_vs_category.drop(
                columns=[COLNAMES.N_QUERIES_THIS_CATEGORY], errors="ingore"
            )
            .drop_duplicates()
            .reset_index(drop=True)
        )

    df_unchanged: pd.DataFrame = df_query_vs_category.query(
        f"{COLNAMES.N_QUERIES_THIS_CATEGORY} >= {min_n_queries_this_category}"
    )
    df_rollup_candidates = Helper.update_categories(
        df_rollup_candidates, df_category_tree
    )
    df_query_vs_category = pd.concat(
        [df_unchanged, df_rollup_candidates], axis=0, ignore_index=True
    ).drop(columns=[COLNAMES.N_QUERIES_THIS_CATEGORY])
    return recursive_rollup_category(
        df_query_vs_category,
        df_category_tree=df_category_tree,
        min_n_queries_this_category=min_n_queries_this_category,
    )

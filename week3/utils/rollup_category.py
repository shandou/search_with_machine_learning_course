"""Utility module for rolling up category tree"""
import pandas as pd

from utils.constants import COLNAMES


class Helper:
    """Helper class for rolling up categories"""

    @staticmethod
    def update_n_queries_this_category(
        df_query_vs_category: pd.DataFrame,
    ) -> pd.DataFrame:
        """Updates number of queries belonging to each category

        Parameters
        ----------
        df_query_vs_category : pd.DataFrame
            | Input query vs. category dataframe.
            Example:
            >>> df_query_vs_category.head(3)
                    category                                query
            0        abcat0101001  Televisiones Panasonic  50 pulgadas
            1        abcat0101001                                Sharp
            2  pcmcat193100050014                                 nook

        Returns
        -------
        pd.DataFrame
            | Query vs. category dataframe with an additional coloumn
            tallying the number of queries associated with each category
            Example:
            >>> df_query_vs_category.head(3)

        """
        if COLNAMES.N_QUERIES_THIS_CATEGORY in df_query_vs_category.columns:
            df_query_vs_category.drop(
                columns=[COLNAMES.N_QUERIES_THIS_CATEGORY],
                inplace=True,
            )
        # NOTE: Duplicates are KEPT when tallying number of queries
        #   associated with each category
        return df_query_vs_category.join(
            df_query_vs_category.groupby(COLNAMES.THIS_CATEGORY).agg(
                **{
                    COLNAMES.N_QUERIES_THIS_CATEGORY: pd.NamedAgg(
                        column=COLNAMES.QUERY,
                        aggfunc=pd.Series.count,
                    )
                }
            ),
            on=COLNAMES.THIS_CATEGORY,
        )

    @staticmethod
    def merge_to_parent_category(
        df_rollup_candidates: pd.DataFrame, df_category_tree: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge child to parent category

        | Achieved by replacing "category" values with their immediate parents
        (child-parent connections obtained from ``df_category_tree``)

        Parameters
        ----------
        df_rollup_candidates : pd.DataFrame
            | Query vs. category dataframe subset with categories
            that contain too few queries
            and thus need to be merged with parent categories
        df_category_tree : pd.DataFrame
            | Category tree dataframe
            Example:
            >>> parents_df.head(3)
                category        parent
            0  abcat0010000      cat00000
            1  abcat0011000  abcat0010000
            2  abcat0011001  abcat0011000

        Returns
        -------
        pd.DataFrame
            Query vs. category dataframe after category rollup

        """
        return (
            df_rollup_candidates.join(
                df_category_tree.set_index(COLNAMES.THIS_CATEGORY),
                on=COLNAMES.THIS_CATEGORY,
                how="left",
            )
            .drop(columns=[COLNAMES.THIS_CATEGORY])
            .rename(columns={COLNAMES.PARENT_CATEGORY: COLNAMES.THIS_CATEGORY})
        )


def recursive_rollup_category(
    df_query_vs_category: pd.DataFrame,
    *,
    df_category_tree: pd.DataFrame,
    min_n_queries_this_category: int,
):
    """Category rollup implemented with recursion

    Parameters
    ----------
    df_query_vs_category : pd.DataFrame
        | Input query vs. category dataframe.
        Example:
        >>> df_query_vs_category.head(3)
                category                                query
        0        abcat0101001  Televisiones Panasonic  50 pulgadas
        1        abcat0101001                                Sharp
        2  pcmcat193100050014                                 nook
    df_category_tree : pd.DataFrame
        | Category tree dataframe
        Example:
        >>> parents_df.head(3)
            category        parent
        0  abcat0010000      cat00000
        1  abcat0011000  abcat0010000
        2  abcat0011001  abcat0011000
    min_n_queries_this_category : int
        | Minimum number of queries that a category must contain;
        Categories with n_queries_this_category < min_n_queries_this_category
        must be merged into their ancestor nodes

    Returns
    -------
    pd.DataFrame
        df_query_vs_category after category rollup

    """
    # 1. Attach or update "n_queries_this_category" column
    df_query_vs_category = Helper.update_n_queries_this_category(
        df_query_vs_category
    )

    # 2. Extract records with number of products per category below
    #   threshold ``min_n_queries_this_category``
    # These are the category candidates that
    #   require recursive merging of leaf nodes into ancestors
    df_rollup_candidates: pd.DataFrame = df_query_vs_category.query(
        f"{COLNAMES.N_QUERIES_THIS_CATEGORY} < {min_n_queries_this_category}"
    )
    if len(df_rollup_candidates) == 0:
        # If no records require rolling up,
        #   the query vs. category dataframe is ready as output
        return (
            df_query_vs_category.drop(
                columns=[COLNAMES.N_QUERIES_THIS_CATEGORY], errors="ignore"
            )
            .drop_duplicates()
            .reset_index(drop=True)
        )

    # 3. Keep track of records that don't require rollup
    df_unchanged: pd.DataFrame = df_query_vs_category.query(
        f"{COLNAMES.N_QUERIES_THIS_CATEGORY} >= {min_n_queries_this_category}"
    )

    # 4. Update category by merging leaf to parent
    df_rollup_candidates = Helper.merge_to_parent_category(
        df_rollup_candidates, df_category_tree
    )

    # 5. Combine rolled-up records and unchanged records
    df_query_vs_category = pd.concat(
        [df_unchanged, df_rollup_candidates], axis=0, ignore_index=True
    ).drop(columns=[COLNAMES.N_QUERIES_THIS_CATEGORY])

    # Recursive call again until exit condition is met
    return recursive_rollup_category(
        df_query_vs_category,
        df_category_tree=df_category_tree,
        min_n_queries_this_category=min_n_queries_this_category,
    )

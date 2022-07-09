import argparse
import glob
import itertools
import multiprocessing
import os
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Generator, List, Tuple

import pandas as pd
from nltk.stem import SnowballStemmer
from tqdm import tqdm

# Contants
COLNAME_CATEGORY: str = "category_code"
COLNAME_PROD_NAME: str = "normalized_product_name"
COLNAME_N_PROD_PER_CATEGORY: str = "n_products_per_category"


def transform_name(product_name: str) -> str:
    """Normalize product name text

    Operations are:
    - Convert to lowercase
    - Exclude non-word character with \W (i.e., [^a-zA-Z0-9_])
    - Word tokens separated by single space only

    Parameters
    ----------
    product_name: str
        | Product name of the catalogue

    Returns
    -------
    str
        | Normalized string

    """
    # IMPLEMENT

    # 1. Lowercase and exclude all non-alphanumerical chars except underscore
    product_name_processed: str = re.sub(r"\W", " ", product_name.lower())

    # 2. Apply Snowball stemmer
    # TODO: Ask instructors about why Snowball stemmer
    #   --a more aggressive stemmer is picked over Porter stemmer
    tokens: List[str] = [
        SnowballStemmer("english").stem(token)
        for token in product_name_processed.split()
    ]

    return " ".join(tokens)


# Directory for product data
directory = r"/workspace/datasets/product_data/products/"

parser = argparse.ArgumentParser(description="Process some integers.")
general = parser.add_argument_group("general")
general.add_argument(
    "--input", default=directory, help="The directory containing product data"
)
general.add_argument(
    "--output",
    default="/workspace/datasets/fasttext/output.fasttext",
    help="the file to output to",
)
general.add_argument(
    "--label",
    default="id",
    help="id is default and needed for downsteam use, but name is helpful for debugging",
)

# Consuming all of the product data, even excluding music and movies,
# takes a few minutes. We can speed that up by taking a representative
# random sample.
general.add_argument(
    "--sample_rate",
    default=1.0,
    type=float,
    help="The rate at which to sample input (default is 1.0)",
)

# IMPLEMENT: Setting min_products removes infrequent categories and
# makes the classifier's task easier.
general.add_argument(
    "--min_products",
    default=0,
    type=int,
    help="The minimum number of products per category (default is 0).",
)

# Configurable category pruning method
#   - filter = simple exclusion of categories with n_products cutoff
#   - rollup = merge child node with parent node in the taxonomy hierarchy
general.add_argument(
    "--pruning_method",
    default="filter",
    type=str,
    choices=["filter", "rollup"],
    help="Method used to prune category taxonomy (default to 'filter').",
)

args = parser.parse_args()
output_file = args.output
path = Path(output_file)
output_dir = path.parent
if os.path.isdir(output_dir) == False:
    os.mkdir(output_dir)

if args.input:
    directory = args.input
# IMPLEMENT: Track the number of items in each category and
#   only output if above the min
min_products = args.min_products
sample_rate = args.sample_rate
names_as_labels = False
if args.label == "name":
    names_as_labels = True
pruning_method: str = args.pruning_method


def filter_category_by_n_products(
    df_all_labels: pd.DataFrame, min_products: int = 1
) -> pd.DataFrame:
    """Filter product category labels by ``min_products`` per category cutoff

    Only categories with n_products > min_products are kept

    Parameters
    ----------
    df_all_labels: pd.DataFrame
        | Complete set of labelled data (category vs. product title)
    min_products: int, default = 1
        | n_products per category cutoff.
        Categories with lower n_products are excluded in output dataframe

    Returns
    -------
    pd.DataFrame
        | labelled data after category filtering

    """
    # Obtain category list with n_products_per_category larger than
    #   ``min_products`` cutoff
    # The cutoff is to ensure each class label has sufficient
    #   data coverage for ML classification
    list_category_keeper: List[str] = list(
        df_all_labels.groupby(COLNAME_CATEGORY)
        .agg(
            **{
                COLNAME_N_PROD_PER_CATEGORY: pd.NamedAgg(
                    column=COLNAME_PROD_NAME, aggfunc=lambda x: len(set(x))
                )
            }
        )
        .query(f"{COLNAME_N_PROD_PER_CATEGORY} >= {min_products}")
        .index
    )

    # Apply category pruning
    if len(list_category_keeper) > 0:
        df_all_labels = df_all_labels.query(
            f"{COLNAME_CATEGORY}.isin(@list_category_keeper)"
        )

    return df_all_labels


def _label_filename(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    labels = []
    for child in root:
        if random.random() > sample_rate:
            continue
        # Check to make sure category name is valid and not in music or movies
        if (
            child.find("name") is not None
            and child.find("name").text is not None
            and child.find("categoryPath") is not None
            and len(child.find("categoryPath")) > 0
            and child.find("categoryPath")[len(child.find("categoryPath")) - 1][0].text
            is not None
            and child.find("categoryPath")[0][0].text == "cat00000"
            and child.find("categoryPath")[1][0].text != "abcat0600000"
        ):
            # Choose last element in categoryPath as the leaf categoryId or name
            if names_as_labels:
                cat = child.find("categoryPath")[len(child.find("categoryPath")) - 1][
                    1
                ].text.replace(" ", "_")
            else:
                cat = child.find("categoryPath")[len(child.find("categoryPath")) - 1][
                    0
                ].text
            # Replace newline chars with spaces so fastText doesn't complain
            name = child.find("name").text.replace("\n", " ")
            labels.append((cat, transform_name(name)))
    return labels


if __name__ == "__main__":

    files: List[str] = glob.glob(f"{directory}/*.xml")

    print("Writing results to %s" % output_file)

    with multiprocessing.Pool() as p:

        all_labels: Generator[List[List[Tuple[str, str]]], None, None] = tqdm(
            p.imap_unordered(_label_filename, files), total=len(files)
        )

        # Package category vs. product name data into pandas dataframe
        #   ``itertools.chain.from_iterable(all_labels)`` is to flatten
        #   nested iterable content list(list())
        df_all_labels: pd.DataFrame = pd.DataFrame(
            itertools.chain.from_iterable(all_labels),
            columns=[COLNAME_CATEGORY, COLNAME_PROD_NAME],
        )

        # Prune taxonomy tree using selected method (filter or rollup)
        if pruning_method == "filter":
            df_all_labels = filter_category_by_n_products(
                df_all_labels, min_products=min_products
            )

        # Add __label__ to category codes values
        df_all_labels[COLNAME_CATEGORY] = df_all_labels[COLNAME_CATEGORY].map(
            "__label__{}".format
        )

        # Write to ``output_file`` as tab-separated csv file (headerless)
        df_all_labels.to_csv(output_file, index=False, header=False, sep="\t")

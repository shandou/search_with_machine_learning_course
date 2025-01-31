import argparse
import csv
import os
import xml.etree.ElementTree as ET

# Useful if you want to perform stemming.
# import nltk
import numpy as np
import pandas as pd

from utils import logger
from utils.constants import COLNAMES
from utils.normalize_query import normalize_query_multiprocessor
from utils.rollup_category import recursive_rollup_category

# stemmer = nltk.stem.PorterStemmer()

categories_file_name = r"/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml"

queries_file_name = r"/workspace/datasets/train.csv"
# output_file_name = r"/workspace/datasets/labeled_query_data.txt"
output_file_name = r"/workspace/datasets/fasttext/labeled_queries.txt"

parser = argparse.ArgumentParser(description="Process arguments.")
general = parser.add_argument_group("general")
general.add_argument(
    "--min_queries",
    default=1,
    help="The minimum number of queries per category label (default is 1)",
)
general.add_argument(
    "--output", default=output_file_name, help="the file to output to"
)

args = parser.parse_args()
output_file_name = args.output

if args.min_queries:
    min_queries = int(args.min_queries)

# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = "cat00000"

tree = ET.parse(categories_file_name)
root = tree.getroot()

# Parse the category XML file to map each category id to its parent category id in a dataframe.
categories, categories_names = [], []
parents = []
for child in root:
    id = child.find("id").text
    cat_path = child.find("path")
    cat_path_ids = [cat.find("id").text for cat in cat_path]
    cat_path_names = [cat.find("name").text for cat in cat_path]
    leaf_id = cat_path_ids[-1]
    leaf_name = cat_path_names[-1]
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        categories_names.append(leaf_name)
        parents.append(cat_path_ids[-2])
parents_df = pd.DataFrame(
    list(zip(categories, parents)), columns=["category", "parent"]
)

# Add auxiliary CSV with category code vs. category name records
# Example: $ head -n 4 product_category_names.csv
#   category,category_name,label
#   abcat0010000,Gift Center,__label__abcat0010000
#   abcat0011000,Her,__label__abcat0011000
#   abcat0011001,Leisure Gifts,__label__abcat0011001
categories_df = pd.DataFrame(
    list(zip(categories, categories_names)),
    columns=["category", "category_name"],
)
categories_df["label"] = "__label__" + categories_df["category"]
categories_df.to_csv(
    "/workspace/datasets/fasttext/product_category_names.csv", index=False
)

# Read the training data into pandas, only keeping queries with non-root categories in our category tree.
df = pd.read_csv(queries_file_name)[["category", "query"]]
df = df[df["category"].isin(categories)]
# NOTE: Duplicated rows are preserved to reflect the true traffic!

# IMPLEMENT ME: Convert queries to lowercase,
#   and optionally implement other normalization, like stemming.
df = normalize_query_multiprocessor(df)


# IMPLEMENT ME: Roll up categories to ancestors to
#   satisfy the minimum number of queries per category.
logger.info(
    "Query dataframe has %s unique categories before applying rollup",
    df[COLNAMES.THIS_CATEGORY].nunique(),
)
logger.info(
    "Apply category rollup with min_n_queries_this_category = %s", min_queries
)
df = recursive_rollup_category(
    df, df_category_tree=parents_df, min_n_queries_this_category=min_queries
)
logger.info(
    "Query dataframe has %s unique categories after applying rollup",
    df[COLNAMES.THIS_CATEGORY].nunique(),
)


# Create labels in fastText format.
df["label"] = "__label__" + df["category"]
# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
df = df[df["category"].isin(categories)]
df["output"] = df["label"] + " " + df["query"]
df[["output"]].to_csv(
    output_file_name,
    header=False,
    sep="|",
    escapechar="\\",
    quoting=csv.QUOTE_NONE,
    index=False,
)

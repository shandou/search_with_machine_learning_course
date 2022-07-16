# A simple client for querying driven by user input on the command line.  Has hooks for the various
# weeks (e.g. query understanding).  See the main section at the bottom of the file
import argparse

# import fileinput
import json
import logging
import os
import subprocess
import warnings
from base64 import encode
from getpass import getpass
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urljoin
from xml.etree.ElementTree import tostringlist

import fasttext
import numpy as np
import numpy.typing as np_typing
import pandas as pd
import sentence_transformers
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer

PRETRAINED_MODEL_NAME: str = "all-MiniLM-L6-v2"


warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(format="%(levelname)s:%(message)s")

# Constant: Path to the trained query classifier
#   min_queries = 10_000, lr = 0.5, epoch = 25
CLASSIFIER_PATH: str = "/workspace/datasets/fasttext/query_classifier.bin"
CLASSNAME_FILE: str = "/workspace/datasets/fasttext/product_category_names.csv"

# expects clicks and impressions to be in the row
def create_prior_queries_from_group(
    click_group,
):  # total impressions isn't currently used, but it mayb worthwhile at some point
    click_prior_query = ""
    # Create a string that looks like:  "query": "1065813^100 OR 8371111^89", where the left side is the doc id and the right side is the weight.  In our case, the number of clicks a document received in the training set
    if click_group is not None:
        for item in click_group.itertuples():
            try:
                click_prior_query += "%s^%.3f  " % (
                    item.doc_id,
                    item.clicks / item.num_impressions,
                )

            except KeyError as ke:
                pass  # nothing to do in this case, it just means we can't find priors for this doc
    return click_prior_query


# expects clicks from the raw click logs, so value_counts() are being passed in
def create_prior_queries(
    doc_ids, doc_id_weights, query_times_seen
):  # total impressions isn't currently used, but it mayb worthwhile at some point
    click_prior_query = ""
    # Create a string that looks like:  "query": "1065813^100 OR 8371111^89", where the left side is the doc id and the right side is the weight.  In our case, the number of clicks a document received in the training set
    click_prior_map = ""  # looks like: '1065813':100, '8371111':809
    if doc_ids is not None and doc_id_weights is not None:
        for idx, doc in enumerate(doc_ids):
            try:
                wgt = doc_id_weights[
                    doc
                ]  # This should be the number of clicks or whatever
                click_prior_query += "%s^%.3f  " % (
                    doc,
                    wgt / query_times_seen,
                )
            except KeyError as ke:
                pass  # nothing to do in this case, it just means we can't find priors for this doc
    return click_prior_query


def predict_query_category(
    user_query: str, min_class_proba: float = 0.5
) -> Union[None, List[str]]:
    """Use query classifier (week3 project level1) to predict category id

    Parameters
    ----------
    user_query : str
        | Search query entered by user
    min_class_proba : float, optional, default=0.5
        | Minimum score for the classification results to be useable for
        subsequent search

    Returns
    -------
    Union[None, List[str]]
        | List of predicted category ids.
        None indicates no useable category id;
        | Example: ['cat02015']

    """

    predicted_category_ids: Union[None, List[str]] = None

    model: fasttext.FastText._FastText = fasttext.load_model(CLASSIFIER_PATH)
    predictions: Tuple[Tuple[str, ...], np.ndarray] = model.predict(user_query)
    ids_filtered: np.ndarray = np.argwhere(
        predictions[1] > min_class_proba
    ).squeeze()

    if ids_filtered.size > 0:
        labels_filtered = np.array(predictions[0])[ids_filtered]

        if isinstance(labels_filtered, str):
            labels_filtered = [labels_filtered]

        predicted_category_ids = [
            label.replace("__label__", "") for label in labels_filtered
        ]
        predicted_category_names: List[str] = [
            subprocess.check_output(["grep", category_id, CLASSNAME_FILE])
            .decode("utf-8")
            .split(",")[1]
            for category_id in predicted_category_ids
        ]
        logger.info("Classifier output is: %s", predictions)
        logger.info(
            "Corresponding category name is: %s", predicted_category_names
        )

    return predicted_category_ids


def create_vector_query(
    user_query: str, size: int = 10, source: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Create query JSON for performing k-NN vector search

    Parameters
    ----------
    user_query : str
        Search query issued by user
    size : int, optional
        Maximum return size, by default 10
    source : Optional[List[str]], optional
        List of index fields being kept in search results, by default None

    Returns
    -------
    Dict[str, Any]
        Query JSON specific to k-NN vector search

    References
    -----------
    | Query object syntax follows examples provided by OpenSearch documentation:
    https://opensearch.org/docs/latest/search-plugins/knn/knn-score-script/

    """
    # Load pretrained sentence embedding model
    model: sentence_transformers.SentenceTransformer = SentenceTransformer(
        PRETRAINED_MODEL_NAME
    )

    # Create query embedding
    query_embedding: np_typing.NDArray[np.float32] = model.encode([user_query])

    # Use query embedding (in the form of single-element array) to
    #   create OpenSearch query JSON
    # TODO: Play with filtering and sorting
    query_obj: Dict[str, Any] = {
        "size": size,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "knn_score",
                    "lang": "knn",
                    "params": {
                        "field": "name_embedding",
                        "query_value": query_embedding[0].tolist(),
                        "space_type": "cosinesimil",
                    },
                },
            }
        },
    }

    if source is not None:  # otherwise use the default and retrieve all source
        query_obj["_source"] = source

    return query_obj


# Hardcoded query here.  Better to use search templates or other query config.
def create_query(
    user_query,
    click_prior_query,
    filters,
    sort="_score",
    sortDir="desc",
    size=10,
    source=None,
    use_synonyms: bool = False,
):

    if use_synonyms:
        # When ``use_synonyms`` toggle is True
        #   Simply replace all occurrences of the fieldname "name"
        #   with "name.synonyms"
        name_field: str = "name.synonyms"
    else:
        name_field = "name"

    if filters is not None:
        print(
            "==================================\n",
            "Classifier-based filter setting is:\n",
            "==================================\n",
        )
        print(json.dumps(filters, indent=4))

    query_obj = {
        "size": size,
        "sort": [{sort: {"order": sortDir}}],
        "query": {
            "function_score": {
                "query": {
                    "bool": {
                        "must": [],
                        "should": [  #
                            {
                                "match": {
                                    name_field: {
                                        "query": user_query,
                                        "fuzziness": "1",
                                        "prefix_length": 2,
                                        # short words are often acronyms
                                        #   or usually not misspelled,
                                        #   so don't edit
                                        "boost": 0.01,
                                    }
                                }
                            },
                            {
                                "match_phrase": {  # near exact phrase match
                                    "name.hyphens": {
                                        "query": user_query,
                                        "slop": 1,
                                        "boost": 50,
                                    }
                                }
                            },
                            {
                                "multi_match": {
                                    "query": user_query,
                                    "type": "phrase",
                                    "slop": "6",
                                    "minimum_should_match": "2<75%",
                                    "fields": [
                                        "name^10",
                                        "name.hyphens^10",
                                        "shortDescription^5",
                                        "longDescription^5",
                                        "department^0.5",
                                        "sku",
                                        "manufacturer",
                                        "features",
                                        "categoryPath",
                                    ],
                                }
                            },
                            {
                                "terms": {
                                    # Lots of SKUs in the query logs, boost by it, split on whitespace so we get a list
                                    "sku": user_query.split(),
                                    "boost": 50.0,
                                }
                            },
                            {  # lots of products have hyphens in them or other weird casing things like iPad
                                "match": {
                                    "name.hyphens": {
                                        "query": user_query,
                                        "operator": "OR",
                                        "minimum_should_match": "2<75%",
                                    }
                                }
                            },
                        ],
                        "minimum_should_match": 1,
                        "filter": filters,  #
                    }
                },
                "boost_mode": "multiply",  # how _score and functions are combined
                "score_mode": "sum",  # how functions are combined
                "functions": [
                    {
                        "filter": {"exists": {"field": "salesRankShortTerm"}},
                        "gauss": {
                            "salesRankShortTerm": {
                                "origin": "1.0",
                                "scale": "100",
                            }
                        },
                    },
                    {
                        "filter": {"exists": {"field": "salesRankMediumTerm"}},
                        "gauss": {
                            "salesRankMediumTerm": {
                                "origin": "1.0",
                                "scale": "1000",
                            }
                        },
                    },
                    {
                        "filter": {"exists": {"field": "salesRankLongTerm"}},
                        "gauss": {
                            "salesRankLongTerm": {
                                "origin": "1.0",
                                "scale": "1000",
                            }
                        },
                    },
                    {"script_score": {"script": "0.0001"}},
                ],
            }
        },
    }
    # print(json.dumps(query_obj, indent=4))
    if click_prior_query is not None and click_prior_query != "":
        query_obj["query"]["function_score"]["query"]["bool"]["should"].append(
            {
                "query_string": {
                    # This may feel like cheating, but it's really not,
                    # esp. in ecommerce where you have all this prior data,
                    # You just can't let the test clicks leak in,
                    # which is why we split on date
                    "query": click_prior_query,
                    "fields": ["_id"],
                }
            }
        )
    if user_query == "*" or user_query == "#":
        # replace the bool
        try:
            query_obj["query"] = {"match_all": {}}
        except:
            print("Couldn't replace query for *")
    if source is not None:  # otherwise use the default and retrieve all source
        query_obj["_source"] = source
    return query_obj


def search(
    client,
    user_query,
    index="bbuy_products",
    sort="_score",
    sortDir="desc",
    use_synonyms: bool = False,
    use_classifier: bool = False,
    use_vector_search: bool = False,
):
    #### W3: classify the query
    #### W3: create filters and boosts
    # Note: you may also want to modify the `create_query` method above
    filters: Union[None, Dict[str, Any]] = None
    if use_classifier:
        pred_category_ids: Union[None, List[str]] = predict_query_category(
            user_query
        )
        if pred_category_ids is not None:
            filters = {
                "multi_match": {
                    "query": ",".join(pred_category_ids),
                    "fields": ["categoryLeaf", "categoryPathIds"],
                }
            }

    SOURCE_FIELDS: List[str] = [
        "name",
        "shortDescription",
        "categoryPath",
        "categoryPathIds",
    ]
    if use_vector_search:
        query_obj = create_vector_query(user_query, source=SOURCE_FIELDS)
    else:
        query_obj = create_query(
            user_query,
            click_prior_query=None,
            filters=filters,
            sort=sort,
            sortDir=sortDir,
            source=SOURCE_FIELDS,
            use_synonyms=use_synonyms,
        )
    logging.info(query_obj)

    response = client.search(query_obj, index=index)
    if (
        response
        and response["hits"]["hits"]
        and len(response["hits"]["hits"]) > 0
    ):
        # hits = response["hits"]["hits"]
        print(json.dumps(response, indent=2))


if __name__ == "__main__":
    host = "localhost"
    port = 9200
    auth = (
        "admin",
        "admin",
    )  # For testing only. Don't store credentials in code.
    parser = argparse.ArgumentParser(description="Build LTR.")
    general = parser.add_argument_group("general")
    general.add_argument(
        "-i",
        "--index",
        default="bbuy_products",
        help="The name of the main index to search",
    )
    general.add_argument(
        "-s", "--host", default="localhost", help="The OpenSearch host name"
    )
    general.add_argument(
        "-p", "--port", type=int, default=9200, help="The OpenSearch port"
    )
    general.add_argument(
        "--user",
        help=(
            "The OpenSearch admin. "
            "If this is set, the program will prompt for password too. "
            "If not set, use default of admin/admin"
        ),
    )
    general.add_argument(
        "--synonyms",
        action="store_true",
        help=(
            "synonyms option. "
            "If this is set, name.synonyms will replace name in search query"
        ),
    )
    general.add_argument(
        "--classifier",
        action="store_true",
        help=(
            "Boolean toggle indicating if query classifier model will be used. "
            "If this is set, classification results will be used to "
            "compose opensearch query"
        ),
    )
    general.add_argument(
        "--vector",
        action="store_true",
        help=(
            "Boolean toggle indicating if vector search will be used. "
            "If this is set, ``create_vector_query`` "
            "(instead of ``create_query``) will be used to create query JSON."
        ),
    )
    args = parser.parse_args()

    if len(vars(args)) == 0:
        parser.print_usage()
        exit()

    if bool(args.synonyms):
        print(
            f"synonyms == {args.synonyms} is requested; "
            "use ``name.synonyms`` in search query"
        )

    host = args.host
    port = args.port
    if args.user:
        password = getpass()
        auth = (args.user, password)

    base_url = "https://{}:{}/".format(host, port)
    opensearch = OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_compress=True,  # enables gzip compression for request bodies
        http_auth=auth,
        # client_cert = client_cert_path,
        # client_key = client_key_path,
        use_ssl=True,
        verify_certs=False,  # set to true if you have certs
        ssl_assert_hostname=False,
        ssl_show_warn=False,
    )
    index_name = args.index
    use_synonyms: bool = args.synonyms
    use_classifier: bool = args.classifier
    use_vector_search: bool = args.vector

    query_prompt = "\nEnter your query (type 'Exit' to exit or hit ctrl-c):"
    while True:
        try:
            query: str = str(input(query_prompt)).rstrip()
        except KeyboardInterrupt:
            break
        else:
            if query.lower() == "exit":
                break
            else:
                search(
                    client=opensearch,
                    user_query=query,
                    index=index_name,
                    use_synonyms=use_synonyms,
                    use_classifier=use_classifier,
                    use_vector_search=use_vector_search,
                )

    # for line in fileinput.input():
    #     query = line.rstrip()
    #     if query == "Exit":
    #         break
    #     search(
    #         client=opensearch,
    #         user_query=query,
    #         index=index_name,
    #         use_synonyms=use_synonyms,
    #     )

    #     print(query_prompt)

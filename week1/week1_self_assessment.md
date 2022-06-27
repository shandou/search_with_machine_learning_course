# Week 1 Project Self-Assessment

**1. Do you understand the steps involved in creating and deploying an LTR model?
Name them and describe what each step does in your own words.**

The following summary is under the premise of using Elasticsearch LTR plugin to
deploy an LTR model.

1. Data collection and preparation

    In this case, we use open-source click data, and fake impression and ranking data by combining a baseline retriever and click distributions.

2. Feature store initialization

    Commond: `python week1/utilities/build_ltr.py --create_ltr_store`

    We initialize an Elasticsearch feature store (endpoint `_ltr/week1`) to store both the feature set and the trained XGBoost model

3. Featureset creation for Elasticsearch's feature logging

    Command: `python week1/utilities/build_ltr.py -f week1/conf/ltr_featureset.json --upload_featureset`

    For Elasticsearch LTR plugin, the phrase "feature logging" means computing feature values by running the [Sltr query](https://elasticsearch-learning-to-rank.readthedocs.io/en/latest/logging-features.html) and extract the values from the `_ltrlog` fields of the query responses.

4. Time-based train-test split

    Command:

    ```bash
    python week1/utilities/build_ltr.py --output_dir /workspace/ltr_output --split_input /workspace/datasets/train.csv  --split_train_rows 1000000 --split_test_rows 1000000
    ```

    Splits all click data into training and testing sets.

5. Target and feature engineering

    Commands:

    ```bash
    python week1/utilities/build_ltr.py --generate_impressions --output_dir \
    /workspace/ltr_output --train_file /workspace/ltr_output/train.csv \
    --synthesize &&
    python week1/utilities/build_ltr.py --ltr_terms_field sku \
    --output_dir /workspace/ltr_output --create_xgb_training \
    -f week1/conf/ltr_featureset.json --click_model heuristic
    ```

    Although synthetic data are used in the absence of impression and ranking records,
    the ideas remain the same--At this step we proximate relevancy grade with

6. Train and save XGBoost model

    Command:

    ```bash
    python week1/utilities/build_ltr.py --output_dir /workspace/ltr_output \
    -x /workspace/ltr_output/training.xgb \
    --xgb_conf week1/conf/xgb-conf.json
    ```

    Model tuning is not included but ordinarily during model development stage,
    we tune the model prior to training.
    Specifically for week1 project, the model is trained on training data with predefined hyperparameters, and the resulting model is saved both as a binary and as a json.
    The model json `/workspace/ltr_output/xgb_model.model.ltr` is
    subsequently deployed to Elasticsearch
    (by being uploaded to the `_ltr/week1` feature store)

7. Model deployment into Elasticsearch's LTR feature store as XGBoost model json

    ```bash
    python week1/utilities/build_ltr.py --upload_ltr_model \
    --xgb_model /workspace/ltr_output/xgb_model.model
    ```

8. Use the model to rerank the products.
   For Elasticsearch LTR plugin, this is achieved via `rescore` function.
   The base score and LTR-based rescore are combined as weighted sum to provide the
   total score for reranking.

9. Performance evaluation on test set

    Note: ordinarily the offline evaluation could occur in at least two stages

    1. Prior to model deployment to examine the generalizability of the model
    2. After model deployment Mini-batch online evaluation to monitor model drift after

    ```bash
    python week1/utilities/build_ltr.py --xgb_test /workspace/ltr_output/test.csv \
    --train_file /workspace/ltr_output/train.csv \
    --output_dir /workspace/ltr_output
    --xgb_test_num_queries 100 \
    --xgb_main_query 1 --xgb_rescore_query_weight 2 && \
    python week1/utilities/build_ltr.py --analyze \
    --output_dir /workspace/ltr_output/
    ```

---

**2. What is a feature and featureset?**

-   Broadly speaking, a feature is a numeric representation of data. Specically for search engines, a feature is a predictor variable that determines query-document relevancy (i.e., the response variable).

-   A _featureset_ is a set of predictor variables that each represents an aspect of the data. A machine-learning model describes the relationships between the featureset and the response variable. For search engines, the featureset could consist of
    -   User-specific features
    -   Query-specific features
    -   Document-specific features
    -   Context-specific features
    -   User-document features
    -   Query-document features

---

**3. What is the difference between precision and recall?**
$$precision = \frac{TP}{TP + FP}$$
$$recall = \frac{TP}{TP + FN}$$
Precision and recall differ in their denominators: Precision's denominator is all the positives among the predicted labels, whereas recall's denominator is all the positives among the true labels.

For search engines, precision is the proportion of relevant documents within the search returns, whereas recall is the ratio of relevant documents within the search returns compared to the total number of relevant documents that _should have been_ returned.

---

**4. What are some of the traps associated with using click data in your model?**

-   Target leakage
-   TODO: What else?

---

**5. What are some of the ways we are faking our data and how would you prevent that in your application?**

1. We fake impression data by running queries through a baseline retriever.
2. We fake ranking data based on distributions of clicks.

To prevent these "data faking", we need to equip our application with instrumentation and web analytics.

---

**6. What is target leakage and why is it a bad thing?**

Target leakage occurs whenver a model is given information that it would not have access to at prediction time. Typical examples include:

-   Data from the future is mixed with the past in forecasting problems
-   Target variables are used to encode categorical features (e.g., mean encoding)

When a model is trained under the influence of target leakage, it gives off an illusion of good predictive power. In actuality, the model has poor generalizability and would exhibit poor performance when being used in production.

---

**7. When can using prior history cause problems in search and LTR?**

When using prior history for feature engineering,
one must beware of the risks of target leakage.

More concretely,

-   If we don't enforce time-based train-test split,
    we induce target leakage by mixing future with the past
-   If we apply similar recipes to construct relevancy grade and features,
    we would leak the target into the predictor variables

---

**8. Submit your project along with your best MRR scores**

My best MRR scores are as the following:

```bash
Simple MRR is 0.249
LTR Simple MRR is 0.382
Hand tuned MRR is 0.406
LTR Hand Tuned MRR is 0.416
```

They are generated with the following featureset and XGBoost hyperparameters:

1. Featureset:

```json
{
    "featureset": {
        "features": [
            {
                "name": "name_match",
                "params": ["keywords"],
                "template_language": "mustache",
                "template": {
                    "match": {
                        "name": "{{keywords}}"
                    }
                }
            },
            {
                "name": "name_match_phrase",
                "params": ["keywords"],
                "template_language": "mustache",
                "template": {
                    "match_phrase": {
                        "name": {
                            "query": "{{keywords}}",
                            "slop": 6
                        }
                    }
                }
            },
            {
                "name": "artistName_match_phrase",
                "params": ["keywords"],
                "template_language": "mustache",
                "template": {
                    "match_phrase": {
                        "artistName": {
                            "query": "{{keywords}}",
                            "slop": 6
                        }
                    }
                }
            },
            {
                "name": "shortDescription_match_phrase",
                "params": ["keywords"],
                "template_language": "mustache",
                "template": {
                    "match_phrase": {
                        "shortDescription": {
                            "query": "{{keywords}}",
                            "slop": 6
                        }
                    }
                }
            },
            {
                "name": "longDescription_match_phrase",
                "params": ["keywords"],
                "template_language": "mustache",
                "template": {
                    "match_phrase": {
                        "longDescription": {
                            "query": "{{keywords}}",
                            "slop": 6
                        }
                    }
                }
            },
            {
                "name": "customerReviewAverage",
                "template_language": "mustache",
                "template": {
                    "function_score": {
                        "functions": [
                            {
                                "field_value_factor": {
                                    "field": "customerReviewAverage",
                                    "missing": 4.5
                                }
                            }
                        ],
                        "query": {
                            "match_all": {}
                        }
                    }
                }
            },
            {
                "name": "customerReviewCount",
                "template_language": "mustache",
                "template": {
                    "function_score": {
                        "functions": [
                            {
                                "field_value_factor": {
                                    "field": "customerReviewCount",
                                    "missing": 2
                                }
                            }
                        ],
                        "query": {
                            "match_all": {}
                        }
                    }
                }
            }
        ]
    }
}
```

2. XGBoost hyperparameters:

```python
num_rounds = 5
xgb_params = {"objective": "reg:logistic"}
```

Hyperparameters not mentioned above all take default values as shown in [XGBoos documentation](https://xgboost.readthedocs.io/en/stable/parameter.html)

Interestingly, the following operations do not improve MRR:

-   Reduce max_depth to 2 or 3
-   Increase number of boosting round `num_rounds` to 50
-   Subsample 0.9
-   Use `rank:pairwise` for pairwise ranking objective function
-   Use `rank:ndcg` for listwise ranking objective function

---

## Remaining questions

> There are two primary ways you can bring prior click data into play:

> -   Offline we could aggregate it and then join it to our products during indexing time as a field on the document, much like sales rank.  We could then use a function score query to access it in our featureset just like price and sales rank. The downside is it is hard to update and, at least as far as the class is concerned, it would require us to overhaul our indexing code!

> -   We can inject it at feature logging time and query time via a query time join.  This is admittedly trickier to make perform, but has the benefit that it is easy to update and allows for more flexibility in our query design.

TODO: How to register query-item pair features that are more complicated to computed,
such as purchase log-based query-item counts, query-specific features,
query-item semantic similarities.

# Week 3 Project Self-Assessment

## Instructor's question set

### 1. Query classification:

**1a: For query classification: How many unique categories did you see in your rolled up training data when you set the minimum number of queries per category to 1000? To 10000?**

| min_n_queries_per_category | n_unique_categories |
| :------------------------- | :------------------ |
| 1                          | 1486                |
| 1000                       | 388                 |
| 10000                      | 70                  |

**1b. For query classification: What were the best values you achieved for R@1, R@3, and R@5? You should have tried at least a few different models, varying the minimum number of queries per category, as well as trying different fastText parameters or query normalization. Report at least 2 of your runs.**

| metrics | best_value | min_queries | fasttext parameters           |
| :------ | :--------- | :---------- | :---------------------------- |
| R@1     | 0.588      | 10_000      | lr = 0.5; epoch = 25          |
| R@2     | 0.723      | 10_000      | (default) lr = 0.1; epoch = 5 |
| R@3     | 0.782      | 10_000      | (default) lr = 0.1; epoch = 5 |

-   Example run 1: Model with best testing performance with `min_queries==1_000`

```bash
min_queries = 1000


# Training
/home/gitpod/fastText-0.9.2/fasttext supervised  -input training_queries.txt -output query_classifier  -lr 0.5 -epoch 25
Read 0M words
Number of words:  7565
Number of labels: 387
Progress: 100.0% words/sec/thread:     316 lr:  0.000000 avg.loss:  3.315475 ETA:   0h 0m 0s


# Testing performances
/home/gitpod/fastText-0.9.2/fasttext test query_classifier.bin  testing_queries.txt  1
N       10000
P@1     0.527
R@1     0.527

/home/gitpod/fastText-0.9.2/fasttext test query_classifier.bin  testing_queries.txt  2
N       10000
P@2     0.325
R@2     0.65

/home/gitpod/fastText-0.9.2/fasttext test query_classifier.bin  testing_queries.txt  3
N       10000
P@3     0.235
R@3     0.705


```

-   Example run 2: Model with best testing performance with `min_queries==10_000`

```bash
min_queries = 10000


# Training
/home/gitpod/fastText-0.9.2/fasttext supervised  -input training_queries.txt -output query_classifier
Read 0M words
Number of words:  7722
Number of labels: 69
Progress: 100.0% words/sec/thread:    1448 lr:  0.000000 avg.loss:  2.141264 ETA:   0h 0m 0s


# Testing performances

/home/gitpod/fastText-0.9.2/fasttext test query_classifier.bin  testing_queries.txt  1
N       10000
P@1     0.583
R@1     0.583

/home/gitpod/fastText-0.9.2/fasttext test query_classifier.bin  testing_queries.txt  2
N       10000
P@2     0.361
R@2     0.723

/home/gitpod/fastText-0.9.2/fasttext test query_classifier.bin  testing_queries.txt  3
N       10000
P@3     0.261
R@3     0.782
```

---

### 2. Integrating query classification with search:

NOTE: The model used for level2 tasks is the last model with the following parameters:

```bash
min_queries = 10_000
lr = 0.5
epoch = 25
```

**2a. For integrating query classification with search: Give 2 or 3 examples of queries where you saw a dramatic positive change in the results because of filtering. Make sure to include the classifier output for those queries.**

Two examples with better results:

1. Query = 'ipad'

-   Without category-based filtering, OpenSearch returns 5657 items that are a mixture of ipad itself and accessories such as `"Apple\u00ae - iPad\u2122 Digital Camera Connection Kit"` and `"Apple\u00ae - Smart Case for Apple\u00ae iPad\u00ae 2 and iPad (3rd Generation) - Dark Gray"`

-   With category-based filtering, the search result size is drastically reduced to 48 and scoped to be for iPad, the tablet:

```bash
# 1. Classification results
Classifier output is: (('__label__pcmcat209000050007',), array([0.69972545]))
Corresponding category name is: ['iPad']

# 2. Filter setting
{
    "multi_match": {
        "query": "pcmcat209000050007",
        "fields": [
            "categoryLeaf",
            "categoryPathIds"
        ]
    }
}

# 3. Excerpt of search results
{
  ...
  "hits": {
    "total": {
      "value": 48,
      "relation": "eq"
    },
    "max_score": 715.55817,
    "hits": [
      {
        "_index": "bbuy_products",
        "_type": "_doc",
        "_id": "1945531",
        "_score": 715.55817,
        "_source": {
          "categoryPathIds": [
            "cat00000",
            "abcat0500000",
            "pcmcat209000050006",
            "pcmcat209000050007"
          ],
          "categoryPath": [
            "Best Buy",
            "Computers & Tablets",
            "Tablets & iPad",
            "iPad"
          ],
          "name": [
            "Apple\u00ae - iPad\u00ae 2 with Wi-Fi - 16GB - Black"
          ],
          "shortDescription": [
            "9.7\" widescreen display; 802.11a/b/g/n Wi-Fi; Bluetooth; iBooks support; measures just 0.34\" thin and weighs only 1.35 lbs."
          ]
        }
      },
      ...
  }
```

2. Query = 'ipod'
   Similar to "ipad", the query "ipod" also benefits from filtering by predicted category:

-   Without category-based filtering, OpenSearch returns 8433 items that are a mixture of ipod itself and accessories such as `"Apple\u00ae - Earbuds for Select Apple\u00ae iPod\u00ae Models"` and `"Rocketfish\u2122 - Premium Vehicle Charger for Apple\u00ae iPad\u2122, iPhone\u00ae and iPod\u00ae"`

-   With category-based filtering, the search result size is drastically reduced to 129 and scoped to be for iPod, the music player:

```bash
# 1. Classification results
Classifier output is: (('__label__abcat0201011',), array([0.64042461]))
Corresponding category name is: ['All iPod & MP3 Players']

# 2. Filter setting
{
    "multi_match": {
        "query": "abcat0201011",
        "fields": [
            "categoryLeaf",
            "categoryPathIds"
        ]
    }
}

# 3. Excerpt of search results
{
  ...,
  "hits": {
    "total": {
      "value": 129,
      "relation": "eq"
    },
    "max_score": 559.9455,
    "hits": [
      {
        "_index": "bbuy_products",
        "_type": "_doc",
        "_id": "3109302",
        "_score": 559.9455,
        "_source": {
          "categoryPathIds": [
            "cat00000",
            "abcat0200000",
            "abcat0201000",
            "abcat0201011"
          ],
          "categoryPath": [
            "Best Buy",
            "Audio & MP3",
            "iPod & MP3 Players",
            "All iPod & MP3 Players"
          ],
          "name": [
            "Apple\u00ae - iPod touch\u00ae 8GB* MP3 Player (4th Generation - Latest Model) - White"
          ],
          "shortDescription": [
            "iOS 5, iCloud, iMessage, FaceTime camera, HD video recording, Retina display, Multi-Touch interface, 3.5\" widescreen display, Wi-Fi web browsing"
          ]
        }
      },
    ...
  }

```

In summary, these positive examples demonstrated the better-controlled search scope via query understanding by classification.

**2b. For integrating query classification with search: Give 2 or 3 examples of queries where filtering hurt the results, either because the classifier was wrong or for some other reason. Again, include the classifier output for those queries.**

Two examples where the results become worse:

1. Query = "apple"

The results have incorrect class label despite the apparently high `pred_proba`:

```bash
# Classifier output
query: 'apple'
Classifier output is: (('__label__cat02015',), array([0.76973659]))
Corresponding category name is: ['Movies & TV Shows']
```

Applying the category filter as follows leads to "no results returned":

```bash
# Filter setting
{
    "multi_match": {
        "query": "cat02015",
        "fields": [
            "categoryLeaf",
            "categoryPathIds"
        ]
    }
}
```

2. Query = 'iphone'

```bash
query: 'iphone'
Classifier output is: (('__label__cat02015',), array([0.76973659]))
Corresponding category name is: ['Movies & TV Shows']
```

Applying the category filter as follows leads to "no results returned":

```bash
# Filter setting
{
    "multi_match": {
        "query": "cat02015",
        "fields": [
            "categoryLeaf",
            "categoryPathIds"
        ]
    }
}
```

In summary, both examples share the common issue that the classifier is wrong. This is the most extreme case when classification-based query understanding harms the search results. Another likely reason is when the search scope become overly narrow. Unfortunately, I do not have enough time to demonstrate such examples in this submission.

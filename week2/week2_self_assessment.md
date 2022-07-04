# Week 2 Project Self-Assessment

## Instructor's question set

**1. For classifying product names to categories:**

**_a. What precision (P@1) were you able to achieve?_**

With simple filter (exclude categories with n_unique_products < 500) and basic normalization applied to product names (details described below), I get testing precision **P@1 = 0.969**.

```bash
# Train / testing data generation steps
# 1. Category filtering
$ python week2/createContentTrainingData.py \
--min_products 500
--output /workspace/datasets/fasttext/pruned_labeled_products.txt

$ wc -l pruned_labeled_products.txt
23902 pruned_labeled_products.txt


# 2. Shuffle to reduce the chance of being affected by systematic biases
$ shuf /workspace/datasets/fasttext/pruned_labeled_products.txt > \
/workspace/datasets/fasttext/shuffled_pruned_labeled_products.txt

# Train-test data preparation
$ cd /workspace/datasets/fasttext/
$ head -n 10000 shuffled_pruned_labeled_products.txt > pruned_training_data.txt
$ tail -n 10000 shuffled_pruned_labeled_products.txt > pruned_test_data.txt
```

**_b. What fastText parameters did you use?_**

-   Learning rate = 1.0
-   Number of epochs = 25
-   Word ngrams = 2

```bash
## fastText parameters as command line arguments

# Training
$ ~/fastText-0.9.2/fasttext supervised \
-input pruned_training_data.txt \
-output product_classifier_pruned_bigram \
-lr 1.0 -epoch 25 -wordNgrams 2

# Testing
$ ~/fastText-0.9.2/fasttext test \
product_classifier_pruned_bigram.bin pruned_test_data.txt
N       10000
P@1     0.969
R@1     0.969
```

**_c. How did you transform the product names?_**

-   Lowercase
-   Exclude non-word characters

    ("word" = alphanumerical character and underscore "\_")

-   Stemming with Snowball stemmer

**_d. How did you prune infrequent category labels, and how did that affect your precision?_**

I simply excluded all categories that have n_unique_products < 500.
It results in a marked improvement in P@1--an increase from 0.615 to 0.969

-   Without category filtering:

```bash
$ ~/fastText-0.9.2/fasttext supervised -input training_data.txt \
-output product_classifier_tune3_lr1_epoch25_bigrams \
-lr 1.0 -epoch 25 -wordNgrams 2
Read 0M words
Number of words:  11191
Number of labels: 1362
Progress: 100.0% words/sec/thread:     861 lr:  0.000000 avg.loss:  1.241515 ETA:   0h 0m 0s


$ ~/fastText-0.9.2/fasttext test \
product_classifier_tune3_lr1_epoch25_bigrams.bin test_data.txt
N       9659
P@1     0.615
R@1     0.615
```

-   With category filtering

```bash
$ ~/fastText-0.9.2/fasttext supervised \
-input pruned_training_data.txt -output product_classifier_pruned_bigram \
-lr 1.0 -epoch 25 -wordNgrams 2
Read 0M words
Number of words:  5338
Number of labels: 25
Progress: 100.0% words/sec/thread:    9425 lr:  0.000000 avg.loss:  0.034142 ETA:   0h 0m 0s

$ ~/fastText-0.9.2/fasttext test \
product_classifier_pruned_bigram.bin pruned_test_data.txt
N       10000
P@1     0.969
R@1     0.969
```

**_e. How did you prune the category tree, and how did that affect your precision?_**

I did not implement the leave node merging approach that is optional.
However, this is exactly the type of approach I want to use
for creating class labels for query classification.

---

**2. For deriving synonyms from content:**

a. What were the results for your best model in the tokens used for evaluation?
The best skipgram model has an average loss of 1.2236 (loss = fastText's default softmax loss)

b. What fastText parameters did you use?

-   Learning rate = 0.05
-   Number of epochs = 25
-   Rare word filter threshold minCount (minimal number of word occurrences) = 20

The corresponding command line is:

```bash
$ ~/fastText-0.9.2/fasttext skipgram \
-input /workspace/datasets/fasttext/normalized_titles.txt \
-output /workspace/datasets/fasttext/normalized_title_model_lrtest \
-minCount 20 -epoch 25 -lr 0.05
Read 1M words
Number of words:  3709
Number of labels: 0
Progress: 100.0% words/sec/thread:    3826 lr:  0.000000 avg.loss:  1.223611 ETA:   0h 0m 0s
```

c. How did you transform the product names?

Ref: [Path to implementation](https://github.com/shandou/search_with_machine_learning_course/blob/97750317bc6da681635a5b1226fce22a6b300339/week2/utilities/synonym_utils.py#L136-L143)

-   Lower case
-   Strip accents (e.g., transform Â -> A)
-   Remove non-word characters (i.e., exclude [^a-za-z0-9_])
-   Lemmatize to reduce inflected word variants to root words (via nltk's wordnet lemmatizer)

```python
# Implementation example in week2/utilities/synonym_utils.py
>>> df_titles[COLNAME_PRODUCT] = df_titles[COLNAME_PRODUCT].apply(
...     lambda x: TextNormalizer(x.lower())
...     .strip_accents()
...     .remove_non_word()
...     .tokenize()
...     .lemmatize()
...     .done()
... )
```

---

**3. For integrating synonyms with search:**

Field mapping after adding synonym filter
```json
...
"name" : {
          "type" : "text",
          "fields" : {
            "hyphens" : {
              "type" : "text",
              "analyzer" : "smarter_hyphens"
            },
            "keyword" : {
              "type" : "keyword",
              "ignore_above" : 2048
            },
            "suggest" : {
              "type" : "completion",
              "analyzer" : "simple",
              "preserve_separators" : true,
              "preserve_position_increments" : true,
              "max_input_length" : 50
            },
            "synonyms" : {
              "type" : "text",
              "analyzer" : "synonym"
            }
          },
          "analyzer" : "english"
        },
...
```

Test the synonym analyzer:

- Request:
```json
GET /bbuy_products/_analyze
{
  "analyzer": "synonym",
  "explain": "true",
  "text": ["iphone"]
}
```

- Response:
```json
...
    "tokenfilters" : [
      {
        "name" : "synonym_filter",
        "tokens" : [
          {
            "token" : "iphone",
            "start_offset" : 0,
            "end_offset" : 6,
            "type" : "<ALPHANUM>",
            "position" : 0,
            "bytes" : "[69 70 68 6f 6e 65]",
            "positionLength" : 1,
            "termFrequency" : 1
          },
          {
            "token" : "apple",
            "start_offset" : 0,
            "end_offset" : 6,
            "type" : "SYNONYM",
            "position" : 0,
            "bytes" : "[61 70 70 6c 65]",
            "positionLength" : 1,
            "termFrequency" : 1
          },
          {
            "token" : "ipod",
            "start_offset" : 0,
            "end_offset" : 6,
            "type" : "SYNONYM",
            "position" : 0,
            "bytes" : "[69 70 6f 64]",
            "positionLength" : 1,
            "termFrequency" : 1
          },
          {
            "token" : "r",
            "start_offset" : 0,
            "end_offset" : 6,
            "type" : "SYNONYM",
            "position" : 0,
            "bytes" : "[72]",
            "positionLength" : 1,
            "termFrequency" : 1
          },
          {
            "token" : "ipad",
            "start_offset" : 0,
            "end_offset" : 6,
            "type" : "SYNONYM",
            "position" : 0,
            "bytes" : "[69 70 61 64]",
            "positionLength" : 1,
            "termFrequency" : 1
          }
        ]
      },
...
```

a. How did you transform the product names (if different than previously)?

I apply the same text normalization steps as level2.


b. What threshold score did you use?

Cosine similarity >= 0.75

c. Were you able to find the additional results by matching synonyms?

Not always for the three test cases listed in the instructions: `earbuds`, `nespresso`, and `dslr`.

| query     | synonyms in synonyms.csv                                                                                 | n_hits (w/o synonyms) | n_hits (with synonyms) | Getting more results with synonym matching? |
|-----------|----------------------------------------------------------------------------------------------------------|-----------------------|------------------------|---------------------------------------------|
| earbuds   | earbuds,aerosport                                                                                        | 1205                  | 1076                   | FALSE                                       |
| nespresso | NULL                                                                                                     | 8                     | 8                      | N/A                                         |
| dslr      | lens,dslr,slr,telephoto<br/>dslr,slr,300mm,lens<br/>300mm,200mm,85mm,250mm,dslr,telephoto,slr,70mm,105mm | 2837                  | 3802                   | TRUE                                        |

Likely explanations:
- The reduced results for `earbuds` are not yet explainable
- The unchanged result count for `nespresso` is because of a lack of synonyms for origin word
- The increased results for `dslr` makes sense given the rich set of synonyms it has


**4. For classifying reviews:**

Level 4 is optional. Due to time constraint, I will have to leave it for a later time and not include it for week2 submission :(

---

## Self guided question-answering articulations

**Q1. For information retrieval, why does stemming appear more often than
lemmatization for normalizing morphological variations?**

A1: Stemming is a faster process than lemmatization as
stemming chops off the word irrespective of the context,
whereas the latter is context-dependent.
Lemmatization is preferred for context analysis,
whereas stemming is recommended when the context is not important.

**Q2: For stemmer, why is Snowball recommended for week2 project level1,
but not Porter?**

A1: TODO


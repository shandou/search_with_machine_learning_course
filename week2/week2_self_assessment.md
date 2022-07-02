# Week 2 Project Self-Assessment

## Instructor's question set

---

**1. For classifying product names to categories:**

___a. What precision (P@1) were you able to achieve?___

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

___b. What fastText parameters did you use?___

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

___c. How did you transform the product names?___

-   Lowercase
-   Exclude non-word characters 

    ("word" = alphanumerical character and underscore "_")

-   Stemming with Snowball stemmer

___d. How did you prune infrequent category labels, and how did that affect your precision?___

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

___e. How did you prune the category tree, and how did that affect your precision?___

I did not implement the leave node merging approach that is optional.
However, this is exactly the type of approach I want to use
for creating class labels for query classification.

---

**2. For deriving synonyms from content:**

a. What were the results for your best model in the tokens used for evaluation?

b. What fastText parameters did you use?

c. How did you transform the product names?

**3. For integrating synonyms with search:**

a. How did you transform the product names (if different than previously)?

b. What threshold score did you use?

c. Were you able to find the additional results by matching synonyms?

**4. For classifying reviews:**

a. What precision (P@1) were you able to achieve?

b. What fastText parameters did you use?

c. How did you transform the review content?

d. What else did you try and learn?

---

## Self guided question and answering articulations

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

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

**2a. For integrating query classification with search: Give 2 or 3 examples of queries where you saw a dramatic positive change in the results because of filtering. Make sure to include the classifier output for those queries.**

TODO

**2b. For integrating query classification with search: Give 2 or 3 examples of queries where filtering hurt the results, either because the classifier was wrong or for some other reason. Again, include the classifier output for those queries.**

TODO

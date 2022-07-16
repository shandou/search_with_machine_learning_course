# Week 4 Project Self-Assessment

## Part 1: Create and index document vectors

### Step 1: Obtain pre-trained model

The model printouts for `sentence-transformers/all-MiniLM-L6-v2` are:

```python
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False})
  (2): Normalize()
)
```

### Step 2: Encode names using the model

Core logic used to encode names ([link to code lines](https://github.com/shandou/search_with_machine_learning_course/blob/bed44dca9c9f46efc87a52bdc0d9e52070d6262b/week4/utilities/index_products.py#L169-L236)):

```python
def index_file(file, index_name, reduced=False):
    logger.info("Creating Model")

    # IMPLEMENT ME: instantiate the sentence transformer model!
    model: sentence_transformers.SentenceTransformer = SentenceTransformer(
        PRETRAINED_MODEL_NAME
    )

    logger.info("Ready to index")

    docs_indexed = 0
    client = get_opensearch()
    logger.info(f"Processing file : {file}")
    tree = etree.parse(file)
    root = tree.getroot()
    children = root.findall("./product")
    docs = []
    names = []
    # IMPLEMENT ME: maintain the names array parallel to docs,
    # and then embed them in bulk and add them to each doc,
    # in the '_source' part of each docs entry, before calling bulk
    # to index them 200 at a time. Make sure to clear the names array
    # when you clear the docs array!
    for child in children:
        doc = {}
        for idx in range(0, len(mappings), 2):
            xpath_expr = mappings[idx]
            key = mappings[idx + 1]
            doc[key] = child.xpath(xpath_expr)
        # print(doc)
        if "productId" not in doc or len(doc["productId"]) == 0:
            continue
        if "name" not in doc or len(doc["name"]) == 0:
            continue
        if reduced and (
            "categoryPath" not in doc
            or "Best Buy" not in doc["categoryPath"]
            or "Movies & Music" in doc["categoryPath"]
        ):
            continue
        docs.append(
            {"_index": index_name, "_id": doc["sku"][0], "_source": doc}
        )
        names.append(doc["name"][0])
        # docs.append({'_index': index_name, '_source': doc})
        docs_indexed += 1
        if docs_indexed % 200 == 0:

            # Generate embeddings for the "name" field in batch
            #   (200 records per batch)
            names_embeddings: np_typing.NDArray[np.float32] = model.encode(
                names
            )

            # Add name_embedding as an additional field
            for i, name_embedding in enumerate(names_embeddings):
                docs[i]["_source"].update({"name_embedding": name_embedding})

            logger.info("Indexing")
            bulk(client, docs, request_timeout=60)
            logger.info(f"{docs_indexed} documents indexed")
            docs = []
            names = []
    if len(docs) > 0:
        bulk(client, docs, request_timeout=60)
        logger.info(f"{docs_indexed} documents indexed")
    return docs_indexed
```

### Step 3: Update index settings and field mapping

Specific changes to
`/workspace/search_with_machine_learning_course/week4/conf/bbuy_products.json`:

```json
...
            "name_embedding": {
                "type": "knn_vector",
                "dimension": 384,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "nmslib"
                }
            },
...
```

References:

1. [OpenSearch's k-NN index configuration references](https://opensearch.org/docs/latest/search-plugins/knn/knn-index#knn_vector-data-type)
2. [Mapping syntax examples for k-NN field](https://opensearch.org/docs/latest/search-plugins/knn/approximate-knn/)

## Step 4: Index

Test results via OpenSearch dashboard dev tool:

-   Query:

```json
GET bbuy_products/_search
{"query": { "match": { "name": "tv"}}}
```

-   Results excerpt:

```json
...
          "longDescription" : [
            """The lowboy design of this 3-shelf TV stand adds a sleek look to your living space while displaying TVs up to 63" and storing your A/V components."""
          ],
          "longDescriptionHtml" : [
            """The lowboy design of this 3-shelf TV stand adds a sleek look to your living space while displaying TVs up to 63" and storing your A/V components."""
          ],
          "features" : [
            """Holds TVs up to 63"
Top shelf supports up to 200 lbs. for reliable strength.""",
            """Hardwood ash veneer with chocolate finish
For a stylish look.""",
            """3 shelves
Support your A/V components. Middle shelf supports up to 51 lbs; bottom shelf supports up to 76 lbs.""",
            """IR-friendly sliding-glass doors
Allow remote access while concealing your A/V components.""",
            """Rear cable management
Keeps cables hidden and organized.""",
            "For manufacturer's parts or assembly information, please contact Sanus toll-free at 1-800-359-5520.",
            "Assembly required.",
            "Put it all together with our professional in-home assembly service"
          ],
          "name_embedding" : [
            -0.003042079508304596,
            0.045866064727306366,
            -0.06841698288917542,
            -0.05629890039563179,
            -0.08066531270742416,
            0.05543777346611023,
            -0.09256944805383682,
            ...
    ...
```

## Part 2: Update query client to perform vector search

### Step 1: Obtain pre-trained model

Similar to steps taken inside `index_products.py`

```python
import sentence_transformers
...
from sentence_transformers import SentenceTransformer

PRETRAINED_MODEL_NAME: str = "all-MiniLM-L6-v2"
...
    model: sentence_transformers.SentenceTransformer = SentenceTransformer(
        PRETRAINED_MODEL_NAME
    )
...
```

### Step 2: Implement nearest-neighbor vector query

By following query examples in [OpenSearch documentation](https://opensearch.org/docs/latest/search-plugins/knn/knn-score-script/),
my query object specific to k-NN vector search looks like the following
([link to relevant code lines](https://github.com/shandou/search_with_machine_learning_course/blob/f8d73ddb2fcacd4b411c9b1384c94fe90f006197/utilities/query.py#L132-L190)):

```python
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
```

Command used to run vector search:

```bash
# PWD == /workspace/search_with_machine_learning_course
$ python utilities/query.py --vector
```

Example searches (with size = 10, source=`["name", "shortDescription", "categoryPath", "categoryPathIds"]`):

1. query = "ipod"

```
{
  "took": 496,
  "timed_out": false,
  "_shards": {
    "total": 1,
    "successful": 1,
    "skipped": 0,
    "failed": 0
  },
  "hits": {
    "total": {
      "value": 10000,
      "relation": "gte"
    },
    "max_score": 1.7386136,
    "hits": [
      {
        "_index": "bbuy_products",
        "_type": "_doc",
        "_id": "8501105",
        "_score": 1.7386136,
        "_source": {
          "categoryPathIds": [
            "cat00000",
            "abcat0200000",
            "abcat0201000",
            "abcat0201009"
          ],
          "categoryPath": [
            "Best Buy",
            "Audio & MP3",
            "iPod & MP3 Players",
            "iPods"
          ],
          "name": [
            "Apple\u00ae - iPod\u00ae shuffle 2GB* MP3 Player - Silver"
          ],
          "shortDescription": [
            "Just over 0.4\" thin and only 0.6 oz.; up to 12 hours of battery life; USB 2.0 compatible through dock connector"
          ]
        }
      },
      {
        "_index": "bbuy_products",
        "_type": "_doc",
        "_id": "8500516",
        "_score": 1.7350397,
        "_source": {
          "categoryPathIds": [
            "cat00000",
            "abcat0200000",
            "abcat0201000",
            "abcat0201009"
          ],
          "categoryPath": [
            "Best Buy",
            "Audio & MP3",
            "iPod & MP3 Players",
            "iPods"
          ],
          "name": [
            "Apple\u00ae - iPod\u00ae nano 8GB* MP3 Player - Silver"
          ],
          "shortDescription": [
            "Measures just under 0.3\" thin and weighs only 1.74 oz.; up to 24 hours of music playback; 2\" color LCD with LED backlight; iTunes"
          ]
        }
      },
      {
        "_index": "bbuy_products",
        "_type": "_doc",
        "_id": "8500491",
        "_score": 1.7342815,
        "_source": {
          "categoryPathIds": [
            "cat00000",
            "abcat0200000",
            "abcat0201000",
            "abcat0201009"
          ],
          "categoryPath": [
            "Best Buy",
            "Audio & MP3",
            "iPod & MP3 Players",
            "iPods"
          ],
          "name": [
            "Apple\u00ae - iPod\u00ae nano 4GB* MP3 Player - Silver"
          ],
          "shortDescription": [
            "Measures just under 0.3\" thin and weighs only 1.74 oz.; up to 24 hours of music playback; 2\" color LCD with LED backlight; iTunes"
          ]
        }
      },
      {
        "_index": "bbuy_products",
        "_type": "_doc",
        "_id": "8500883",
        "_score": 1.7337548,
        "_source": {
          "categoryPathIds": [
            "cat00000",
            "abcat0200000",
            "abcat0201000",
            "abcat0201009"
          ],
          "categoryPath": [
            "Best Buy",
            "Audio & MP3",
            "iPod & MP3 Players",
            "iPods"
          ],
          "name": [
            "Apple\u00ae - iPod\u00ae shuffle 2GB* MP3 Player - Purple"
          ],
          "shortDescription": [
            "Just over 0.4\" thin and only 0.6 oz.; up to 12 hours of battery life; USB 2.0 compatible through dock connector"
          ]
        }
      },
      {
        "_index": "bbuy_products",
        "_type": "_doc",
        "_id": "8500552",
        "_score": 1.7319539,
        "_source": {
          "categoryPathIds": [
            "cat00000",
            "abcat0200000",
            "abcat0201000",
            "abcat0201004"
          ],
          "categoryPath": [
            "Best Buy",
            "Audio & MP3",
            "iPod & MP3 Players",
            "Music & Photo & Video"
          ],
          "name": [
            "Apple\u00ae - iPod\u00ae"
          ],
          "shortDescription": []
        }
      },
      {
        "_index": "bbuy_products",
        "_type": "_doc",
        "_id": "8771947",
        "_score": 1.7298236,
        "_source": {
          "categoryPathIds": [
            "cat00000",
            "abcat0200000",
            "abcat0201000",
            "abcat0201009"
          ],
          "categoryPath": [
            "Best Buy",
            "Audio & MP3",
            "iPod & MP3 Players",
            "iPods"
          ],
          "name": [
            "Apple\u00ae - iPod nano\u00ae 8GB* MP3 Player - Purple"
          ],
          "shortDescription": [
            "New curved aluminum and glass design and stunning features; Genius music playlist; view photos and videos in either portrait or landscape."
          ]
        }
      },
      {
        "_index": "bbuy_products",
        "_type": "_doc",
        "_id": "9229792",
        "_score": 1.7292821,
        "_source": {
          "categoryPathIds": [
            "cat00000",
            "abcat0200000",
            "abcat0201000",
            "abcat0201009"
          ],
          "categoryPath": [
            "Best Buy",
            "Audio & MP3",
            "iPod & MP3 Players",
            "iPods"
          ],
          "name": [
            "Apple\u00ae - iPod shuffle\u00ae 2GB* MP3 Player - Silver"
          ],
          "shortDescription": [
            "Only 0.3\" thin and 1.8\" tall; up to 10 hours of battery life; includes Apple Earphones with remote"
          ]
        }
      },
      {
        "_index": "bbuy_products",
        "_type": "_doc",
        "_id": "8501551",
        "_score": 1.7285709,
        "_source": {
          "categoryPathIds": [
            "cat00000",
            "abcat0200000",
            "abcat0201000",
            "abcat0201009"
          ],
          "categoryPath": [
            "Best Buy",
            "Audio & MP3",
            "iPod & MP3 Players",
            "iPods"
          ],
          "name": [
            "Apple\u00ae - iPod\u00ae shuffle 1GB* MP3 Player - Purple"
          ],
          "shortDescription": [
            "Just over 0.4\" thin and only 0.6 oz.; up to 12 hours of battery life; USB 2.0 compatible through dock connector"
          ]
        }
      },
      {
        "_index": "bbuy_products",
        "_type": "_doc",
        "_id": "9229505",
        "_score": 1.7269006,
        "_source": {
          "categoryPathIds": [
            "cat00000",
            "abcat0200000",
            "abcat0201000",
            "abcat0201009"
          ],
          "categoryPath": [
            "Best Buy",
            "Audio & MP3",
            "iPod & MP3 Players",
            "iPods"
          ],
          "name": [
            "Apple\u00ae - iPod shuffle\u00ae 4GB* MP3 Player - Silver"
          ],
          "shortDescription": [
            "Only 0.3\" thin and 1.8\"tall; up to 10 hours of battery life; includes Apple Earphones with remote"
          ]
        }
      },
      {
        "_index": "bbuy_products",
        "_type": "_doc",
        "_id": "9229532",
        "_score": 1.7269006,
        "_source": {
          "categoryPathIds": [
            "cat00000",
            "abcat0200000",
            "abcat0201000",
            "abcat0201009"
          ],
          "categoryPath": [
            "Best Buy",
            "Audio & MP3",
            "iPod & MP3 Players",
            "iPods"
          ],
          "name": [
            "Apple\u00ae - iPod shuffle\u00ae 4GB* MP3 Player - Silver"
          ],
          "shortDescription": [
            "Only 0.3\" thin and 1.8\"tall; up to 10 hours of battery life; includes Apple Earphones with remote"
          ]
        }
      }
    ]
  }
}
```

2. query = "earbuds"

```json
{
    "took": 505,
    "timed_out": false,
    "_shards": {
        "total": 1,
        "successful": 1,
        "skipped": 0,
        "failed": 0
    },
    "hits": {
        "total": {
            "value": 10000,
            "relation": "gte"
        },
        "max_score": 1.7466335,
        "hits": [
            {
                "_index": "bbuy_products",
                "_type": "_doc",
                "_id": "5143501",
                "_score": 1.7466335,
                "_source": {
                    "categoryPathIds": [
                        "cat00000",
                        "pcmcat248700050021",
                        "pcmcat244200050008",
                        "pcmcat236800050000"
                    ],
                    "categoryPath": [
                        "Best Buy",
                        "Home",
                        "Cartoon & Popular Characters",
                        "Hello Kitty Merchandise"
                    ],
                    "name": ["Hello Kitty - Earbud Headphones"],
                    "shortDescription": [
                        "Sound-isolating design; 40mm drivers; ferrite magnets"
                    ]
                }
            },
            {
                "_index": "bbuy_products",
                "_type": "_doc",
                "_id": "2642289",
                "_score": 1.7466334,
                "_source": {
                    "categoryPathIds": [
                        "cat00000",
                        "pcmcat248700050021",
                        "pcmcat244200050008",
                        "pcmcat236800050000"
                    ],
                    "categoryPath": [
                        "Best Buy",
                        "Home",
                        "Cartoon & Popular Characters",
                        "Hello Kitty Merchandise"
                    ],
                    "name": ["Hello Kitty - Earbud Headphones"],
                    "shortDescription": [
                        "Hello Kitty design; sound-isolating design; 10mm drivers; ferrite magnet"
                    ]
                }
            },
            {
                "_index": "bbuy_products",
                "_type": "_doc",
                "_id": "3793876",
                "_score": 1.6926315,
                "_source": {
                    "categoryPathIds": [
                        "cat00000",
                        "abcat0200000",
                        "abcat0204000",
                        "pcmcat143000050007"
                    ],
                    "categoryPath": [
                        "Best Buy",
                        "Audio & MP3",
                        "Headphones",
                        "Earbud Headphones"
                    ],
                    "name": ["Sennheiser - Earbud Headphones"],
                    "shortDescription": [
                        "From our expanded online assortment; insulates against noise; neodymium magnets; dynamic drivers"
                    ]
                }
            },
            {
                "_index": "bbuy_products",
                "_type": "_doc",
                "_id": "1535257",
                "_score": 1.6926315,
                "_source": {
                    "categoryPathIds": [
                        "cat00000",
                        "abcat0200000",
                        "abcat0204000",
                        "pcmcat144700050004"
                    ],
                    "categoryPath": [
                        "Best Buy",
                        "Audio & MP3",
                        "Headphones",
                        "All Headphones"
                    ],
                    "name": ["Sennheiser - Earbud Headphones"],
                    "shortDescription": [
                        "From our expanded online assortment; smart remote with microphone; insulates against noise"
                    ]
                }
            },
            {
                "_index": "bbuy_products",
                "_type": "_doc",
                "_id": "9834152",
                "_score": 1.6926315,
                "_source": {
                    "categoryPathIds": [
                        "cat00000",
                        "abcat0200000",
                        "abcat0204000",
                        "pcmcat143000050007"
                    ],
                    "categoryPath": [
                        "Best Buy",
                        "Audio & MP3",
                        "Headphones",
                        "Earbud Headphones"
                    ],
                    "name": ["Sennheiser - Earbud Headphones"],
                    "shortDescription": [
                        "From our expanded online assortment; built-in volume control; 3-size adapter set; noise-canceling design; ear debris guards; cleaning tool; carrying pouch"
                    ]
                }
            },
            {
                "_index": "bbuy_products",
                "_type": "_doc",
                "_id": "9265145",
                "_score": 1.6926315,
                "_source": {
                    "categoryPathIds": [
                        "cat00000",
                        "abcat0200000",
                        "abcat0204000",
                        "pcmcat143000050007"
                    ],
                    "categoryPath": [
                        "Best Buy",
                        "Audio & MP3",
                        "Headphones",
                        "Earbud Headphones"
                    ],
                    "name": ["Sennheiser - Earbud Headphones"],
                    "shortDescription": [
                        "From our expanded online assortment; earbud design; neodymium magnet; Basswind technology; foam ear cushions; symmetrical cord"
                    ]
                }
            },
            {
                "_index": "bbuy_products",
                "_type": "_doc",
                "_id": "4863468",
                "_score": 1.691933,
                "_source": {
                    "categoryPathIds": [
                        "cat00000",
                        "pcmcat242800050021",
                        "pcmcat208400050005",
                        "pcmcat186400050010"
                    ],
                    "categoryPath": [
                        "Best Buy",
                        "Health, Fitness & Sports",
                        "Portable Fitness Electronics",
                        "Fitness Audio"
                    ],
                    "name": ["iHome - Behind-the-Neck Earbud Headphones"],
                    "shortDescription": [
                        "Folding design; cord clip; adjustable cord length; interchangeable color ear cushions"
                    ]
                }
            },
            {
                "_index": "bbuy_products",
                "_type": "_doc",
                "_id": "9724996",
                "_score": 1.6837424,
                "_source": {
                    "categoryPathIds": [
                        "cat00000",
                        "abcat0500000",
                        "abcat0515000",
                        "abcat0515038",
                        "abcat0515040"
                    ],
                    "categoryPath": [
                        "Best Buy",
                        "Computers & Tablets",
                        "Accessories",
                        "Speakers & Headsets",
                        "Computer Headsets"
                    ],
                    "name": ["Rocketfish\u2122 - Earbud Headphones"],
                    "shortDescription": [
                        "20Hz - 20kHz frequency response; PC adapter for 3.5mm jack; 1 pair comply ear foam tip ear pieces and 3-size silicone ear pieces; cable clip; carrying case"
                    ]
                }
            },
            {
                "_index": "bbuy_products",
                "_type": "_doc",
                "_id": "3726418",
                "_score": 1.6836314,
                "_source": {
                    "categoryPathIds": [
                        "cat00000",
                        "abcat0200000",
                        "abcat0204000",
                        "abcat0204003"
                    ],
                    "categoryPath": [
                        "Best Buy",
                        "Audio & MP3",
                        "Headphones",
                        "Sport Headphones"
                    ],
                    "name": ["Koss - Ear Buds"],
                    "shortDescription": [
                        "4 extra ear cushions; frequency response: 10Hz - 20kHz"
                    ]
                }
            },
            {
                "_index": "bbuy_products",
                "_type": "_doc",
                "_id": "9751308",
                "_score": 1.6803071,
                "_source": {
                    "categoryPathIds": [
                        "cat00000",
                        "abcat0200000",
                        "abcat0204000",
                        "pcmcat143000050011"
                    ],
                    "categoryPath": [
                        "Best Buy",
                        "Audio & MP3",
                        "Headphones",
                        "Over-Ear & On-Ear Headphones"
                    ],
                    "name": ["Sennheiser - Earbud Headset"],
                    "shortDescription": [
                        "From our expanded online assortment; earbud design; built-in microphone; in-line volume control; 3 sizes of silicone ear sleeves"
                    ]
                }
            }
        ]
    }
}
```

---

## Instructor question

**What are you the most proud of over the past 4 weeks?**

1. Submitted all four projects
2. Already putting learning to work: Upon learning the benefits of query classification, I have heightened its priority in our project road map
3. Seeing vector search in action
4. Had a great time learning from the instructors and fellow students :)

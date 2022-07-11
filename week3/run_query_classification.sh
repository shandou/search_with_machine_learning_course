#!/bin/bash
# Bash script for selectively run query classification commands

######################################################
# Preparation
#   Includes the following actions:
#   - Reset command line arguments' values
#   - Validate and parse stdin arguments
######################################################

# Unset parameters to start from clean slate
min_queries=unset
shuffle=unset
train=unset
learning_rate=unset
epoch=unset
word_ngrams=unset
test=unset
top_n=unset


script_dir="/workspace/search_with_machine_learning_course/week3/"
data_dir="/workspace/datasets/fasttext/"
query_data_path=${data_dir}labeled_queries.txt
classifier_name="query_classifier"
fasttext="${HOME}/fastText-0.9.2/fasttext"
virtual_envname="search_with_ml"


usage()
{
    echo "Usage: bash run_query_classifiation.sh
    [--min_queries=MIN_QUERIES]
    [--shuf]
    [--train] [--learning_rate=learning_rate] [--epoch=epoch]
    [--ngrams=word_ngrams]
    [--test] [--top_n=top_n]"
    exit 2
}


PARSED_ARGUMENTS=$(getopt -a -n run_query_classification -o "" \
    --long min_queries:,shuf,train,learning_rate:,epoch:,ngrams:,test,top_n: \
-- "$@")
VALID_ARGUMENTS=$?
if [[ "$VALID_ARGUMENTS" != "0" ]]
then
    usage
fi

echo "PARSED_ARGUMENTS is $PARSED_ARGUMENTS"
eval set -- "$PARSED_ARGUMENTS"

while :
do
    echo $1
    case "$1" in
        --min_queries) min_queries="$2"; shift 2 ;;
        --shuf) shuffle=1; shift ;;
        --train) train=1; shift ;;
        --learning_rate) learning_rate="$2"; shift 2 ;;
        --epoch) epoch="$2"; shift 2 ;;
        --ngrams) word_ngrams="$2"; shift 2 ;;
        --test) test=1; shift ;;
        --top_n) top_n="$2"; shift 2 ;;
        --) shift; break ;;
    esac
done

# Go to directory that stores artifacts needed by fasttext classifier
cd ${data_dir}

# Activate python environment if necessary
if [[ $VIRTUAL_ENV != *"${virtual_envname}"* ]]
then
    echo "Current virtual environment is ${VIRTUAL_ENV} \
    and not ${virtual_envname}"
    cmd="pyenv activate ${virtual_envname}"
    echo "Run '${cmd}' to activate designated virtual environment..."
    ${cmd}
fi


######################################################
# Prune categories and update source of training data
######################################################

# 1. Created labelled query data with category rollup criterion
#   n_queries_per_category >= min_queries
if [[ $min_queries != "unset" ]]
then
    echo "Prepare labelled query data using min_queries = ${min_queries}"
    cmd_parts=(
        "python ${script_dir}create_labeled_queries.py "
        "--min_queries ${min_queries} --output ${query_data_path}"
    )
    cmd=${cmd_parts[@]}
    echo "Prepare labelled data: ${cmd}"
    ${cmd}
fi


######################################################
# Shuffle data, perform train-test split
######################################################

if [[ $shuffle != "unset" ]]
then
    echo "Shuffle labelled queries and perform 50000:10000 train-test split"
    shuf labeled_queries.txt > shuffled_labeled_queries.txt
    head -n 50000 shuffled_labeled_queries.txt > training_queries.txt
    tail -n 10000 shuffled_labeled_queries.txt > testing_queries.txt
fi



######################################################
# Train query classifier
######################################################

fasttext_training_params=""
if [[ $learning_rate != "unset" ]]
then
    fasttext_training_params="${fasttext_training_params} -lr ${learning_rate}"
fi

if [[ $epoch != "unset" ]]
then
    fasttext_training_params="${fasttext_training_params} -epoch ${epoch}"
fi

if [[ $word_ngrams != "unset" ]]
then
    fasttext_training_params=\
    "${fasttext_training_params} -wordNgrams ${word_ngrams}"
fi



# 2. Train classifier model on training set
if [[ $train != "unset" ]]
then
    echo "==================="
    echo "Train classifier..."
    echo "==================="
    echo "Training parameters = ${fasttext_training_params}"
    cmd_parts=(
        "${fasttext} supervised "
        "-input training_queries.txt -output ${classifier_name}"
        "${fasttext_training_params}"
    )
    cmd=${cmd_parts[@]}
    echo "BEGIN TRAINING: ${cmd}"
    ${cmd}
fi


######################################################
# Test query classifier
######################################################

fasttext_testing_params=""
if [[ $top_n != "unset" ]]
then
    fasttext_testing_params="${fasttext_testing_params} ${top_n}"
fi

if [[ $test != "unset" ]]
then
    echo "==================="
    echo "Test classifier..."
    echo "==================="
    echo "Testing parameters = ${fasttext_testing_params}"
    cmd_parts=(
        "${fasttext} test ${classifier_name}.bin "
        "testing_queries.txt"
        "${fasttext_testing_params}"
    )
    cmd=${cmd_parts[@]}
    echo "BEGIN TESTING: ${cmd}"
    ${cmd}
fi

# Return to starting path
cd -
import csv
from nltk import *
import numpy as np
from collections import defaultdict
import pandas as pd

test_data = "roccio_benchmark.csv"

def read_data_file(data_file):
    """
        Input:  csv file with sets of items in each row (item appears only once per row),
        Output: list of lists
    """
    with open(data_file, 'rU') as csv_file:
        reader = csv.reader(csv_file)
        # data = [{'text': row[0], 'class':row[1]} for row in reader]
        data = []
        for row in reader:
            doc = dict()
            doc['category'] = row[1]
            text = row[0]                       # extract text
            text = word_tokenize(text)     # tokenize text
            text = [SnowballStemmer("english").stem(t) for t in text]  # get stems
            doc['text'] = text
            data.append(doc)
    return data


# def tf_idf(term, doc, DOCUMENTS):

def get_vocabulary(DOCUMENTS):
    return sorted(list(reduce(lambda x,y: x.union(y), [set(doc) for doc in DOCUMENTS])))

def get_bag_of_words(DOCUMENTS, vocabulary):
    bag_of_words = []
    for doc in DOCUMENTS:
        counts = defaultdict(int)
        for term in doc:
            counts[term] += 1
        bag_of_words.append(counts)

    bag_of_words_df = pd.DataFrame(bag_of_words)
    return bag_of_words_df

if __name__ == '__main__':
    data = read_data_file(test_data)
    DOCUMENTS = [doc['text'] for doc in data]
    # temp = [set(doc['text']) for doc in data]
    vocabulary = get_vocabulary(DOCUMENTS)

    # matrix =
    #
    bag_of_words = get_bag_of_words(DOCUMENTS, vocabulary)

    # for i in bag_of_words:
    #     print i

    print bag_of_words
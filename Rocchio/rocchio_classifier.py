# Start: 22.03.2016
# Last version: 23.03.2016
"""
    Python implementation of the Rocchio classification algorithm (Nearest centroid classifier)

    Command line:
        $ python rocchio_classifier.py train.csv test.csv

    Python shell:
        >> from rocchio_classifier import Rocchio_classifier
        >> train_data = read_data_file(train.csv, 'train')
        >> classifier = Rocchio_classifier(train_data)
        >> test_data = read_data_file(test.csv, 'test')
        >> predictions = classifier.classify(test_data)
        >> print_predictions(predictions)
"""


import csv
from nltk import *
import argparse
import os
from collections import defaultdict
import pandas as pd
from math import *

# default_train_file = "rocchio_benchmark_train.csv"
# default_test_file = "rocchio_benchmark_test.csv"


def read_data_file(data_file, data_type):
    with open(data_file, 'rU') as csv_file:
        reader = csv.reader(csv_file)
        data = []
        if data_type == 'train':
            for row in reader:
                doc = dict()
                doc['text'] = process_text(row[0])
                doc['category'] = row[1]
                data.append(doc)
        elif data_type == 'test':
            data = [row[0] for row in reader]
    return data


def process_text(text):
    text = word_tokenize(text)     # tokenize text
    text = [SnowballStemmer("english").stem(t) for t in text]  # get stems
    return text


class Rocchio_classifier(object):


    def __init__(self, train_data):
        self.word_counts = self.get_word_counts_in_corpus(train_data)
        self.N = len(self.word_counts)
        self.vector_space_model = self.get_vector_space_model()
        self.vocabulary = self.get_vocabulary()
        self.centroids = self.get_centroids()


    def get_vocabulary(self):
        return filter(lambda term: term != 'CATEGORY', self.vector_space_model.columns.values)

    @staticmethod
    def get_word_counts(text):
        counts = defaultdict(int)
        for term in text:
            counts[term] += 1
        return counts



    def get_word_counts_in_corpus(self, data):
        word_counts = []



        for doc in data:
            counts = self.get_word_counts(doc['text'])
            doc_counts = {'text': counts, 'category': doc['category']}
            word_counts.append(doc_counts)

        return word_counts

    @staticmethod
    def tf_idf(term, doc, N, word_counts):
        term_freq = doc[term]
        max_term_freq = max(doc.values())
        TF = 0.5 + 0.5*(term_freq/max_term_freq)

        def term_freq_in_corpus(term, word_counts):
            freq_in_corpus = 0
            for doc in word_counts:
                if term in doc['text']:
                    freq_in_corpus += 1
            return freq_in_corpus

        n = term_freq_in_corpus(term, word_counts)
        if n == 0:
            n = 1
        IDF = log(N/n, 10)   # log in base of 10
        return TF*IDF


    @staticmethod
    def get_weights_vector(doc, N, word_counts):
        vector_weights = dict()
        for term in doc:
            vector_weights[term] = Rocchio_classifier.tf_idf(term, doc, N, word_counts)
        return vector_weights


    def get_vector_space_model(self):

        vector_space_model = []

        def vector_space_model_in_dataframe(data):
            data_for_df = []
            for doc in data:
                text = doc['text']
                text['CATEGORY'] = doc['category']
                data_for_df.append(text)

            return pd.DataFrame(data_for_df).fillna(0)


        for doc in self.word_counts:
            vector_weights = Rocchio_classifier.get_weights_vector(doc['text'], self.N, self.word_counts)
            doc_weights = {'text': vector_weights, 'category': doc['category']}
            vector_space_model.append(doc_weights)

        return vector_space_model_in_dataframe(vector_space_model)



    """ Easy way to get cetroids """
    def get_centroids(self):
        centroids = dict()
        categories = self.vector_space_model['CATEGORY'].unique()


        for category in categories:
            docs_in_category = self.vector_space_model[self.vector_space_model['CATEGORY'] == category]
            docs_in_category = docs_in_category[self.vocabulary]
            centroid = {term: docs_in_category[term].mean() for term in docs_in_category}
            centroids[category] = centroid
        return centroids

    """ More sofisticated way to get cetroids """
    """
    def get_term_weight_in_prototype(term, docs_in_category, docs_out_category):
        b = 16
        c = 4
        category_size = float(len(docs_in_category))
        other_categories_size = float(len(docs_out_category))

        avg_term_weight_in_category = docs_in_category[term].sum()/category_size
        avg_term_weight_out_category = docs_out_category[term].sum()/other_categories_size

        prototype_term_weight = b*avg_term_weight_in_category - c*avg_term_weight_out_category
        if prototype_term_weight < 0:
            prototype_term_weight = 0
        return prototype_term_weight


    def get_prototype_vector(bag_of_words):
        PROTOTYPES = dict()
        categories = bag_of_words['CATEGORY'].unique()

        for category in categories:
            prototype = dict()
            docs_in_category = bag_of_words[bag_of_words['CATEGORY'] == category]
            docs_out_category = bag_of_words[bag_of_words['CATEGORY'] != category]

            for term in docs_in_category:
                if term != 'CATEGORY':
                    prototype[term] = get_term_weight_in_prototype(term, docs_in_category, docs_out_category)
            PROTOTYPES[category] = prototype

        return PROTOTYPES

    """
    def classify(self, test_data):
        predictions = []

        for doc in test_data:
            text = process_text(doc)
            present_in_vocabulary = False
            for term in text:
                if term in self.vocabulary:
                    present_in_vocabulary = True
                    break

            if not present_in_vocabulary:
                print "ERROR: This text can not be classified. Any of terms is not present in vocabulary of Classifier."
                prediction = None
            else:
                counts = self.get_word_counts(text)
                weights_vector = Rocchio_classifier.get_weights_vector(counts, self.N, self.word_counts)
                weights_vector = {term: weights_vector[term] if term in weights_vector else 0.0 for term in self.vocabulary}

                magnitudes = []
                for category in self.centroids:
                    centroid = self.centroids[category]
                    difference = [centroid[term] - weights_vector[term] for term in centroid]
                    magnitude = sqrt(sum([x**2 for x in difference]))
                    magnitudes.append((magnitude, category))

                prediction = sorted(magnitudes, key=lambda tup: tup[0])[0][1]
            predictions.append({'text': doc, 'category': prediction})

        return predictions

    @staticmethod
    def print_predictions(predictions):
        for p in predictions:
            print p['text']
            print "Class:", p['category']
            print


### MAIN ###
if __name__ == '__main__':

    def valid_path(filepath):
        if not os.path.exists(filepath):
            raise argparse.ArgumentTypeError("{} is not a valid path".format(filepath))
        else:
            return filepath


    parser = argparse.ArgumentParser()
    parser.add_argument("train_file", help="CSV-file with data. Each row must have the following format: <string>,<class>", type=valid_path)
    parser.add_argument("test_file", help="CSV-file with data. Each row must have the following format: <string>", type=valid_path)
    args = parser.parse_args()


    train_file = args.train_file
    test_file = args.test_file


    train_data = read_data_file(train_file, 'train')
    classifier = Rocchio_classifier(train_data)

    test_data = read_data_file(test_file, 'test')
    predictions = classifier.classify(test_data)
    classifier.print_predictions(predictions)
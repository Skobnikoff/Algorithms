# coding: utf-8
import nltk
import string
from string import digits
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np


data_folder = "Data/"
csv_bag_of_words = "BagOfWords/bag_of_words.csv"



def words_counts_in_one_document(file_path):
    with open(file_path, 'r') as file_object:
        text = file_object.read()

    text = text.translate(None, string.punctuation)  # remove punctuation
    text = text.translate(None, digits)  # remove digits
    tokens = nltk.word_tokenize(text)  # get tokens
    tokens = [re.sub(r'[^\w]', '', t) for t in tokens]  # remove any special characters
    tokens = [token.lower() for token in tokens]  # lowercase tokens
    tokens = [i for i in tokens if i not in stopwords.words('english')]  # remove english stopwords
    tokens = [SnowballStemmer("english").stem(t) for t in tokens]  # get stems
    tokens.sort()

    prev = tokens[0]
    count = 0
    counts = {}

    for t in tokens:
        if t == prev:
            count += 1
            prev = t
        else:
            counts[prev] = count
            count = 1
            prev = t

    counts[prev] = count

    return counts


#===== MAIN =====#
if __name__ == '__main__':

    """ List files in data folder """
    files_to_process = [join(data_folder, f) for f in listdir(data_folder) if isfile(join(data_folder, f))]
    num_of_files = len(files_to_process)

    """ Add word counts of each individual file to a list.
        Get a list of unique words present in the corpus"""
    distinct_tokens = set()
    temp_bag_of_words = []
    for index, file in enumerate(files_to_process):
        counts = words_counts_in_one_document(file)
        temp_bag_of_words.append(counts)
        distinct_tokens.update(counts.keys())

    distinct_tokens = sorted(list(distinct_tokens))

    """ From list of dictionaries to data frame """
    df_bag_of_words = pd.DataFrame(None, columns=distinct_tokens, index=np.array(range(num_of_files)))
    for index, line in enumerate(temp_bag_of_words):
        for word in line:
            df_bag_of_words[word][index] = line[word]

    print df_bag_of_words

    """ Write bag of words to csv file """
    df_bag_of_words.to_csv(csv_bag_of_words, index=False)

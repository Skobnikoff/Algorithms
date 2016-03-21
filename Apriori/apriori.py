# First version: 14.03.2016
# Last version: 21.03.2016
"""
    Python implementation of the Apriori algorithm

    Command line:
        $ python apriori.py datafile.csv -s minSupport  -c minConfidence
        $ python apriori.py datafile.csv -s 0.33 -c 0.7

    Python shell:
        >> from apriori import *
        >> data = read_data_file(datafile.csv)
        >> association_rules = get_association_rules(data, minsup, minconf)
        >> print_rules(association_rules)
"""


from collections import defaultdict
import csv
import argparse
import os
from itertools import chain, combinations

default_min_sup = 0.01
default_min_conf = 0.8



def generate_candidates(Fprev):
    k = len(Fprev[0])+1
    Ck_temp=[]
    i=1
    for f in Fprev:
        for m in range(i, len(Fprev)):
            u = f.union(Fprev[m])

            if len(u) == k and u not in Ck_temp:
                Ck_temp.append(u)
        i+=1

    Ck = []
    for c in Ck_temp:
        k_subsets_of_c = list(chain.from_iterable(combinations(c,n) for n in range(k-1, k)))
        k_subsets_of_c = [frozenset(i) for i in k_subsets_of_c]
        filtered = filter(lambda subset: subset in Fprev, k_subsets_of_c)

        if len(filtered) == len(k_subsets_of_c):
            Ck.append(c)

    return Ck


def candidate_frequency(items, transaction_list):
    Ck = defaultdict(int)
    for i in items:
        for t in transaction_list:
            if i.issubset(t):
                Ck[i] += 1
    return Ck


def frequent_sets(Ck_count, N, MINSUP):
        Fk = []
        for c in Ck_count:
            if Ck_count[c]/float(N) >= MINSUP:
                Fk.append(c)
        return Fk



def get_association_rules(data, MINSUP, MINCONF):
    """
        Apriori algorithm.
        Input:  list of lists,
                min support,
                min confidence
        Output: list of dictionaries
    """
    N = len(data)

    transaction_list = [frozenset(line) for line in data]
    ITEMS = set([frozenset([item]) for line in data for item in line])


    """ PART 1: generate frequent itemsets """
    freq_sets = []
    support_of_all_subsets = defaultdict()
    k = 1
    Fprev = ITEMS
    while len(Fprev) != 0:
        if k==1:
            Ck = ITEMS
        else:
            Ck = generate_candidates(Fprev)

        Ck_count = candidate_frequency(Ck, transaction_list)
        support_of_all_subsets.update(Ck_count)
        Fk = frequent_sets(Ck_count, N, MINSUP)
        freq_sets.append(Fk)

        Fprev = Fk
        k+=1


    """ PART 2: find association rules """
    association_rules = []
    freq_sets_for_rules = freq_sets[1:-1]

    for sets in freq_sets_for_rules:
        for s in sets:
            k = len(s)
            antecedents = list(chain.from_iterable(combinations(s,n) for n in range(1, k)))
            antecedents = [frozenset(i) for i in antecedents]

            consequents = [s-i for i in antecedents]

            for i, antecedent in enumerate(antecedents):
                consequent = consequents[i]
                prefreq = support_of_all_subsets[antecedent ]
                s_freq = support_of_all_subsets[s]
                postfreq = support_of_all_subsets[consequent]/float(N)

                confidence = float(s_freq)/prefreq
                lift = confidence/postfreq

                if confidence >= MINCONF:
                    rule_support = support_of_all_subsets[s]/float(N)
                    rule = {'antecedent': set(antecedent),
                            'consequent': set(consequent),
                            'confidence': confidence,
                            'support': rule_support,
                            'lift': lift
                    }
                    association_rules.append(rule)


    return association_rules


def print_rules(association_rules):

    for rule in association_rules:
        print list(rule['antecedent']), "==>", list(rule['consequent'])
        print "confidence:", rule['confidence']
        print "support:",    rule['support']
        print "lift:",       rule['lift']
        print


def read_data_file(data_file):
    """
        Input:  csv file with sets of items in each row (item appears only once per row),
        Output: list of lists
    """
    with open(data_file, 'rU') as csv_file:
        reader = csv.reader(csv_file)
        data = [row for row in reader]
        data = [filter(lambda x: x != '', line) for line in data]
    return data




if __name__ == '__main__':

    def restricted_float(x):
        x = float(x)
        if x < 0.0 or x > 1.0:
            raise argparse.ArgumentTypeError("{} not in range [0.0, 1.0]".format(x))
        return x


    def valid_path(filepath):
        if not os.path.exists(filepath):
            raise argparse.ArgumentTypeError("{} is not a valid path".format(filepath))
        else:
            return filepath


    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", help="CSV-file with data", type=valid_path)
    parser.add_argument("-s", "--support", help="minimal support for frequent itemsets", type=restricted_float, default=default_min_sup)
    parser.add_argument("-c", "--confidence", help="minimal confidence for association rules", type=restricted_float, default=default_min_conf)
    args = parser.parse_args()


    data_file = args.filepath
    minsup = args.support
    minconf = args.confidence


    data = read_data_file(data_file)
    association_rules = get_association_rules(data, minsup, minconf)
    print_rules(association_rules)



def get_entropy(target):
    """
    Calculate and return the entropy of the given data set.
    Entropy measures how informative or rich in diversity the data set is.
    Higher is the entropy, more heterogeneous the data set is and vice versa.

    :param target: pandas.Series. Contains the classification target we aim to predict.
    :return: int entropy (>=0).
    """
    import math

    entropy = 0
    for klass in set(target):
        subset = target[target == klass]
        klass_weight = len(subset) / len(target)
        entropy += klass_weight * math.log2(1 / klass_weight)

    return round(entropy, 6)

def get_information_gain(data_set, attribute):
    """
    Calculate and return the information gain obtained by splitting the data set on the given attribute.
    Information gain = (entropy (information richness) of the given data set) - (total sum of entropies of all subsets
    obtained by splitting the given data set on the given attribute).
    :return:
    """
    #TODO
    return


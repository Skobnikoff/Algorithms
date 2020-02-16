

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


def get_post_split_entropy(data_set, attr, target_col='class'):
    """
    Calculate and return the entropy of the given data set after it is split on the given attribute.
    This entropy measures how informative or rich in diversity the data set becomes after the split.
    It should be smaller then the entropy of the data set before the split.

    :param data_set: pandas.DataFrame. Contains the training data 'X' AND the classification target 'y'.
    :param attr: str. The name of the attribute (data frame column name) to split the data frame on.
    :param target_col: str. The name of a column that contains the classification target 'y'.
    :return: int entropy (>=0).
    """
    entropy = 0
    for value in set(data_set[attr]):
        subset = data_set[data_set[attr] == value].copy()
        value_weight = len(subset) / len(data_set)
        entropy += value_weight * get_entropy(subset[target_col])

    return round(entropy, 6)

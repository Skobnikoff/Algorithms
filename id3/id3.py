import pandas

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


def get_best_split(data_set, unused_attributes, target_col='class'):

    best_information_gain = None
    selected_attr = None

    pre_split_entropy = get_entropy(data_set[target_col])
    print("\n> pre_split_entropy:", pre_split_entropy)

    for attr in unused_attributes:
        print("\n>", attr)

        post_split_entropy = get_post_split_entropy(data_set, attr)
        print(">> post_split_entropy:", post_split_entropy)

        information_gain = round(pre_split_entropy - post_split_entropy, 6)
        print(">> information_gain:", information_gain)

        if best_information_gain is None or information_gain > best_information_gain:
            best_information_gain = information_gain
            selected_attr = attr

    print("\n> best_information_gain:", best_information_gain)
    print("> selected_attr:", selected_attr)

    return selected_attr


class Node:

    def __init__(self, attr=None, branches=None, klass=None):
        self.attr = attr
        self.branches = branches
        self.klass = klass

    def __str__(self):
        if self.klass is not None:
            return "{{Class: {}}}\n".format(self.klass)
        return ("{{Attribute: '{}', Branches: {}}}\n".format(self.attr, self.branches))

    def __repr__(self):
        if self.klass is not None:
            return "{{Class: {}}}\n".format(self.klass)
        return ("{{Attribute: '{}', Branches: {}}}\n".format(self.attr, self.branches))


def run_id3_algorithm(
        data_set: pandas.DataFrame,
        unused_attributes: set,
        unique_attr_values: dict,
        target_col: str = 'class'):

    unique_classes = data_set[target_col].unique()
    if len(unique_classes) == 1:
        Node(klass=unique_classes[0])

    major_class = data_set[target_col].mode()[0]

    if len(unused_attributes) == 0:
        return Node(klass=major_class)

    node = Node()
    node.attr = get_best_split(data_set, unused_attributes)
    node.branches = {}

    for value in unique_attr_values[node.attr]:
        print(value)
        if value not in data_set[node.attr].values:
            node.branches[value] = Node(klass=major_class)
        else:
            node.branches[value] = run_id3_algorithm(data_set=data_set[data_set[node.attr] == value],
                                                     unused_attributes=unused_attributes - {node.attr},
                                                     unique_attr_values=unique_attr_values,
                                                     target_col=target_col)
    return node

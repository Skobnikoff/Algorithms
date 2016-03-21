def max_len_of_consequence(association_rules):
    rule = reduce(lambda rule1, rule2: rule1 if (len_of_consequence(rule1) > len_of_consequence(rule2)) else rule2, association_rules)
    return len_of_consequence(rule)


def len_of_rule(rule):
    return len(rule[0])+len(rule[1])


def len_of_consequence(rule):
    return len(rule[1])

def len_of_antecedent(rule):
    return len(rule[0])


def k_item_rules(association_rules, k):
    k_item_rules = filter(lambda rule: len_of_rule(rule) in k, association_rules)
    return k_item_rules


def k_item_antecedent_rules(association_rules, k):
    k_item_rules = filter(lambda rule: len_of_antecedent(rule) in k, association_rules)
    return k_item_rules


def k_item_consequent_rules(association_rules, k):
    """
    :arg: list of tuples. Each tuple contains: (preset: frozenset, postset: frozenset, confidence: float, support: float)
    :return: list of tuples. Each tuple contains: (preset: frozenset, postset: frozenset, confidence: float, support: float)
    """
    k_item_rules = filter(lambda rule: len_of_consequence(rule) in k, association_rules)
    return k_item_rules


def filter_out(association_rules, items):
    rules = association_rules
    for item in items:
        rules = filter(lambda rule: item not in rule[0].union(rule[1]), rules)
    return rules


def filter_in(association_rules, items):
    rules = association_rules
    for item in items:
        rules = filter(lambda rule: item in rule[0].union(rule[1]), rules)
    return rules

def sort_rules_by(association_rules, criteria):

    if criteria == 'length of reason':
        return sorted(association_rules, key=lambda rule: len(rule[0]))
    elif criteria == 'length of consequence':
        return sorted(association_rules, key=lambda rule: len(rule[1]))
    elif criteria == 'confidence':
        return sorted(association_rules, key=lambda rule: rule[2], reverse=True)
    elif criteria == 'support':
        return sorted(association_rules, key=lambda rule: rule[3], reverse=True)
    elif criteria == 'lift':
        return sorted(association_rules, key=lambda rule: rule[4], reverse=True)
    else:
        print "Specify in 'criteria' argument one of the following : 'length of reason', 'length of consequence', 'confidence', 'support', 'lift"






# #### TEST
#
# association_rules = [
# (frozenset(['Jamaica', 'BLACK']), frozenset(['MBE']), 1.0, 0.011267605633802818),
# (frozenset(['10018']), frozenset(['MBE', 'New York']), 0.6666666666666666, 0.018309859154929577),
# (frozenset(['MBE', '10018']), frozenset(['New York']), 1.0, 0.018309859154929577),
# (frozenset(['10018', 'New York']), frozenset(['MBE']), 0.6666666666666666, 0.018309859154929577),
# (frozenset(['HISPANIC', 'New York']), frozenset(['MBE']), 1.0, 0.05070422535211268),
# (frozenset(['1', '2', '3', '4']), frozenset(['A', 'B', 'C', 'D']), 1.0, 0.021830985915492956),
# (frozenset(['HISPANIC', 'WBE', 'New York']), frozenset(['MBE']), 1.0, 0.014084507042253521),
# (frozenset(['WBE', '10001']), frozenset(['NON-MINORITY', 'New York']), 0.84, 0.014788732394366197),
# (frozenset(['NON-MINORITY']), frozenset(['WBE', 'New York', 'jkl']), 1.0, 0.014788732394366197)]
#
# # for i in  k_item_consequent_rules(association_rules, range(3, max_len_of_consequence(association_rules)+1)):
# #     print(i)
#
# # for i in filter_in(association_rules, ['MBE', 'New York']):
# #     print(i)
#
# criteria = 'support'
# for i in sort_rules_by(association_rules, criteria):
#     print i[3]
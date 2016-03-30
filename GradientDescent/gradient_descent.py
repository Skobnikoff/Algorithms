# Start: 30.03.2016
import csv
import numpy as np
import pprint

data_file = "regression_banchmark.csv"
pp = pprint.PrettyPrinter(indent=4)

class GradientDescentRegressor():

    def __init__(self, data):
        """
        :param data: numpy array
        """
        self.data = np.matrix(data)
        self.target = None
        self.n = data.shape[1]              # number of features (columns, variables)
        self.n_param = self.n+1             # number of parameters (from teta_0 to teta_n)
        self.parameters = np.matrix( np.zeros((1, self.n_param)) )    # initialize n+1 parameters (from teta_0 to teta_1)


    def train(self, target):
        """

        :param target: numpy array
        :return:
        """
        self.target = target


def read_data_file(data_file):
    with open(data_file, 'rU') as csvfile:
        has_header = csv.Sniffer().has_header(csvfile.readline())
        dialect = csv.Sniffer().sniff(csvfile.readline(), [',','\t'])
        csvfile.seek(0)
        reader = csv.reader(csvfile, dialect)
        data = [line for line in reader]

    if has_header:
        data = data[1:]

    return np.array(data, dtype=float)






### MAIN ###
if __name__ == '__main__':

    data = read_data_file(data_file)
    target = data[:,-1:]
    data = data[:,:-1]

    gdp = GradientDescentRegressor(data)
    pp.pprint(gdp.data)

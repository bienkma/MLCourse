__author__ = 'bienkma'
import numpy as np
import csv


def load_data_set():
    """
    Load WDBC dataset link: http://pages.cs.wisc.edu/~olvi/uwmp/cancer.html
    :return: numpy array
    """
    data_set_file_name = "WDBC.dat"
    with open(data_set_file_name, "r") as data_set_file:
        data_set_list = list(csv.reader(data_set_file))

    return np.asarray(data_set_list)


def delta(X, y):
    """
    Delta = (X^T * X)^-1 * (X^T * y)
    X^T is transpose matrix of X matrix
    y is matrix label of X matrix
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    XT = np.transpose(X)
    return np.multiply(np.linalg.inv(np.matmul(XT, X)), (np.matmul(XT, y)))

def convert(Matrix):
    """
    replace M = 1 & B = 0
    """
    for it in xrange(len(Matrix)):
        if Matrix[it] == 'M':
            Matrix[it] = 1
        else:
            Matrix[it] = 0
    return np.asanyarray(Matrix, dtype=float)

if __name__ == '__main__':

    X = load_data_set()[:, 2:]
    y = load_data_set()[:, 1]
    print convert(y)
    print("Delta is {0}".format(delta(X, y)))
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
    return np.dot(np.linalg.inv(np.dot(X.T, X)), (np.dot(X.T, y)))

def convert(Matrix):
    """
    replace M = 1 & B = 0
    """
    for it in xrange(len(Matrix)):
        if Matrix[it] == 'M':
            Matrix[it] = 0
        else:
            Matrix[it] = 1
    return np.asanyarray(Matrix, dtype=float)

def qr_solution(Q,R,y):
    """
    R * delta = Q^T*y
    :return: delta
    """
    vp = np.dot(Q.T, y)
    x = np.linalg.solve(R, vp)
    return x

if __name__ == '__main__':
    bias = np.ones((569,1), dtype=float)
    # Choice 10 feature important
    X = load_data_set()[:, 2:12]
    X = np.append(bias, X, 1)
    y = load_data_set()[:, 1]
    y = convert(y)
    delta_result = delta(X, y)
    print("Standard equations solution. Delta = {0}".format(delta_result))
    # calculate qr
    Q, R = np.linalg.qr(X)
    qr_solution_result = qr_solution(Q, R, y)
    print "QR solution. Delta: {0}".format(qr_solution_result)
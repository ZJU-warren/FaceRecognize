import numpy as np
from keras import backend as K
from keras.layers import Dense
from keras.layers.core import Lambda
from keras.layers.merge import add


_neg = Lambda(lambda x: -x)
_square = Lambda(lambda x: x ** 2)
_sqrt = Lambda(lambda x: K.sqrt(x))
_sum = Dense(1, kernel_initializer='ones', bias_initializer='zeros')
_sum.trainable = False


def compute_accuracy(y_true, y_pred, threshold=1):
    pred = y_pred.ravel() < threshold
    return np.mean(pred == y_true.ravel())


def sum_all(x):
    return _sum(x)


def neg(x):
    return _neg(x)


def subtract(x1, x2):
    return add([x1, neg(x2)])


def square(x):
    return _square(x)


def sqrt(x):
    return _sqrt(x)


def norm(x):
    return sqrt(sum_all(square(x)))


def vec(matrix):
    return matrix.reshape(-1, 1)


def mat(vector, N):
    return vector.reshape(-1, N)


def join(a, b):
    return np.concatenate([a, b])



from functools import reduce
import operator as op


import numpy as np


def interp1(x, xp, fp):
    """
    Interpolate the 1d piecewise linear interpolant to a function with given 
    values at discrete data points 

    :param x: The x-coordinates of the interpolated values
    :param xp: The x-coordinates of the data points
    :param fp: the y-coordinates of the data points. Multiple functions can be
               interpolated, as long as they lie along the last dimension of 
               the array.
    :return: the interpolated values
    """

    # numpy.interp only accepts 1-D sequences, so all the input data will need
    # to be reshaped
    if isinstance(x, np.ndarray) and x.ndim > 1:
        x = np.reshape(x, -1)
    if isinstance(xp, np.ndarray) and xp.ndim > 1:
        xp = np.reshape(xp, -1)
    y_shape = (*fp.shape[:-1], len(x))
    y = np.zeros(y_shape)
    for i_row in np.ndindex(fp.shape[:-1]):
        y[i_row] = np.interp(x, xp, fp[i_row])
    return y


def is_between(x, a, b, inclusive=(True, True)):
    """Determine the elements that are between a and b"""
    ops = tuple(np.less_equal if inc else np.less for inc in inclusive)
    return np.logical_and(ops[0](a, x), ops[1](x, b))


def product(sequence):
    """
    Computes the product of all elements in the sequence

    :param sequence: an iterable containing things that can be multiplied
    :return: the product
    """
    return reduce(op.mul, sequence, 1)


def to_matrix(array: np.ndarray):
    """reshapes an array so that it is 2-dimensional"""
    if array.ndim == 1:
        return array.reshape(1, array.size)
    elif array.ndim == 2:
        return array.reshape(array.shape)
    else:
        return array.reshape(product(array.shape[:-1]), -1)

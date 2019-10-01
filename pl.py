import numpy as np


def p(s, q):
    ss, qs = np.meshgrid(s, q, sparse=False, indexing='ij')

    e = np.exp(ss - qs)
    return e / (e + 1)


def dpds(s, q):
    ss, qs = np.meshgrid(s, q, sparse=False, indexing='ij')

    e = np.exp(ss - qs)

    return e / (e + 1) ** 2


def dpdq(s, q):
    ss, qs = np.meshgrid(s, q, sparse=False, indexing='ij')

    e = np.exp(ss - qs)

    return -e / (e + 1) ** 2
